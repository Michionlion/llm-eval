from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

DEFAULT_SYSTEM = "You are a helpful assistant."
DEFAULT_USER_INSTRUCTIONS = (
    "Keep generating as much plain text as possible. "
    "Do not be concise. Avoid lists or formatting. End only when you run out of tokens.\n"
)
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
)

_SSE_DATA_RE = re.compile(r"^data:\s*(.*)\s*$")


# ----------------------------
# Data models (same intent as old script)
# ----------------------------

@dataclass
class RequestResult:
    job_id: str
    idx: int
    base_url: str
    model: str
    target_prompt_tokens: int
    parallel: int
    runs: int
    warmup: int
    max_output_tokens: int
    temperature: float
    timeout_s: float
    stream_ttft: bool
    include_usage_in_stream: bool
    chars_per_token_est: float
    calibration_prompt_tokens: int
    calibration_char_len: int
    calibration_iters: int
    calibration_ts_utc: str

    run_mode: str  # "stream" | "nonstream"
    warmup_run: bool
    ok: bool
    error: str

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_reasoning: int
    completion_tokens_visible: int

    wall_s: float
    ttft_s: float

    completion_toks_per_s: float
    completion_toks_per_s_reasoning: float
    completion_toks_per_s_visible: float
    prefill_toks_per_s: float
    decode_toks_per_s: float
    decode_toks_per_s_reasoning: float
    decode_toks_per_s_visible: float


@dataclass
class SummaryRow:
    timestamp_utc: str
    base_url: str
    model: str

    # Token-based prompt target (library still builds prompt text under the hood)
    target_prompt_tokens: int

    parallel: int
    runs: int
    warmup: int

    max_output_tokens: int
    temperature: float
    timeout_s: float
    chars_per_token_est: float

    stream_ttft: bool
    include_usage_in_stream: bool
    calibration_tolerance_tokens: int
    calibration_max_iters: int

    calibration_prompt_tokens: int
    calibration_char_len: int
    calibration_iters: int
    calibration_ts_utc: str

    warmup_ttft_count: int
    warmup_ttft_first_s: float
    warmup_ttft_avg_s: float
    warmup_load_est_s: float

    # totals
    request_count: int
    success_count: int
    error_count: int

    prompt_tokens_total: int
    completion_tokens_total: int
    completion_tokens_reasoning_total: int
    completion_tokens_visible_total: int
    wall_s: float

    # aggregate throughput (batch)
    completion_toks_per_s_total: float
    completion_toks_per_s_reasoning_total: float
    completion_toks_per_s_visible_total: float
    stream_toks_per_s_total: float
    nonstream_toks_per_s_total: float
    stream_toks_per_s_reasoning_total: float
    stream_toks_per_s_visible_total: float
    nonstream_toks_per_s_reasoning_total: float
    nonstream_toks_per_s_visible_total: float
    prefill_toks_per_s_total: float

    # per-request distributions (completion tok/s)
    completion_tps_mean: float
    completion_tps_stdev: float
    completion_tps_p50: float
    completion_tps_p90: float
    completion_tps_reasoning_mean: float
    completion_tps_reasoning_stdev: float
    completion_tps_reasoning_p50: float
    completion_tps_reasoning_p90: float
    completion_tps_visible_mean: float
    completion_tps_visible_stdev: float
    completion_tps_visible_p50: float
    completion_tps_visible_p90: float

    # TTFT distributions (seconds)
    ttft_mean: float
    ttft_stdev: float
    ttft_p50: float
    ttft_p90: float

    # prefill/decode approx (only for requests where ttft>0 and usage known)
    prefill_tps_mean: float
    prefill_tps_stdev: float
    prefill_tps_p50: float
    prefill_tps_p90: float
    decode_tps_mean: float
    decode_tps_stdev: float
    decode_tps_p50: float
    decode_tps_p90: float

    errors_json: str  # up to N errors


@dataclass(frozen=True)
class CalibrationKey:
    base_url: str
    model: str
    target_prompt_tokens: int
    system: str
    instructions: str


@dataclass
class CalibrationResult:
    prompt: str
    prompt_tokens: int
    # informational
    char_len: int
    iters: int
    ts_utc: str


# ----------------------------
# Events / callbacks (live)
# ----------------------------

@dataclass
class StreamUpdate:
    job_id: str
    idx: int
    model: str

    phase: str  # "connecting" | "prefill" | "decode" | "done" | "error"
    t_rel_s: float

    # rolling
    ttft_s: float
    recv_chars: int
    recv_chars_visible: int
    recv_chars_reasoning: int
    recv_events: int

    # estimated live decode throughput
    est_decode_toks_per_s: float
    est_decode_visible_toks_per_s: float
    est_decode_reasoning_toks_per_s: float


OnRequestDone = Callable[[RequestResult], Awaitable[None]]
OnStreamUpdate = Callable[[StreamUpdate], Awaitable[None]]
OnWarmupDone = Callable[[RequestResult], Awaitable[None]]
OnStatusUpdate = Callable[[str], Awaitable[None]]


@dataclass
class BenchConfig:
    base_url: str
    api_key: Optional[str]
    model: str

    target_prompt_tokens: int
    parallel: int
    runs: int
    warmup: int

    max_output_tokens: int = 256
    temperature: float = 0.0
    timeout_s: float = 900.0

    stream_ttft: bool = True
    include_usage_in_stream: bool = True
    # fallback_usage removed; recorded requests always include a non-streaming call

    # live estimation heuristic for streaming decode rate
    chars_per_token_est: float = 4.0

    # calibration
    calibration_tolerance_tokens: int = 8
    calibration_max_iters: int = 18


# ----------------------------
# Utilities
# ----------------------------

def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _truncate(s: str, n: int = 700) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _set_max_tokens(payload: Dict[str, Any], max_tokens: int) -> None:
    # LiteLLM expects OpenAI-style params and maps them to provider-specific fields.
    payload["max_tokens"] = max_tokens


def make_messages(prompt: str, system: str = DEFAULT_SYSTEM) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _extract_usage(data: Dict[str, Any]) -> Tuple[int, int, int]:
    usage = data.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    if pt is None or ct is None:
        raise RuntimeError("Missing usage.prompt_tokens or usage.completion_tokens.")
    if tt is None:
        tt = int(pt) + int(ct)
    return int(pt), int(ct), int(tt)


def _extract_usage_with_reasoning(data: Dict[str, Any]) -> Tuple[int, int, int, int]:
    pt, ct, tt = _extract_usage(data)
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens")
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    try:
        reasoning_ct = int(reasoning) if reasoning is not None else 0
    except Exception:
        reasoning_ct = 0
    return pt, ct, tt, reasoning_ct


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_stdev(vals: List[float]) -> float:
    return statistics.pstdev(vals) if len(vals) >= 2 else 0.0


def _derive_rates(pt: int, ct: int, wall: float, ttft: float) -> Tuple[float, float, float]:
    completion_tps = (ct / wall) if (wall > 0 and ct > 0) else 0.0
    prefill_tps = (pt / ttft) if (ttft > 0 and pt > 0) else 0.0
    decode_tps = (ct / (wall - ttft)) if (ttft > 0 and wall > ttft and ct > 0) else 0.0
    return completion_tps, prefill_tps, decode_tps


def _derive_completion_rates(ct: int, wall: float, ttft: float) -> Tuple[float, float]:
    completion_tps = (ct / wall) if (wall > 0 and ct > 0) else 0.0
    decode_tps = (ct / (wall - ttft)) if (ttft > 0 and wall > ttft and ct > 0) else 0.0
    return completion_tps, decode_tps


# ----------------------------
# Prompt construction (still char-based internally; caller only uses tokens)
# ----------------------------

def build_prompt_with_filler(filler_chars: int, instructions: str = DEFAULT_USER_INSTRUCTIONS) -> str:
    header = instructions + "\nBEGIN FILLER\n"
    footer = "\nEND FILLER\n"
    filler_target = max(0, filler_chars)

    reps = (filler_target // len(FILLER_SENTENCE)) + 1
    filler = (FILLER_SENTENCE * reps)[:filler_target]
    return header + filler + footer


def _estimate_filler_chars(cfg: BenchConfig, instructions: str) -> int:
    header = instructions + "\nBEGIN FILLER\n"
    footer = "\nEND FILLER\n"
    overhead_chars = len(header) + len(footer)
    est_total_chars = int(cfg.target_prompt_tokens * cfg.chars_per_token_est)
    return max(0, est_total_chars - overhead_chars)


def _prompt_guess(cfg: BenchConfig, instructions: str) -> str:
    filler_chars = _estimate_filler_chars(cfg, instructions)
    return build_prompt_with_filler(filler_chars, instructions)


def _calibration_fields(calib: CalibrationResult) -> Dict[str, Any]:
    return {
        "calibration_prompt_tokens": calib.prompt_tokens,
        "calibration_char_len": calib.char_len,
        "calibration_iters": calib.iters,
        "calibration_ts_utc": calib.ts_utc,
    }


def _cache_bust_prefix(tag: str) -> str:
    seed = f"{tag}|{time.time_ns()}|{time.perf_counter_ns()}"
    code = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"[bench-cache-bust:{code}]\n"


def _prefix_cache_bust(prompt: str, tag: Optional[str]) -> str:
    if not tag:
        return prompt
    return _cache_bust_prefix(tag) + prompt


# ----------------------------
# HTTP calls
# ----------------------------

async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    api_key: Optional[str],
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = await client.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


async def _usage_prompt_tokens_only(
    client: httpx.AsyncClient,
    cfg: BenchConfig,
    prompt: str,
    system: str,
) -> int:
    """
    Low-cost call to get prompt_tokens quickly.
    Uses max_tokens=1 (still includes full prompt in tokenization, which is what we want).
    """
    url = _join_url(cfg.base_url, "/v1/chat/completions")
    payload = {
        "model": cfg.model,
        "messages": make_messages(prompt, system),
        "temperature": 0.0,
        "stream": False,
    }
    _set_max_tokens(payload, 1)
    data = await _post_json(client, url, cfg.api_key, payload, cfg.timeout_s)
    pt, _ct, _tt = _extract_usage(data)
    return pt


async def _one_request_nonstream(
    client: httpx.AsyncClient,
    cfg: BenchConfig,
    prompt: str,
    system: str,
    *,
    cache_bust_tag: Optional[str] = None,
) -> Tuple[float, Tuple[int, int, int, int]]:
    url = _join_url(cfg.base_url, "/v1/chat/completions")
    prompt_for_request = _prefix_cache_bust(prompt, cache_bust_tag)
    payload = {
        "model": cfg.model,
        "messages": make_messages(prompt_for_request, system),
        "temperature": cfg.temperature,
        "stream": False,
    }
    _set_max_tokens(payload, cfg.max_output_tokens)
    t0 = time.perf_counter()
    data = await _post_json(client, url, cfg.api_key, payload, cfg.timeout_s)
    wall = time.perf_counter() - t0
    usage = _extract_usage_with_reasoning(data)
    return wall, usage


async def _one_request_stream_live(
    client: httpx.AsyncClient,
    cfg: BenchConfig,
    prompt: str,
    system: str,
    *,
    job_id: str,
    idx: int,
    on_stream_update: Optional[OnStreamUpdate],
    cache_bust_tag: Optional[str] = None,
) -> Tuple[float, float, Optional[Tuple[int, int, int, int]]]:
    """
    Streaming request with:
      - TTFT measurement
      - live decode throughput estimate during stream (heuristic)
      - attempts to capture usage from stream when include_usage_in_stream=True
    Returns: (wall_s, ttft_s, usage_tuple_or_none)
    """
    url = _join_url(cfg.base_url, "/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    prompt_for_request = _prefix_cache_bust(prompt, cache_bust_tag)

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": make_messages(prompt_for_request, system),
        "temperature": cfg.temperature,
        "stream": True,
    }
    _set_max_tokens(payload, cfg.max_output_tokens)
    if cfg.include_usage_in_stream:
        payload["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    got_first = False
    ttft = 0.0

    recv_chars = 0
    recv_chars_visible = 0
    recv_chars_reasoning = 0
    recv_events = 0

    usage_pt = usage_ct = usage_tt = 0
    usage_reasoning = 0
    saw_usage = False

    async def emit(phase: str) -> None:
        if not on_stream_update:
            return
        t_rel = time.perf_counter() - t0
        # Live estimate: chars received since TTFT / (t - ttft) / chars_per_token
        est_total = 0.0
        est_visible = 0.0
        est_reasoning = 0.0
        if got_first:
            denom = max(1e-6, t_rel - ttft)
            est_visible = (recv_chars_visible / denom) / max(1e-6, cfg.chars_per_token_est)
            est_reasoning = (recv_chars_reasoning / denom) / max(1e-6, cfg.chars_per_token_est)
            est_total = est_visible + est_reasoning
        await on_stream_update(
            StreamUpdate(
                job_id=job_id,
                idx=idx,
                model=cfg.model,
                phase=phase,
                t_rel_s=t_rel,
                ttft_s=ttft if got_first else 0.0,
                recv_chars=recv_chars,
                recv_chars_visible=recv_chars_visible,
                recv_chars_reasoning=recv_chars_reasoning,
                recv_events=recv_events,
                est_decode_toks_per_s=est_total,
                est_decode_visible_toks_per_s=est_visible,
                est_decode_reasoning_toks_per_s=est_reasoning,
            )
        )

    await emit("connecting")

    try:
        async with client.stream("POST", url, headers=headers, json=payload, timeout=cfg.timeout_s) as r:
            r.raise_for_status()

            async for line in r.aiter_lines():
                if not line:
                    continue
                m = _SSE_DATA_RE.match(line)
                if not m:
                    continue
                raw = m.group(1)
                if raw == "[DONE]":
                    break

                try:
                    chunk = json.loads(raw)
                except Exception:
                    continue

                # token deltas (handle content, tool calls, or other payloads)
                choices = chunk.get("choices") or []
                if choices:
                    choice0 = choices[0] or {}
                    delta = (choice0.get("delta") or {})
                    message = (choice0.get("message") or {})
                    content = delta.get("content")
                    msg_content = message.get("content")
                    text = choice0.get("text")
                    reasoning = delta.get("reasoning_content") or message.get("reasoning_content")
                    tool_calls = delta.get("tool_calls") or message.get("tool_calls")
                    func_call = delta.get("function_call") or message.get("function_call")

                    if content is not None:
                        recv_events += 1
                        recv_chars += len(content)
                        recv_chars_visible += len(content)
                    if msg_content is not None:
                        recv_events += 1
                        recv_chars += len(msg_content)
                        recv_chars_visible += len(msg_content)
                    if text is not None:
                        recv_events += 1
                        recv_chars += len(text)
                        recv_chars_visible += len(text)
                    if reasoning is not None:
                        recv_events += 1
                        recv_chars += len(reasoning)
                        recv_chars_reasoning += len(reasoning)

                    has_payload = (
                        content is not None
                        or msg_content is not None
                        or text is not None
                        or reasoning is not None
                        or tool_calls
                        or func_call
                    )
                    if not got_first and has_payload:
                        got_first = True
                        ttft = time.perf_counter() - t0
                        await emit("prefill")
                    elif got_first and has_payload:
                        await emit("decode")

                # usage (often appears only near the end)
                if "usage" in chunk and chunk["usage"]:
                    try:
                        u = chunk["usage"]
                        usage_pt = int(u.get("prompt_tokens") or 0)
                        usage_ct = int(u.get("completion_tokens") or 0)
                        usage_tt = int(u.get("total_tokens") or (usage_pt + usage_ct))
                        details = u.get("completion_tokens_details") or {}
                        usage_reasoning = int(details.get("reasoning_tokens") or u.get("reasoning_tokens") or 0)
                        saw_usage = True
                    except Exception:
                        pass

        wall = time.perf_counter() - t0
        usage: Optional[Tuple[int, int, int, int]] = (
            (usage_pt, usage_ct, usage_tt, usage_reasoning) if saw_usage else None
        )
        await emit("done")
        return wall, (ttft if got_first else 0.0), usage

    except Exception as e:
        await emit("error")
        raise RuntimeError(f"{type(e).__name__}: {_truncate(str(e), 1200)}") from e


# ----------------------------
# Calibration (token-targeted)
# ----------------------------

class CalibrationCache:
    """
    In-memory cache with optional on-disk persistence.
    Stored as JSON mapping key->result.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._mem: Dict[str, CalibrationResult] = {}
        if path:
            self._load()

    def _k(self, key: CalibrationKey) -> str:
        return json.dumps(asdict(key), sort_keys=True)

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self._mem[k] = CalibrationResult(**v)
        except Exception:
            # ignore corrupt cache
            self._mem = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            data = {k: asdict(v) for k, v in self._mem.items()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get(self, key: CalibrationKey) -> Optional[CalibrationResult]:
        return self._mem.get(self._k(key))

    def put(self, key: CalibrationKey, value: CalibrationResult) -> None:
        self._mem[self._k(key)] = value
        self._save()


async def calibrate_prompt(
    client: httpx.AsyncClient,
    cfg: BenchConfig,
    *,
    system: str = DEFAULT_SYSTEM,
    instructions: str = DEFAULT_USER_INSTRUCTIONS,
    cache: Optional[CalibrationCache] = None,
    on_stream_update: Optional[OnStreamUpdate] = None,  # optional: UI can show "calibrating"
) -> CalibrationResult:
    """
    Binary-search filler_chars until usage.prompt_tokens ~= target_prompt_tokens within tolerance.
    Returns a prompt string ready for benchmarking.
    """
    key = CalibrationKey(cfg.base_url, cfg.model, cfg.target_prompt_tokens, system, instructions)
    if cache:
        hit = cache.get(key)
        if hit:
            return hit

    target = cfg.target_prompt_tokens
    tol = max(0, int(cfg.calibration_tolerance_tokens))

    # Establish rough brackets for filler size.
    # Start from a proportional guess: ~4 chars/token ⇒ filler ~ target*4, plus overhead.
    lo_chars = 0
    hi_chars = max(2000, int(target * cfg.chars_per_token_est * 2.0))

    async def pt_for(filler_chars: int) -> int:
        prompt = build_prompt_with_filler(filler_chars, instructions)
        return await _usage_prompt_tokens_only(client, cfg, prompt, system)

    # Grow upper bound until we exceed target (or cap iterations).
    iters = 0
    pt_hi = await pt_for(hi_chars)
    iters += 1
    while pt_hi < target and iters < 8:
        hi_chars *= 2
        pt_hi = await pt_for(hi_chars)
        iters += 1

    # Binary search in [lo, hi]
    best_chars = hi_chars
    best_pt = pt_hi

    for _ in range(cfg.calibration_max_iters):
        mid = (lo_chars + hi_chars) // 2
        pt = await pt_for(mid)
        iters += 1

        # Track best by absolute distance
        if abs(pt - target) < abs(best_pt - target):
            best_chars, best_pt = mid, pt

        if abs(pt - target) <= tol:
            best_chars, best_pt = mid, pt
            break

        if pt < target:
            lo_chars = mid + 1
        else:
            hi_chars = mid - 1

    prompt = build_prompt_with_filler(best_chars, instructions)
    result = CalibrationResult(
        prompt=prompt,
        prompt_tokens=best_pt,
        char_len=len(prompt),
        iters=iters,
        ts_utc=_now_utc_iso(),
    )
    if cache:
        cache.put(key, result)
    return result


# ----------------------------
# Benchmark runner (library)
# ----------------------------

async def run_benchmark(
    cfg: BenchConfig,
    *,
    job_id: str,
    system: str = DEFAULT_SYSTEM,
    instructions: str = DEFAULT_USER_INSTRUCTIONS,
    cache: Optional[CalibrationCache] = None,
    on_request_done: Optional[OnRequestDone] = None,
    on_stream_update: Optional[OnStreamUpdate] = None,
    on_warmup_done: Optional[OnWarmupDone] = None,
    on_status_update: Optional[OnStatusUpdate] = None,
) -> Tuple[SummaryRow, List[RequestResult], CalibrationResult, List[RequestResult]]:
    """
    Full benchmark run (warmup + recorded).
    Returns: (summary, per_request_results, calibration_result, warmup_results)
    """
    prompt: str
    calib: CalibrationResult

    async with httpx.AsyncClient() as client:
        key = CalibrationKey(cfg.base_url, cfg.model, cfg.target_prompt_tokens, system, instructions)
        calib_cached = cache.get(key) if cache else None
        if calib_cached:
            prompt = calib_cached.prompt
            calib = calib_cached
        else:
            prompt = _prompt_guess(cfg, instructions)
            calib = CalibrationResult(
                prompt=prompt,
                prompt_tokens=0,
                char_len=len(prompt),
                iters=0,
                ts_utc=_now_utc_iso(),
            )
        warmup_count = max(3, cfg.warmup)
        meta_common = {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "target_prompt_tokens": cfg.target_prompt_tokens,
            "parallel": cfg.parallel,
            "runs": cfg.runs,
            "warmup": warmup_count,
            "max_output_tokens": cfg.max_output_tokens,
            "temperature": cfg.temperature,
            "timeout_s": cfg.timeout_s,
            "stream_ttft": bool(cfg.stream_ttft),
            "include_usage_in_stream": bool(cfg.include_usage_in_stream),
            "chars_per_token_est": cfg.chars_per_token_est,
        }
        warmup_calib_fields = _calibration_fields(calib)

        # Warmup (sequential)
        warmup_ttf_ts: list[float] = []
        warmup_results: List[RequestResult] = []
        warm_cfg = BenchConfig(**{**cfg.__dict__, "warmup": warmup_count})
        for i in range(warmup_count):
            try:
                if cfg.stream_ttft:
                    _wall, warm_ttft, _usage = await _one_request_stream_live(
                        client,
                        warm_cfg,
                        prompt,
                        system,
                        job_id=job_id,
                        idx=-(i + 1),
                        on_stream_update=None,
                        cache_bust_tag=f"{job_id}:warmup:{i}:stream",
                    )
                    if warm_ttft > 0:
                        warmup_ttf_ts.append(warm_ttft)
                    if _usage is None:
                        pt = ct = tt = rt = 0
                    else:
                        pt, ct, tt, rt = _usage
                    visible_ct = max(0, ct - rt)
                    completion_tps, prefill_tps, decode_tps = _derive_rates(pt, ct, _wall, warm_ttft)
                    completion_tps_reasoning, decode_tps_reasoning = _derive_completion_rates(rt, _wall, warm_ttft)
                    completion_tps_visible, decode_tps_visible = _derive_completion_rates(visible_ct, _wall, warm_ttft)
                    warmup_calib_prompt_tokens = pt if pt > 0 else warmup_calib_fields["calibration_prompt_tokens"]
                    warmup_results.append(
                        RequestResult(
                            job_id=job_id,
                            idx=-(i + 1),
                            **meta_common,
                            **warmup_calib_fields,
                            calibration_prompt_tokens=warmup_calib_prompt_tokens,
                            run_mode="stream",
                            warmup_run=True,
                            ok=True,
                            error="",
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            total_tokens=tt,
                            completion_tokens_reasoning=rt,
                            completion_tokens_visible=visible_ct,
                            wall_s=_wall,
                            ttft_s=warm_ttft,
                            completion_toks_per_s=completion_tps,
                            completion_toks_per_s_reasoning=completion_tps_reasoning,
                            completion_toks_per_s_visible=completion_tps_visible,
                            prefill_toks_per_s=prefill_tps,
                            decode_toks_per_s=decode_tps,
                            decode_toks_per_s_reasoning=decode_tps_reasoning,
                            decode_toks_per_s_visible=decode_tps_visible,
                        )
                    )
                    if on_warmup_done:
                        await on_warmup_done(warmup_results[-1])
                else:
                    _wall, (pt, ct, tt, rt) = await _one_request_nonstream(
                        client,
                        warm_cfg,
                        prompt,
                        system,
                        cache_bust_tag=f"{job_id}:warmup:{i}:nonstream",
                    )
                    visible_ct = max(0, ct - rt)
                    completion_tps, prefill_tps, decode_tps = _derive_rates(pt, ct, _wall, 0.0)
                    completion_tps_reasoning, _decode_tps_reasoning = _derive_completion_rates(rt, _wall, 0.0)
                    completion_tps_visible, _decode_tps_visible = _derive_completion_rates(visible_ct, _wall, 0.0)
                    warmup_calib_prompt_tokens = pt if pt > 0 else warmup_calib_fields["calibration_prompt_tokens"]
                    warmup_results.append(
                        RequestResult(
                            job_id=job_id,
                            idx=-(i + 1),
                            **meta_common,
                            **warmup_calib_fields,
                            calibration_prompt_tokens=warmup_calib_prompt_tokens,
                            run_mode="nonstream",
                            warmup_run=True,
                            ok=True,
                            error="",
                            prompt_tokens=pt,
                            completion_tokens=ct,
                            total_tokens=tt,
                            completion_tokens_reasoning=rt,
                            completion_tokens_visible=visible_ct,
                            wall_s=_wall,
                            ttft_s=0.0,
                            completion_toks_per_s=completion_tps,
                            completion_toks_per_s_reasoning=completion_tps_reasoning,
                            completion_toks_per_s_visible=completion_tps_visible,
                            prefill_toks_per_s=prefill_tps,
                            decode_toks_per_s=0.0,
                            decode_toks_per_s_reasoning=0.0,
                            decode_toks_per_s_visible=0.0,
                        )
                    )
                    if on_warmup_done:
                        await on_warmup_done(warmup_results[-1])
            except Exception as e:
                warmup_results.append(
                    RequestResult(
                        job_id=job_id,
                        idx=-(i + 1),
                        **meta_common,
                        **warmup_calib_fields,
                        run_mode="stream",
                        warmup_run=True,
                        ok=False,
                        error=f"warmup[{i}]: {type(e).__name__}: {_truncate(str(e), 1200)}",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        completion_tokens_reasoning=0,
                        completion_tokens_visible=0,
                        wall_s=0.0,
                        ttft_s=0.0,
                        completion_toks_per_s=0.0,
                        completion_toks_per_s_reasoning=0.0,
                        completion_toks_per_s_visible=0.0,
                        prefill_toks_per_s=0.0,
                        decode_toks_per_s=0.0,
                        decode_toks_per_s_reasoning=0.0,
                        decode_toks_per_s_visible=0.0,
                    )
                )
                if on_warmup_done:
                    await on_warmup_done(warmup_results[-1])

        if not calib_cached:
            warmup_pts = [r.prompt_tokens for r in warmup_results if r.ok and r.prompt_tokens > 0]
            warmup_pt = int(round(sum(warmup_pts) / len(warmup_pts))) if warmup_pts else 0
            warmup_iters = 0
            if warmup_pt == 0:
                try:
                    warmup_pt = await _usage_prompt_tokens_only(client, cfg, prompt, system)
                    warmup_iters = 1
                except Exception:
                    warmup_pt = 0
            tol = max(0, int(cfg.calibration_tolerance_tokens))
            if warmup_pt > 0 and abs(warmup_pt - cfg.target_prompt_tokens) <= tol:
                calib = CalibrationResult(
                    prompt=prompt,
                    prompt_tokens=warmup_pt,
                    char_len=len(prompt),
                    iters=warmup_iters,
                    ts_utc=_now_utc_iso(),
                )
                if cache:
                    cache.put(key, calib)
            else:
                calib = await calibrate_prompt(
                    client,
                    cfg,
                    system=system,
                    instructions=instructions,
                    cache=cache,
                    on_stream_update=on_stream_update,
                )
                prompt = calib.prompt

        record_calib_fields = _calibration_fields(calib)
        results: List[RequestResult] = []
        errors: List[str] = []

        async def one(idx: int) -> None:
            async def attempt_once(attempt_no: int) -> tuple[RequestResult, RequestResult, List[str]]:
                errors_local: List[str] = []
                pair_seq = 0

                async def run_pair(
                    run_label: str,
                ) -> tuple[float, float, str, float, Optional[Tuple[int, int, int, int]], str]:
                    nonlocal pair_seq
                    pair_seq += 1
                    tag_base = (
                        f"{job_id}:req:{idx}:attempt:{attempt_no}:run:{run_label}:pair:{pair_seq}"
                    )
                    stream_wall = 0.0
                    ttft = 0.0
                    stream_err = ""
                    if cfg.stream_ttft:
                        try:
                            stream_wall, ttft, _stream_usage = await _one_request_stream_live(
                                client,
                                cfg,
                                prompt,
                                system,
                                job_id=job_id,
                                idx=idx,
                                on_stream_update=on_stream_update,
                                cache_bust_tag=f"{tag_base}:stream",
                            )
                        except Exception as e:
                            stream_err = f"stream[{idx}]: {type(e).__name__}: {_truncate(str(e), 1200)}"
                    else:
                        stream_err = ""

                    nonstream_wall = 0.0
                    nonstream_usage: Optional[Tuple[int, int, int, int]] = None
                    nonstream_err = ""
                    try:
                        nonstream_wall, nonstream_usage = await _one_request_nonstream(
                            client,
                            cfg,
                            prompt,
                            system,
                            cache_bust_tag=f"{tag_base}:nonstream",
                        )
                    except Exception as e:
                        nonstream_err = f"nonstream[{idx}]: {type(e).__name__}: {_truncate(str(e), 1200)}"

                    return stream_wall, ttft, stream_err, nonstream_wall, nonstream_usage, nonstream_err

                stream_wall, ttft, stream_err, nonstream_wall, nonstream_usage, nonstream_err = await run_pair(
                    "base"
                )

                if stream_err and cfg.stream_ttft:
                    errors_local.append(stream_err)
                if nonstream_err:
                    errors_local.append(nonstream_err)

                if not errors_local and stream_wall > 0 and stream_wall < 10.0:
                    stream_wall2, ttft2, stream_err2, nonstream_wall2, nonstream_usage2, nonstream_err2 = await run_pair(
                        "avg"
                    )
                    if stream_err2 or nonstream_err2:
                        errors_local.append(
                            f"run[{idx}] retry failed: {stream_err2 or nonstream_err2}"
                        )
                    else:
                        stream_wall = (stream_wall + stream_wall2) / 2.0
                        ttft = (ttft + ttft2) / 2.0
                        nonstream_wall = (nonstream_wall + nonstream_wall2) / 2.0
                        if nonstream_usage is not None and nonstream_usage2 is not None:
                            n_pt = int(round((nonstream_usage[0] + nonstream_usage2[0]) / 2.0))
                            n_ct = int(round((nonstream_usage[1] + nonstream_usage2[1]) / 2.0))
                            n_tt = int(round((nonstream_usage[2] + nonstream_usage2[2]) / 2.0))
                            n_rt = int(round((nonstream_usage[3] + nonstream_usage2[3]) / 2.0))
                            nonstream_usage = (n_pt, n_ct, n_tt, n_rt)

                if nonstream_usage is None:
                    n_pt = n_ct = n_tt = n_rt = 0
                else:
                    n_pt, n_ct, n_tt, n_rt = nonstream_usage

                visible_ct = max(0, n_ct - n_rt)
                completion_tps, prefill_tps, decode_tps = _derive_rates(n_pt, n_ct, stream_wall, ttft)
                nonstream_completion_tps, _prefill, _decode = _derive_rates(n_pt, n_ct, nonstream_wall, 0.0)
                completion_tps_reasoning, decode_tps_reasoning = _derive_completion_rates(n_rt, stream_wall, ttft)
                completion_tps_visible, decode_tps_visible = _derive_completion_rates(visible_ct, stream_wall, ttft)
                nonstream_completion_tps_reasoning, _nonstream_decode_tps_reasoning = _derive_completion_rates(
                    n_rt, nonstream_wall, 0.0
                )
                nonstream_completion_tps_visible, _nonstream_decode_tps_visible = _derive_completion_rates(
                    visible_ct, nonstream_wall, 0.0
                )

                stream_rr = RequestResult(
                    job_id=job_id,
                    idx=idx,
                    **meta_common,
                    **record_calib_fields,
                    run_mode="stream",
                    warmup_run=False,
                    ok=(stream_err == ""),
                    error=stream_err,
                    prompt_tokens=n_pt,
                    completion_tokens=n_ct,
                    total_tokens=n_tt,
                    completion_tokens_reasoning=n_rt,
                    completion_tokens_visible=visible_ct,
                    wall_s=stream_wall,
                    ttft_s=ttft,
                    completion_toks_per_s=completion_tps,
                    completion_toks_per_s_reasoning=completion_tps_reasoning,
                    completion_toks_per_s_visible=completion_tps_visible,
                    prefill_toks_per_s=prefill_tps,
                    decode_toks_per_s=decode_tps,
                    decode_toks_per_s_reasoning=decode_tps_reasoning,
                    decode_toks_per_s_visible=decode_tps_visible,
                )

                nonstream_rr = RequestResult(
                    job_id=job_id,
                    idx=idx,
                    **meta_common,
                    **record_calib_fields,
                    run_mode="nonstream",
                    warmup_run=False,
                    ok=(nonstream_err == ""),
                    error=nonstream_err,
                    prompt_tokens=n_pt,
                    completion_tokens=n_ct,
                    total_tokens=n_tt,
                    completion_tokens_reasoning=n_rt,
                    completion_tokens_visible=visible_ct,
                    wall_s=nonstream_wall,
                    ttft_s=0.0,
                    completion_toks_per_s=nonstream_completion_tps,
                    completion_toks_per_s_reasoning=nonstream_completion_tps_reasoning,
                    completion_toks_per_s_visible=nonstream_completion_tps_visible,
                    prefill_toks_per_s=0.0,
                    decode_toks_per_s=0.0,
                    decode_toks_per_s_reasoning=0.0,
                    decode_toks_per_s_visible=0.0,
                )

                return stream_rr, nonstream_rr, errors_local

            retry_delays_s = [1, 5, 15, 30]
            attempt = 0
            while True:
                stream_rr, nonstream_rr, errors_local = await attempt_once(attempt)
                if errors_local:
                    if on_stream_update:
                        await on_stream_update(
                            StreamUpdate(
                                job_id=job_id,
                                idx=idx,
                                model=cfg.model,
                                phase="error",
                                t_rel_s=0.0,
                                ttft_s=0.0,
                                recv_chars=0,
                                recv_chars_visible=0,
                                recv_chars_reasoning=0,
                                recv_events=0,
                                est_decode_toks_per_s=0.0,
                                est_decode_visible_toks_per_s=0.0,
                                est_decode_reasoning_toks_per_s=0.0,
                            )
                        )
                    if attempt < len(retry_delays_s):
                        wait_s = retry_delays_s[attempt]
                        attempt += 1
                        if on_status_update:
                            for remaining in range(wait_s, 0, -1):
                                await on_status_update(
                                    f"Retrying {cfg.model} (req {idx}) in {remaining}s "
                                    f"(retry {attempt}/{len(retry_delays_s)})…"
                                )
                                await asyncio.sleep(1)
                            await on_status_update("")
                        else:
                            await asyncio.sleep(wait_s)
                        continue

                    if on_stream_update:
                        await on_stream_update(
                            StreamUpdate(
                                job_id=job_id,
                                idx=idx,
                                model=cfg.model,
                                phase="error_final",
                                t_rel_s=0.0,
                                ttft_s=0.0,
                                recv_chars=0,
                                recv_chars_visible=0,
                                recv_chars_reasoning=0,
                                recv_events=0,
                                est_decode_toks_per_s=0.0,
                                est_decode_visible_toks_per_s=0.0,
                                est_decode_reasoning_toks_per_s=0.0,
                            )
                        )
                    errors.extend(errors_local)

                results.append(stream_rr)
                results.append(nonstream_rr)
                if on_request_done:
                    await on_request_done(stream_rr)
                    await on_request_done(nonstream_rr)
                return

        # Recorded: wall time across all batches for aggregate throughput
        t0 = time.perf_counter()
        idx = 0
        for _ in range(cfg.runs):
            await asyncio.gather(*[one(idx + j) for j in range(cfg.parallel)])
            idx += cfg.parallel
        wall_total = time.perf_counter() - t0

    stream_results = [r for r in results if r.run_mode == "stream"]
    nonstream_results = [r for r in results if r.run_mode == "nonstream"]
    ok_stream = [r for r in stream_results if r.ok]
    ok_nonstream = [r for r in nonstream_results if r.ok]

    prompt_total = sum(r.prompt_tokens for r in ok_nonstream)
    compl_total = sum(r.completion_tokens for r in ok_nonstream)
    reasoning_total = sum(r.completion_tokens_reasoning for r in ok_nonstream)
    visible_total = sum(r.completion_tokens_visible for r in ok_nonstream)

    stream_wall_sum = sum(r.wall_s for r in ok_stream if r.wall_s > 0)
    nonstream_wall_sum = sum(r.wall_s for r in ok_nonstream if r.wall_s > 0)
    stream_ct_total = sum(r.completion_tokens for r in ok_stream)
    nonstream_ct_total = sum(r.completion_tokens for r in ok_nonstream)
    stream_reasoning_ct_total = sum(r.completion_tokens_reasoning for r in ok_stream)
    nonstream_reasoning_ct_total = sum(r.completion_tokens_reasoning for r in ok_nonstream)
    stream_visible_ct_total = sum(r.completion_tokens_visible for r in ok_stream)
    nonstream_visible_ct_total = sum(r.completion_tokens_visible for r in ok_nonstream)

    stream_tps_total = (stream_ct_total / stream_wall_sum) if stream_wall_sum > 0 else 0.0
    nonstream_tps_total = (nonstream_ct_total / nonstream_wall_sum) if nonstream_wall_sum > 0 else 0.0
    stream_reasoning_tps_total = (
        (stream_reasoning_ct_total / stream_wall_sum) if stream_wall_sum > 0 else 0.0
    )
    nonstream_reasoning_tps_total = (
        (nonstream_reasoning_ct_total / nonstream_wall_sum) if nonstream_wall_sum > 0 else 0.0
    )
    stream_visible_tps_total = (
        (stream_visible_ct_total / stream_wall_sum) if stream_wall_sum > 0 else 0.0
    )
    nonstream_visible_tps_total = (
        (nonstream_visible_ct_total / nonstream_wall_sum) if nonstream_wall_sum > 0 else 0.0
    )

    tps_vals = [v for v in (stream_tps_total, nonstream_tps_total) if v > 0]
    completion_tps_total = (sum(tps_vals) / len(tps_vals)) if tps_vals else 0.0
    reasoning_tps_vals = [v for v in (stream_reasoning_tps_total, nonstream_reasoning_tps_total) if v > 0]
    completion_tps_reasoning_total = (sum(reasoning_tps_vals) / len(reasoning_tps_vals)) if reasoning_tps_vals else 0.0
    visible_tps_vals = [v for v in (stream_visible_tps_total, nonstream_visible_tps_total) if v > 0]
    completion_tps_visible_total = (sum(visible_tps_vals) / len(visible_tps_vals)) if visible_tps_vals else 0.0

    prefill_ttft_sum = sum(r.ttft_s for r in ok_stream if r.ttft_s > 0 and r.prompt_tokens > 0)
    prefill_prompt_sum = sum(r.prompt_tokens for r in ok_stream if r.ttft_s > 0 and r.prompt_tokens > 0)
    prefill_tps_total = (prefill_prompt_sum / prefill_ttft_sum) if prefill_ttft_sum > 0 else 0.0

    compl_tps_vals = sorted([r.completion_toks_per_s for r in ok_stream if r.completion_toks_per_s > 0])
    compl_tps_reasoning_vals = sorted(
        [r.completion_toks_per_s_reasoning for r in ok_stream if r.completion_toks_per_s_reasoning > 0]
    )
    compl_tps_visible_vals = sorted(
        [r.completion_toks_per_s_visible for r in ok_stream if r.completion_toks_per_s_visible > 0]
    )
    ttft_vals = sorted([r.ttft_s for r in ok_stream if r.ttft_s > 0])
    prefill_vals = sorted([r.prefill_toks_per_s for r in ok_stream if r.prefill_toks_per_s > 0])
    decode_vals = sorted([r.decode_toks_per_s for r in ok_stream if r.decode_toks_per_s > 0])

    warmup_ttft_count = len(warmup_ttf_ts)
    warmup_ttft_first_s = warmup_ttf_ts[0] if warmup_ttf_ts else 0.0
    warmup_ttft_avg_s = _safe_mean(warmup_ttf_ts[1:]) if len(warmup_ttf_ts) > 1 else 0.0
    warmup_load_est_s = (
        max(0.0, warmup_ttft_first_s - warmup_ttft_avg_s) if len(warmup_ttf_ts) > 1 else 0.0
    )

    request_count = cfg.runs * cfg.parallel
    ok_by_idx: Dict[int, Dict[str, bool]] = {}
    for r in stream_results:
        ok_by_idx.setdefault(r.idx, {})["stream"] = r.ok
    for r in nonstream_results:
        ok_by_idx.setdefault(r.idx, {})["nonstream"] = r.ok
    success_count = sum(
        1 for status in ok_by_idx.values() if status.get("stream") and status.get("nonstream")
    )
    error_count = max(0, request_count - success_count)

    summary = SummaryRow(
        timestamp_utc=_now_utc_iso(),
        base_url=cfg.base_url,
        model=cfg.model,
        target_prompt_tokens=cfg.target_prompt_tokens,
        parallel=cfg.parallel,
        runs=cfg.runs,
        warmup=warmup_count,
        max_output_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
        timeout_s=cfg.timeout_s,
        chars_per_token_est=cfg.chars_per_token_est,
        stream_ttft=bool(cfg.stream_ttft),
        include_usage_in_stream=bool(cfg.include_usage_in_stream),
        calibration_tolerance_tokens=cfg.calibration_tolerance_tokens,
        calibration_max_iters=cfg.calibration_max_iters,
        calibration_prompt_tokens=calib.prompt_tokens,
        calibration_char_len=calib.char_len,
        calibration_iters=calib.iters,
        calibration_ts_utc=calib.ts_utc,
        warmup_ttft_count=warmup_ttft_count,
        warmup_ttft_first_s=warmup_ttft_first_s,
        warmup_ttft_avg_s=warmup_ttft_avg_s,
        warmup_load_est_s=warmup_load_est_s,

        request_count=request_count,
        success_count=success_count,
        error_count=error_count,

        prompt_tokens_total=prompt_total,
        completion_tokens_total=compl_total,
        completion_tokens_reasoning_total=reasoning_total,
        completion_tokens_visible_total=visible_total,
        wall_s=wall_total,

        completion_toks_per_s_total=completion_tps_total,
        completion_toks_per_s_reasoning_total=completion_tps_reasoning_total,
        completion_toks_per_s_visible_total=completion_tps_visible_total,
        stream_toks_per_s_total=stream_tps_total,
        nonstream_toks_per_s_total=nonstream_tps_total,
        stream_toks_per_s_reasoning_total=stream_reasoning_tps_total,
        stream_toks_per_s_visible_total=stream_visible_tps_total,
        nonstream_toks_per_s_reasoning_total=nonstream_reasoning_tps_total,
        nonstream_toks_per_s_visible_total=nonstream_visible_tps_total,
        prefill_toks_per_s_total=prefill_tps_total,

        completion_tps_mean=_safe_mean(compl_tps_vals),
        completion_tps_stdev=_safe_stdev(compl_tps_vals),
        completion_tps_p50=_percentile(compl_tps_vals, 0.50),
        completion_tps_p90=_percentile(compl_tps_vals, 0.90),
        completion_tps_reasoning_mean=_safe_mean(compl_tps_reasoning_vals),
        completion_tps_reasoning_stdev=_safe_stdev(compl_tps_reasoning_vals),
        completion_tps_reasoning_p50=_percentile(compl_tps_reasoning_vals, 0.50),
        completion_tps_reasoning_p90=_percentile(compl_tps_reasoning_vals, 0.90),
        completion_tps_visible_mean=_safe_mean(compl_tps_visible_vals),
        completion_tps_visible_stdev=_safe_stdev(compl_tps_visible_vals),
        completion_tps_visible_p50=_percentile(compl_tps_visible_vals, 0.50),
        completion_tps_visible_p90=_percentile(compl_tps_visible_vals, 0.90),

        ttft_mean=_safe_mean(ttft_vals),
        ttft_stdev=_safe_stdev(ttft_vals),
        ttft_p50=_percentile(ttft_vals, 0.50),
        ttft_p90=_percentile(ttft_vals, 0.90),

        prefill_tps_mean=_safe_mean(prefill_vals),
        prefill_tps_stdev=_safe_stdev(prefill_vals),
        prefill_tps_p50=_percentile(prefill_vals, 0.50),
        prefill_tps_p90=_percentile(prefill_vals, 0.90),
        decode_tps_mean=_safe_mean(decode_vals),
        decode_tps_stdev=_safe_stdev(decode_vals),
        decode_tps_p50=_percentile(decode_vals, 0.50),
        decode_tps_p90=_percentile(decode_vals, 0.90),

        errors_json=json.dumps(errors[:30]),
    )
    return summary, results, calib, warmup_results


# ----------------------------
# CSV helpers (optional; TUI can use these directly)
# ----------------------------

def append_csv(path: Path, row_dict: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        # preserve deterministic column order
        fields = list(row_dict.keys())
        writer = __import__("csv").DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def load_time_row_from_summary(summary: SummaryRow) -> Dict[str, Any]:
    return {
        "timestamp_utc": summary.timestamp_utc,
        "base_url": summary.base_url,
        "model": summary.model,
        "target_prompt_tokens": summary.target_prompt_tokens,
        "parallel": summary.parallel,
        "warmup": summary.warmup,
        "warmup_ttft_count": summary.warmup_ttft_count,
        "warmup_ttft_first_s": summary.warmup_ttft_first_s,
        "warmup_ttft_avg_s": summary.warmup_ttft_avg_s,
        "warmup_load_est_s": summary.warmup_load_est_s,
        "calibration_prompt_tokens": summary.calibration_prompt_tokens,
        "calibration_char_len": summary.calibration_char_len,
        "calibration_iters": summary.calibration_iters,
        "calibration_ts_utc": summary.calibration_ts_utc,
    }
