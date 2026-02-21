from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from bench_lib import (
    BenchConfig,
    CalibrationCache,
    RequestResult,
    StreamUpdate,
    append_csv,
    load_time_row_from_summary,
    run_benchmark,
)

_STOP_REQUESTED = False


def _now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[{_now_utc_iso()}] Received signal {signum}; will stop after current job.", flush=True)


def _register_signals() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _request_stop)
        except Exception:
            pass


def _load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "base_url",
        "api_key",
        "summary_csv",
        "requests_csv",
        "load_times_csv",
        "models",
        "pars",
        "context_sizes",
        "out_tokens",
        "runs",
        "warmup",
        "timeout_s",
        "chars_per_token_est",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Config missing keys: {', '.join(missing)}")
    return data


def _job_profile_for_context(
    *,
    ctx: int,
    max_ctx: int,
    runs: int,
    out_tokens: int,
    long_ctx_threshold: float,
    long_ctx_runs: int,
    out_tokens_long_ctx: int,
) -> tuple[int, int]:
    ratio = (ctx / max(1, max_ctx))
    if ratio >= long_ctx_threshold:
        return (
            max(1, min(runs, long_ctx_runs)),
            max(1, min(out_tokens, out_tokens_long_ctx)),
        )
    return runs, out_tokens


async def run_headless(config: Dict[str, Any]) -> int:
    base_url = str(config["base_url"])
    api_key = str(config["api_key"])
    summary_csv = Path(str(config["summary_csv"]))
    requests_csv = Path(str(config["requests_csv"]))
    load_times_csv = Path(str(config["load_times_csv"]))
    cache_path_raw = str(config.get("calibration_cache") or "")
    cache = CalibrationCache(Path(cache_path_raw)) if cache_path_raw else None

    models: List[str] = [str(x) for x in config["models"]]
    pars: List[int] = [int(x) for x in config["pars"]]
    context_sizes: List[int] = [int(x) for x in config["context_sizes"]]
    out_tokens = int(config["out_tokens"])
    runs = int(config["runs"])
    out_tokens_long_ctx = int(config.get("out_tokens_long_ctx", max(16, out_tokens // 2)))
    long_ctx_runs = int(config.get("long_ctx_runs", 1))
    long_ctx_threshold = float(config.get("long_ctx_threshold", 0.75))
    warmup = int(config["warmup"])
    timeout_s = float(config["timeout_s"])
    chars_per_token_est = float(config["chars_per_token_est"])

    out_tokens = max(1, out_tokens)
    out_tokens_long_ctx = max(1, out_tokens_long_ctx)
    long_ctx_runs = max(1, long_ctx_runs)
    long_ctx_threshold = min(1.0, max(0.0, long_ctx_threshold))

    total_jobs = len(models) * len(pars) * len(context_sizes)
    max_ctx = max(context_sizes) if context_sizes else 1
    total_requests = 0
    for _m in models:
        for ctx in context_sizes:
            for p in pars:
                job_runs, _job_out_tokens = _job_profile_for_context(
                    ctx=ctx,
                    max_ctx=max_ctx,
                    runs=runs,
                    out_tokens=out_tokens,
                    long_ctx_threshold=long_ctx_threshold,
                    long_ctx_runs=long_ctx_runs,
                    out_tokens_long_ctx=out_tokens_long_ctx,
                )
                total_requests += job_runs * p
    done_requests = 0
    started_at = time.time()

    print(f"[{_now_utc_iso()}] Headless benchmark started.", flush=True)
    print(
        f"[{_now_utc_iso()}] jobs={total_jobs}, logical_requests={total_requests}, "
        f"models={models}, pars={pars}, contexts={context_sizes}",
        flush=True,
    )

    job_no = 0
    for model in models:
        for ctx in context_sizes:
            for par in pars:
                if _STOP_REQUESTED:
                    elapsed = time.time() - started_at
                    print(
                        f"[{_now_utc_iso()}] Stop requested before next job. Elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    return 130

                job_no += 1
                job_id = f"{model}||{par}||{ctx}"
                job_runs, job_out_tokens = _job_profile_for_context(
                    ctx=ctx,
                    max_ctx=max_ctx,
                    runs=runs,
                    out_tokens=out_tokens,
                    long_ctx_threshold=long_ctx_threshold,
                    long_ctx_runs=long_ctx_runs,
                    out_tokens_long_ctx=out_tokens_long_ctx,
                )
                print(
                    f"[{_now_utc_iso()}] [{job_no}/{total_jobs}] Starting model={model} ctx={ctx} par={par} "
                    f"runs={job_runs} out_tokens={job_out_tokens}",
                    flush=True,
                )
                job_start = time.time()

                async def on_request_done(
                    rr: RequestResult,
                    *,
                    _job_id: str = job_id,
                    _requests_csv: Path = requests_csv,
                    _total_requests: int = total_requests,
                ) -> None:
                    nonlocal done_requests
                    append_csv(_requests_csv, asdict(rr))
                    if rr.run_mode == "nonstream":
                        done_requests += 1
                        state = "ok" if rr.ok else "err"
                        print(
                            f"[{_now_utc_iso()}] req {done_requests}/{_total_requests} "
                            f"job={_job_id} idx={rr.idx} {state} wall={rr.wall_s:.2f}s",
                            flush=True,
                        )

                async def on_warmup_done(
                    rr: RequestResult,
                    *,
                    _job_id: str = job_id,
                    _requests_csv: Path = requests_csv,
                ) -> None:
                    append_csv(_requests_csv, asdict(rr))
                    state = "ok" if rr.ok else "err"
                    print(
                        f"[{_now_utc_iso()}] warmup job={_job_id} idx={rr.idx} {state} ttft={rr.ttft_s:.3f}s",
                        flush=True,
                    )

                async def on_stream_update(
                    update: StreamUpdate,
                    *,
                    _job_id: str = job_id,
                ) -> None:
                    if update.phase in {"error", "error_final"}:
                        print(
                            f"[{_now_utc_iso()}] stream {update.phase} job={_job_id} idx={update.idx}",
                            flush=True,
                        )

                async def on_status_update(
                    text: str,
                    *,
                    _job_id: str = job_id,
                ) -> None:
                    if text:
                        print(f"[{_now_utc_iso()}] status {_job_id}: {text}", flush=True)

                cfg = BenchConfig(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    target_prompt_tokens=ctx,
                    parallel=par,
                    runs=job_runs,
                    warmup=warmup,
                    max_output_tokens=job_out_tokens,
                    temperature=0.0,
                    timeout_s=timeout_s,
                    stream_ttft=True,
                    include_usage_in_stream=True,
                    chars_per_token_est=chars_per_token_est,
                )

                try:
                    summary, _results, _calib, _warmups = await run_benchmark(
                        cfg,
                        job_id=job_id,
                        cache=cache,
                        on_request_done=on_request_done,
                        on_stream_update=on_stream_update,
                        on_warmup_done=on_warmup_done,
                        on_status_update=on_status_update,
                    )
                    append_csv(summary_csv, asdict(summary))
                    append_csv(load_times_csv, load_time_row_from_summary(summary))

                    job_elapsed = time.time() - job_start
                    print(
                        f"[{_now_utc_iso()}] [{job_no}/{total_jobs}] Done model={model} ctx={ctx} par={par} "
                        f"success={summary.success_count}/{summary.request_count} "
                        f"total_tps={summary.completion_toks_per_s_total:.2f} "
                        f"elapsed={job_elapsed:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    job_elapsed = time.time() - job_start
                    print(
                        f"[{_now_utc_iso()}] [{job_no}/{total_jobs}] ERROR model={model} ctx={ctx} par={par} "
                        f"{type(exc).__name__}: {exc} elapsed={job_elapsed:.1f}s",
                        flush=True,
                    )

    elapsed = time.time() - started_at
    print(
        f"[{_now_utc_iso()}] Headless benchmark complete. "
        f"logical_requests_done={done_requests}/{total_requests}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless/background benchmark runner")
    parser.add_argument("--config", required=True, help="Path to JSON config produced by bench_tui.py")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 2

    _register_signals()

    try:
        config = _load_config(cfg_path)
        return asyncio.run(run_headless(config))
    except Exception as exc:
        print(f"Fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
