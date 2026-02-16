# LLMBench TUI — Codebase Context for Codex

This repository is a Textual TUI + benchmarking library for OpenAI‑compatible chat completion endpoints (LiteLLM + Ollama is the common backend). It measures TTFT, streaming speed, and non‑streaming speed across models, parallelism, and context sizes, and writes `requests.csv`, `summary.csv`, and `load_times.csv`.

## Entry Points

- `bench_tui.py` — Textual app (primary UI).
- `bench_lib.py` — Benchmarking library and HTTP logic.
- `run.sh` — `poetry install` then run the TUI.

## High‑Level Flow

1. TUI loads models on startup (`LLMBenchTUI.on_mount` → `load_models`).
2. User selects models + parallelism + input sizes; TUI sweeps context sizes (1/8 → full by default, controlled by `SWEEP_COUNT`).
3. For each (model, ctx, par) job, TUI calls `bench_lib.run_benchmark`.
4. `run_benchmark` does:
   - **Warmup**: sequential streaming requests only (min 3) using a cached or estimated prompt.
   - **Calibration**: seeded from warmup usage; if warmup prompt is outside tolerance, extra calibration calls happen after warmup.
   - **Recorded run**: for each logical request index, performs **two calls**:
     - streaming call (TTFT + streaming wall time)
     - non‑streaming call (usage + non‑stream wall time)
   - Emits **two RequestResult rows per recorded request**: `run_mode="stream"` and `run_mode="nonstream"`.
5. TUI updates the table live, writes CSVs, and displays status/ETA.

## Key Files + Responsibilities

### `bench_tui.py`

- Builds UI: model selection, settings, bench progress table.
- Table columns include: Model, Ctx, Par, State, Reqs, Warmup, Job elapsed, Live est tok/s, Stream tok/s, Total tok/s, Prefill tok/s, TTFT, Errors, Last update.
- `run_pressed` validates config, clamps warmup to ≥3, builds context sweep, starts `_run_benchmarks`.
- `_run_benchmarks` is sequential per (model, ctx, par) to reduce interference.
- **Warmup tracking**: `job_warmup_total` + `job_warmup_done` and live updates via `on_warmup_done`.
- **Progress**: `done_requests` increments on **non‑stream** completion (a logical request is “done” after both calls).
- **Total tok/s (UI)**: mean of stream total and non‑stream total. Stream total is tokens / stream wall; non‑stream total is tokens / non‑stream wall.
- **Errors**: tracked per logical request index (deduped across stream + non‑stream).
- **Retry indicator**: if a request fails, UI state can show “rerunning”.
- **Retry countdown**: status line can show per‑second retry timers (1s/5s/15s/30s).

### `bench_lib.py`

- `run_benchmark` orchestrates warmup-first calibration and recorded runs.
- **Warmup min 3** enforced inside `run_benchmark`.
- **Two‑call recorded requests**:
  - Streaming call: `_one_request_stream_live`
  - Non‑streaming call: `_one_request_nonstream`
  - Returns two RequestResult rows with the same `idx` but different `run_mode`.
- **Token usage**: Non‑stream call is the source of usage; stream rows reuse non‑stream usage for rates.
- **Reasoning vs visible**: completion tokens are split into `reasoning` and `visible` when usage provides reasoning token counts.
- **Summary stats**:
  - `stream_toks_per_s_total` = total stream completion tokens / total stream wall.
  - `nonstream_toks_per_s_total` = total non‑stream completion tokens / total non‑stream wall.
  - `completion_toks_per_s_total` = mean of stream+non‑stream totals (when both exist).
  - `success_count`/`error_count` computed per **logical request index** (stream + non‑stream must both succeed).
  - Distributions (mean/stdev/p50/p90) are for **stream runs**.
- Warmup TTFT fields in summary: first TTFT, avg of subsequent, load estimate (first − avg).
- Retries: per‑request backoff at 1s, 5s, 15s, 30s; after final failure the stream update phase is `"error_final"`.

## CSV Schema Notes

### `requests.csv`

- **Two rows per recorded request** (`run_mode` distinguishes).
- Warmup rows are logged too.
- Important fields:
  - `run_mode`: `"stream"` or `"nonstream"`.
  - `warmup_run`: `True` for warmups.
  - `wall_s`: wall time for that specific run.
  - `ttft_s`: only meaningful for stream rows.
  - `completion_toks_per_s`: per‑run tokens/sec.
  - `completion_tokens_reasoning` / `completion_tokens_visible`: split completion tokens (when available).
  - `completion_toks_per_s_reasoning` / `completion_toks_per_s_visible`: per‑run throughput split.
  - `prefill_toks_per_s` and `decode_toks_per_s`: stream rows only (0 for non‑stream).

### `summary.csv`

- One row per **(model × ctx × par)**.
- Aggregates only recorded runs (excludes warmup).
- `request_count` = total logical requests (runs × parallel).
- `success_count` = logical requests where **both** stream + non‑stream succeeded.
- `completion_toks_per_s_total` = mean(stream_total, nonstream_total).
- Reasoning/visible totals and per‑request throughput stats are also included.

### `load_times.csv`

- One row per **(model × ctx × par)**, derived from warmup TTFT.
- Includes: `warmup_ttft_*`, `warmup_load_est_s`, and calibration metadata.

## Environment / Config

Key env vars (used in TUI):

- `BASE_URL`
- `LITELLM_KEY`
- `TARGET_PROMPT_TOKENS`, `OUT_TOKENS`, `RUNS`, `WARMUP`, `TIMEOUT_S`, `CHARS_PER_TOKEN_EST`, `SWEEP_COUNT`
- CSV paths: `SUMMARY_CSV`, `REQUESTS_CSV`
- `LOAD_TIMES_CSV`
- `CALIBRATION_CACHE` (JSON cache path)

## Design/Behavioral Rationale

- **Two‑call recorded requests**: ensures consistent usage and measures both streaming and non‑stream latency/throughput.
- **Warmup streaming only**: used to estimate model load time via TTFT deltas.
- **Progress/ETA**: based on logical request completion (non‑stream done).
- **Total tok/s in UI**: mean of stream + non‑stream totals (matches `summary.csv`).
- **No fallback usage**: removed entirely; non‑stream call is always done.

## Known “Gotchas”

- `requests.csv` schema has changed multiple times; start fresh files after schema changes.
- `stream` rows reuse non‑stream usage when stream usage is missing.
- Warmup rows are logged to requests.csv and update the UI Warmup column live.

## Where to Look for Changes

- Stream logic: `bench_lib._one_request_stream_live`
- Non‑stream logic: `bench_lib._one_request_nonstream`
- Orchestration: `bench_lib.run_benchmark`
- UI updates + metrics aggregation: `bench_tui._on_request_done`
