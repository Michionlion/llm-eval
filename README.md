# LLMBench TUI

This project benchmarks OpenAI-compatible chat completion endpoints with a Textual TUI (`bench_tui.py`) and a reusable benchmarking library (`bench_lib.py`). It measures TTFT, streaming decode speed, throughput, and error rates across models, parallelism settings, and prompt context sizes.

## Quick Start

- Set `BASE_URL` to your OpenAI-compatible endpoint and `LITELLM_KEY` (if required).
- Optionally set `SWEEP_COUNT` to control the number of context sweep steps (default 8).
- Run `python bench_tui.py`.

The TUI loads models, lets you select parallelism and target prompt tokens, then sweeps context lengths (1/`SWEEP_COUNT` → full) for each model/parallel combo.

## Outputs

CSV output paths can be overridden with `SUMMARY_CSV`, `REQUESTS_CSV`, and `LOAD_TIMES_CSV`.

### summary.csv

One row per **(model × context size × parallel)** run, with aggregate stats.

Column order (exact, as written):

1. `timestamp_utc` — ISO timestamp for the run.
2. `base_url` — API base URL.
3. `model` — model name/id.
4. `target_prompt_tokens` — requested prompt length in tokens (context sweep value).
5. `parallel` — number of concurrent requests per run.
6. `runs` — number of batches.
7. `warmup` — number of warmup requests.
8. `max_output_tokens` — max tokens per completion.
9. `temperature` — sampling temperature.
10. `timeout_s` — per-request timeout.
11. `chars_per_token_est` — heuristic used for live decode estimate.
12. `stream_ttft` — whether streaming + TTFT measurement was enabled.
13. `include_usage_in_stream` — whether usage was requested in stream responses.
14. `calibration_tolerance_tokens` — target token tolerance for calibration.
15. `calibration_max_iters` — max calibration iterations.
16. `calibration_prompt_tokens` — actual prompt tokens achieved by calibration.
17. `calibration_char_len` — prompt character length.
18. `calibration_iters` — iterations used during calibration.
19. `calibration_ts_utc` — calibration timestamp.
20. `warmup_ttft_count` — number of warmup TTFT samples captured.
21. `warmup_ttft_first_s` — TTFT of the first warmup request.
22. `warmup_ttft_avg_s` — average TTFT of warmup requests excluding the first.
23. `warmup_load_est_s` — estimated model load time (first − avg subsequent warmup TTFT).
24. `request_count` — total recorded request pairs.
25. `success_count` — number of request pairs where both stream + non-stream succeeded.
26. `error_count` — number of request pairs with any failure.
27. `prompt_tokens_total` — total prompt tokens across successful non-streaming runs.
28. `completion_tokens_total` — total completion tokens across successful non-streaming runs.
29. `completion_tokens_reasoning_total` — total reasoning completion tokens across successful non-streaming runs.
30. `completion_tokens_visible_total` — total visible completion tokens across successful non-streaming runs.
31. `wall_s` — elapsed wall time for the recorded run (excludes warmup).
32. `completion_toks_per_s_total` — mean of stream + non-stream aggregate completion tok/s (when both available).
33. `completion_toks_per_s_reasoning_total` — mean aggregate reasoning tok/s (stream + non-stream, when available).
34. `completion_toks_per_s_visible_total` — mean aggregate visible tok/s (stream + non-stream, when available).
35. `stream_toks_per_s_total` — aggregate streaming throughput (completion tokens / total streaming time).
36. `nonstream_toks_per_s_total` — aggregate non-streaming throughput (completion tokens / total non-streaming time).
37. `stream_toks_per_s_reasoning_total` — aggregate streaming reasoning throughput.
38. `stream_toks_per_s_visible_total` — aggregate streaming visible throughput.
39. `nonstream_toks_per_s_reasoning_total` — aggregate non-streaming reasoning throughput.
40. `nonstream_toks_per_s_visible_total` — aggregate non-streaming visible throughput.
41. `prefill_toks_per_s_total` — aggregate prompt prefill throughput (prompt tokens / total TTFT).
42. `completion_tps_mean` — mean per-request completion tokens/sec (streaming runs).
43. `completion_tps_stdev` — stdev of per-request completion tokens/sec (streaming runs).
44. `completion_tps_p50` — median per-request completion tokens/sec (streaming runs).
45. `completion_tps_p90` — p90 per-request completion tokens/sec (streaming runs).
46. `completion_tps_reasoning_mean` — mean per-request reasoning tokens/sec (streaming runs).
47. `completion_tps_reasoning_stdev` — stdev of per-request reasoning tokens/sec (streaming runs).
48. `completion_tps_reasoning_p50` — median per-request reasoning tokens/sec (streaming runs).
49. `completion_tps_reasoning_p90` — p90 per-request reasoning tokens/sec (streaming runs).
50. `completion_tps_visible_mean` — mean per-request visible tokens/sec (streaming runs).
51. `completion_tps_visible_stdev` — stdev of per-request visible tokens/sec (streaming runs).
52. `completion_tps_visible_p50` — median per-request visible tokens/sec (streaming runs).
53. `completion_tps_visible_p90` — p90 per-request visible tokens/sec (streaming runs).
54. `ttft_mean` — mean TTFT (seconds).
55. `ttft_stdev` — stdev of TTFT.
56. `ttft_p50` — median TTFT.
57. `ttft_p90` — p90 TTFT.
58. `prefill_tps_mean` — mean prompt prefill tokens/sec.
59. `prefill_tps_stdev` — stdev of prompt prefill tokens/sec.
60. `prefill_tps_p50` — median prompt prefill tokens/sec.
61. `prefill_tps_p90` — p90 prompt prefill tokens/sec.
62. `decode_tps_mean` — mean streaming decode tokens/sec (excludes TTFT).
63. `decode_tps_stdev` — stdev of streaming decode tokens/sec.
64. `decode_tps_p50` — median streaming decode tokens/sec.
65. `decode_tps_p90` — p90 streaming decode tokens/sec.
66. `errors_json` — JSON array of up to 30 error strings.

### load_times.csv

One row per **(model × context size × parallel)** run, focused on warmup-derived load time.

Column order (exact, as written):

1. `timestamp_utc` — ISO timestamp for the run.
2. `base_url` — API base URL.
3. `model` — model name/id.
4. `target_prompt_tokens` — requested prompt length in tokens (context sweep value).
5. `parallel` — number of concurrent requests per run.
6. `warmup` — number of warmup requests.
7. `warmup_ttft_count` — number of warmup TTFT samples captured.
8. `warmup_ttft_first_s` — TTFT of the first warmup request.
9. `warmup_ttft_avg_s` — average TTFT of warmup requests excluding the first.
10. `warmup_load_est_s` — estimated model load time (first − avg subsequent warmup TTFT).
11. `calibration_prompt_tokens` — actual prompt tokens achieved by calibration.
12. `calibration_char_len` — prompt character length.
13. `calibration_iters` — iterations used during calibration.
14. `calibration_ts_utc` — calibration timestamp.

### requests.csv

One row per **request** (success or failure), including full config metadata.

Column order (exact, as written):

1. `job_id` — job identifier (in this app: `model||parallel||context_tokens`).
2. `idx` — request index within the job.
3. `base_url` — API base URL.
4. `model` — model name/id.
5. `target_prompt_tokens` — target prompt size for this job.
6. `parallel` — parallelism for this job.
7. `runs` — number of batches.
8. `warmup` — warmup requests.
9. `max_output_tokens` — max tokens per completion.
10. `temperature` — sampling temperature.
11. `timeout_s` — timeout for this request.
12. `stream_ttft` — streaming/TTFT enabled.
13. `include_usage_in_stream` — usage requested in stream responses.
14. `chars_per_token_est` — heuristic used for live decode estimate.
15. `calibration_prompt_tokens` — actual calibrated prompt token count.
16. `calibration_char_len` — calibrated prompt length in characters.
17. `calibration_iters` — calibration iterations.
18. `calibration_ts_utc` — calibration timestamp.
19. `run_mode` — `"stream"` or `"nonstream"`.
20. `warmup_run` — whether this row is from a warmup request.
21. `ok` — success flag.
22. `error` — error string (empty if ok).
23. `prompt_tokens` — prompt tokens for this request.
24. `completion_tokens` — completion tokens for this request.
25. `total_tokens` — total tokens for this request.
26. `completion_tokens_reasoning` — reasoning completion tokens for this request.
27. `completion_tokens_visible` — visible completion tokens for this request.
28. `wall_s` — wall time for this run (streaming or non-streaming).
29. `ttft_s` — time to first token (seconds, stream runs only).
30. `completion_toks_per_s` — completion tokens / wall time (per run).
31. `completion_toks_per_s_reasoning` — reasoning tokens / wall time (per run).
32. `completion_toks_per_s_visible` — visible tokens / wall time (per run).
33. `prefill_toks_per_s` — prompt tokens / TTFT (stream runs only).
34. `decode_toks_per_s` — completion tokens / (wall time − TTFT) (stream runs only).
35. `decode_toks_per_s_reasoning` — reasoning tokens / (wall time − TTFT) (stream runs only).
36. `decode_toks_per_s_visible` — visible tokens / (wall time − TTFT) (stream runs only).

### calibration_cache.json

On-disk cache of calibration results keyed by base URL, model, and target prompt tokens. This prevents re-calibrating prompts across runs.

## Notes

- Each recorded request runs **two calls**: one streaming (for TTFT/stream timing) and one non-streaming (for usage). Warmups remain streaming-only.
- Calibration is now seeded from warmup runs: the first requests are warmups, and additional calibration calls only happen after warmup when the warmup prompt is outside tolerance.
- The calibrated prompt **text** itself is intentionally not written to CSV to keep files small. If you want it included, we can add an opt-in column.
- Errors are retried with backoff (1s, 5s, 15s, 30s). In the TUI, the Status line shows a per-second countdown before each retry.
- For thinking models that emit `reasoning_content`, streaming TTFT/live estimates count reasoning output as well as visible content; usage splits reasoning vs visible tokens when available.
