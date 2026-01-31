from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import httpx
from dotenv import load_dotenv
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    ProgressBar,
    RichLog,
    SelectionList,
    Static,
)
from rich.text import Text

from bench_lib import (
    BenchConfig,
    CalibrationCache,
    RequestResult,
    StreamUpdate,
    append_csv,
    load_time_row_from_summary,
    run_benchmark,
)

load_dotenv()


def fetch_models_sync(base_url: str, api_key: str) -> List[str]:
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = httpx.get(url, headers=headers, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    return sorted([m["id"] for m in data.get("data", []) if "id" in m])


class LLMBenchTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #pre_run { height: 1fr; }
    #cfg { border: solid green; padding: 1; }
    #model_row { height: 8; }
    #prompt_row > .spin_container { width: 1fr; }
    #settings_row > .spin_container { width: 1fr; }
    .spin_container { width: 1fr; }
    .spin_field { width: 1fr; }
    .spin_field Input { width: 1fr; }
    .spin_buttons { width: 3; }
    .spin_buttons Button { height: 1; padding: 0 0; }
    #bench_panel { height: 1fr; }
    #progress_row { height: 1; }
    #bench_table { height: 1fr; }
    #pre_log { height: 4; }
    DataTable { height: 1fr; }
    .hidden { display: none; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("BASE_URL", "")
        self.api_key = os.getenv("LITELLM_KEY", "")

        self.summary_csv = Path(os.getenv("SUMMARY_CSV", "summary.csv"))
        self.requests_csv = Path(os.getenv("REQUESTS_CSV", "requests.csv"))
        self.load_times_csv = Path(os.getenv("LOAD_TIMES_CSV", "load_times.csv"))
        self.cache_path = Path(os.getenv("CALIBRATION_CACHE", "./.cache/calibration_cache.json"))
        self.cache = CalibrationCache(self.cache_path)

        self.models: List[str] = []
        self.context_sizes: List[int] = []
        self.row_key_to_id: Dict[Tuple[str, int, int], str] = {}
        self._pre_status_has_selection = False
        self._last_selection_status = ""
        self._last_focus_row: str | None = None
        self._run_timer = None
        self._eta_s: float | None = None
        self._status_base = "idle"
        self._status_override: str | None = None
        self._spin_meta = {
            "target_prompt_tokens": {"min": 1, "kind": "int", "mode": "pow2"},
            "sweep_count": {"min": 1, "kind": "int", "mode": "step", "step": 1},
            "runs": {"min": 1, "kind": "int", "mode": "step", "step": 1},
            "warmup": {"min": 1, "kind": "int", "mode": "step", "step": 1},
            "timeout_s": {"min": 1.0, "kind": "float", "mode": "step", "step": 1.0},
            "chars_per_token_est": {"min": 0.1, "kind": "float", "mode": "step", "step": 0.1},
        }
        self.job_start_time: Dict[Tuple[str, int, int], float] = {}
        self.job_done: set[Tuple[str, int, int]] = set()
        self.job_warmup_total: Dict[Tuple[str, int, int], int] = {}
        self.job_warmup_done: Dict[Tuple[str, int, int], int] = {}

        # global progress
        self.total_requests = 0
        self.done_requests = 0
        self.req_wall_samples: List[float] = []
        self.start_time: float | None = None

        # per-job state
        self.job_req_total: Dict[Tuple[str, int, int], int] = {}
        self.job_req_done: Dict[Tuple[str, int, int], int] = {}
        self.job_errs: Dict[Tuple[str, int, int], int] = {}
        self.job_last_ttft: Dict[Tuple[str, int, int], float] = {}
        self.job_last_live_est_tps: Dict[Tuple[str, int, int], float] = {}
        self.job_last_prefill_tps: Dict[Tuple[str, int, int], float] = {}
        self.job_last_stream_tps: Dict[Tuple[str, int, int], float] = {}
        self.job_last_total_tps: Dict[Tuple[str, int, int], float] = {}
        self.job_completion_sum_total: Dict[Tuple[str, int, int], int] = {}
        self.job_completion_sum_nonstream: Dict[Tuple[str, int, int], int] = {}
        self.job_wall_time_sum: Dict[Tuple[str, int, int], float] = {}
        self.job_nonstream_wall_sum: Dict[Tuple[str, int, int], float] = {}
        self.job_completion_sum_stream: Dict[Tuple[str, int, int], int] = {}
        self.job_stream_time_sum: Dict[Tuple[str, int, int], float] = {}
        self.job_prompt_sum: Dict[Tuple[str, int, int], int] = {}
        self.job_ttft_sum: Dict[Tuple[str, int, int], float] = {}
        self.job_stream_wall_by_idx: Dict[Tuple[str, int, int], Dict[int, float]] = {}
        self.job_err_idxs: Dict[Tuple[str, int, int], set[int]] = {}
        self.table_col_keys = {
            "model": "model",
            "ctx": "ctx",
            "par": "par",
            "state": "state",
            "reqs": "reqs",
            "warmup": "warmup",
            "elapsed": "run_elapsed",
            "live_est": "live_est_tps",
            "prefill": "prefill_tps",
            "stream": "stream_tps",
            "total": "total_tps",
            "ttft": "ttft",
            "errors": "errors",
            "updated": "last_update",
        }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with VerticalScroll(id="pre_run"):
            with Vertical(id="cfg"):
                yield Label(f"BASE_URL: {self.base_url}")
                yield Label("Model selection (default all):")

                with Horizontal(id="model_row"):
                    yield SelectionList(id="model_select")

                yield Label("Benchmark settings (token-targeted prompt):")
                with Horizontal(id="prompt_row"):
                    yield from self._spin_input(
                        label="Target prompt tokens",
                        value=os.getenv("TARGET_PROMPT_TOKENS", "16384"),
                        input_id="target_prompt_tokens",
                        placeholder="target_prompt_tokens",
                    )
                    yield from self._spin_input(
                        label="Sweep count",
                        value=os.getenv("SWEEP_COUNT", "8"),
                        input_id="sweep_count",
                        placeholder="sweep_count",
                    )
                with Horizontal(id="settings_row"):
                    yield from self._spin_input(
                        label="Runs",
                        value=os.getenv("RUNS", "2"),
                        input_id="runs",
                        placeholder="runs",
                    )
                    yield from self._spin_input(
                        label="Warmup",
                        value=os.getenv("WARMUP", "3"),
                        input_id="warmup",
                        placeholder="warmup",
                    )
                    yield from self._spin_input(
                        label="Timeout (s)",
                        value=os.getenv("TIMEOUT_S", "900"),
                        input_id="timeout_s",
                        placeholder="timeout_s",
                    )
                    yield from self._spin_input(
                        label="Chars/token est",
                        value=os.getenv("CHARS_PER_TOKEN_EST", "4.0"),
                        input_id="chars_per_token_est",
                        placeholder="chars/token est",
                    )
                yield Horizontal(
                    Checkbox(label="Parallel 1", value=True, id="par1"),
                    Checkbox(label="Parallel 2", value=True, id="par2"),
                    Checkbox(label="Parallel 4", value=False, id="par4"),
                    Checkbox(label="Parallel 8", value=False, id="par8"),
                )

            with Horizontal(id="controls"):
                yield Button("Select All", id="select_all")
                yield Button("Select None", id="select_none")
                yield Button("Run Benchmarks", id="run", variant="success")
            yield Static("Status: idle", id="pre_status_text")
            yield RichLog(id="pre_log", wrap=True, highlight=False, max_lines=200)
            with Horizontal(id="load_models_row", classes="hidden"):
                yield LoadingIndicator(id="load_models_spinner")
                yield Label("Loading models…", id="load_models_label")

        with Vertical(id="bench_panel", classes="hidden"):
            with Horizontal(id="progress_row"):
                yield Static("Status: idle", id="status_text")
                yield ProgressBar(total=100, id="global_bar")
                yield Label("ETA: --", id="eta")

            table = DataTable(id="bench_table")
            table.add_column("Model", key=self.table_col_keys["model"])
            table.add_column("Ctx", key=self.table_col_keys["ctx"])
            table.add_column("Par", key=self.table_col_keys["par"])
            table.add_column("State", key=self.table_col_keys["state"])
            table.add_column("Reqs", key=self.table_col_keys["reqs"])
            table.add_column("Warmup", key=self.table_col_keys["warmup"])
            table.add_column("Job elapsed", key=self.table_col_keys["elapsed"])
            table.add_column("Live est tok/s", key=self.table_col_keys["live_est"])
            table.add_column("Stream tok/s", key=self.table_col_keys["stream"])
            table.add_column("Total tok/s", key=self.table_col_keys["total"])
            table.add_column("Prefill tok/s", key=self.table_col_keys["prefill"])
            table.add_column("TTFT", key=self.table_col_keys["ttft"])
            table.add_column("Errors", key=self.table_col_keys["errors"])
            table.add_column("Last update", key=self.table_col_keys["updated"])
            yield table
            yield Button("Back to setup", id="back_to_setup", classes="hidden")
        yield Footer()

    def _set_status(self, text: str) -> None:
        self._status_base = text
        if self._status_override:
            return
        self.query_one("#status_text", Static).update(f"Status: {text}")

    def _set_status_override(self, text: str) -> None:
        if text:
            self._status_override = text
            self.query_one("#status_text", Static).update(f"Status: {text}")
            return
        self._status_override = None
        self.query_one("#status_text", Static).update(f"Status: {self._status_base}")

    def _set_pre_status(self, text: str) -> None:
        self.query_one("#pre_status_text", Static).update(f"Status: {text}")

    def _spin_input(
        self,
        *,
        label: str,
        value: str,
        input_id: str,
        placeholder: str,
    ) -> ComposeResult:
        with Vertical(classes="spin_container"):
            yield Label(label)
            with Horizontal(classes="spin_field"):
                yield Input(value=value, id=input_id, placeholder=placeholder)
                with Vertical(classes="spin_buttons"):
                    yield Button("▲", id=f"{input_id}__up")
                    yield Button("▼", id=f"{input_id}__down")

    def _format_selected_models(self) -> str:
        models = self._selected_models()
        if not models:
            return "Selected models (0): none"
        max_show = 6
        if len(models) <= max_show:
            return f"Selected models ({len(models)}): " + ", ".join(models)
        shown = ", ".join(models[:max_show])
        return f"Selected models ({len(models)}): {shown}, +{len(models) - max_show} more"

    def _update_selected_models_status(self) -> None:
        if not self.models and not self._pre_status_has_selection:
            return
        status = self._format_selected_models()
        if status == self._last_selection_status:
            return
        self._set_pre_status(status)
        self._log_pre_message(status)
        self._pre_status_has_selection = True
        self._last_selection_status = status

    def _set_loading_models(self, loading: bool) -> None:
        row = self.query_one("#load_models_row", Horizontal)
        if loading:
            row.remove_class("hidden")
        else:
            row.add_class("hidden")

    def _apply_spin_adjust(self, input_id: str, direction: str) -> None:
        meta = self._spin_meta.get(input_id)
        if not meta:
            return
        try:
            input_widget = self.query_one(f"#{input_id}", Input)
        except Exception:
            return
        try:
            val = float(input_widget.value)
        except Exception:
            val = float(meta["min"])
        mode = meta.get("mode", "step")
        if mode == "pow2":
            if direction == "up":
                val *= 2.0
            elif direction == "down":
                val /= 2.0
        else:
            step = float(meta.get("step", 1.0))
            if direction == "up":
                val += step
            elif direction == "down":
                val -= step
        min_val = float(meta["min"])
        if val < min_val:
            val = min_val
        if meta["kind"] == "int":
            input_widget.value = str(int(val))
        else:
            input_widget.value = f"{val:.6g}"

    @on(Button.Pressed)
    def _on_spin_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if not button_id or "__" not in button_id:
            return
        input_id, direction = button_id.rsplit("__", 1)
        if direction not in ("up", "down"):
            return
        if input_id not in self._spin_meta:
            return
        self._apply_spin_adjust(input_id, direction)
        event.stop()

    def _safe_update_cell(self, table: DataTable, row_key: str, column_key: str, value: object) -> None:
        try:
            table.update_cell(row_key, column_key, value)
        except Exception as e:
            self.log.warning("Skipping update for row=%s col=%s: %s", row_key, column_key, e)

    def _focus_row(self, row_key: str) -> None:
        table = self.query_one("#bench_table", DataTable)
        if self._last_focus_row == row_key:
            return
        try:
            row_index = table.get_row_index(row_key)
        except Exception:
            return
        table.move_cursor(row=row_index, column=0, scroll=True)
        self._last_focus_row = row_key

    def _format_elapsed(self, elapsed_s: float | None) -> str:
        if elapsed_s is None or elapsed_s < 0:
            return "--"
        seconds = int(elapsed_s)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _update_run_elapsed(self) -> None:
        if self.start_time is None:
            return
        now = time.time()
        elapsed = self._format_elapsed(now - self.start_time)
        table = self.query_one("#bench_table", DataTable)
        for key, row_key in self.row_key_to_id.items():
            if key in self.job_done:
                continue
            start = self.job_start_time.get(key)
            if start is None:
                continue
            self._safe_update_cell(table, row_key, self.table_col_keys["elapsed"], self._format_elapsed(now - start))

        eta_label = self.query_one("#eta", Label)
        if self._eta_s is None:
            eta_label.update(f"ETA: -- | Elapsed: {elapsed}")
        else:
            eta_label.update(f"ETA: {self._eta_s:,.0f}s | Elapsed: {elapsed}")

    def _state_render(self, state: str) -> object:
        if state == "error":
            return Text(state, style="bold red")
        if state == "rerunning":
            return Text(state, style="bold yellow")
        if state in ("running", "calibrating"):
            return Text(state, style="yellow")
        if state == "done":
            return Text(state, style="green")
        if state == "pending":
            return Text(state, style="dim")
        return state

    def _errors_render(self, errors: int) -> object:
        if errors > 0:
            return Text(str(errors), style="bold red")
        return str(errors)

    def _current_target_prompt_tokens(self) -> int:
        try:
            return max(1, int(self.query_one("#target_prompt_tokens", Input).value))
        except Exception:
            return 1

    def _parse_job_id(self, job_id: str) -> Tuple[str, int, int]:
        parts = job_id.split("||")
        if len(parts) >= 3:
            model, par_s, ctx_s = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            model, par_s = parts
            ctx_s = str(self.context_sizes[0] if self.context_sizes else 0)
        else:
            model = parts[0]
            par_s = "0"
            ctx_s = str(self.context_sizes[0] if self.context_sizes else 0)
        return model, int(par_s), int(ctx_s)

    def _log_pre_message(self, message: str) -> None:
        log = self.query_one("#pre_log", RichLog)
        msg = message.strip().replace("\n", " ")
        if len(msg) > 200:
            msg = msg[:197] + "..."
        log.write(msg)

    def _par_list(self) -> List[int]:
        pars = []
        if self.query_one("#par1", Checkbox).value: pars.append(1)
        if self.query_one("#par2", Checkbox).value: pars.append(2)
        if self.query_one("#par4", Checkbox).value: pars.append(4)
        if self.query_one("#par8", Checkbox).value: pars.append(8)
        return pars

    def _selected_models(self) -> List[str]:
        sel = self.query_one("#model_select", SelectionList)
        # SelectionList stores (value, label) pairs; selected values:
        return [str(v) for v in sel.selected]

    def _refresh_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self.row_key_to_id.clear()

        models = self._selected_models()
        pars = self._par_list()
        context_sizes = self.context_sizes or [self._current_target_prompt_tokens()]

        for m in models:
            for ctx in context_sizes:
                for p in pars:
                    row_key = f"{m}||{p}||{ctx}"
                    table.add_row(
                        m,
                        str(ctx),
                        str(p),
                        self._state_render("pending"),
                        "-",
                        "0/0",
                        "--",
                        "",
                        "",
                        "",
                        "",
                        "",
                        self._errors_render(0),
                        "",
                        key=row_key,
                    )
                    self.row_key_to_id[(m, p, ctx)] = row_key
    
    async def on_mount(self) -> None:
        # Populate the model list on startup.
        self.call_later(self.load_models)

    @on(Button.Pressed, "#load_models")
    def load_models(self) -> None:
        self._set_loading_models(True)
        if not self.base_url:
            self._set_pre_status("Missing BASE_URL in environment")
            self._log_pre_message("Missing BASE_URL in environment")
            self._set_loading_models(False)
            return
        if not self.api_key:
            self._set_pre_status("Missing LITELLM_KEY in environment")
            self._log_pre_message("Missing LITELLM_KEY in environment")
            self._set_loading_models(False)
            return
        self._set_pre_status("Fetching models…")
        self._log_pre_message("Fetching models…")
        self.refresh()

        try:
            models = fetch_models_sync(self.base_url, self.api_key)
            self.models = list(dict.fromkeys(models))
        except Exception as e:
            self.log.error("Failed to fetch models", exc_info=True)
            self._set_pre_status(f"Failed to fetch models: {e}")
            self._log_pre_message(f"Failed to fetch models: {e}")
            return
        finally:
            self._set_loading_models(False)

        sel = self.query_one("#model_select", SelectionList)
        sel.clear_options()
        # default all selected
        sel.add_options([(m, m, True) for m in self.models])

        self._refresh_table()
        self._log_pre_message(f"Loaded {len(self.models)} models")
        self._update_selected_models_status()

    @on(Button.Pressed, "#select_all")
    def select_all(self) -> None:
        sel = self.query_one("#model_select", SelectionList)
        sel.select_all()
        self._refresh_table()
        self._update_selected_models_status()

    @on(Button.Pressed, "#select_none")
    def select_none(self) -> None:
        sel = self.query_one("#model_select", SelectionList)
        sel.deselect_all()
        self._refresh_table()
        self._update_selected_models_status()

    @on(SelectionList.SelectedChanged, "#model_select")
    def on_model_selection_changed(self) -> None:
        self._update_selected_models_status()

    async def _on_stream_update(self, su: StreamUpdate) -> None:
        # job_id format: model||par||ctx
        model, par, ctx = self._parse_job_id(su.job_id)
        key = (model, par, ctx)

        start = time.time() - max(0.0, su.t_rel_s)
        prev_start = self.job_start_time.get(key)
        if prev_start is None or start < prev_start:
            self.job_start_time[key] = start
            if self.start_time is None or start < self.start_time:
                self.start_time = start

        if su.ttft_s > 0:
            self.job_last_ttft[key] = su.ttft_s
        self.job_last_live_est_tps[key] = su.est_decode_toks_per_s

        row_id = self.row_key_to_id.get(key)
        if row_id is None:
            return

        table = self.query_one(DataTable)
        if su.phase in ("prefill", "decode"):
            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("running"))
            self._focus_row(row_id)
        elif su.phase == "error":
            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("rerunning"))
        elif su.phase == "error_final":
            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("error"))
        # show live estimate
        self._safe_update_cell(
            table,
            row_id,
            self.table_col_keys["live_est"],
            f"{self.job_last_live_est_tps.get(key, 0.0):.2f}",
        )
        if key in self.job_last_ttft:
            self._safe_update_cell(table, row_id, self.table_col_keys["ttft"], f"{self.job_last_ttft[key]:.3f}s")
        self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))

    async def _on_status_update(self, text: str) -> None:
        self._set_status_override(text)

    async def _on_request_done(self, rr: RequestResult) -> None:
        model, par, ctx = self._parse_job_id(rr.job_id)
        key = (model, par, ctx)
        is_stream = rr.run_mode == "stream"

        if is_stream:
            start = time.time() - max(0.0, rr.wall_s)
            prev_start = self.job_start_time.get(key)
            if prev_start is None or start < prev_start:
                self.job_start_time[key] = start
                if self.start_time is None or start < self.start_time:
                    self.start_time = start

        if not rr.ok:
            self._log_pre_message(rr.error)
            err_idx = self.job_err_idxs.get(key)
            if err_idx is not None and rr.idx not in err_idx:
                err_idx.add(rr.idx)
                self.job_errs[key] = self.job_errs.get(key, 0) + 1

        # stream metrics
        if is_stream:
            if rr.wall_s > 0:
                self.job_stream_wall_by_idx[key][rr.idx] = rr.wall_s
            if rr.ok and rr.wall_s > 0 and rr.completion_tokens > 0:
                self.job_completion_sum_total[key] = self.job_completion_sum_total.get(key, 0) + rr.completion_tokens
                self.job_wall_time_sum[key] = self.job_wall_time_sum.get(key, 0.0) + rr.wall_s
            stream_time = max(0.0, rr.wall_s - rr.ttft_s)
            if rr.ok and stream_time > 0 and rr.completion_tokens > 0:
                self.job_completion_sum_stream[key] = self.job_completion_sum_stream.get(key, 0) + rr.completion_tokens
                self.job_stream_time_sum[key] = self.job_stream_time_sum.get(key, 0.0) + stream_time
            if rr.ok and rr.ttft_s > 0 and rr.prompt_tokens > 0:
                self.job_prompt_sum[key] = self.job_prompt_sum.get(key, 0) + rr.prompt_tokens
                self.job_ttft_sum[key] = self.job_ttft_sum.get(key, 0.0) + rr.ttft_s
            if rr.ttft_s > 0:
                self.job_last_ttft[key] = rr.ttft_s

        # non-stream metrics (logical request completion)
        if not is_stream:
            self.done_requests += 1
            self.job_req_done[key] = self.job_req_done.get(key, 0) + 1
            if rr.ok and rr.wall_s > 0 and rr.completion_tokens > 0:
                self.job_completion_sum_nonstream[key] = (
                    self.job_completion_sum_nonstream.get(key, 0) + rr.completion_tokens
                )
                self.job_nonstream_wall_sum[key] = self.job_nonstream_wall_sum.get(key, 0.0) + rr.wall_s
            stream_wall = self.job_stream_wall_by_idx.get(key, {}).pop(rr.idx, 0.0)
            total_wall = 0.0
            if rr.wall_s > 0 and stream_wall > 0:
                total_wall = rr.wall_s + stream_wall
            elif rr.wall_s > 0:
                total_wall = rr.wall_s
            elif stream_wall > 0:
                total_wall = stream_wall
            if total_wall > 0:
                self.req_wall_samples.append(total_wall)

        total_time = self.job_wall_time_sum.get(key, 0.0)
        nonstream_time = self.job_nonstream_wall_sum.get(key, 0.0)
        stream_time_sum = self.job_stream_time_sum.get(key, 0.0)
        ttft_sum = self.job_ttft_sum.get(key, 0.0)

        stream_total_tps = (
            self.job_completion_sum_total.get(key, 0) / total_time if total_time > 0 else 0.0
        )
        nonstream_total_tps = (
            self.job_completion_sum_nonstream.get(key, 0) / nonstream_time if nonstream_time > 0 else 0.0
        )
        tps_vals = [v for v in (stream_total_tps, nonstream_total_tps) if v > 0]
        self.job_last_total_tps[key] = (sum(tps_vals) / len(tps_vals)) if tps_vals else 0.0

        self.job_last_stream_tps[key] = (
            self.job_completion_sum_stream.get(key, 0) / stream_time_sum if stream_time_sum > 0 else 0.0
        )
        self.job_last_prefill_tps[key] = (
            self.job_prompt_sum.get(key, 0) / ttft_sum if ttft_sum > 0 else 0.0
        )

        row_id = self.row_key_to_id.get(key)
        if row_id is not None:
            table = self.query_one(DataTable)
            total = self.job_req_total.get(key, 0)
            done = self.job_req_done.get(key, 0)
            errs = self.job_errs.get(key, 0)

            if done >= total and total > 0:
                state = "error" if errs > 0 else "done"
            else:
                state = "running"
            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render(state))
            self._safe_update_cell(table, row_id, self.table_col_keys["reqs"], f"{done}/{total}")
            stream = self.job_last_stream_tps.get(key, 0.0)
            self._safe_update_cell(
                table,
                row_id,
                self.table_col_keys["stream"],
                f"{stream:.2f}",
            )
            total_tps = self.job_last_total_tps.get(key, 0.0)
            self._safe_update_cell(
                table,
                row_id,
                self.table_col_keys["total"],
                f"{total_tps:.2f}",
            )
            prefill = self.job_last_prefill_tps.get(key, 0.0)
            self._safe_update_cell(
                table,
                row_id,
                self.table_col_keys["prefill"],
                f"{prefill:.2f}",
            )
            ttft = self.job_last_ttft.get(key, 0.0)
            if ttft > 0:
                self._safe_update_cell(table, row_id, self.table_col_keys["ttft"], f"{ttft:.3f}s")
            self._safe_update_cell(table, row_id, self.table_col_keys["errors"], self._errors_render(errs))
            self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))

        # global progress / ETA
        bar = self.query_one("#global_bar", ProgressBar)
        pct = int((self.done_requests / max(1, self.total_requests)) * 100)
        bar.update(progress=pct)

        if self.start_time and self.req_wall_samples:
            avg = sum(self.req_wall_samples[-30:]) / min(30, len(self.req_wall_samples))
            remaining = max(0, self.total_requests - self.done_requests)
            self._eta_s = remaining * avg
            self._update_run_elapsed()

        # write per-request CSV
        append_csv(self.requests_csv, asdict(rr))

    async def _on_warmup_done(self, rr: RequestResult) -> None:
        model, par, ctx = self._parse_job_id(rr.job_id)
        key = (model, par, ctx)
        self.job_warmup_done[key] = self.job_warmup_done.get(key, 0) + 1
        if not rr.ok:
            self._log_pre_message(rr.error)
        row_id = self.row_key_to_id.get(key)
        if row_id is not None:
            table = self.query_one(DataTable)
            total = self.job_warmup_total.get(key, 0)
            done = self.job_warmup_done.get(key, 0)
            self._safe_update_cell(table, row_id, self.table_col_keys["warmup"], f"{done}/{total}")
            self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))
        append_csv(self.requests_csv, asdict(rr))

    @on(Button.Pressed, "#run")
    async def run_pressed(self) -> None:
        models = self._selected_models()
        pars = self._par_list()
        if not self.base_url:
            self._set_pre_status("Missing BASE_URL.")
            self._log_pre_message("Missing BASE_URL.")
            return
        if not models:
            self._set_pre_status("No models selected.")
            self._log_pre_message("No models selected.")
            return
        if not pars:
            self._set_pre_status("No parallelism selected.")
            self._log_pre_message("No parallelism selected.")
            return
        if not self.api_key:
            self._set_pre_status("Missing LITELLM_KEY.")
            self._log_pre_message("Missing LITELLM_KEY.")
            return

        try:
            target_prompt_tokens = int(self.query_one("#target_prompt_tokens", Input).value)
            runs = int(self.query_one("#runs", Input).value)
            warmup = int(self.query_one("#warmup", Input).value)
            timeout_s = float(self.query_one("#timeout_s", Input).value)
            chars_per_token_est = float(self.query_one("#chars_per_token_est", Input).value)
            sweep_count = int(self.query_one("#sweep_count", Input).value)
        except Exception as e:
            self._set_pre_status(f"Bad numeric input: {e}")
            self._log_pre_message(f"Bad numeric input: {e}")
            return
        try:
            out_tokens = int(os.getenv("OUT_TOKENS", "512"))
        except Exception as e:
            self._set_pre_status(f"Bad OUT_TOKENS env: {e}")
            self._log_pre_message(f"Bad OUT_TOKENS env: {e}")
            return
        if sweep_count < 1:
            sweep_count = 1
            sweep_input = self.query_one("#sweep_count", Input)
            sweep_input.value = str(sweep_count)
            self._set_pre_status("Sweep count raised to 1 (minimum).")
            self._log_pre_message("Sweep count raised to 1 (minimum).")
        if warmup < 3:
            warmup = 3
            warmup_input = self.query_one("#warmup", Input)
            warmup_input.value = str(warmup)
            self._set_pre_status("Warmup raised to 3 (minimum).")
            self._log_pre_message("Warmup raised to 3 (minimum).")

        self.query_one("#pre_run").add_class("hidden")
        self.query_one("#bench_panel").remove_class("hidden")

        max_tokens = max(1, target_prompt_tokens)
        context_sizes: List[int] = []
        if sweep_count == 1:
            context_sizes = [max_tokens]
        else:
            for i in range(1, sweep_count + 1):
                size = max(1, (max_tokens * i + (sweep_count - 1)) // sweep_count)
                if not context_sizes or size != context_sizes[-1]:
                    context_sizes.append(size)
            if context_sizes[-1] != max_tokens:
                context_sizes.append(max_tokens)
        self.context_sizes = context_sizes

        self.run_worker(
            self._run_benchmarks(
                models,
                pars,
                context_sizes,
                out_tokens,
                runs,
                warmup,
                timeout_s,
                chars_per_token_est,
            ),
            name="benchmarks",
            group="benchmarks",
            exclusive=True,
        )

    async def _run_benchmarks(
        self,
        models: List[str],
        pars: List[int],
        context_sizes: List[int],
        out_tokens: int,
        runs: int,
        warmup: int,
        timeout_s: float,
        chars_per_token_est: float,
    ) -> None:
        # init per-job counters
        self.job_req_total.clear()
        self.job_req_done.clear()
        self.job_errs.clear()
        self.job_last_ttft.clear()
        self.job_last_live_est_tps.clear()
        self.job_last_prefill_tps.clear()
        self.job_last_stream_tps.clear()
        self.job_last_total_tps.clear()
        self.job_completion_sum_total.clear()
        self.job_completion_sum_nonstream.clear()
        self.job_wall_time_sum.clear()
        self.job_nonstream_wall_sum.clear()
        self.job_completion_sum_stream.clear()
        self.job_stream_time_sum.clear()
        self.job_prompt_sum.clear()
        self.job_ttft_sum.clear()
        self.job_stream_wall_by_idx.clear()
        self.job_err_idxs.clear()
        self.job_start_time.clear()
        self.job_done.clear()
        self.job_warmup_total.clear()
        self.job_warmup_done.clear()
        self._eta_s = None

        self.total_requests = 0
        self.query_one("#pre_log", RichLog).clear()
        self.query_one("#back_to_setup", Button).add_class("hidden")
        for m in models:
            for ctx in context_sizes:
                for p in pars:
                    key = (m, p, ctx)
                    self.job_req_total[key] = runs * p
                    self.job_req_done[key] = 0
                    self.job_errs[key] = 0
                    self.job_completion_sum_total[key] = 0
                    self.job_completion_sum_nonstream[key] = 0
                    self.job_wall_time_sum[key] = 0.0
                    self.job_nonstream_wall_sum[key] = 0.0
                    self.job_completion_sum_stream[key] = 0
                    self.job_stream_time_sum[key] = 0.0
                    self.job_prompt_sum[key] = 0
                    self.job_ttft_sum[key] = 0.0
                    self.job_stream_wall_by_idx[key] = {}
                    self.job_err_idxs[key] = set()
                    self.job_warmup_total[key] = warmup
                    self.job_warmup_done[key] = 0
                    self.total_requests += runs * p

        self.done_requests = 0
        self.req_wall_samples = []
        self.start_time = None
        self._last_focus_row = None
        if self._run_timer is not None:
            self._run_timer.stop()
        self._run_timer = self.set_interval(1.0, self._update_run_elapsed, pause=False)

        self._refresh_table()

        self._set_status(
            f"Running {len(models)} models × {len(context_sizes)} contexts × {len(pars)} parallel options ({self.total_requests} requests)…"
        )

        # Run sequentially per (model, parallel) to reduce cross-interference.
        for m in models:
            for ctx in context_sizes:
                for p in pars:
                    job_id = f"{m}||{p}||{ctx}"
                    key = (m, p, ctx)
                    row_id = self.row_key_to_id.get(key)
                    if row_id is not None:
                        table = self.query_one(DataTable)
                        self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("calibrating"))
                        self._safe_update_cell(table, row_id, self.table_col_keys["reqs"], f"0/{self.job_req_total[key]}")
                        self._safe_update_cell(table, row_id, self.table_col_keys["warmup"], f"0/{warmup}")
                        self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))
                        self._focus_row(row_id)
                    cfg = BenchConfig(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        model=m,
                        target_prompt_tokens=ctx,
                        parallel=p,
                        runs=runs,
                        warmup=warmup,
                        max_output_tokens=out_tokens,
                        temperature=0.0,
                        timeout_s=timeout_s,
                        stream_ttft=True,
                        include_usage_in_stream=True,
                        chars_per_token_est=chars_per_token_est,
                    )

                    try:
                        summary, results, calib, warmup_results = await run_benchmark(
                            cfg,
                            job_id=job_id,
                            cache=self.cache,
                            on_request_done=self._on_request_done,
                            on_stream_update=self._on_stream_update,
                            on_warmup_done=self._on_warmup_done,
                            on_status_update=self._on_status_update,
                        )

                        if getattr(summary, "warmup_ttft_count", 0) >= 2:
                            self._log_pre_message(
                                f"Warmup load est {m}: {summary.warmup_load_est_s:.2f}s "
                                f"(first {summary.warmup_ttft_first_s:.2f}s, avg {summary.warmup_ttft_avg_s:.2f}s)"
                            )
                        elif getattr(summary, "warmup_ttft_count", 0) == 1:
                            self._log_pre_message(
                                f"Warmup TTFT {m}: {summary.warmup_ttft_first_s:.2f}s (need 2+ warmups for load est)"
                            )

                        # write one summary row
                        append_csv(self.summary_csv, asdict(summary))
                        append_csv(self.load_times_csv, load_time_row_from_summary(summary))

                        if row_id is not None:
                            table = self.query_one(DataTable)
                            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("done"))
                            self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))
                            if key not in self.job_done and key in self.job_start_time:
                                self._safe_update_cell(
                                    table,
                                    row_id,
                                    self.table_col_keys["elapsed"],
                                    self._format_elapsed(time.time() - self.job_start_time[key]),
                                )
                                self.job_done.add(key)
                    except Exception as e:
                        self.log.error("Benchmark failed for %s (par=%s)", m, p, exc_info=True)
                        self.job_errs[key] = self.job_errs.get(key, 0) + 1
                        self._log_pre_message(f"{m} (par={p}): {type(e).__name__}: {e}")
                        if row_id is not None:
                            table = self.query_one(DataTable)
                            self._safe_update_cell(table, row_id, self.table_col_keys["state"], self._state_render("error"))
                            self._safe_update_cell(table, row_id, self.table_col_keys["errors"], self._errors_render(self.job_errs[key]))
                            self._safe_update_cell(table, row_id, self.table_col_keys["updated"], time.strftime("%H:%M:%S"))
                            if key not in self.job_done and key in self.job_start_time:
                                self._safe_update_cell(
                                    table,
                                    row_id,
                                    self.table_col_keys["elapsed"],
                                    self._format_elapsed(time.time() - self.job_start_time[key]),
                                )
                                self.job_done.add(key)
                        self._set_status(f"Error running {m} (par={p}): {e}")
                        continue

        self._set_status("Complete.")
        if self._run_timer is not None:
            self._run_timer.stop()
            self._run_timer = None
        total_elapsed = self._format_elapsed(time.time() - self.start_time) if self.start_time else "--"
        self._log_pre_message(f"Run complete in {total_elapsed}.")
        self.query_one("#back_to_setup", Button).remove_class("hidden")

        # finalize ETA display
        self.query_one("#eta", Label).update(f"ETA: 0s | Elapsed: {total_elapsed}")

    @on(Button.Pressed, "#back_to_setup")
    def back_to_setup(self) -> None:
        self.query_one("#bench_panel").add_class("hidden")
        self.query_one("#pre_run").remove_class("hidden")
        self._update_selected_models_status()

if __name__ == "__main__":
    LLMBenchTUI().run()
