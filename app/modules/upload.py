"""Tab 0 - Data upload (single file or batch directory)."""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    dispatch_reader,
    raw_has_light,
    raw_summary_row,
    scan_batch_directory,
)
from modules._common import activity_series, empty_fig  # noqa: E402

try:
    from importlib.resources import files as _res_files
except Exception:  # pragma: no cover
    _res_files = None

_FORMAT_CHOICES = {
    "auto": "Auto-detect (by extension)",
    "awd": "AWD (Actiwatch)",
    "agd": "AGD (ActiGraph)",
    "rpx": "RPX / Respironics (.csv)",
    "dqt": "DQT (.csv)",
    "mesa": "MESA (.csv)",
    "tal": "TAL / MotionWare",
    "atr": "ATR (.txt)",
}


# The folder picker runs in a short-lived *subprocess* rather than in-process.
# Creating a tkinter root inside the Shiny server (which owns the asyncio event
# loop and, on macOS, the main thread) is fragile and can wedge the session.
# A subprocess opens the native dialog cleanly and exits as soon as the user
# picks a folder, so nothing tkinter-related lingers in the server process.
_DIALOG_SCRIPT = (
    "import tkinter as tk\n"
    "from tkinter import filedialog\n"
    "r = tk.Tk()\n"
    "r.withdraw()\n"
    "r.wm_attributes('-topmost', True)\n"
    "p = filedialog.askdirectory(title='Select batch directory')\n"
    "r.destroy()\n"
    "print(p or '')\n"
)


def _pick_folder_via_dialog() -> "str | None":
    """Open a native OS folder picker (in a subprocess) and return the path.

    Returns None if the user cancels or no display is available.
    """
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DIALOG_SCRIPT],
            capture_output=True,
            text=True,
            timeout=600,
        )
        path = (proc.stdout or "").strip()
        return path or None
    except Exception:
        return None


# The Browse button is always offered; if the dialog cannot open (e.g. a
# headless server), the click simply falls back to manual entry.
_DIALOG_AVAILABLE = True


@module.ui
def upload_ui():
    # Include upper-case variants so a file picker does not grey out files
    # whose extension is capitalised (e.g. PATIENT.AWD vs .awd).
    accept = sorted(SUPPORTED_EXTENSIONS) + sorted(
        e.upper() for e in SUPPORTED_EXTENSIONS
    )
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_radio_buttons(
                "load_mode",
                "Loading mode",
                {"single": "Single file", "batch": "Batch directory"},
                selected="single",
            ),
            ui.input_select(
                "fmt",
                "File format",
                choices=_FORMAT_CHOICES,
                selected="auto",
            ),
            ui.panel_conditional(
                "input.load_mode === 'single'",
                ui.input_file(
                    "file",
                    "Upload an actigraphy file",
                    accept=accept,
                    multiple=False,
                ),
                ui.input_action_button(
                    "use_example",
                    "Use example file",
                    class_="btn-outline-secondary btn-sm",
                ),
            ),
            ui.panel_conditional(
                "input.load_mode === 'batch'",
                (
                    ui.input_action_button(
                        "browse_btn",
                        "Browse…",
                        class_="btn-outline-secondary btn-sm",
                    )
                    if _DIALOG_AVAILABLE
                    else None
                ),
                # Editable path field: the Browse button pre-fills it, and
                # users can still paste/type a path directly.
                ui.input_text(
                    "batch_dir",
                    "Directory path",
                    placeholder="/path/to/root",
                    width="100%",
                ),
                ui.input_action_button(
                    "load_batch",
                    "Load batch",
                    class_="btn-primary btn-sm",
                ),
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "Both single-file and batch (multi-subject) modes are supported "
                "— use the radio buttons in the sidebar to switch between them. "
                "All subsequent analysis tabs operate on whichever recording is "
                "currently active."
            ),
            class_="text-muted small mb-3",
        ),
        # Loading indicator shown while a file/batch is being read.
        ui.output_ui("load_status"),
        # ---- single-file outputs ----
        ui.panel_conditional(
            "input.load_mode === 'single'",
            ui.h4("Recording summary"),
            ui.output_data_frame("single_summary"),
            ui.hr(),
            ui.input_radio_buttons(
                "plot_channel",
                "Channel",
                {"activity": "Activity", "light": "Light"},
                selected="activity",
                inline=True,
            ),
            output_widget("single_plot"),
        ),
        # ---- batch outputs ----
        ui.panel_conditional(
            "input.load_mode === 'batch'",
            ui.layout_columns(
                ui.value_box("Files loaded", ui.output_text("vb_loaded")),
                ui.value_box("Failed", ui.output_text("vb_failed")),
                ui.value_box("Factors", ui.output_text("vb_factors")),
                col_widths=[4, 4, 4],
            ),
            ui.h4("Loaded recordings"),
            ui.output_data_frame("batch_meta"),
            ui.hr(),
            ui.h4("Activity overview (first subjects)"),
            output_widget("batch_overview_plot"),
        ),
    )


@module.server
def upload_server(input, output, session, rv_single, rv_batch, rv_mode):
    # Local store for the single Raw object's metadata display.
    _single_raw = reactive.Value(None)
    # Persisted last load error so a failure stays visible (not just a toast).
    _load_error = reactive.Value(None)

    # -- keep the global mode in sync and clear the inactive store ----------
    @reactive.effect
    def _sync_mode():
        mode = input.load_mode()
        rv_mode.set(mode)
        if mode == "single":
            rv_batch.set(None)
        else:
            rv_single.set(None)
            _single_raw.set(None)

    # Name of the file currently being loaded (for status / messages).
    _pending_name = reactive.Value("")

    # ---------------------------------------------------------------------
    # Single-file loading (runs off the main thread so the UI stays
    # responsive and a loading indicator can be shown).
    # ---------------------------------------------------------------------
    @reactive.extended_task
    async def _single_task(path: str, fmt):
        # Offload the blocking read to a worker thread so the event loop (and
        # thus the loading indicator) stays responsive while it runs.
        return await asyncio.to_thread(
            lambda: dispatch_reader(Path(path), fmt=fmt)
        )

    @reactive.effect
    @reactive.event(input.file)
    def _on_file():
        finfo = input.file()
        if not finfo:
            return
        src = finfo[0]
        # Preserve the original extension so the dispatcher can detect format.
        suffix = Path(src["name"]).suffix
        tmp = Path(tempfile.mkdtemp()) / (Path(src["name"]).stem + suffix)
        shutil.copy(src["datapath"], tmp)
        fmt = input.fmt()
        fmt = None if fmt == "auto" else fmt
        _pending_name.set(src["name"])
        _single_task(str(tmp), fmt)

    @reactive.effect
    @reactive.event(input.use_example)
    def _on_example():
        if _res_files is None:
            ui.notification_show(
                "Example data not available in this environment.",
                type="warning",
            )
            return
        example = _res_files("circstudio.data") / "example_01.AWD"
        _pending_name.set("example_01.AWD")
        # Force AWD so we never mis-detect the example file.
        _single_task(str(example), "awd")

    @reactive.effect
    def _single_done():
        status = _single_task.status()
        if status in ("initial", "running"):
            return
        name = _pending_name() or "file"
        if status == "error":
            try:
                _single_task.result()
                exc = "unknown error"
            except Exception as e:  # noqa: BLE001
                exc = e
            msg = (
                f"Could not read '{name}'. {exc}\n"
                "Tip: if this is a .csv/.txt/.mtn file, pick the exact format "
                "in the 'File format' dropdown instead of Auto-detect."
            )
            _load_error.set(msg)
            ui.notification_show(msg, type="error", duration=12)
            print("[upload] " + msg)
            return
        # success
        raw = _single_task.result()
        _load_error.set(None)
        rv_single.set(raw)
        _single_raw.set(raw)
        ui.notification_show(f"Loaded {name}.", type="message", duration=4)

    @render.ui
    def load_status():
        single = _single_task.status()
        batch = _batch_task.status()
        if single == "running" or batch == "running":
            what = "file" if single == "running" else "batch directory"
            return ui.div(
                ui.tags.div(
                    class_="spinner-border spinner-border-sm text-primary",
                    role="status",
                ),
                ui.tags.strong(f"  Loading {what}…  ",
                               style="margin-left:10px;"),
                ui.tags.span("please wait", class_="text-muted"),
                class_="d-flex align-items-center my-2",
            )
        return ui.div()

    @render.data_frame
    def single_summary():
        raw = _single_raw.get()
        if raw is None:
            err = _load_error.get()
            if err:
                return pd.DataFrame({"load error": err.split("\n")})
            return pd.DataFrame({"info": ["No file loaded yet."]})
        s = raw_summary_row(raw)
        df = pd.DataFrame(
            {
                "property": [
                    "Recording duration",
                    "Sampling frequency",
                    "Number of epochs",
                    "Start timestamp",
                    "End timestamp",
                    "Light channel present",
                ],
                "value": [
                    s["duration"],
                    s["sampling_freq"],
                    s["n_epochs"],
                    s["start_time"],
                    s["end_time"],
                    s["has_light"],
                ],
            }
        )
        return render.DataGrid(df, width="100%")

    @render_widget
    def single_plot():
        raw = _single_raw.get()
        if raw is None:
            return empty_fig("Upload a file or click 'Use example file'.")
        channel = input.plot_channel()
        if channel == "light" and not raw_has_light(raw):
            return empty_fig("This recording has no light channel.")
        try:
            return raw.plot(mode=channel)
        except Exception as exc:
            return empty_fig(f"Could not plot: {exc}")

    # ---------------------------------------------------------------------
    # Batch loading
    # ---------------------------------------------------------------------
    @reactive.extended_task
    async def _browse_task():
        # Run the (blocking) subprocess dialog off the event loop.
        return await asyncio.to_thread(_pick_folder_via_dialog)

    @reactive.effect
    @reactive.event(input.browse_btn)
    def _open_folder_dialog():
        _browse_task()

    @reactive.effect
    def _browse_done():
        status = _browse_task.status()
        if status in ("initial", "running"):
            return
        if status == "error":
            ui.notification_show(
                "Folder dialog unavailable. Please type or paste the path.",
                type="warning", duration=6,
            )
            return
        path = _browse_task.result()
        if path:
            # Pre-fill the (still editable) path field with the chosen folder.
            ui.update_text("batch_dir", value=path)
        else:
            ui.notification_show(
                "No folder selected — you can also type the path manually.",
                type="warning", duration=5,
            )

    @reactive.extended_task
    async def _batch_task(path: str, fmt):
        return await asyncio.to_thread(
            lambda: scan_batch_directory(Path(path), fmt=fmt)
        )

    @reactive.effect
    @reactive.event(input.load_batch)
    def _on_load_batch():
        path = (input.batch_dir() or "").strip()
        if not path:
            ui.notification_show(
                "Please select or enter a directory path.", type="warning"
            )
            return
        fmt = input.fmt()
        fmt = None if fmt == "auto" else fmt
        _batch_task(path, fmt)

    @reactive.effect
    def _batch_done():
        status = _batch_task.status()
        if status in ("initial", "running"):
            return
        if status == "error":
            try:
                _batch_task.result()
                exc = "unknown error"
            except Exception as e:  # noqa: BLE001
                exc = e
            ui.notification_show(
                f"Batch load failed: {exc}", type="error", duration=8
            )
            return

        collection = _batch_task.result()
        rv_batch.set(collection)
        msg = (
            f"Loaded {len(collection)} file(s); "
            f"{len(collection.errors)} issue(s)."
        )
        if collection.errors:
            preview = "\n".join(collection.errors[:6])
            ui.notification_show(
                msg + "\n" + preview, type="warning", duration=10,
            )
        else:
            ui.notification_show(msg, type="message", duration=5)

    @render.text
    def vb_loaded():
        b = rv_batch()
        return "0" if b is None else str(len(b))

    @render.text
    def vb_failed():
        b = rv_batch()
        return "0" if b is None else str(len(b.errors))

    @render.text
    def vb_factors():
        b = rv_batch()
        if b is None or not b.factor_names:
            return "none"
        return ", ".join(b.factor_names)

    @render.data_frame
    def batch_meta():
        b = rv_batch()
        if b is None or len(b) == 0:
            return pd.DataFrame({"info": ["No batch loaded yet."]})
        return render.DataGrid(b.to_metadata_df(), width="100%")

    @render_widget
    def batch_overview_plot():
        b = rv_batch()
        if b is None or len(b) == 0:
            return empty_fig("Load a batch directory to see the overview.")

        max_rows = 12
        entries = b.entries[:max_rows]

        # Colour subjects by their first factor level.
        levels = b.levels_for_factor(0) if b.factor_names else []
        palette = (
            go.Figure().layout.template.layout.colorway
            or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        )
        color_map = {lv: palette[i % len(palette)] for i, lv in enumerate(levels)}

        titles = [
            f"{e.subject_id}"
            + (f" [{'/'.join(e.factor_levels)}]" if e.factor_levels else "")
            for e in entries
        ]
        # Each subject gets an INDEPENDENT x-axis: recordings may span
        # different dates / periods / lengths, so a shared timeline would
        # misalign them. Separate axes show every individual on its own.
        fig = make_subplots(
            rows=len(entries),
            cols=1,
            shared_xaxes=False,
            subplot_titles=titles,
            vertical_spacing=max(0.02, 0.12 / max(1, len(entries))),
        )
        for i, e in enumerate(entries, start=1):
            act = activity_series(e.raw)
            if act is None:
                continue
            lvl = e.factor_levels[0] if e.factor_levels else None
            color = color_map.get(lvl, palette[0])
            fig.add_trace(
                go.Scatter(
                    x=act.index.astype(str),
                    y=act.values,
                    line=dict(width=0.7, color=color),
                    name=e.subject_id,
                    showlegend=False,
                ),
                row=i,
                col=1,
            )
        fig.update_layout(
            height=max(220, 130 * len(entries)),
            margin=dict(l=40, r=20, t=30, b=30),
            title=(
                None
                if len(b) <= max_rows
                else f"Showing first {max_rows} of {len(b)} subjects"
            ),
        )
        return fig
