"""Tab 8 - Mathematical models of circadian rhythms."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis.models.math_models import (  # noqa: E402
    Breslow13,
    ESRI,
    Forger,
    HannaySP,
    HannayTP,
    Hilaire07,
    Jewett,
    ModelComparer,
    Skeldon23,
)
from circstudio.analysis.models.light_tools import Light  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    light_series,
    subject_controls,
)

MODELS = {
    "Forger": Forger,
    "Jewett": Jewett,
    "HannaySP": HannaySP,
    "HannayTP": HannayTP,
    "Hilaire07": Hilaire07,
    "Breslow13": Breslow13,
    "Skeldon23": Skeldon23,
}

_SKIP_PARAMS = {"self", "data", "inputs", "time"}


def _model_params(name: str) -> dict:
    """Tunable numeric/bool parameters read from a model's __init__ signature."""
    cls = MODELS[name]
    out = {}
    for k, v in inspect.signature(cls.__init__).parameters.items():
        if k in _SKIP_PARAMS or k.startswith("initial_condition"):
            continue
        d = v.default
        if d is inspect.Parameter.empty or d is None:
            continue
        if isinstance(d, bool) or isinstance(d, (int, float)):
            out[k] = d
    return out


def _hhmm(hour) -> str:
    try:
        total = int(float(hour) % 24 * 3600)
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        return "n/a"


def _mean_sd_clock(hours_arr):
    arr = np.asarray(hours_arr, dtype=float) % 24
    if arr.size == 0:
        return "n/a", "n/a"
    return _hhmm(np.mean(arr)), f"{np.std(arr):.2f}h"


def _build_model(name, light, params, equilibrate, loops):
    """Build and (optionally) equilibrate a model. Pure/blocking — safe to run
    in a worker thread (does not touch Shiny ``input``)."""
    cls = MODELS[name]
    model = cls(data=light, **params)
    if name == "Skeldon23":
        model.run()
    elif equilibrate:
        try:
            model.get_initial_conditions(loops, data=light)
            model.initialize_model_states()
        except Exception:
            pass
    return model


@module.ui
def circ_models_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_select("model", "Model", choices=list(MODELS.keys()),
                            selected="Forger"),
            ui.input_radio_buttons(
                "source", "Light source",
                {"light": "Use light channel", "synthetic": "Synthetic schedule"},
                selected="synthetic",
            ),
            ui.panel_conditional(
                "input.source === 'synthetic'",
                ui.input_numeric("syn_days", "Total days", value=10),
                ui.input_numeric("syn_on", "Light-on hours/day", value=16),
                ui.input_numeric("syn_start_hour", "Light-on start hour", value=8),
                ui.input_numeric("syn_low", "Night lux", value=0),
                ui.input_numeric("syn_high", "Day lux", value=1000),
                ui.input_numeric("syn_bins", "Bins per hour", value=6),
                ui.input_text("syn_date", "Start date", value="2020-01-01"),
            ),
            ui.input_switch("equilibrate", "Equilibrate initial conditions", False),
            ui.input_numeric("loops", "Equilibration loops", value=10),
            ui.accordion(
                ui.accordion_panel("Model parameters",
                                   ui.output_ui("param_inputs")),
                open=False,
            ),
            ui.input_action_button("run", "Run model", class_="btn-primary btn-sm"),
            ui.output_ui("run_status"),
            width=340,
        ),
        ui.div(
            ui.p(
                "These mathematical models simulate the dynamics of the human "
                "circadian pacemaker in response to a light input schedule. Fed with "
                "the recorded or synthetic light intensity time series, they integrate "
                "coupled differential equations to predict the phase and amplitude of "
                "the oscillator over time, including physiological markers such as "
                "DLMO (dim-light melatonin onset) and CBTmin (core body temperature "
                "minimum)."
            ),
            class_="text-muted small mb-3",
        ),
        ui.navset_tab(
            ui.nav_panel("State trajectories", output_widget("traj_plot")),
            ui.nav_panel(
                "Amplitude & Phase",
                ui.input_radio_buttons("phase_unit", "Phase unit",
                                       {"rad": "radians", "hours": "hours"},
                                       selected="rad", inline=True),
                output_widget("amp_plot"),
                output_widget("phase_plot"),
            ),
            ui.nav_panel(
                "DLMO & CBTmin",
                ui.layout_columns(
                    ui.value_box("DLMO (mean ± SD)", ui.output_text("dlmo_stat")),
                    ui.value_box("CBTmin (mean ± SD)", ui.output_text("cbt_stat")),
                    col_widths=[6, 6],
                ),
                ui.h5("DLMO events"),
                ui.output_data_frame("dlmo_tbl"),
                ui.h5("CBTmin events"),
                ui.output_data_frame("cbt_tbl"),
            ),
            ui.nav_panel(
                "ESRI",
                ui.input_numeric("esri_window", "Window size (days)", value=4.0),
                ui.input_numeric("esri_step", "Time step (hours)", value=1.0),
                ui.input_numeric("esri_amp", "Initial amplitude", value=0.1),
                ui.input_action_button("run_esri", "Run ESRI",
                                       class_="btn-primary btn-sm"),
                ui.output_ui("esri_status"),
                ui.layout_columns(
                    ui.value_box("Mean ESRI", ui.output_text("esri_mean")),
                    ui.value_box("SD ESRI", ui.output_text("esri_sd")),
                    col_widths=[6, 6],
                ),
                output_widget("esri_plot"),
            ),
            ui.nav_panel(
                "Model comparison",
                ui.layout_columns(
                    ui.input_select("model_a", "Model A", choices=list(MODELS.keys()),
                                    selected="Forger"),
                    ui.input_select("model_b", "Model B", choices=list(MODELS.keys()),
                                    selected="HannaySP"),
                    col_widths=[6, 6],
                ),
                ui.input_action_button("run_cmp", "Compare",
                                       class_="btn-primary btn-sm"),
                ui.output_ui("cmp_status"),
                output_widget("cmp_amp_plot"),
                output_widget("cmp_phase_plot"),
                output_widget("cmp_diff_plot"),
            ),
        ),
    )


@module.server
def circ_models_server(input, output, session, rv_single, rv_batch, rv_mode,
                       rv_active_subject):
    _model = reactive.Value(None)
    _esri = reactive.Value(None)
    _cmp = reactive.Value(None)

    # -- batch subject / factor selectors ----------------------------------
    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    # -- dynamic parameter inputs ------------------------------------------
    @render.ui
    def param_inputs():
        params = _model_params(input.model())
        controls = []
        for k, d in params.items():
            if isinstance(d, bool):
                controls.append(ui.input_switch(f"p_{k}", k, d))
            else:
                controls.append(ui.input_numeric(f"p_{k}", k, value=d))
        return ui.div(*controls)

    def _get_params(name: str) -> dict:
        out = {}
        for k, d in _model_params(name).items():
            try:
                v = input[f"p_{k}"]()
                out[k] = bool(v) if isinstance(d, bool) else float(v)
            except Exception:
                out[k] = d
        return out

    def _get_light():
        if input.source() == "synthetic":
            try:
                sched = Light.create(
                    total_days=int(input.syn_days()),
                    light_on_hours=float(input.syn_on()),
                    bins_per_hour=int(input.syn_bins()),
                    schedule_starts_at=float(input.syn_start_hour()),
                    low=float(input.syn_low()),
                    high=float(input.syn_high()),
                    start=(input.syn_date() or "2020-01-01").strip(),
                )
                return sched.synthetic_light
            except Exception as exc:
                ui.notification_show(f"Synthetic light error: {exc}",
                                     type="error", duration=8)
                return None
        raw = get_active(input, rv_mode, rv_single, rv_batch)
        light = light_series(raw)
        if light is None:
            ui.notification_show(
                "No light channel in this recording; use a synthetic schedule.",
                type="warning", duration=8)
        return light

    # -- run primary model (offloaded to a worker thread) -------------------
    # Extended task — runs the ODE integration off the event loop so the
    # spinner stays responsive while the model integrates.
    @reactive.extended_task
    async def _model_task(name: str, light, params: dict, equilibrate: bool,
                          loops: int):
        def _build_blocking():
            return _build_model(name, light, params, equilibrate, loops)
        return await asyncio.to_thread(_build_blocking)

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        light = _get_light()
        if light is None:
            return
        _model_task(
            input.model(),
            light,
            _get_params(input.model()),
            bool(input.equilibrate()),
            int(input.loops()),
        )

    @reactive.effect
    def _run_done():
        status = _model_task.status()
        if status in ("initial", "running"):
            return
        if status == "error":
            try:
                _model_task.result()
            except Exception as exc:
                ui.notification_show(f"Model failed: {exc}", type="error",
                                     duration=10)
            return
        # success
        name = input.model()
        light = _get_light()        # reconstruct light reference for storage
        model = _model_task.result()
        _model.set((name, model, light))
        ui.notification_show(f"{name} integrated.", type="message")

    @render.ui
    def run_status():
        status = _model_task.status()
        if status == "running":
            return ui.div(
                ui.tags.div(
                    class_="spinner-border spinner-border-sm text-primary",
                    role="status",
                ),
                ui.tags.strong("  Integrating model…  ",
                               style="margin-left:8px;"),
                ui.tags.span("please wait", class_="text-muted"),
                class_="d-flex align-items-center mt-2",
            )
        return ui.div()

    # -- trajectories -------------------------------------------------------
    @render_widget
    def traj_plot():
        st = _model.get()
        if not st:
            return empty_fig("Configure a model and click 'Run model'.")
        name, model, light = st
        try:
            fig = model.plot(states=True)
        except Exception:
            fig = go.Figure()
            ms = np.asarray(model.model_states)
            x = light.index.astype(str)
            for i in range(ms.shape[1]):
                fig.add_trace(go.Scatter(x=x, y=ms[:, i], name=f"state {i}"))
            fig.update_layout(title="Model states")
        # DLMO / CBTmin vertical markers
        try:
            t0 = light.index[0]
            for h in np.asarray(model.dlmos()):
                ts = t0 + pd.to_timedelta(float(h), unit="h")
                fig.add_vline(x=str(ts), line=dict(color="orange", dash="dot"))
            for h in np.asarray(model.cbt()):
                ts = t0 + pd.to_timedelta(float(h), unit="h")
                fig.add_vline(x=str(ts), line=dict(color="purple", dash="dash"))
        except Exception:
            pass
        fig.update_layout(margin=dict(l=40, r=20, t=40, b=30))
        return fig

    # -- amplitude & phase --------------------------------------------------
    @render_widget
    def amp_plot():
        st = _model.get()
        if not st:
            return empty_fig("Run a model first.")
        _name, model, light = st
        try:
            amp = np.asarray(model.amplitude())
        except Exception as exc:
            return empty_fig(f"Amplitude error: {exc}")
        fig = go.Figure(go.Scatter(x=light.index.astype(str), y=amp,
                                   mode="lines", name="Amplitude"))
        fig.update_layout(title="Amplitude", xaxis_title="Time",
                          yaxis_title="Amplitude", margin=dict(l=40, r=20, t=40, b=30))
        return fig

    @render_widget
    def phase_plot():
        st = _model.get()
        if not st:
            return empty_fig("Run a model first.")
        _name, model, light = st
        try:
            ph = np.asarray(model.phase())
        except Exception as exc:
            return empty_fig(f"Phase error: {exc}")
        unit = input.phase_unit()
        y = ph if unit == "rad" else (ph / (2 * np.pi) * 24) % 24
        fig = go.Figure(go.Scatter(x=light.index.astype(str), y=y, mode="lines",
                                   name="Phase"))
        fig.update_layout(title="Phase", xaxis_title="Time",
                          yaxis_title=f"Phase ({unit})",
                          margin=dict(l=40, r=20, t=40, b=30))
        return fig

    # -- DLMO & CBTmin ------------------------------------------------------
    @render.text
    def dlmo_stat():
        st = _model.get()
        if not st:
            return "—"
        try:
            m, s = _mean_sd_clock(st[1].dlmos())
            return f"{m} ± {s}"
        except Exception:
            return "n/a"

    @render.text
    def cbt_stat():
        st = _model.get()
        if not st:
            return "—"
        try:
            m, s = _mean_sd_clock(st[1].cbt())
            return f"{m} ± {s}"
        except Exception:
            return "n/a"

    @render.data_frame
    def dlmo_tbl():
        st = _model.get()
        if not st:
            return pd.DataFrame({"info": ["Run a model first."]})
        try:
            d = np.asarray(st[1].dlmos())
            return render.DataGrid(pd.DataFrame({
                "event": range(1, len(d) + 1),
                "time_hours": np.round(d, 2),
                "clock": [_hhmm(h) for h in d]}), width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    @render.data_frame
    def cbt_tbl():
        st = _model.get()
        if not st:
            return pd.DataFrame({"info": ["Run a model first."]})
        try:
            c = np.asarray(st[1].cbt())
            return render.DataGrid(pd.DataFrame({
                "event": range(1, len(c) + 1),
                "time_hours": np.round(c, 2),
                "clock": [_hhmm(h) for h in c]}), width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    # -- ESRI (offloaded to a worker thread) --------------------------------
    @reactive.extended_task
    async def _esri_task(light, window: float, step: float, amp: float):
        def _build_blocking():
            return ESRI(data=light,
                        window_size_days=window,
                        esri_time_step_hours=step,
                        initial_amplitude=amp)
        return await asyncio.to_thread(_build_blocking)

    @reactive.effect
    @reactive.event(input.run_esri)
    def _run_esri():
        light = _get_light()
        if light is None:
            return
        _esri_task(
            light,
            float(input.esri_window()),
            float(input.esri_step()),
            float(input.esri_amp()),
        )

    @reactive.effect
    def _esri_done():
        status = _esri_task.status()
        if status in ("initial", "running"):
            return
        if status == "error":
            try:
                _esri_task.result()
            except Exception as exc:
                ui.notification_show(f"ESRI failed: {exc}", type="error",
                                     duration=10)
            return
        _esri.set(_esri_task.result())
        ui.notification_show("ESRI computed.", type="message")

    @render.ui
    def esri_status():
        if _esri_task.status() == "running":
            return ui.div(
                ui.tags.div(
                    class_="spinner-border spinner-border-sm text-primary",
                    role="status",
                ),
                ui.tags.strong("  Computing ESRI…  ",
                               style="margin-left:8px;"),
                ui.tags.span("please wait", class_="text-muted"),
                class_="d-flex align-items-center mt-2",
            )
        return ui.div()

    @render.text
    def esri_mean():
        e = _esri.get()
        return "—" if e is None else f"{float(e.mean):.4f}"

    @render.text
    def esri_sd():
        e = _esri.get()
        return "—" if e is None else f"{float(e.std):.4f}"

    @render_widget
    def esri_plot():
        e = _esri.get()
        if e is None:
            return empty_fig("Click 'Run ESRI'.")
        rv = e.raw_values
        fig = go.Figure(go.Scatter(x=list(rv.index), y=rv["esri"].values,
                                   mode="lines", name="ESRI"))
        fig.update_layout(title="ESRI over time", xaxis_title="Window start (h)",
                          yaxis_title="ESRI", margin=dict(l=40, r=20, t=40, b=30))
        return fig

    # -- model comparison (offloaded to a worker thread) --------------------
    @reactive.extended_task
    async def _cmp_task(a: str, b: str, light, equilibrate: bool, loops: int):
        def _build_blocking():
            ma = _build_model(a, light, _model_params(a), equilibrate, loops)
            mb = _build_model(b, light, _model_params(b), equilibrate, loops)
            return (a, ma, b, mb, light)
        return await asyncio.to_thread(_build_blocking)

    @reactive.effect
    @reactive.event(input.run_cmp)
    def _run_cmp():
        light = _get_light()
        if light is None:
            return
        _cmp_task(
            input.model_a(),
            input.model_b(),
            light,
            bool(input.equilibrate()),
            int(input.loops()),
        )

    @reactive.effect
    def _cmp_done():
        status = _cmp_task.status()
        if status in ("initial", "running"):
            return
        if status == "error":
            try:
                _cmp_task.result()
            except Exception as exc:
                ui.notification_show(f"Comparison failed: {exc}", type="error",
                                     duration=10)
            return
        _cmp.set(_cmp_task.result())
        ui.notification_show("Comparison ready.", type="message")

    @render.ui
    def cmp_status():
        if _cmp_task.status() == "running":
            return ui.div(
                ui.tags.div(
                    class_="spinner-border spinner-border-sm text-primary",
                    role="status",
                ),
                ui.tags.strong("  Comparing models…  ",
                               style="margin-left:8px;"),
                ui.tags.span("please wait", class_="text-muted"),
                class_="d-flex align-items-center mt-2",
            )
        return ui.div()

    def _cmp_fig(getter, title):
        st = _cmp.get()
        if not st:
            return empty_fig("Pick two models and click 'Compare'.")
        a, ma, b, mb, light = st
        x = light.index.astype(str)
        fig = go.Figure()
        try:
            fig.add_trace(go.Scatter(x=x, y=np.asarray(getter(ma)), name=a))
            fig.add_trace(go.Scatter(x=x, y=np.asarray(getter(mb)), name=b))
        except Exception as exc:
            return empty_fig(f"Error: {exc}")
        fig.update_layout(title=title, margin=dict(l=40, r=20, t=40, b=30),
                          legend=dict(orientation="h"))
        return fig

    @render_widget
    def cmp_amp_plot():
        return _cmp_fig(lambda m: m.amplitude(), "Amplitude: A vs B")

    @render_widget
    def cmp_phase_plot():
        return _cmp_fig(lambda m: m.phase(), "Phase: A vs B")

    @render_widget
    def cmp_diff_plot():
        st = _cmp.get()
        if not st:
            return empty_fig("Pick two models and click 'Compare'.")
        a, ma, b, mb, light = st
        x = light.index.astype(str)
        fig = go.Figure()
        try:
            da = np.asarray(ma.amplitude()) - np.asarray(mb.amplitude())
            dp = np.asarray(ma.phase()) - np.asarray(mb.phase())
            fig.add_trace(go.Scatter(x=x, y=da, name="Δ amplitude"))
            fig.add_trace(go.Scatter(x=x, y=dp, name="Δ phase (rad)"))
        except Exception as exc:
            return empty_fig(f"Difference needs equal-length states: {exc}")
        fig.update_layout(title=f"Difference ({a} − {b})",
                          margin=dict(l=40, r=20, t=40, b=30),
                          legend=dict(orientation="h"))
        return fig
