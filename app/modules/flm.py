"""Tab 6 - Functional Linear Modelling (FLM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import FLM, daily_profile  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)


@module.ui
def flm_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_select(
                "basis", "Basis", ["fourier", "spline"], selected="fourier"
            ),
            ui.input_slider(
                "max_order", "Max order / harmonics", min=1, max=30, value=10,
                step=1,
            ),
            ui.help_text(
                "Number of basis functions (harmonics for Fourier, knots for "
                "spline). Higher values capture finer detail but may overfit."
            ),
            ui.input_switch(
                "smooth_only", "Smoothing only (full series)", False
            ),
            ui.input_action_button(
                "run", "Run FLM", class_="btn-primary btn-sm"
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "Functional Linear Modelling represents the activity time series as "
                "a smooth function built from a set of basis functions (Fourier "
                "harmonics or spline knots), capturing systematic structure in its "
                "shape. Comparing reconstructed waveforms across subjects or conditions "
                "reveals differences in activity morphology that are not apparent from "
                "scalar summary statistics."
            ),
            class_="text-muted small mb-3",
        ),
        ui.output_ui("rmse_box"),
        ui.h4("FLM output"),
        output_widget("flm_plot"),
    )


@module.server
def flm_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _state = reactive.Value(None)  # dict of results

    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        raw = get_active(input, rv_mode, rv_single, rv_batch)
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return
        s = raw.activity
        freq = s.index.freq
        try:
            flm = FLM(
                basis=input.basis(),
                sampling_freq=freq,
                max_order=int(input.max_order()),
            )
            if input.smooth_only():
                smoothed = np.asarray(flm.smooth_timeseries(s))
                _state.set(
                    {
                        "mode": "smooth",
                        "x": s.index.astype(str),
                        "raw": s.values,
                        "fit": smoothed,
                    }
                )
            else:
                flm.fit(s)
                fitted = np.asarray(flm.evaluate())
                profile = daily_profile(s)
                prof = np.asarray(profile.values, dtype=float)
                # Align lengths (fourier returns samples/day = profile length).
                n = min(len(prof), len(fitted))
                rmse = float(
                    np.sqrt(np.nanmean((prof[:n] - fitted[:n]) ** 2))
                )
                _state.set(
                    {
                        "mode": "profile",
                        "x": list(range(n)),
                        "raw": prof[:n],
                        "fit": fitted[:n],
                        "rmse": rmse,
                    }
                )
        except Exception as exc:
            ui.notification_show(
                f"FLM failed: {exc}", type="error", duration=8
            )
            return
        ui.notification_show("FLM complete.", type="message")

    @render.ui
    def rmse_box():
        st = _state.get()
        if not st or st.get("mode") != "profile":
            return ui.div()
        return ui.layout_columns(
            ui.value_box("Fit RMSE", f"{st['rmse']:.3f}"),
            col_widths=[12],
        )

    @render_widget
    def flm_plot():
        st = _state.get()
        if not st:
            return empty_fig("Set options and click 'Run FLM'.")
        fig = go.Figure()
        if st["mode"] == "smooth":
            fig.add_trace(
                go.Scatter(
                    x=st["x"], y=st["raw"], name="Raw",
                    line=dict(width=0.6, color="#bbb"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st["x"], y=st["fit"], name="Smoothed",
                    line=dict(width=1.5, color="#1f77b4"),
                )
            )
            fig.update_layout(xaxis_title="DateTime")
        else:
            fig.add_trace(
                go.Scatter(
                    x=st["x"], y=st["raw"], name="Daily profile",
                    line=dict(width=1, color="#888"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st["x"], y=st["fit"], name="FLM fit",
                    line=dict(width=2, color="#d62728"),
                )
            )
            fig.update_layout(xaxis_title="Epoch within day")
        fig.update_layout(
            yaxis_title="Activity",
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h"),
        )
        return fig
