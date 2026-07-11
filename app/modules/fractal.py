"""Tab 5 - Fractal / MFDFA analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import Fractal  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)


def _n_array(n_min, n_max, n_points):
    arr = np.unique(
        np.logspace(np.log10(n_min), np.log10(n_max), int(n_points)).astype(int)
    )
    return arr[arr >= 4]


@module.ui
def fractal_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_slider(
                "qrange", "q-order range", min=-5, max=5, value=[-3, 3],
                step=1,
            ),
            ui.help_text(
                "Range of statistical moments used to probe different fluctuation "
                "scales. Negative q emphasises small fluctuations; positive q "
                "emphasises large ones."
            ),
            ui.input_numeric("nq", "Number of q values", value=7, min=3, max=21),
            ui.input_numeric("deg", "Detrending poly order", value=1, min=1, max=4),
            ui.help_text(
                "Polynomial order used to remove local trends before computing "
                "fluctuations. Order 1 removes linear trends; higher orders remove "
                "more complex drifts."
            ),
            ui.input_numeric("nmin", "n min (epochs)", value=16, min=4),
            ui.input_numeric("nmax", "n max (epochs)", value=1440, min=10),
            ui.input_numeric("npoints", "n points", value=30, min=5, max=100),
            ui.input_action_button(
                "run", "Run MFDFA", class_="btn-primary btn-sm"
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "Multifractal Detrended Fluctuation Analysis (MFDFA) characterises "
                "how the statistical properties of the activity signal scale across "
                "different time windows. The generalised Hurst exponent h(q) quantifies "
                "these scaling properties for each statistical moment q — constant h "
                "across all q indicates monofractality, while variation in h indicates "
                "multifractality. The singularity spectrum summarises the full range of "
                "local scaling behaviours in a single curve."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Fluctuation function F(n)"),
        output_widget("flux_plot"),
        ui.hr(),
        ui.h4("Generalized Hurst exponent h(q)"),
        ui.output_data_frame("hurst_table"),
        ui.hr(),
        ui.h4("Multifractal (singularity) spectrum"),
        output_widget("spectrum_plot"),
    )


@module.server
def fractal_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _state = reactive.Value(None)  # (n_array, q_array, Fm, hurst)

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
        qlo, qhi = input.qrange()
        q_array = np.linspace(float(qlo), float(qhi), int(input.nq()))
        # MFDFA is undefined at q=0; nudge it away if present.
        q_array = np.where(np.abs(q_array) < 1e-6, 1e-6, q_array)
        n_array = _n_array(input.nmin(), input.nmax(), input.npoints())
        try:
            Fm = Fractal.mfdfa(
                raw.activity, n_array, q_array, deg=int(input.deg())
            )
            hurst = [
                Fractal.generalized_hurst_exponent(Fm[:, j], n_array)
                for j in range(len(q_array))
            ]
        except Exception as exc:
            ui.notification_show(
                f"MFDFA failed: {exc}", type="error", duration=8
            )
            return
        _state.set((n_array, q_array, Fm, hurst))
        ui.notification_show("MFDFA complete.", type="message")

    @render_widget
    def flux_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Set parameters and click 'Run MFDFA'.")
        n_array, q_array, Fm, _ = st
        fig = go.Figure()
        for j, q in enumerate(q_array):
            fig.add_trace(
                go.Scatter(
                    x=n_array,
                    y=Fm[:, j],
                    mode="lines+markers",
                    name=f"q={q:.1f}",
                    marker=dict(size=4),
                )
            )
        fig.update_layout(
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="n (epochs, log)",
            yaxis_title="F(n) (log)",
            margin=dict(l=40, r=20, t=20, b=30),
        )
        return fig

    @render.data_frame
    def hurst_table():
        st = _state.get()
        if st is None:
            return pd.DataFrame({"info": ["No analysis yet."]})
        _, q_array, _, hurst = st
        df = pd.DataFrame(
            {
                "q": [round(float(q), 2) for q in q_array],
                "h(q)": [round(float(h), 4) for h, _e in hurst],
                "std_err": [round(float(e), 4) for _h, e in hurst],
            }
        )
        return render.DataGrid(df, width="100%")

    @render_widget
    def spectrum_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Run MFDFA to see the spectrum.")
        _, q_array, _, hurst = st
        q = np.asarray(q_array, dtype=float)
        if not (q.min() < 0 < q.max()):
            return empty_fig(
                "Spectrum requires q to span negative to positive values."
            )
        h = np.asarray([hh for hh, _ in hurst], dtype=float)
        # tau(q) = q*h(q) - 1; alpha = dtau/dq; f(alpha) = q*alpha - tau.
        order = np.argsort(q)
        q, h = q[order], h[order]
        tau = q * h - 1.0
        alpha = np.gradient(tau, q)
        falpha = q * alpha - tau
        fig = go.Figure(
            go.Scatter(x=alpha, y=falpha, mode="lines+markers")
        )
        fig.update_layout(
            xaxis_title="alpha (singularity strength)",
            yaxis_title="f(alpha)",
            margin=dict(l=40, r=20, t=20, b=30),
        )
        return fig
