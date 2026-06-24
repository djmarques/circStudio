"""Tab 4 - LIDS analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import LIDS  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)


@module.ui
def lids_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_select(
                "lids_func", "LIDS function", ["lids"], selected="lids"
            ),
            ui.input_select(
                "fit_func",
                "Fit function",
                ["cosine", "chirp", "modchirp"],
                selected="cosine",
            ),
            ui.input_slider(
                "win", "Smoothing width (min)", min=5, max=120, value=30,
                step=5,
            ),
            ui.input_action_button(
                "run", "Run LIDS", class_="btn-primary btn-sm"
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "LIDS (Locomotor Inactivity During Sleep) transforms the wrist activity "
                "signal during sleep to track the homeostatic build-up and dissipation "
                "of sleep pressure. The Munich Rhythmicity Index (MRI) quantifies the "
                "depth of the resulting oscillation: higher values indicate a stronger, "
                "more consolidated sleep structure."
            ),
            class_="text-muted small mb-3",
        ),
        ui.layout_columns(
            ui.value_box("Munich Rhythmicity Index (MRI)", ui.output_text("mri")),
            col_widths=[12],
        ),
        ui.h4("LIDS-transformed series"),
        output_widget("transform_plot"),
        ui.hr(),
        ui.h4("LIDS with fitted model"),
        output_widget("fit_plot"),
    )


@module.server
def lids_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _state = reactive.Value(None)  # (lids, transformed, mri)

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
        try:
            lids = LIDS(
                lids_func=input.lids_func(), fit_func=input.fit_func()
            )
            tr = lids.lids_transform(raw.activity, win_td=f"{int(input.win())}min")
            lids.lids_fit(tr)
            mri = float(lids.lids_mri(tr, lids._fit_results.params))
        except Exception as exc:
            ui.notification_show(
                f"LIDS failed: {exc}", type="error", duration=8
            )
            return
        _state.set((lids, tr, mri))
        ui.notification_show("LIDS analysis complete.", type="message")

    @render.text
    def mri():
        st = _state.get()
        return "—" if st is None else f"{st[2]:.4f}"

    @render_widget
    def transform_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Set options and click 'Run LIDS'.")
        _, tr, _ = st
        fig = go.Figure(
            go.Scatter(
                x=tr.index.astype(str), y=tr.values, line=dict(width=1)
            )
        )
        fig.update_layout(
            title="LIDS transform",
            xaxis_title="DateTime",
            yaxis_title="LIDS",
            margin=dict(l=40, r=20, t=40, b=30),
        )
        return fig

    @render_widget
    def fit_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Run LIDS to see the fitted model.")
        lids, tr, _ = st
        try:
            x = np.arange(tr.index.size)
            yfit = lids._fit_func(x, lids._fit_results.params)
        except Exception as exc:
            return empty_fig(f"Could not evaluate fit: {exc}")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tr.index.astype(str), y=tr.values, name="LIDS",
                line=dict(width=1, color="#888"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tr.index.astype(str), y=np.asarray(yfit), name="Fit",
                line=dict(width=2, color="#d62728"),
            )
        )
        fig.update_layout(
            xaxis_title="DateTime",
            yaxis_title="LIDS",
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h"),
        )
        return fig
