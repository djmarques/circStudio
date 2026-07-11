"""Tab 7 - Singular Spectrum Analysis (SSA)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import SSA  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)


@module.ui
def ssa_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_text("window", "Window length", value="24h"),
            ui.help_text(
                "Embedding dimension in time units. Typically set to one full cycle "
                "length (e.g., 24h). Larger windows resolve lower-frequency components."
            ),
            ui.input_text(
                "resample",
                "Resample frequency (for speed)",
                value="10min",
            ),
            ui.input_slider(
                "ncomp", "Number of components", min=2, max=30, value=10,
                step=1,
            ),
            ui.help_text(
                "Number of singular components to extract. Inspect the variance "
                "plot to decide how many components capture meaningful structure "
                "versus noise."
            ),
            ui.output_ui("group_ctrl"),
            ui.input_action_button(
                "run", "Run SSA", class_="btn-primary btn-sm"
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "Singular Spectrum Analysis (SSA) decomposes the activity time series "
                "into additive components — trend, oscillatory patterns, and noise "
                "— without assuming a fixed parametric model. The variance-explained "
                "bar chart indicates which components capture the most signal energy, "
                "and the W-correlation matrix reveals which components are statistically "
                "related and can be grouped for reconstruction."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Variance explained"),
        output_widget("variance_plot"),
        ui.hr(),
        ui.h4("W-correlation matrix"),
        output_widget("wcorr_plot"),
        ui.hr(),
        ui.h4("Reconstructed signal"),
        output_widget("recon_plot"),
    )


@module.server
def ssa_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _state = reactive.Value(None)  # (ssa, series, ncomp)

    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    @render.ui
    def group_ctrl():
        n = int(input.ncomp())
        choices = [str(i) for i in range(n)]
        return ui.input_selectize(
            "group",
            "Components to reconstruct",
            choices=choices,
            selected=["0", "1"],
            multiple=True,
        )

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        raw = get_active(input, rv_mode, rv_single, rv_batch)
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return
        s = raw.activity
        resample = (input.resample() or "").strip()
        try:
            if resample:
                s = s.resample(resample).mean().dropna()
                s = s.asfreq(resample) if s.index.freq is None else s
                s = s.interpolate()
            ssa = SSA(s, window_length=input.window())
            ssa.fit()
        except Exception as exc:
            ui.notification_show(
                f"SSA failed: {exc}", type="error", duration=10
            )
            return
        _state.set((ssa, s, int(input.ncomp())))
        ui.notification_show("SSA complete.", type="message")

    @render_widget
    def variance_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Set options and click 'Run SSA'.")
        ssa, _, ncomp = st
        ve = np.asarray(ssa.variance_explained)
        k = min(ncomp, len(ve))
        fig = go.Figure(
            go.Bar(
                x=[f"C{i}" for i in range(k)],
                y=ve[:k] * 100.0,
                marker_color="#1f77b4",
            )
        )
        fig.update_layout(
            yaxis_title="Variance explained (%)",
            xaxis_title="Component",
            margin=dict(l=40, r=20, t=20, b=30),
        )
        return fig

    @render_widget
    def wcorr_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Run SSA to see the W-correlation matrix.")
        ssa, _, ncomp = st
        k = int(min(ncomp, ssa.L))
        try:
            wc = ssa.w_correlation_matrix(k)
        except Exception as exc:
            return empty_fig(f"Could not compute W-correlation: {exc}")
        labels = [f"C{i}" for i in range(k)]
        fig = go.Figure(
            go.Heatmap(
                z=np.abs(wc), x=labels, y=labels, colorscale="Viridis",
                zmin=0, zmax=1,
            )
        )
        fig.update_layout(margin=dict(l=40, r=20, t=20, b=30))
        return fig

    @render_widget
    def recon_plot():
        st = _state.get()
        if st is None:
            return empty_fig("Run SSA to reconstruct a component group.")
        ssa, s, _ = st
        try:
            group = [int(g) for g in (input.group() or [])]
        except Exception:
            group = [0]
        if not group:
            group = [0]
        try:
            rec = ssa.reconstruct_signal(group)
        except Exception as exc:
            return empty_fig(f"Reconstruction failed: {exc}")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=s.index.astype(str), y=s.values, name="Original",
                line=dict(width=0.6, color="#bbb"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rec.index.astype(str), y=rec.values,
                name=f"Reconstructed {group}",
                line=dict(width=1.5, color="#1f77b4"),
            )
        )
        fig.update_layout(
            xaxis_title="DateTime",
            yaxis_title="Activity",
            margin=dict(l=40, r=20, t=20, b=30),
            legend=dict(orientation="h"),
        )
        return fig
