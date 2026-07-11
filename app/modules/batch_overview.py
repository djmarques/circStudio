"""Tab 0b - Batch overview (grouped metrics & plots)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import IS, IV, l5, m10, ra  # noqa: E402
from modules._common import empty_fig  # noqa: E402

_METRICS = ["IS", "IV", "L5", "M10", "RA"]


def _compute_metrics(series) -> dict:
    """Compute the five circadian metrics for one activity series."""
    out = {}
    for name, fn in (("IS", IS), ("IV", IV)):
        try:
            out[name] = float(fn(series))
        except Exception:
            out[name] = float("nan")
    for name, fn in (("L5", l5), ("M10", m10)):
        try:
            res = fn(series)
            out[name] = float(res[1] if isinstance(res, tuple) else res)
        except Exception:
            out[name] = float("nan")
    try:
        out["RA"] = float(ra(series))
    except Exception:
        out["RA"] = float("nan")
    return out


@module.ui
def batch_overview_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_action_button(
                "run_metrics",
                "Run metrics on all subjects",
                class_="btn-primary btn-sm",
            ),
            ui.hr(),
            ui.input_select(
                "box_metric", "Boxplot metric", choices=_METRICS, selected="IS"
            ),
            ui.output_ui("subject_picker"),
            width=320,
        ),
        ui.div(
            ui.p(
                "This tab summarises interdaily stability, intradaily variability, "
                "and other circadian metrics computed across all loaded subjects. "
                "Use the subject selector at the bottom of the sidebar to set the "
                "active recording used by the individual-level analysis tabs."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Grouped metrics"),
        ui.output_data_frame("metrics_table"),
        ui.hr(),
        ui.h4("Metric distribution by factor"),
        output_widget("metrics_box"),
    )


@module.server
def batch_overview_server(input, output, session, rv_batch, rv_active_subject):
    _results = reactive.Value(None)  # cached metrics DataFrame

    @reactive.effect
    @reactive.event(input.run_metrics)
    def _run():
        batch = rv_batch()
        if batch is None or len(batch) == 0:
            ui.notification_show("No batch loaded.", type="warning")
            return
        rows = []
        for e in batch.entries:
            row = {"subject_id": e.subject_id}
            for i, fname in enumerate(batch.factor_names):
                row[fname] = (
                    e.factor_levels[i] if i < len(e.factor_levels) else ""
                )
            row.update(_compute_metrics(e.raw.activity))
            rows.append(row)
        _results.set(pd.DataFrame(rows))
        ui.notification_show(
            f"Computed metrics for {len(rows)} subject(s).", type="message"
        )

    @render.data_frame
    def metrics_table():
        df = _results.get()
        if df is None:
            return pd.DataFrame(
                {"info": ["Click 'Run metrics on all subjects'."]}
            )
        show = df.copy()
        for c in _METRICS:
            if c in show:
                show[c] = show[c].round(4)
        return render.DataGrid(show, width="100%")

    @render_widget
    def metrics_box():
        df = _results.get()
        if df is None:
            return empty_fig("Run metrics to see the distribution.")
        metric = input.box_metric()
        if metric not in df:
            return empty_fig("Metric not available.")

        batch = rv_batch()
        factor = batch.factor_names[0] if batch and batch.factor_names else None
        fig = go.Figure()
        if factor and factor in df:
            for level, sub in df.groupby(factor):
                fig.add_trace(
                    go.Box(y=sub[metric], name=str(level), boxpoints="all")
                )
            fig.update_layout(xaxis_title=factor)
        else:
            fig.add_trace(go.Box(y=df[metric], name="all", boxpoints="all"))
        fig.update_layout(
            yaxis_title=metric,
            title=f"{metric} by {factor}" if factor else metric,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        return fig

    # -- subject selector that drives the default subject in analysis tabs --
    @render.ui
    def subject_picker():
        batch = rv_batch()
        if batch is None or len(batch) == 0:
            return ui.p("No batch loaded.")
        return ui.input_select(
            "active_subject",
            "Active subject (used by analysis tabs)",
            choices=batch.subject_ids(),
        )

    @reactive.effect
    def _sync_active_subject():
        try:
            val = input.active_subject()
        except Exception:
            return
        if val:
            rv_active_subject.set(val)
