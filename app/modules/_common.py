"""Helpers shared by the analysis tab modules.

These functions implement the per-tab subject/factor selectors used in batch
mode and a couple of small Plotly utilities. They are deliberately kept
free of module-specific state so every analysis tab can reuse them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import ui

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import active_raw  # noqa: E402


def as_series(x) -> Optional[pd.Series]:
    """Coerce activity/light data to a clean 1-D pandas Series.

    Some readers (e.g. AGD) return the light channel as a one-column
    DataFrame; several metric functions only accept a Series, so this picks a
    single column (preferring a ``LIGHT`` column) while preserving the
    DatetimeIndex and its frequency.
    """
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return None
        col = "LIGHT" if "LIGHT" in x.columns else x.columns[0]
        return x[col]
    return pd.Series(np.ravel(x))


def activity_series(raw) -> Optional[pd.Series]:
    if raw is None:
        return None
    return as_series(getattr(raw, "activity", None))


def light_series(raw) -> Optional[pd.Series]:
    if raw is None:
        return None
    s = as_series(getattr(raw, "light", None))
    if s is None or len(s) == 0:
        return None
    return s


def empty_fig(message: str = "No data loaded.") -> go.Figure:
    """A blank Plotly figure carrying an informational annotation."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16, color="#888"),
            )
        ],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def factor_filter_controls(rv_mode, rv_batch):
    """Return the factor-level filter select (batch mode only) or ``None``."""
    if rv_mode() != "batch":
        return None
    batch = rv_batch()
    if batch is None or len(batch) == 0:
        return None
    if not batch.factor_names:
        return None
    levels = ["(all)"] + batch.levels_for_factor(0)
    return ui.input_select(
        "factor_filter",
        "Filter by first factor",
        choices=levels,
        selected="(all)",
    )


def subject_controls(input, rv_mode, rv_batch, default=None):
    """Return the subject select for the current mode/filter, or ``None``.

    ``default`` (e.g. the globally selected subject from the Batch Overview
    tab) is used as the initially selected value when it is among the choices.
    """
    if rv_mode() != "batch":
        return None
    batch = rv_batch()
    if batch is None or len(batch) == 0:
        return None

    level = None
    if batch.factor_names:
        try:
            sel = input.factor_filter()
            if sel and sel != "(all)":
                level = sel
        except Exception:
            level = None

    entries = batch.filter(0, level) if level else batch.entries
    choices = [e.subject_id for e in entries]
    if not choices:
        choices = batch.subject_ids()
    selected = default if default in choices else None
    return ui.input_select(
        "subject", "Subject", choices=choices, selected=selected
    )


def selected_subject(input) -> Optional[str]:
    """Read the per-tab subject selector value if present."""
    try:
        val = input.subject()
        return val or None
    except Exception:
        return None


def get_active(input, rv_mode, rv_single, rv_batch):
    """Resolve the Raw object this tab should use (single or batch)."""
    return active_raw(rv_mode, rv_single, rv_batch, selected_subject(input))
