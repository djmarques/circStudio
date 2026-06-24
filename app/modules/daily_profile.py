"""Tab 1 - Daily profile & circadian metrics (Activity + Light).

All metrics are computed once over the whole recording and shown in plain
tables, split into Activity and Light sub-tabs. No per-period computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import (  # noqa: E402
    daily_profile,
    daily_profile_auc,
    temporal_centroid,
    spectral_centroid,
    IS,
    IV,
    l5,
    m10,
    ra,
    lmx,
    adat,
    get_time_barycentre,
    kRA,
    kAR,
    summary_stats,
    light_exposure,
    TAT,
    VAT,
    mlit,
)
from circstudio.analysis.sleep.sleep import AonT, AoffT  # noqa: E402
from modules._common import (  # noqa: E402
    activity_series,
    empty_fig,
    factor_filter_controls,
    get_active,
    light_series,
    subject_controls,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _hhmm(value) -> str:
    try:
        if isinstance(value, pd.Timedelta):
            total = int(value.total_seconds())
        elif isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M")
        else:
            total = int(float(value) * 3600)
        total %= 86400
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        return "n/a"


def _num(value, nd=4):
    try:
        return round(float(value), nd)
    except Exception:
        return value


def _scalar(fn):
    try:
        return fn(), True
    except Exception:
        return "n/a", False


def _as_value(res, take=1):
    return res[take] if isinstance(res, tuple) else res


def _scalarize(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        arr = np.asarray(x).ravel()
        return float(arr[0]) if arr.size == 1 else float(np.nanmean(arr))
    return float(x)


def _count(x):
    try:
        return int(x.shape[0]) if isinstance(x, (pd.Series, pd.DataFrame)) \
            else int(np.asarray(x).size)
    except Exception:
        return len(x)


def _build_table(spec):
    """spec: list of (name, fn, unit). Returns a Metric|Value|Unit DataFrame."""
    rows = []
    for name, fn, unit in spec:
        val, ok = _scalar(fn)
        if ok and unit == "HH:MM":
            val = _hhmm(val)
        elif ok:
            val = _num(val)
        rows.append((name, val, unit))
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@module.ui
def daily_profile_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_radio_buttons(
                "profile_signal", "Profile plot signal",
                choices=["Activity", "Light"], selected="Activity", inline=True,
            ),
            ui.input_switch("cyclic", "Cyclic profile", False),
            ui.input_switch("logscale", "Log scale", False),
            ui.hr(),
            ui.h6("Activity options"),
            ui.input_numeric("whs", "AonT/AoffT window half-size (pts)", value=12),
            ui.input_text("lmx_len", "LMX window length", value="5h"),
            ui.input_switch("lmx_lowest", "LMX lowest (else highest)", True),
            ui.h6("Light options"),
            ui.input_numeric("threshold", "Threshold (lux)", value=10),
            ui.input_text("bins", "Summary bins", value="1h"),
            ui.input_text("start_time", "Start time (HH:MM:SS, optional)", value=""),
            ui.input_text("stop_time", "Stop time (HH:MM:SS, optional)", value=""),
            ui.input_select(
                "agg", "Light exposure aggregation",
                ["mean", "median", "std", "min", "max"], selected="mean",
            ),
            ui.input_action_button("run", "Run metrics", class_="btn-primary btn-sm"),
            width=340,
        ),
        ui.div(
            ui.p(
                "The average daily profile is the mean activity (or light) waveform "
                "folded across 24 hours, providing a composite picture of the typical "
                "rest-activity cycle. Circadian metrics such as IS (interdaily stability), "
                "IV (intradaily variability), M10, and L5 quantify the regularity and "
                "structure of this pattern."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Average daily profile"),
        output_widget("profile_plot"),
        ui.hr(),
        ui.navset_tab(
            ui.nav_panel(
                "Activity metrics",
                ui.output_data_frame("activity_table"),
            ),
            ui.nav_panel(
                "Light metrics",
                ui.output_data_frame("light_table"),
            ),
            ui.nav_panel(
                "Light summary statistics",
                ui.output_data_frame("summary_stats_table"),
            ),
        ),
    )


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------
@module.server
def daily_profile_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _results = reactive.Value(None)

    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    def _raw():
        return get_active(input, rv_mode, rv_single, rv_batch)

    def _opt(txt):
        txt = (txt or "").strip()
        return txt or None

    # -- compute on Run -----------------------------------------------------
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        raw = _raw()
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return

        act = activity_series(raw)
        light = light_series(raw)

        # ---- Activity metrics ----
        act_df = None
        if act is not None:
            whs = int(input.whs() or 12)
            lmx_len = (input.lmx_len() or "5h").strip() or "5h"
            lowest = bool(input.lmx_lowest())
            act_df = _build_table([
                ("AUC of daily profile", lambda: daily_profile_auc(act), ""),
                ("Activity onset (AonT)", lambda: AonT(act, whs=whs), "HH:MM"),
                ("Activity offset (AoffT)", lambda: AoffT(act, whs=whs), "HH:MM"),
                ("Temporal centroid", lambda: temporal_centroid(act), "HH:MM"),
                ("Spectral centroid", lambda: spectral_centroid(act), ""),
                ("IS (interdaily stability)", lambda: IS(act), ""),
                ("IV (intradaily variability)", lambda: IV(act), ""),
                ("L5", lambda: _as_value(l5(act)), ""),
                ("L5 onset", lambda: _as_value(l5(act), 0), "HH:MM"),
                ("M10", lambda: _as_value(m10(act)), ""),
                ("M10 onset", lambda: _as_value(m10(act), 0), "HH:MM"),
                ("RA (relative amplitude)", lambda: ra(act), ""),
                ("LMX (%s, %s)" % (lmx_len, "low" if lowest else "high"),
                 lambda: _as_value(lmx(act, length=lmx_len, lowest=lowest)), ""),
                ("ADAT", lambda: adat(act), "counts"),
                ("Time barycentre", lambda: get_time_barycentre(act), "epoch idx"),
                ("kRA", lambda: kRA(act), ""),
                ("kAR", lambda: kAR(act), ""),
            ])

        # ---- Light metrics ----
        light_df = None
        ss_df = None
        if light is not None:
            thr = float(input.threshold() or 10)
            st, sp = _opt(input.start_time()), _opt(input.stop_time())
            agg = input.agg()
            light_df = _build_table([
                ("AUC of daily light profile", lambda: daily_profile_auc(light), ""),
                ("Light exposure (%s)" % agg,
                 lambda: _scalarize(light_exposure(light, threshold=thr, start_time=st, stop_time=sp, agg=agg)), "lux"),
                ("Time above threshold (TAT)",
                 lambda: _scalarize(TAT(light, threshold=thr, start_time=st, stop_time=sp, oformat="minute")), "min"),
                ("Values above threshold (count)",
                 lambda: _count(VAT(light, threshold=thr)), "epochs"),
                ("Mean light timing (MLiT)",
                 lambda: _scalarize(mlit(light, threshold=thr)), "min since midnight"),
                ("Temporal centroid", lambda: temporal_centroid(light), "HH:MM"),
                ("Spectral centroid", lambda: spectral_centroid(light), ""),
            ])
            try:
                ss_df = summary_stats(light, bins=(input.bins() or "1h")).reset_index()
            except Exception as exc:
                ss_df = pd.DataFrame({"info": [f"summary_stats error: {exc}"]})

        _results.set({"activity": act_df, "light": light_df, "summary": ss_df})
        ui.notification_show("Metrics computed.", type="message")

    # -- outputs ------------------------------------------------------------
    @render_widget
    def profile_plot():
        raw = _raw()
        if raw is None:
            return empty_fig("Load a recording on the Data Upload tab.")
        if input.profile_signal() == "Light":
            s = light_series(raw)
            if s is None:
                return empty_fig("No light channel in this recording.")
        else:
            s = activity_series(raw)
        try:
            return daily_profile(
                s, cyclic=bool(input.cyclic()), plot=True,
                log=bool(input.logscale()),
            )
        except Exception as exc:
            return empty_fig(f"Could not build profile: {exc}")

    @render.data_frame
    def activity_table():
        r = _results.get()
        if not r:
            return pd.DataFrame({"info": ["Click 'Run metrics'."]})
        df = r.get("activity")
        if df is None:
            return pd.DataFrame({"info": ["No activity channel."]})
        return render.DataGrid(df, width="100%")

    @render.data_frame
    def light_table():
        r = _results.get()
        if not r:
            return pd.DataFrame({"info": ["Click 'Run metrics'."]})
        df = r.get("light")
        if df is None:
            return pd.DataFrame({"info": ["No light channel in this recording."]})
        return render.DataGrid(df, width="100%")

    @render.data_frame
    def summary_stats_table():
        r = _results.get()
        if not r or r.get("summary") is None:
            return pd.DataFrame(
                {"info": ["Requires a light channel; click 'Run metrics'."]}
            )
        return render.DataGrid(r["summary"], width="100%")
