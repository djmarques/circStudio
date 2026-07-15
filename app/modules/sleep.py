"""Tab 2 - Sleep scoring and full sleep-metric reporting."""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis.sleep.sleep import (  # noqa: E402
    AonT,
    AoffT,
    CSM,
    Cole_Kripke,
    Crespo,
    Crespo_AoT,
    Oakley,
    Roenneberg,
    Roenneberg_AoT,
    Sadeh,
    Scripps,
    SleepMidPoint,
    SleepProfile,
    SleepRegularityIndex,
    SoD,
    fSoD,
    active_bouts,
    active_durations,
    main_sleep_bouts,
    sleep_bouts,
    sleep_durations,
    waso,
)
from modules._common import (  # noqa: E402
    activity_series,
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)

# UI label -> scoring function name (as defined in sleep.py global namespace).
_ALGO_FUNC = {
    "Cole-Kripke": "Cole_Kripke",
    "Sadeh": "Sadeh",
    "Scripps": "Scripps",
    "Oakley": "Oakley",
    "Roenneberg": "Roenneberg",
    "Crespo": "Crespo",
    "CSM": "CSM",
}
# Functions usable directly via globals()[algo] with just `data`.
_ARGFREE = {"Roenneberg", "Sadeh", "Scripps", "Oakley"}


def _ck_settings(freq: pd.Timedelta) -> str:
    if freq <= pd.Timedelta("10s"):
        return "10sec_max_non_overlap"
    if freq <= pd.Timedelta("30s"):
        return "30sec_max_non_overlap"
    return "mean"


def _zcm_series(raw):
    df = getattr(raw, "df", None)
    if df is not None and hasattr(df, "columns"):
        for c in ("ZCMn", "ZCM"):
            if c in df.columns:
                return df[c]
    return None


def _summary_algo(ui_algo: str) -> str:
    """Pick a function name usable by SoD/SleepProfile/... (globals()[algo]).

    Cole-Kripke/Crespo/CSM need extra arguments not forwarded by those helpers,
    so they fall back to Roenneberg (the package default) for these panels.
    """
    fn = _ALGO_FUNC.get(ui_algo, "Roenneberg")
    return fn if fn in _ARGFREE else "Roenneberg"


def _score(ui_algo, raw, threshold, rescoring):
    """Return a binary sleep/wake Series (1=sleep) for the overlay plot."""
    act = activity_series(raw)
    freq = pd.Timedelta(act.index.freq)
    if ui_algo == "Cole-Kripke":
        s = Cole_Kripke(
            act, settings=_ck_settings(freq), threshold=threshold,
            rescoring=rescoring,
        )
    elif ui_algo == "Sadeh":
        s = Sadeh(act)
    elif ui_algo == "Scripps":
        s = Scripps(act)
    elif ui_algo == "Oakley":
        s = Oakley(act)
    elif ui_algo == "Roenneberg":
        s = Roenneberg(act)
    elif ui_algo == "Crespo":
        s = Crespo(act, frequency=freq)
    elif ui_algo == "CSM":
        zcm = _zcm_series(raw)
        if zcm is None:
            raise ValueError("No ZCM channel available for CSM.")
        s = (CSM(zcm) == 1).astype(int)
    else:
        raise ValueError(f"Unknown algorithm: {ui_algo}")
    return s.astype(float)


def _bouts_to_df(bouts) -> pd.DataFrame:
    """Convert a list of bout Series into a start/end/duration table."""
    rows = []
    for b in bouts:
        try:
            start, end = b.index[0], b.index[-1]
            rows.append(
                {
                    "start": str(start),
                    "end": str(end),
                    "duration_min": round(
                        (end - start) / pd.Timedelta("1min"), 1
                    ),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        {"info": ["No bouts found."]}
    )


def _durations_minutes(durations) -> list:
    out = []
    for d in durations:
        try:
            if isinstance(d, pd.Timedelta):
                out.append(d / pd.Timedelta("1min"))
            else:
                out.append(float(d))
        except Exception:
            continue
    return out


def _hhmm(value) -> str:
    try:
        if isinstance(value, pd.Timedelta):
            total = int(value.total_seconds()) % 86400
        elif isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M")
        else:
            total = int(float(value) * 3600) % 86400
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        return "n/a"


def _dur_opt(txt):
    txt = (txt or "").strip()
    if not txt:
        return None
    return f"{txt}min"


@module.ui
def sleep_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.output_ui("algo_ctrl"),
            ui.panel_conditional(
                "input.algo === 'Cole-Kripke'",
                ui.input_slider("threshold", "Threshold", min=0.1, max=5.0,
                                value=1.0, step=0.1),
                ui.input_switch("rescoring", "Webster rescoring", True),
            ),
            ui.input_numeric("dur_min", "Bout duration min (min, blank=none)",
                             value=None),
            ui.input_numeric("dur_max", "Bout duration max (min, blank=none)",
                             value=None),
            ui.input_numeric("whs", "AonT/AoffT window half-size (pts)",
                             value=12),
            ui.input_file("diary", "Sleep diary (optional)",
                          accept=[".ods", ".xls", ".xlsx", ".csv"]),
            ui.input_action_button("run", "Run", class_="btn-primary btn-sm"),
            width=340,
        ),
        ui.div(
            ui.p(
                "Sleep scoring applies epoch-by-epoch algorithms (Cole-Kripke, Sadeh, "
                "Scripps, and others) to classify each minute of the recording as sleep "
                "or wake based on the activity count and its local context. The resulting "
                "estimates reflect the most likely sleep-wake pattern but are not a "
                "substitute for clinical assessment or polysomnography."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Activity with sleep/wake overlay"),
        output_widget("overlay_plot"),
        ui.hr(),
        ui.navset_tab(
            ui.nav_panel(
                "Bout-level statistics",
                ui.h5("Sleep bouts"),
                ui.output_data_frame("sleep_bouts_tbl"),
                ui.h5("Active bouts"),
                ui.output_data_frame("active_bouts_tbl"),
                ui.h5("Main sleep bouts"),
                ui.output_data_frame("main_bouts_tbl"),
                ui.h5("Sleep-duration distribution"),
                output_widget("sleep_dur_hist"),
                ui.h5("Active-duration distribution"),
                output_widget("active_dur_hist"),
            ),
            ui.nav_panel(
                "Daily sleep summary",
                ui.layout_columns(
                    ui.value_box("Sleep regularity index",
                                 ui.output_text("vb_sri")),
                    ui.value_box("SoD (mean)", ui.output_text("vb_sod")),
                    ui.value_box("fSoD (mean)", ui.output_text("vb_fsod")),
                    ui.value_box("WASO mean (min)", ui.output_text("vb_waso")),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.h5("Sleep midpoint"),
                ui.output_data_frame("midpoint_tbl"),
                ui.h5("Sleep profile (24h)"),
                output_widget("sleep_profile_plot"),
                ui.h5("WASO per day"),
                ui.output_data_frame("waso_tbl"),
            ),
            ui.nav_panel(
                "Onset / offset",
                ui.layout_columns(
                    ui.value_box("Activity onset (AonT)",
                                 ui.output_text("vb_aont")),
                    ui.value_box("Activity offset (AoffT)",
                                 ui.output_text("vb_aofft")),
                    col_widths=[6, 6],
                ),
                ui.output_ui("aot_section"),
            ),
            ui.nav_panel(
                "All sleep metrics",
                ui.download_button("download_summary", "Download CSV"),
                ui.output_data_frame("all_metrics_tbl"),
            ),
        ),
    )


@module.server
def sleep_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _state = reactive.Value(None)

    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    @render.ui
    def algo_ctrl():
        raw = get_active(input, rv_mode, rv_single, rv_batch)
        has_zcm = _zcm_series(raw) is not None
        choices = ["Cole-Kripke", "Sadeh", "Scripps", "Oakley",
                   "Roenneberg", "Crespo"]
        if has_zcm:
            choices.append("CSM")
        sel = ui.input_select("algo", "Algorithm", choices=choices,
                              selected="Cole-Kripke")
        if not has_zcm:
            return ui.div(
                sel,
                ui.help_text("CSM hidden: no ZCM channel in this recording."),
            )
        return sel

    def _raw():
        return get_active(input, rv_mode, rv_single, rv_batch)

    def _threshold():
        try:
            return float(input.threshold())
        except Exception:
            return 1.0

    def _rescoring():
        try:
            return bool(input.rescoring())
        except Exception:
            return True

    # diary upload
    @reactive.effect
    @reactive.event(input.diary)
    def _on_diary():
        finfo = input.diary()
        raw = _raw()
        if not finfo or raw is None:
            return
        src = finfo[0]
        tmp = Path(tempfile.mkdtemp()) / src["name"]
        shutil.copy(src["datapath"], tmp)
        try:
            raw.read_sleep_diary(str(tmp))
            ui.notification_show("Sleep diary attached.", type="message")
        except Exception as exc:
            ui.notification_show(f"Diary error: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        raw = _raw()
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return
        try:
            algo = input.algo()
        except Exception:
            algo = "Cole-Kripke"
        try:
            s = _score(algo, raw, _threshold(), _rescoring())
        except Exception as exc:
            ui.notification_show(f"Scoring failed: {exc}", type="error",
                                 duration=8)
            return
        _state.set(
            {"raw": raw, "algo": algo, "sleep": s,
             "act": activity_series(raw)}
        )
        ui.notification_show("Scoring complete.", type="message")

    # -- overlay ------------------------------------------------------------
    @render_widget
    def overlay_plot():
        st = _state.get()
        if not st:
            return empty_fig("Load a recording and click 'Run'.")
        act, s = st["act"], st["sleep"]
        ymax = float(np.nanmax(act.values)) if len(act) else 1.0
        sleep = s.reindex(act.index, method="nearest")
        shade = np.where(sleep.values == 1, ymax, 0.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=act.index.astype(str), y=shade, fill="tozeroy", mode="none",
            fillcolor="rgba(100,149,237,0.25)", name="Sleep"))
        fig.add_trace(go.Scatter(
            x=act.index.astype(str), y=act.values,
            line=dict(width=0.8, color="#333"), name="Activity"))
        fig.update_layout(margin=dict(l=40, r=20, t=20, b=30),
                          xaxis_title="DateTime", yaxis_title="Activity",
                          legend=dict(orientation="h"))
        return fig

    # -- bout-level ---------------------------------------------------------
    def _alg():
        st = _state.get()
        return _summary_algo(st["algo"]) if st else "Roenneberg"

    @render.data_frame
    def sleep_bouts_tbl():
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})
        try:
            b = sleep_bouts(st["act"], duration_min=_dur_opt_num("dur_min", input),
                            duration_max=_dur_opt_num("dur_max", input), algo=_alg())
            return render.DataGrid(_bouts_to_df(b), width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    @render.data_frame
    def active_bouts_tbl():
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})
        try:
            b = active_bouts(st["act"], duration_min=_dur_opt_num("dur_min", input),
                             duration_max=_dur_opt_num("dur_max", input), algo=_alg())
            return render.DataGrid(_bouts_to_df(b), width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    @render.data_frame
    def main_bouts_tbl():
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})
        try:
            # Pass RAW activity: main_sleep_bouts runs Roenneberg internally,
            # which expects raw counts. Passing the pre-scored 0/1 series
            # (st["sleep"]) double-scores and inverts sleep/wake, yielding the
            # active period instead of the main sleep bout.
            df, _mean = main_sleep_bouts(st["act"], report="major")
            out = df.copy()
            out["duration_min"] = (out["duration"] / pd.Timedelta("1min")).round(1)
            out["is_major"] = True
            out = out[["date", "start_time", "stop_time", "duration_min", "is_major"]]
            out = out.astype({"date": str, "start_time": str, "stop_time": str})
            return render.DataGrid(out, width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    def _dur_hist(fn):
        st = _state.get()
        if not st:
            return empty_fig("Run scoring first.")
        try:
            durs = _durations_minutes(
                fn(st["act"], duration_min=_dur_opt_num("dur_min", input),
                   duration_max=_dur_opt_num("dur_max", input), algo=_alg())
            )
        except Exception as exc:
            return empty_fig(f"Error: {exc}")
        if not durs:
            return empty_fig("No durations.")
        fig = go.Figure(go.Histogram(x=durs, nbinsx=30))
        mean, sd = float(np.mean(durs)), float(np.std(durs))
        fig.update_layout(
            xaxis_title="Duration (min)", yaxis_title="Count",
            title=f"mean={mean:.1f}  sd={sd:.1f}  n={len(durs)}",
            margin=dict(l=40, r=20, t=40, b=30))
        return fig

    @render_widget
    def sleep_dur_hist():
        return _dur_hist(sleep_durations)

    @render_widget
    def active_dur_hist():
        return _dur_hist(active_durations)

    # -- daily summary ------------------------------------------------------
    @render.text
    def vb_sri():
        st = _state.get()
        if not st:
            return "—"
        try:
            return f"{float(SleepRegularityIndex(st['act'], algo=_alg())):.2f}"
        except Exception:
            return "n/a"

    @render.text
    def vb_sod():
        st = _state.get()
        if not st:
            return "—"
        try:
            v = SoD(st["act"], algo=_alg())
            v = np.nanmean(np.asarray(v, dtype=float)) if hasattr(v, "__len__") else float(v)
            return f"{v:.2f}"
        except Exception:
            return "n/a"

    @render.text
    def vb_fsod():
        st = _state.get()
        if not st:
            return "—"
        try:
            v = fSoD(st["act"], algo=_alg())
            v = np.nanmean(np.asarray(v, dtype=float)) if hasattr(v, "__len__") else float(v)
            return f"{v:.3f}"
        except Exception:
            return "n/a"

    def _waso_pair():
        st = _state.get()
        if not st:
            return None
        try:
            return waso(st["act"], frequency=pd.Timedelta(st["act"].index.freq),
                        algo="Cole-Kripke", settings="mean")
        except Exception:
            return None

    @render.text
    def vb_waso():
        wp = _waso_pair()
        return "n/a" if wp is None else f"{float(wp[1]):.1f}"

    @render.data_frame
    def midpoint_tbl():
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})
        try:
            smp = SleepMidPoint(st["act"], to_td=True, algo=_alg())
            if isinstance(smp, pd.Series):
                df = pd.DataFrame({"day": [str(i) for i in smp.index],
                                   "midpoint": [_hhmm(v) for v in smp.values]})
            else:
                df = pd.DataFrame({"metric": ["Sleep midpoint (mean)"],
                                   "value": [_hhmm(smp)]})
            return render.DataGrid(df, width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    @render_widget
    def sleep_profile_plot():
        st = _state.get()
        if not st:
            return empty_fig("Run scoring first.")
        try:
            sp = SleepProfile(st["act"], algo=_alg())
            fig = go.Figure(go.Scatter(x=[str(i) for i in sp.index],
                                       y=sp.values, mode="lines"))
            fig.update_layout(xaxis_title="Time of day",
                              yaxis_title="Sleep probability",
                              margin=dict(l=40, r=20, t=20, b=30))
            return fig
        except Exception as exc:
            return empty_fig(f"Error: {exc}")

    @render.data_frame
    def waso_tbl():
        wp = _waso_pair()
        if wp is None:
            return pd.DataFrame({"info": ["WASO unavailable."]})
        series = wp[0]
        df = pd.DataFrame({"day": [str(i) for i in series.index],
                           "WASO_min": [round(float(v), 1) for v in series.values]})
        return render.DataGrid(df, width="100%")

    # -- onset/offset -------------------------------------------------------
    @render.text
    def vb_aont():
        st = _state.get()
        if not st:
            return "—"
        try:
            return _hhmm(AonT(st["act"], whs=int(input.whs() or 12)))
        except Exception:
            return "n/a"

    @render.text
    def vb_aofft():
        st = _state.get()
        if not st:
            return "—"
        try:
            return _hhmm(AoffT(st["act"], whs=int(input.whs() or 12)))
        except Exception:
            return "n/a"

    @render.ui
    def aot_section():
        st = _state.get()
        if not st:
            return ui.div()
        algo = st["algo"]
        try:
            if algo == "Roenneberg":
                aont, aofft = Roenneberg_AoT(st["act"])
            elif algo == "Crespo":
                aont, aofft = Crespo_AoT(
                    st["act"], frequency=pd.Timedelta(st["act"].index.freq))
            else:
                return ui.help_text(
                    "Algorithm-specific onset/offset is available for "
                    "Roenneberg and Crespo.")
        except Exception as exc:
            return ui.help_text(f"Onset/offset error: {exc}")

        def _fmt(x):
            if isinstance(x, pd.Series):
                return ", ".join(_hhmm(v) for v in x.values[:8])
            return _hhmm(x)

        return ui.tags.table(
            ui.tags.tr(ui.tags.th(f"{algo} onset"), ui.tags.td(_fmt(aont))),
            ui.tags.tr(ui.tags.th(f"{algo} offset"), ui.tags.td(_fmt(aofft))),
            class_="table table-sm",
        )

    # -- comprehensive table + CSV -----------------------------------------
    @reactive.calc
    def _all_metrics_df():
        st = _state.get()
        if not st:
            return pd.DataFrame(columns=["Metric", "Value", "Unit"])
        act, s, alg = st["act"], st["sleep"], _alg()
        freq = pd.Timedelta(act.index.freq)
        epoch_min = freq / pd.Timedelta("1min")
        rows = []

        # TST / efficiency from main sleep period
        try:
            vals = s.values
            idx = np.where(vals == 1)[0]
            tst = int(np.sum(vals == 1)) * epoch_min
            n_days = max(1, (act.index[-1] - act.index[0]) / pd.Timedelta("1D"))
            rows.append(("Total sleep time (mean/day)", round(tst / n_days / 60, 2), "hours"))
            if len(idx):
                span = vals[idx[0]:idx[-1] + 1]
                eff = 100.0 * np.sum(span == 1) / len(span)
            else:
                eff = 0.0
            rows.append(("Sleep efficiency", round(eff, 1), "%"))
        except Exception:
            rows.append(("Total sleep time (mean/day)", "n/a", "hours"))
            rows.append(("Sleep efficiency", "n/a", "%"))

        def _try(label, fn, unit, fmt=None):
            try:
                v = fn()
                if fmt:
                    v = fmt(v)
                elif isinstance(v, (int, float, np.floating)):
                    v = round(float(v), 2)
                rows.append((label, v, unit))
            except Exception:
                rows.append((label, "n/a", unit))

        wp = _waso_pair()
        rows.append(("WASO (mean)", round(float(wp[1]), 1) if wp else "n/a", "minutes"))
        _try("Number of sleep bouts (mean/day)",
             lambda: len(sleep_bouts(act, algo=alg)) /
             max(1, (act.index[-1] - act.index[0]) / pd.Timedelta("1D")), "—")
        _try("Sleep midpoint (mean)",
             lambda: SleepMidPoint(act, to_td=True, algo=alg), "HH:MM",
             fmt=lambda v: _hhmm(v.mean() if isinstance(v, pd.Series) else v))
        _try("Sleep regularity index",
             lambda: SleepRegularityIndex(act, algo=alg), "—")
        _try("Sleep onset (SoD, mean)",
             lambda: SoD(act, algo=alg), "HH:MM",
             fmt=lambda v: _hhmm(np.nanmean(np.asarray(v, float)) if hasattr(v, "__len__") else v))
        _try("Functional SoD (mean)",
             lambda: fSoD(act, algo=alg), "HH:MM",
             fmt=lambda v: _hhmm(np.nanmean(np.asarray(v, float)) if hasattr(v, "__len__") else v))
        _try("Activity onset (AonT)",
             lambda: AonT(act, whs=int(input.whs() or 12)), "HH:MM", fmt=_hhmm)
        _try("Activity offset (AoffT)",
             lambda: AoffT(act, whs=int(input.whs() or 12)), "HH:MM", fmt=_hhmm)
        return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])

    @render.data_frame
    def all_metrics_tbl():
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring to populate this table."]})
        return render.DataGrid(_all_metrics_df(), width="100%")

    @render.download(filename="sleep_metrics.csv")
    def download_summary():
        buf = io.StringIO()
        _all_metrics_df().to_csv(buf, index=False)
        yield buf.getvalue()


def _dur_opt_num(name, input):
    """Read a numeric duration input (minutes) -> '<n>min' string or None."""
    try:
        v = getattr(input, name)()
    except Exception:
        return None
    if v is None or v == "":
        return None
    try:
        return f"{int(float(v))}min"
    except Exception:
        return None
