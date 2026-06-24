"""Tab 1 - Preprocessing (non-wear detection, masking, filtering).

This module provides:
- Automatic non-wear detection (Choi 2011 or Troiano 2008 algorithms)
- Manual mask period entry (start/stop datetime)
- Mask import from CSV / XLSX / ODS file
- Filter application: resample, binarize, impute NaN
- Single interactive Plotly figure: activity trace with mask overlay
- Mask export to CSV
- Batch mode with subject/factor selectors

The preprocessing module works on a *local deep copy* of the active Raw object
so that changes here do not propagate to other analysis tabs.
"""

from __future__ import annotations

import copy
import io
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import active_raw  # noqa: E402, F401
from modules._common import (  # noqa: E402
    activity_series,
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
    selected_subject,
)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# JavaScript: listen for Plotly drawrect events and send the drawn rectangle's
# x0/x1 coordinates to Shiny. The plot itself uses a numeric x-axis internally,
# but the true timestamp labels are stored in trace.customdata. This lets us
# draw rectangles robustly while still populating Start/Stop with real dates.
_SELECT_LISTENER_JS = ui.tags.script(ui.HTML("""
(function() {
    function nearestPlotOutputId(plotEl) {
        var el = plotEl;

        while (el) {
            if (el.id && (el.id === "plot_main" || el.id.endsWith("-plot_main"))) {
                return el.id;
            }
            el = el.parentElement;
        }

        return null;
    }

    function inputIdFromOutputId(outputId) {
        if (outputId === "plot_main") {
            return "selected_range";
        }

        return outputId.replace(/plot_main$/, "selected_range");
    }

    function isNumberLike(v) {
        var n = Number(v);
        return Number.isFinite(n);
    }

    function mapNumericRangeToTimestamps(x0, x1, plotEl) {
        var trace = ((plotEl.data || [])[0] || {});
        var labels = trace.customdata || [];

        if (!labels || labels.length === 0) {
            return {x0: String(x0), x1: String(x1)};
        }

        if (!isNumberLike(x0) || !isNumberLike(x1)) {
            return {x0: String(x0), x1: String(x1)};
        }

        var n0 = Number(x0);
        var n1 = Number(x1);

        var lo = Math.floor(Math.min(n0, n1));
        var hi = Math.ceil(Math.max(n0, n1));

        lo = Math.max(0, Math.min(lo, labels.length - 1));
        hi = Math.max(0, Math.min(hi, labels.length - 1));

        return {
            x0: String(labels[lo]),
            x1: String(labels[hi])
        };
    }

    function extractRange(eventData, plotEl) {
        var x0;
        var x1;

        // Case 1: Plotly sends the full shapes array.
        if (Array.isArray(eventData.shapes) && eventData.shapes.length > 0) {
            var s = eventData.shapes[eventData.shapes.length - 1];

            if (s && s.x0 !== undefined && s.x1 !== undefined) {
                x0 = s.x0;
                x1 = s.x1;
            }
        }

        // Case 2: Plotly sends keys such as shapes[0].x0 / shapes[0].x1.
        var shapeUpdates = {};

        Object.keys(eventData).forEach(function(k) {
            var m = k.match(/^shapes\\[(\\d+)\\]\\.x([01])$/);

            if (m) {
                var idx = parseInt(m[1], 10);
                var coord = "x" + m[2];

                if (!shapeUpdates[idx]) {
                    shapeUpdates[idx] = {};
                }

                shapeUpdates[idx][coord] = eventData[k];
            }
        });

        var idxs = Object.keys(shapeUpdates)
            .map(Number)
            .sort(function(a, b) { return a - b; });

        if (idxs.length > 0) {
            var last = shapeUpdates[idxs[idxs.length - 1]];

            if (last.x0 !== undefined) {
                x0 = last.x0;
            }

            if (last.x1 !== undefined) {
                x1 = last.x1;
            }
        }

        // Case 3: Plotly sends newshape.x0 / newshape.x1.
        if (eventData["newshape.x0"] !== undefined) {
            x0 = eventData["newshape.x0"];
        }

        if (eventData["newshape.x1"] !== undefined) {
            x1 = eventData["newshape.x1"];
        }

        // Final fallback: read the most recent shape from the plot layout.
        if ((x0 === undefined || x1 === undefined) && plotEl.layout && plotEl.layout.shapes) {
            var shapes = plotEl.layout.shapes;

            if (shapes.length > 0) {
                var lastShape = shapes[shapes.length - 1];
                x0 = lastShape.x0;
                x1 = lastShape.x1;
            }
        }

        if (x0 === undefined || x1 === undefined) {
            return null;
        }

        return mapNumericRangeToTimestamps(x0, x1, plotEl);
    }

    function bindPlot(plotEl) {
        if (plotEl._circStudioDrawBound) {
            return;
        }

        plotEl._circStudioDrawBound = true;

        plotEl.on("plotly_relayout", function(eventData) {
            if (!eventData) {
                return;
            }

            var keys = Object.keys(eventData);

            var isShapeEvent = keys.some(function(k) {
                return (
                    k === "shapes" ||
                    k.indexOf("shapes[") === 0 ||
                    k.indexOf("newshape") === 0
                );
            });

            if (!isShapeEvent) {
                return;
            }

            var range = extractRange(eventData, plotEl);

            if (!range || !window.Shiny) {
                return;
            }

            var outputId = nearestPlotOutputId(plotEl);

            if (!outputId) {
                return;
            }

            var inputId = inputIdFromOutputId(outputId);

            Shiny.setInputValue(
                inputId,
                {
                    x0: range.x0,
                    x1: range.x1,
                    nonce: Math.random()
                },
                {priority: "event"}
            );
        });
    }

    function bindAllPlots() {
        if (!window.Shiny) {
            return;
        }

        document.querySelectorAll(".js-plotly-plot").forEach(bindPlot);
    }

    function start() {
        bindAllPlots();

        new MutationObserver(bindAllPlots).observe(
            document.body,
            {childList: true, subtree: true}
        );

        // Extra safety: shinywidgets can re-render Plotly outputs dynamically.
        setInterval(bindAllPlots, 1000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
"""))


@module.ui
def preprocessing_ui():
    return ui.TagList(
        _SELECT_LISTENER_JS,
        ui.layout_sidebar(
            ui.sidebar(
                # ---- non-wear detection ------------------------------------
                ui.h6("Automatic Non-Wear Detection"),
                ui.input_select(
                    "nw_method",
                    "Algorithm",
                    choices={
                        "choi": "Choi et al. (2011) — recommended",
                        "troiano": "Troiano et al. (2008) — NHANES",
                    },
                    selected="choi",
                ),
                ui.input_text(
                    "nw_min_length",
                    "Minimum non-wear length",
                    value="90min",
                    placeholder="e.g. 60min, 2h",
                ),
                ui.input_numeric(
                    "nw_spike_tolerance",
                    "Spike tolerance (epochs)",
                    value=2,
                    min=0,
                    max=10,
                    step=1,
                ),
                ui.input_numeric(
                    "nw_spike_max_counts",
                    "Max spike counts",
                    value=100,
                    min=0,
                    step=10,
                ),
                ui.panel_conditional(
                    "input.nw_method === 'choi'",
                    ui.input_text(
                        "nw_window_size",
                        "Neighbourhood window size",
                        value="30min",
                        placeholder="e.g. 30min, 1h",
                    ),
                ),
                ui.input_action_button(
                    "btn_detect",
                    "Detect Non-Wear",
                    class_="btn-primary btn-sm w-100",
                ),
                ui.hr(),

                # ---- manual mask -------------------------------------------
                ui.h6("Manual Mask Period"),
                ui.input_text(
                    "mask_start",
                    "Start (YYYY-MM-DD HH:MM:SS)",
                    placeholder="2024-01-01 00:00:00",
                ),
                ui.input_text(
                    "mask_stop",
                    "Stop (YYYY-MM-DD HH:MM:SS)",
                    placeholder="2024-01-01 08:00:00",
                ),
                ui.input_action_button(
                    "btn_add_period",
                    "Add Period",
                    class_="btn-outline-secondary btn-sm w-100",
                ),
                ui.p(
                    "Or draw a rectangle on the chart — Start/Stop will be "
                    "filled in automatically.",
                    class_="text-muted small mt-1",
                ),
                ui.hr(),

                # ---- import mask from file ---------------------------------
                ui.h6("Import Mask from File"),
                ui.input_file(
                    "mask_file",
                    "Upload mask log (.csv / .xlsx / .ods)",
                    accept=[".csv", ".xlsx", ".xls", ".ods"],
                    multiple=False,
                ),
                ui.input_action_button(
                    "btn_import_mask",
                    "Import Mask",
                    class_="btn-outline-secondary btn-sm w-100",
                ),
                ui.hr(),

                # ---- apply filters -----------------------------------------
                ui.h6("Apply Filters"),
                ui.input_text(
                    "filt_resample",
                    "Resample epoch (blank = no change)",
                    placeholder="e.g. 1min, 30s",
                    value="",
                ),
                ui.input_checkbox("filt_binarize", "Binarize", value=False),
                ui.panel_conditional(
                    "input.filt_binarize",
                    ui.input_numeric(
                        "filt_threshold",
                        "Binarize threshold",
                        value=0,
                        min=0,
                    ),
                ),
                ui.input_checkbox("filt_impute", "Impute NaN", value=False),
                ui.panel_conditional(
                    "input.filt_impute",
                    ui.input_select(
                        "filt_impute_method",
                        "Imputation method",
                        choices={"mean": "Mean", "median": "Median"},
                        selected="mean",
                    ),
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "btn_apply",
                        "Apply",
                        class_="btn-success btn-sm",
                    ),
                    ui.input_action_button(
                        "btn_reset",
                        "Reset",
                        class_="btn-outline-danger btn-sm",
                    ),
                    col_widths=[6, 6],
                ),
                ui.hr(),

                ui.input_action_button(
                    "btn_export_to_tabs",
                    "Apply & Export to Analysis Tabs",
                    class_="btn-primary btn-sm w-100",
                ),
                ui.p(
                    "Pushes the preprocessed recording to all other tabs.",
                    class_="text-muted small mt-1",
                ),
                ui.hr(),

                # ---- batch mode selectors ----------------------------------
                ui.output_ui("batch_controls"),
                width=320,
            ),

            # ---- main panel ------------------------------------------------
            ui.output_ui("status_msg"),
            ui.h5("Activity trace"),
            ui.p(
                "Draw a rectangle on the chart to populate the Start/Stop "
                "fields, then click Add Period. Use the toolbar (top-right) "
                "to zoom or pan. Red band = non-wear; grey band = NaN after "
                "filter.",
                class_="text-muted small",
            ),
            output_widget("plot_main", height="430px"),
            ui.hr(),
            ui.download_button(
                "btn_export_mask",
                "Export mask to CSV",
                class_="btn-outline-secondary btn-sm",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

@module.server
def preprocessing_server(
    input,
    output,
    session,
    rv_single,
    rv_batch,
    rv_mode,
    rv_active_subject,
):  # noqa: PLR0915

    # Local reactive state ---------------------------------------------------
    # Deep copy of the active Raw so we never mutate the shared reactive value.
    _raw_local = reactive.Value(None)

    # Status message to surface to the user.
    _status = reactive.Value("")

    # Keep the local copy in sync with the active subject --------------------
    @reactive.effect
    def _sync_raw():
        raw = get_active(input, rv_mode, rv_single, rv_batch)

        if raw is not None:
            _raw_local.set(copy.deepcopy(raw))
        else:
            _raw_local.set(None)

        _status.set("")

    # ---- batch mode selectors ---------------------------------------------
    @render.ui
    def batch_controls():
        mode = rv_mode()

        if mode != "batch":
            return ui.div()

        factor_ctrl = factor_filter_controls(rv_mode, rv_batch)
        subj_ctrl = subject_controls(
            input,
            rv_mode,
            rv_batch,
            default=rv_active_subject(),
        )

        controls = []
        if factor_ctrl:
            controls.append(factor_ctrl)
        if subj_ctrl:
            controls.append(subj_ctrl)

        if controls:
            return ui.div(ui.hr(), ui.h6("Batch"), *controls)

        return ui.div()

    # ---- status message ----------------------------------------------------
    @render.ui
    def status_msg():
        msg = _status()

        if not msg:
            return ui.div()

        colour = "danger" if msg.startswith("Error") else "success"

        return ui.div(
            ui.tags.div(
                msg,
                class_=f"alert alert-{colour} alert-dismissible py-2 px-3",
                role="alert",
            ),
            class_="mb-2",
        )

    # ---- automatic non-wear detection -------------------------------------
    @reactive.effect
    @reactive.event(input.btn_detect)
    def _on_detect():
        raw = _raw_local()

        if raw is None:
            _status.set("Error: No recording loaded. Please upload a file first.")
            return

        method = input.nw_method()
        min_length = (input.nw_min_length() or "90min").strip()
        spike_tol = int(input.nw_spike_tolerance())
        spike_max = int(input.nw_spike_max_counts())

        kwargs = {}
        if method == "choi":
            kwargs["window_size"] = (input.nw_window_size() or "30min").strip()

        try:
            # Preserve manually-added mask periods before detection, so that
            # automatic and manual periods are always additive.
            existing_mask = raw._mask.copy() if raw._mask is not None else None

            raw.detect_nonwear(
                method=method,
                min_length=min_length,
                spike_tolerance=spike_tol,
                spike_max_counts=spike_max,
                **kwargs,
            )

            # Merge: union of non-wear periods. Since mask uses 0 = non-wear
            # and 1 = wear, taking the minimum makes non-wear "win".
            if existing_mask is not None and raw._mask is not None:
                aligned = existing_mask.reindex(raw._mask.index, fill_value=1)
                raw._mask = raw._mask.combine(aligned, min)

            m = raw.mask
            n_nonwear = int((m == 0).sum()) if m is not None else 0

            if hasattr(raw.frequency, "total_seconds"):
                epoch_sec = raw.frequency.total_seconds()
            else:
                epoch_sec = 60

            nonwear_hours = round(n_nonwear * epoch_sec / 3600, 1)

            # Count distinct non-wear runs.
            n_runs = 0
            if m is not None and n_nonwear > 0:
                n_runs = int((m.diff().fillna(m.iloc[0] - 1) < 0).sum())

            # Deep copy so Shiny sees a new object reference and re-renders.
            _raw_local.set(copy.deepcopy(raw))

            _status.set(
                f"Non-wear detection complete ({method.capitalize()}). "
                f"Flagged {n_nonwear} epochs ({nonwear_hours} h) "
                f"across {n_runs} separate period(s). "
                f"Zoom in on the plot to inspect individual periods."
            )

        except Exception as exc:  # noqa: BLE001
            _status.set(f"Error during detection: {exc}")

    # ---- manual add mask period -------------------------------------------
    @reactive.effect
    @reactive.event(input.btn_add_period)
    def _on_add_period():
        raw = _raw_local()

        if raw is None:
            _status.set("Error: No recording loaded.")
            return

        start = (input.mask_start() or "").strip()
        stop = (input.mask_stop() or "").strip()

        if not start or not stop:
            _status.set("Error: Please enter both a start and a stop time.")
            return

        try:
            raw.add_mask_period(start, stop)
            _raw_local.set(copy.deepcopy(raw))
            _status.set(f"Mask period added: {start} → {stop}.")

        except Exception as exc:  # noqa: BLE001
            _status.set(f"Error adding mask period: {exc}")

    # ---- drawn rectangle → populate Start/Stop text inputs -----------------
    @reactive.effect
    @reactive.event(input.selected_range)
    def _on_drawn_rect():
        data = input.selected_range()

        if not data:
            return

        x0 = str(data.get("x0", "")).strip()
        x1 = str(data.get("x1", "")).strip()

        if not x0 or not x1:
            return

        try:
            start = pd.to_datetime(x0)
            stop = pd.to_datetime(x1)

            if start > stop:
                start, stop = stop, start

            start_s = start.strftime("%Y-%m-%d %H:%M:%S")
            stop_s = stop.strftime("%Y-%m-%d %H:%M:%S")

        except Exception:
            # Fallback for unusual date formats emitted by Plotly.
            start_s, stop_s = sorted([x0, x1])

        ui.update_text("mask_start", value=start_s)
        ui.update_text("mask_stop", value=stop_s)

        _status.set(
            f"Selection: {start_s} → {stop_s}. Click 'Add Period' to confirm."
        )

    # ---- import mask from file --------------------------------------------
    @reactive.effect
    @reactive.event(input.btn_import_mask)
    def _on_import_mask():
        raw = _raw_local()

        if raw is None:
            _status.set("Error: No recording loaded.")
            return

        finfo = input.mask_file()

        if not finfo:
            _status.set("Error: Please select a mask log file first.")
            return

        src = finfo[0]
        suffix = Path(src["name"]).suffix
        tmp = Path(tempfile.mkdtemp()) / (Path(src["name"]).stem + suffix)

        shutil.copy(src["datapath"], tmp)

        try:
            raw.add_mask_periods(str(tmp))
            _raw_local.set(copy.deepcopy(raw))
            _status.set(f"Mask imported from '{src['name']}'.")

        except Exception as exc:  # noqa: BLE001
            _status.set(f"Error importing mask: {exc}")

    # ---- apply filters -----------------------------------------------------
    @reactive.effect
    @reactive.event(input.btn_apply)
    def _on_apply():
        raw = _raw_local()

        if raw is None:
            _status.set("Error: No recording loaded.")
            return

        new_freq = (input.filt_resample() or "").strip() or None
        binarize = bool(input.filt_binarize())
        threshold = float(input.filt_threshold()) if binarize else 0
        impute = bool(input.filt_impute())
        imp_method = input.filt_impute_method() if impute else "mean"
        has_mask = raw._mask is not None

        try:
            raw.apply_filters(
                new_freq=new_freq,
                binarize=binarize,
                threshold=threshold,
                apply_mask=has_mask,
                impute_nan=impute,
                imputation_method=imp_method,
            )

            _raw_local.set(copy.deepcopy(raw))
            _status.set("Filters applied.")

        except Exception as exc:  # noqa: BLE001
            _status.set(f"Error applying filters: {exc}")

    # ---- reset -------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.btn_reset)
    def _on_reset():
        raw = get_active(input, rv_mode, rv_single, rv_batch)

        if raw is not None:
            _raw_local.set(copy.deepcopy(raw))

        _status.set("Reset to original recording.")

    # ---- activity plot: mask overlay + post-filter state -------------------
    @render_widget
    def plot_main():
        raw = _raw_local()

        if raw is None:
            return empty_fig("Upload a file on the 'Data Upload' tab first.")

        act = activity_series(raw)

        if act is None:
            return empty_fig("No activity data available.")

        fig = go.Figure()

        # Internal numeric x-axis: robust for long actigraphy recordings and
        # reliable for rectangle drawing.
        x_pos = np.arange(len(act), dtype=float)

        # Keep the real timestamps separately. These are used for hover text and
        # for populating the Start/Stop fields after a rectangle is drawn.
        if isinstance(act.index, pd.DatetimeIndex):
            time_labels = act.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
            parsed_dates = act.index
        else:
            time_labels = [str(v) for v in act.index]
            parsed_dates = pd.to_datetime(time_labels, errors="coerce")

        # Use a small number of readable tick labels instead of showing every
        # epoch on the x-axis.
        if len(act) > 1:
            tick_idx = np.unique(
                np.linspace(
                    0,
                    len(act) - 1,
                    num=min(7, len(act)),
                    dtype=int,
                )
            )
        else:
            tick_idx = np.array([0])

        tick_vals = tick_idx.astype(float).tolist()

        if len(parsed_dates) == len(act) and pd.Series(parsed_dates).notna().mean() > 0.9:
            date_min = pd.to_datetime(parsed_dates).min()
            date_max = pd.to_datetime(parsed_dates).max()
            span_days = (date_max - date_min).total_seconds() / 86400

            if span_days > 2:
                tick_text = [
                    pd.to_datetime(parsed_dates[i]).strftime("%b %d")
                    for i in tick_idx
                ]
            else:
                tick_text = [
                    pd.to_datetime(parsed_dates[i]).strftime("%b %d<br>%H:%M")
                    for i in tick_idx
                ]
        else:
            tick_text = [time_labels[i] for i in tick_idx]

        # Convert activity to plain numeric values. This avoids silent rendering
        # problems when the Series uses pandas nullable/object dtypes.
        y_plot = pd.to_numeric(act, errors="coerce").to_numpy(dtype=float)

        color = "#2ca02c" if np.isnan(y_plot).any() else "#1f77b4"

        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=y_plot,
                mode="lines",
                line=dict(width=0.8, color=color),
                name="Activity",
                customdata=time_labels,
                hovertemplate=(
                    "Date/Time: %{customdata}<br>"
                    "Activity counts: %{y}<extra></extra>"
                ),
            )
        )

        # --- non-wear mask: one rectangle per contiguous period --------------
        mask = None

        try:
            mask = raw.mask
        except Exception:
            pass

        if mask is not None:
            aligned = mask.reindex(act.index)
            is_nw = (aligned == 0).values

            edges = np.diff(is_nw.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            for s, e in zip(starts, ends):
                fig.add_shape(
                    type="rect",
                    xref="x",
                    x0=x_pos[s],
                    x1=x_pos[min(e, len(x_pos) - 1)],
                    yref="paper",
                    y0=0,
                    y1=0.07,
                    fillcolor="rgba(214,39,40,0.75)",
                    line=dict(width=0),
                    layer="above",
                )

        # --- NaN regions: grey band just above red --------------------------
        if np.isnan(y_plot).any():
            is_nan = np.isnan(y_plot)

            edges_n = np.diff(is_nan.astype(int), prepend=0, append=0)
            nan_starts = np.where(edges_n == 1)[0]
            nan_ends = np.where(edges_n == -1)[0]

            for s, e in zip(nan_starts, nan_ends):
                fig.add_shape(
                    type="rect",
                    xref="x",
                    x0=x_pos[s],
                    x1=x_pos[min(e, len(x_pos) - 1)],
                    yref="paper",
                    y0=0.07,
                    y1=0.12,
                    fillcolor="rgba(127,127,127,0.6)",
                    line=dict(width=0),
                    layer="above",
                )

        finite_y = y_plot[np.isfinite(y_plot)]

        if len(finite_y):
            y_min_data = float(np.nanmin(finite_y))
            y_max_data = float(np.nanmax(finite_y))

            # For actigraphy, values are usually non-negative. Keeping the
            # lower reference at zero makes the signal easier to interpret.
            y_lower_ref = min(0.0, y_min_data)
            y_upper_ref = y_max_data

            y_span = y_upper_ref - y_lower_ref

            if y_span <= 0:
                y_span = max(abs(y_upper_ref), 1.0)

            # Add visual breathing room below zero and above the maximum.
            y_pad = 0.05 * y_span

            y_min = y_lower_ref - y_pad
            y_max = y_upper_ref + y_pad

        else:
            y_min = -1.0
            y_max = 1.0

        fig.update_layout(
            xaxis=dict(
                title="Date/Time",
                type="linear",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            yaxis=dict(
                title="Activity counts",
                range=[y_min, y_max],
                zeroline=True,
            ),
            margin=dict(l=40, r=20, t=30, b=45),
            height=430,
            autosize=True,
            showlegend=False,
            dragmode="drawrect",
            newshape=dict(
                line=dict(color="rgba(214,39,40,0.7)", width=1.5),
                fillcolor="rgba(214,39,40,0.15)",
            ),
        )

        return fig

    # ---- export preprocessed recording to all analysis tabs ----------------
    @reactive.effect
    @reactive.event(input.btn_export_to_tabs)
    def _on_export_to_tabs():
        raw = _raw_local()

        if raw is None:
            _status.set("Error: No recording loaded.")
            return

        try:
            mode = rv_mode()

            if mode == "single":
                rv_single.set(copy.deepcopy(raw))
            else:
                # In batch mode update the active subject in the collection.
                batch = rv_batch()
                subj = selected_subject(input)

                if batch is not None and subj:
                    batch[subj] = copy.deepcopy(raw)
                    rv_batch.set(copy.deepcopy(batch))

            _status.set("Recording exported to all analysis tabs.")

        except Exception as exc:  # noqa: BLE001
            _status.set(f"Error exporting: {exc}")

    # ---- export mask to CSV ------------------------------------------------
    @render.download(filename="mask_export.csv")
    def btn_export_mask():
        raw = _raw_local()
        mask = None

        if raw is not None:
            try:
                mask = raw.mask
            except Exception:
                pass

        if mask is None:
            # Return an empty CSV with headers only.
            buf = io.StringIO()
            pd.DataFrame(columns=["Start_time", "Stop_time"]).to_csv(
                buf,
                index=False,
            )
            yield buf.getvalue()
            return

        # Convert the binary mask series to start/stop intervals of non-wear.
        in_nonwear = (mask == 0).astype(int)
        edges = in_nonwear.diff().fillna(in_nonwear.iloc[0])

        starts = mask.index[edges == 1].tolist()
        ends = mask.index[edges == -1].tolist()

        if in_nonwear.iloc[0] == 1:
            starts = [mask.index[0]] + starts

        if in_nonwear.iloc[-1] == 1:
            ends = ends + [mask.index[-1]]

        rows = [
            {"Start_time": s, "Stop_time": e}
            for s, e in zip(starts, ends)
        ]

        df = pd.DataFrame(rows, columns=["Start_time", "Stop_time"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)

        yield buf.getvalue()