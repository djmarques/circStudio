"""Tab 3 - Cosinor analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lmfit import Parameters
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from circstudio.analysis import Cosinor  # noqa: E402
from modules._common import (  # noqa: E402
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)


@module.ui
def cosinor_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),
            ui.input_slider(
                "period", "Period (min)", min=240, max=2880, value=1440,
                step=30,
            ),
            ui.input_select(
                "method",
                "Fitting method",
                ["leastsq", "least_squares", "differential_evolution"],
                selected="leastsq",
            ),
            ui.input_numeric("mesor", "Mesor (init)", value=50),
            ui.input_numeric("amplitude", "Amplitude (init)", value=50),
            ui.input_numeric("acrophase", "Acrophase (init, rad)", value=3.14),
            ui.input_action_button(
                "run", "Run fit", class_="btn-primary btn-sm"
            ),
            width=320,
        ),
        ui.div(
            ui.p(
                "Cosinor analysis fits a cosine function to the activity time series "
                "to characterise the dominant 24-hour rhythm. The fitted parameters "
                "— mesor (rhythmic mean), amplitude (half the peak-to-trough range), "
                "and acrophase (clock time of the peak) — describe the level, strength, "
                "and timing of the oscillation."
            ),
            class_="text-muted small mb-3",
        ),
        ui.h4("Raw data with best-fit cosine"),
        output_widget("fit_plot"),
        ui.hr(),
        ui.h4("Fit parameters"),
        ui.output_data_frame("params_table"),
    )


@module.server
def cosinor_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    _fit = reactive.Value(None)  # (model, params, series)

    @render.ui
    def factor_filter_ctrl():
        return factor_filter_controls(rv_mode, rv_batch)

    @render.ui
    def subject_ctrl():
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    @reactive.effect
    @reactive.event(input.run)
    def _do_fit():
        raw = get_active(input, rv_mode, rv_single, rv_batch)
        if raw is None:
            ui.notification_show(
                "Load a recording first.", type="warning"
            )
            return
        s = raw.activity
        params = Parameters()
        params.add("Amplitude", value=float(input.amplitude()), min=0)
        params.add(
            "Acrophase", value=float(input.acrophase()), min=0, max=2 * np.pi
        )
        params.add("Period", value=float(input.period()), min=0)
        params.add("Mesor", value=float(input.mesor()), min=0)
        try:
            model = Cosinor()
            result = model.fit(s, params=params, method=input.method())
        except Exception as exc:
            ui.notification_show(
                f"Fit failed: {exc}", type="error", duration=8
            )
            return
        _fit.set((model, result, s))
        ui.notification_show("Cosinor fit complete.", type="message")

    @render_widget
    def fit_plot():
        data = _fit.get()
        if data is None:
            return empty_fig("Set parameters and click 'Run fit'.")
        model, result, s = data
        try:
            return model.plot(s, result.params)
        except Exception as exc:
            return empty_fig(f"Could not plot fit: {exc}")

    @render.data_frame
    def params_table():
        data = _fit.get()
        if data is None:
            return pd.DataFrame({"info": ["No fit yet."]})
        model, result, s = data
        p = result.params
        period_min = float(p["Period"].value)
        acrophase_rad = float(p["Acrophase"].value)
        # Convert acrophase (radians) to clock hours of the peak.
        acrophase_hours = (acrophase_rad / (2 * np.pi)) * (period_min / 60.0)

        # R-squared from best-fit residuals.
        try:
            fitted = model.best_fit(s, p)
            ss_res = float(np.nansum((s.values - fitted.values) ** 2))
            ss_tot = float(np.nansum((s.values - np.nanmean(s.values)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        except Exception:
            r2 = float("nan")

        df = pd.DataFrame(
            {
                "parameter": [
                    "Mesor",
                    "Amplitude",
                    "Acrophase (hours)",
                    "Period (hours)",
                    "R-squared",
                    "Fit success",
                ],
                "value": [
                    round(float(p["Mesor"].value), 4),
                    round(float(p["Amplitude"].value), 4),
                    round(acrophase_hours, 4),
                    round(period_min / 60.0, 4),
                    round(r2, 4),
                    bool(result.success),
                ],
            }
        )
        return render.DataGrid(df, width="100%")
