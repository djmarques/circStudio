"""
Tab 1 - Daily profile & circadian metrics (Activity + Light).

All metrics are computed once over the whole recording and shown in plain
tables, split into Activity and Light sub-tabs. No per-period computation.
"""
# Libraries for data processing and plotting
import numpy as np
import pandas as pd

# Shiny imports for UI and server functionality
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

# Library imports used to locate and import the circStudio package
from pathlib import Path
from import_paths import _add_import_path

# Add the parent directory of this file to the import path to ensure that
# the circStudio package can be imported correctly
app_dir = Path(__file__).resolve().parent.parent
_add_import_path(app_dir)

# Import circStudio modules after adding the import path
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
# Value-formatting and table-building helpers
# --------------------------------------------------------------------------
def _hhmm(value) -> str:
    """Convert a time-like metric value to a clock time in ``HH:MM`` format.

    Timedeltas are interpreted as elapsed seconds, timestamps retain their
    clock time, and numeric values are interpreted as decimal hours. Values are
    wrapped to a 24-hour clock. Invalid values are displayed as ``"n/a"`` so a
    single formatting error does not prevent the table from rendering.
    """
    try:
        # Convert a duration to its total number of seconds.
        if isinstance(value, pd.Timedelta):
            total_seconds = int(value.total_seconds())

        # A timestamp already contains a clock time and can be formatted directly.
        elif isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M")

        # Treat other numeric results as decimal hours and convert them to seconds.
        else:
            total_seconds = int(float(value) * 3600)

        # Wrap durations longer than one day onto a standard 24-hour clock.
        total_seconds %= 86400

        # Derive the hour and minute components from the wrapped second count.
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        # Display both components with two digits for consistent table formatting.
        return f"{hours:02d}:{minutes:02d}"

    # Return a stable placeholder if the result cannot be interpreted as a time.
    except Exception:
        return "n/a"


def _num(value, nd=4):
    """Round a numeric metric for display while preserving non-numeric values."""
    try:
        # Convert NumPy and other numeric scalar types to float before rounding.
        return round(float(value), nd)

    # Leave placeholders or other non-numeric results unchanged.
    except Exception:
        return value


def _scalar(fn):
    """Run one metric function and report whether it completed successfully."""
    try:
        # Return both the calculated value and a success flag used for formatting.
        return fn(), True

    # Isolate metric failures so the remaining metrics can still be calculated.
    except Exception:
        return "n/a", False


def _as_value(result, take=1):
    """Extract one component from tuple results or return a scalar unchanged."""
    # Some circadian functions return ``(onset, value)`` rather than one scalar.
    if isinstance(result, tuple):
        return result[take]

    # Scalar metric results require no extraction.
    return result


def _scalarize(value):
    """Reduce a pandas result to one float for display in the metric table."""
    # Flatten pandas objects because some analysis functions return one-cell tables.
    if isinstance(value, (pd.Series, pd.DataFrame)):
        values = np.asarray(value).ravel()

        # Preserve an existing scalar when the pandas object contains one value.
        if values.size == 1:
            return float(values[0])

        # Summarize multi-value results with a mean that ignores missing values.
        return float(np.nanmean(values))

    # Convert an existing scalar or NumPy scalar directly to a Python float.
    return float(value)


def _count(value):
    """Count values in an array-like result returned by an exposure function."""
    try:
        # For pandas objects, count rows to retain the original epoch convention.
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return int(value.shape[0])

        # For other array-like objects, count all elements after NumPy conversion.
        return int(np.asarray(value).size)

    # Fall back to the object's length when NumPy conversion is unavailable.
    except Exception:
        return len(value)


def _build_table(metric_specification):
    """Calculate a set of metrics and return a display-ready table.

    Parameters
    ----------
    metric_specification:
        Sequence of ``(metric name, calculation function, unit)`` tuples. Each
        function is called independently so that one failed metric is shown as
        ``"n/a"`` without removing successful results from the table.

    Returns
    -------
    pandas.DataFrame
        Table with ``Metric``, ``Value``, and ``Unit`` columns.
    """
    # Collect one display row for each requested metric.
    rows = []

    # Calculate and format the metrics in the order defined by the caller.
    for metric_name, metric_function, unit in metric_specification:
        value, calculation_succeeded = _scalar(metric_function)

        # Format timing metrics as clock times rather than decimal values.
        if calculation_succeeded and unit == "HH:MM":
            value = _hhmm(value)

        # Round other successfully calculated metrics for concise display.
        elif calculation_succeeded:
            value = _num(value)

        # Retain the metric label and unit even when its calculation failed.
        rows.append((metric_name, value, unit))

    # Convert the ordered rows into the common table structure used by the UI.
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


# --------------------------------------------------------------------------
# User interface
# --------------------------------------------------------------------------
# Register this function as the UI component of a reusable Shiny module.
@module.ui
def daily_profile_ui():
    """Create the controls, profile plot, and metric tables for this tab."""
    # Place analysis controls in a sidebar and results in the main content area.
    return ui.layout_sidebar(
        ui.sidebar(
            # Render factor and subject selectors from the currently loaded data.
            ui.output_ui("factor_filter_ctrl"),
            ui.output_ui("subject_ctrl"),

            # Choose which measurement is folded into the displayed 24-hour profile.
            ui.input_radio_buttons(
                "profile_signal",
                "Profile plot signal",
                choices=["Activity", "Light"],
                selected="Activity",
                inline=True,
            ),

            # Control the visual representation of the daily profile.
            ui.input_switch("cyclic", "Cyclic profile", False),
            ui.input_switch("logscale", "Log scale", False),
            ui.hr(),

            # Collect parameters used only by the activity calculations.
            ui.h6("Activity options"),
            ui.input_numeric(
                "whs",
                "AonT/AoffT window half-size (pts)",
                value=12,
            ),
            ui.input_text("lmx_len", "LMX window length", value="5h"),
            ui.input_switch("lmx_lowest", "LMX lowest (else highest)", True),

            # Collect thresholds, time limits, and aggregation settings for light.
            ui.h6("Light options"),
            ui.input_numeric("threshold", "Threshold (lux)", value=10),
            ui.input_text("bins", "Summary bins", value="1h"),
            ui.input_text(
                "start_time",
                "Start time (HH:MM:SS, optional)",
                value="",
            ),
            ui.input_text(
                "stop_time",
                "Stop time (HH:MM:SS, optional)",
                value="",
            ),
            ui.input_select(
                "agg",
                "Light exposure aggregation",
                ["mean", "median", "std", "min", "max"],
                selected="mean",
            ),

            # Trigger metric calculation explicitly after parameters are selected.
            ui.input_action_button(
                "run",
                "Run metrics",
                class_="btn-primary btn-sm",
            ),
            width=340,
        ),

        # Briefly explain the daily profile and representative circadian metrics.
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

        # Reserve space for the reactive 24-hour profile figure.
        ui.h4("Average daily profile"),
        output_widget("profile_plot"),
        ui.hr(),

        # Separate activity, light, and binned light summaries into sub-tabs.
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
# Server logic
# --------------------------------------------------------------------------
# Register this function as the server component of the Shiny module.
@module.server
def daily_profile_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    """Calculate and render daily profiles and whole-recording metrics.

    The server selects the active single or batch recording, extracts its
    activity and light signals, and calculates metrics when ``Run metrics`` is
    clicked. The resulting tables are stored in one reactive value so each
    output is updated from the same calculation run.
    """
    # Store all tables produced by the most recent metric calculation.
    _results = reactive.Value(None)

    # Render factor filters that reflect the current single- or batch-data mode.
    @render.ui
    def factor_filter_ctrl():
        """Display factor-filter controls for the currently loaded data."""
        return factor_filter_controls(rv_mode, rv_batch)

    # Render the subject selector and preserve the shared active-subject state.
    @render.ui
    def subject_ctrl():
        """Display subject controls for the current batch selection."""
        return subject_controls(
            input,
            rv_mode,
            rv_batch,
            rv_active_subject(),
        )

    def _raw():
        """Return the recording selected by the current mode and UI controls."""
        return get_active(input, rv_mode, rv_single, rv_batch)

    def _opt(text):
        """Convert an optional text input to stripped text or ``None``."""
        # Treat whitespace-only time limits as unspecified analysis boundaries.
        text = (text or "").strip()
        return text or None

    # Recalculate metrics only in response to a click on the Run button.
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        """Calculate activity and light metrics for the active recording."""
        # Resolve the active recording from the current loading mode and subject.
        raw = _raw()

        # Stop early when no recording is available for analysis.
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return

        # Extract the available activity and light time series independently.
        activity = activity_series(raw)
        light = light_series(raw)

        # ------------------------------------------------------------------
        # Activity metrics
        # ------------------------------------------------------------------
        # Use ``None`` to indicate that the recording has no activity channel.
        activity_table = None

        # Calculate activity metrics only when an activity signal is available.
        if activity is not None:
            # Use the default onset/offset half-window if the numeric input is empty.
            window_half_size = int(input.whs() or 12)

            # Use a five-hour LMX window if the text input is empty or whitespace.
            lmx_length = (input.lmx_len() or "5h").strip() or "5h"

            # Choose whether LMX identifies the lowest or highest activity window.
            use_lowest_lmx = bool(input.lmx_lowest())

            # Define the activity metrics, calculation calls, and display units.
            activity_table = _build_table(
                [
                    (
                        "AUC of daily profile",
                        lambda: daily_profile_auc(activity),
                        "",
                    ),
                    (
                        "Activity onset (AonT)",
                        lambda: AonT(activity, whs=window_half_size),
                        "HH:MM",
                    ),
                    (
                        "Activity offset (AoffT)",
                        lambda: AoffT(activity, whs=window_half_size),
                        "HH:MM",
                    ),
                    (
                        "Temporal centroid",
                        lambda: temporal_centroid(activity),
                        "HH:MM",
                    ),
                    (
                        "Spectral centroid",
                        lambda: spectral_centroid(activity),
                        "",
                    ),
                    ("IS (interdaily stability)", lambda: IS(activity), ""),
                    ("IV (intradaily variability)", lambda: IV(activity), ""),
                    ("L5", lambda: _as_value(l5(activity)), ""),
                    (
                        "L5 onset",
                        lambda: _as_value(l5(activity), 0),
                        "HH:MM",
                    ),
                    ("M10", lambda: _as_value(m10(activity)), ""),
                    (
                        "M10 onset",
                        lambda: _as_value(m10(activity), 0),
                        "HH:MM",
                    ),
                    ("RA (relative amplitude)", lambda: ra(activity), ""),
                    (
                        "LMX (%s, %s)"
                        % (
                            lmx_length,
                            "low" if use_lowest_lmx else "high",
                        ),
                        lambda: _as_value(
                            lmx(
                                activity,
                                length=lmx_length,
                                lowest=use_lowest_lmx,
                            )
                        ),
                        "",
                    ),
                    ("ADAT", lambda: adat(activity), "counts"),
                    (
                        "Time barycentre",
                        lambda: get_time_barycentre(activity),
                        "epoch idx",
                    ),
                    ("kRA", lambda: kRA(activity), ""),
                    ("kAR", lambda: kAR(activity), ""),
                ]
            )

        # ------------------------------------------------------------------
        # Light metrics
        # ------------------------------------------------------------------
        # Use ``None`` placeholders when the recording has no light channel.
        light_table = None
        summary_table = None

        # Calculate light metrics only when a light signal is available.
        if light is not None:
            # Use 10 lux as the default threshold when the numeric input is empty.
            threshold = float(input.threshold() or 10)

            # Convert blank time limits to ``None`` for unrestricted calculations.
            start_time = _opt(input.start_time())
            stop_time = _opt(input.stop_time())

            # Read the summary function selected for light-exposure values.
            aggregation = input.agg()

            # Define the light metrics, calculation calls, and display units.
            light_table = _build_table(
                [
                    (
                        "AUC of daily light profile",
                        lambda: daily_profile_auc(light),
                        "",
                    ),
                    (
                        "Light exposure (%s)" % aggregation,
                        lambda: _scalarize(
                            light_exposure(
                                light,
                                threshold=threshold,
                                start_time=start_time,
                                stop_time=stop_time,
                                agg=aggregation,
                            )
                        ),
                        "lux",
                    ),
                    (
                        "Time above threshold (TAT)",
                        lambda: _scalarize(
                            TAT(
                                light,
                                threshold=threshold,
                                start_time=start_time,
                                stop_time=stop_time,
                                oformat="minute",
                            )
                        ),
                        "min",
                    ),
                    (
                        "Values above threshold (count)",
                        lambda: _count(VAT(light, threshold=threshold)),
                        "epochs",
                    ),
                    (
                        "Mean light timing (MLiT)",
                        lambda: _scalarize(mlit(light, threshold=threshold)),
                        "min since midnight",
                    ),
                    (
                        "Temporal centroid",
                        lambda: temporal_centroid(light),
                        "HH:MM",
                    ),
                    (
                        "Spectral centroid",
                        lambda: spectral_centroid(light),
                        "",
                    ),
                ]
            )

            # Calculate binned light statistics separately because they form a table.
            try:
                summary_bins = input.bins() or "1h"
                summary_table = summary_stats(
                    light,
                    bins=summary_bins,
                ).reset_index()

            # Display the calculation error without preventing other outputs.
            except Exception as exc:
                summary_table = pd.DataFrame(
                    {"info": [f"summary_stats error: {exc}"]}
                )

        # Update all three outputs together from the same calculation run.
        _results.set(
            {
                "activity": activity_table,
                "light": light_table,
                "summary": summary_table,
            }
        )

        # Confirm to the user that the requested metric run has completed.
        ui.notification_show("Metrics computed.", type="message")

    # ----------------------------------------------------------------------
    # Reactive outputs
    # ----------------------------------------------------------------------
    # Rebuild the profile plot when its recording or display controls change.
    @render_widget
    def profile_plot():
        """Render the average 24-hour activity or light profile."""
        # Resolve the recording selected by the current mode and subject controls.
        raw = _raw()

        # Show an instructional placeholder until a recording is loaded.
        if raw is None:
            return empty_fig("Load a recording on the Data Upload tab.")

        # Extract light when it is the selected profile signal.
        if input.profile_signal() == "Light":
            profile_series = light_series(raw)

            # Explain why a light profile cannot be drawn for this recording.
            if profile_series is None:
                return empty_fig("No light channel in this recording.")

        # Otherwise, use activity as the profile signal.
        else:
            profile_series = activity_series(raw)

        # Fold the selected signal across 24 hours using the display options.
        try:
            return daily_profile(
                profile_series,
                cyclic=bool(input.cyclic()),
                plot=True,
                log=bool(input.logscale()),
            )

        # Replace plotting failures with a user-readable placeholder figure.
        except Exception as exc:
            return empty_fig(f"Could not build profile: {exc}")

    # Render the activity table whenever the stored metric results change.
    @render.data_frame
    def activity_table():
        """Display the most recently calculated activity metrics."""
        # Read the complete result bundle from the latest calculation run.
        results = _results.get()

        # Prompt for a calculation until results have been generated.
        if not results:
            return pd.DataFrame({"info": ["Click 'Run metrics'."]})

        # Retrieve the activity table from the shared result bundle.
        table = results.get("activity")

        # Report when the active recording contains no activity channel.
        if table is None:
            return pd.DataFrame({"info": ["No activity channel."]})

        # Present the metrics in Shiny's interactive data-grid component.
        return render.DataGrid(table, width="100%")

    # Render the light table whenever the stored metric results change.
    @render.data_frame
    def light_table():
        """Display the most recently calculated light metrics."""
        # Read the complete result bundle from the latest calculation run.
        results = _results.get()

        # Prompt for a calculation until results have been generated.
        if not results:
            return pd.DataFrame({"info": ["Click 'Run metrics'."]})

        # Retrieve the light table from the shared result bundle.
        table = results.get("light")

        # Report when the active recording contains no light channel.
        if table is None:
            return pd.DataFrame(
                {"info": ["No light channel in this recording."]}
            )

        # Present the metrics in Shiny's interactive data-grid component.
        return render.DataGrid(table, width="100%")

    # Render the binned light summary whenever the stored results change.
    @render.data_frame
    def summary_stats_table():
        """Display the most recently calculated binned light statistics."""
        # Read the complete result bundle from the latest calculation run.
        results = _results.get()

        # Explain that this output requires both light data and a metric run.
        if not results or results.get("summary") is None:
            return pd.DataFrame(
                {"info": ["Requires a light channel; click 'Run metrics'."]}
            )

        # Present the summary statistics in Shiny's interactive data grid.
        return render.DataGrid(results["summary"], width="100%")