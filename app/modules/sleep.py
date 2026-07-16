"""
Tab 2 - Sleep scoring and full sleep-metric reporting.

This module defines the user interface and server logic for classifying an
actigraphy recording into sleep and wake epochs. It also summarises sleep and
activity bouts, daily sleep timing, wake after sleep onset (WASO), activity
onset and offset, and a consolidated set of sleep metrics.

The selected recording is retrieved from the shared application state. Results
are calculated only after the user clicks ``Run`` and are then stored in a
module-level reactive value so that all tables and plots use the same scoring
result.
"""
# Standard library imports for file handling and temporary directories
import io
import shutil
import tempfile
from pathlib import Path

# Third-party imports for numerical and DataFrame operations
import numpy as np
import pandas as pd

# Third-party imports for interactive plotting
import plotly.graph_objects as go

# Shiny imports for building the web application interface and reactive components
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

# Import the shared helper used to locate the local circStudio package.
from import_paths import _add_import_path

# Add the application directory to Python's search path before local imports.
app_dir = Path(__file__).resolve().parent.parent
_add_import_path(app_dir)

# Import the sleep-scoring algorithms and summary metrics used by this tab.
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

# Import helpers shared by the circStudio Shiny modules.
from modules._common import (  # noqa: E402
    activity_series,
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
)

# Map each user-facing algorithm label to its function name in ``sleep.py``.
_ALGO_FUNC = {
    "Cole-Kripke": "Cole_Kripke",
    "Sadeh": "Sadeh",
    "Scripps": "Scripps",
    "Oakley": "Oakley",
    "Roenneberg": "Roenneberg",
    "Crespo": "Crespo",
    "CSM": "CSM",
}

# Identify algorithms that the summary functions can call using activity alone.
_ARGFREE = {"Roenneberg", "Sadeh", "Scripps", "Oakley"}


def _ck_settings(freq: pd.Timedelta) -> str:
    """Select the Cole-Kripke aggregation setting for the recording epoch.

    Parameters
    ----------
    freq:
        Duration of one activity epoch.

    Returns
    -------
    str
        Cole-Kripke setting appropriate for 10-second, 30-second, or longer
        epochs.
    """
    # Preserve non-overlapping maxima for recordings sampled every 10 seconds.
    if freq <= pd.Timedelta("10s"):
        return "10sec_max_non_overlap"

    # Use the equivalent 30-second setting for epochs up to 30 seconds.
    if freq <= pd.Timedelta("30s"):
        return "30sec_max_non_overlap"

    # Average longer epochs, including the common one-minute resolution.
    return "mean"


def _zcm_series(raw):
    """Return the recording's zero-crossing-mode channel when available."""
    # Read the underlying data table without assuming that every Raw object has it.
    df = getattr(raw, "df", None)

    # Search both supported zero-crossing-mode column names in priority order.
    if df is not None and hasattr(df, "columns"):
        for c in ("ZCMn", "ZCM"):
            if c in df.columns:
                return df[c]

    # Signal that CSM scoring cannot be used for this recording.
    return None


def _summary_algo(ui_algo: str) -> str:
    """Choose an algorithm compatible with the sleep-summary functions.

    The summary functions resolve algorithm names internally and pass only the
    activity data. Cole-Kripke, Crespo, and CSM require extra arguments, so the
    summary panels use Roenneberg for those selections.
    """
    # Translate the interface label into the function name used by sleep.py.
    fn = _ALGO_FUNC.get(ui_algo, "Roenneberg")

    # Keep directly supported algorithms and otherwise use the package default.
    return fn if fn in _ARGFREE else "Roenneberg"


def _score(ui_algo, raw, threshold, rescoring):
    """Score every activity epoch as sleep or wake.

    Parameters
    ----------
    ui_algo:
        User-facing name of the selected scoring algorithm.
    raw:
        Active circStudio recording.
    threshold:
        Cole-Kripke decision threshold.
    rescoring:
        Whether Cole-Kripke should apply Webster rescoring rules.

    Returns
    -------
    pandas.Series
        Binary sleep/wake series stored as floating-point values, where
        ``1`` denotes sleep and ``0`` denotes wake.
    """
    # Extract the activity channel in the common Series format used by scorers.
    act = activity_series(raw)

    # Convert the Series sampling frequency into a Timedelta for algorithm setup.
    freq = pd.Timedelta(act.index.freq)

    # Run Cole-Kripke with settings matched to the recording resolution.
    if ui_algo == "Cole-Kripke":
        s = Cole_Kripke(
            act,
            settings=_ck_settings(freq),
            threshold=threshold,
            rescoring=rescoring,
        )

    # The following algorithms require only the activity-count series.
    elif ui_algo == "Sadeh":
        s = Sadeh(act)
    elif ui_algo == "Scripps":
        s = Scripps(act)
    elif ui_algo == "Oakley":
        s = Oakley(act)
    elif ui_algo == "Roenneberg":
        s = Roenneberg(act)

    # Crespo additionally requires the recording epoch duration.
    elif ui_algo == "Crespo":
        s = Crespo(act, frequency=freq)

    # CSM operates on zero-crossing-mode data rather than the activity channel.
    elif ui_algo == "CSM":
        zcm = _zcm_series(raw)

        # Stop with a user-facing error when the required channel is absent.
        if zcm is None:
            raise ValueError("No ZCM channel available for CSM.")

        # Convert CSM's sleep category to the shared binary convention.
        s = (CSM(zcm) == 1).astype(int)

    # Reject unexpected values rather than silently applying another algorithm.
    else:
        raise ValueError(f"Unknown algorithm: {ui_algo}")

    # Standardise the output type used by downstream alignment and plotting.
    return s.astype(float)


def _bouts_to_df(bouts) -> pd.DataFrame:
    """Convert a list of bout Series into a start/end/duration table."""
    # Accumulate one summary row for each valid bout.
    rows = []

    # Derive temporal boundaries from the first and last epochs of each bout.
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

        # Ignore malformed bouts so that other valid bouts remain reportable.
        except Exception:
            continue

    # Return the populated table when at least one valid bout was found.
    if rows:
        return pd.DataFrame(rows)

    # Provide an informative table instead of an empty output component.
    return pd.DataFrame({"info": ["No bouts found."]})


def _durations_minutes(durations) -> list:
    """Convert duration values into numeric minutes for plotting."""
    # Store only values that can be represented as minutes.
    out = []
    for d in durations:
        try:
            # Convert pandas durations by dividing by a one-minute interval.
            if isinstance(d, pd.Timedelta):
                out.append(d / pd.Timedelta("1min"))

            # Preserve already numeric durations through explicit conversion.
            else:
                out.append(float(d))

        # Skip malformed values rather than preventing the histogram from rendering.
        except Exception:
            continue

    return out


def _hhmm(value) -> str:
    """Format a duration, timestamp, or decimal hour as ``HH:MM``."""
    try:
        # Convert durations to seconds and wrap values longer than one day.
        if isinstance(value, pd.Timedelta):
            total = int(value.total_seconds()) % 86400

        # Timestamps already provide an unambiguous clock-time formatter.
        elif isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M")

        # Interpret other numeric values as decimal hours within a 24-hour day.
        else:
            total = int(float(value) * 3600) % 86400

        # Separate total seconds into zero-padded hours and minutes.
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"

    # Keep display components usable when a metric cannot be converted.
    except Exception:
        return "n/a"


def _dur_opt(txt):
    """Convert optional minute text into the duration syntax used by metrics."""
    # Treat missing values and surrounding whitespace uniformly.
    txt = (txt or "").strip()

    # Use ``None`` to indicate that no duration limit should be applied.
    if not txt:
        return None

    # Append the unit required by pandas-compatible duration parsing.
    return f"{txt}min"


# Register the function as the user interface for a reusable Shiny module.
@module.ui
def sleep_ui():
    """Build the controls, plots, tables, and metric panels for sleep scoring."""
    # Place analysis controls in a fixed-width sidebar beside the result panels.
    return ui.layout_sidebar(
        ui.sidebar(
            # Filter batch recordings by their experimental factor when applicable.
            ui.output_ui("factor_filter_ctrl"),

            # Select the active subject from the current single or batch dataset.
            ui.output_ui("subject_ctrl"),

            # Show only algorithms supported by the active recording's channels.
            ui.output_ui("algo_ctrl"),

            # Display Cole-Kripke-specific options only when that algorithm is active.
            ui.panel_conditional(
                "input.algo === 'Cole-Kripke'",
                ui.input_slider(
                    "threshold",
                    "Threshold",
                    min=0.1,
                    max=5.0,
                    value=1.0,
                    step=0.1,
                ),
                ui.input_switch("rescoring", "Webster rescoring", True),
            ),

            # Let the user restrict which bout durations enter the summaries.
            ui.input_numeric(
                "dur_min",
                "Bout duration min (min, blank=none)",
                value=None,
            ),
            ui.input_numeric(
                "dur_max",
                "Bout duration max (min, blank=none)",
                value=None,
            ),

            # Control the half-window used to estimate activity onset and offset.
            ui.input_numeric(
                "whs",
                "AonT/AoffT window half-size (pts)",
                value=12,
            ),

            # Accept common spreadsheet formats for an optional sleep diary.
            ui.input_file(
                "diary",
                "Sleep diary (optional)",
                accept=[".ods", ".xls", ".xlsx", ".csv"],
            ),

            # Start scoring explicitly so changes to settings are not run immediately.
            ui.input_action_button("run", "Run", class_="btn-primary btn-sm"),
            width=340,
        ),

        # Explain the interpretation and clinical limitations of actigraphy scoring.
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

        # Reserve the first result area for the activity and sleep/wake overlay.
        ui.h4("Activity with sleep/wake overlay"),
        output_widget("overlay_plot"),
        ui.hr(),

        # Organise detailed results into focused tab panels.
        ui.navset_tab(
            # Report individual bouts and their duration distributions.
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

            # Report daily timing, regularity, WASO, and the 24-hour profile.
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

            # Report general and algorithm-specific activity onset and offset.
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

            # Combine the principal sleep metrics into a downloadable table.
            ui.nav_panel(
                "All sleep metrics",
                ui.download_button("download_summary", "Download CSV"),
                ui.output_data_frame("all_metrics_tbl"),
            ),
        ),
    )


# Register the function as the server logic for each sleep-scoring module instance.
@module.server
def sleep_server(
    input, output, session, rv_single, rv_batch, rv_mode, rv_active_subject
):
    """Run sleep scoring and update all reactive outputs for the active recording.

    Parameters
    ----------
    input, output, session:
        Shiny objects that provide this module's inputs, outputs, and session.
    rv_single:
        Shared reactive value containing the active single recording.
    rv_batch:
        Shared reactive value containing the loaded batch collection.
    rv_mode:
        Shared reactive value identifying single-file or batch mode.
    rv_active_subject:
        Shared reactive value identifying the selected batch subject.

    Notes
    -----
    The private ``_state`` value is populated only after successful scoring. It
    stores the recording, selected algorithm, binary sleep series, and activity
    series so every downstream output is based on the same analysis run.
    """
    # Store the most recent successful scoring result for all output components.
    _state = reactive.Value(None)

    # Rebuild the experimental-factor filter when the loading mode or batch changes.
    @render.ui
    def factor_filter_ctrl():
        """Render controls for filtering subjects by batch factor."""
        return factor_filter_controls(rv_mode, rv_batch)

    # Rebuild the subject selector from the currently available recordings.
    @render.ui
    def subject_ctrl():
        """Render the subject selector for single-file or batch mode."""
        return subject_controls(input, rv_mode, rv_batch, rv_active_subject())

    # Render algorithm choices supported by the active recording's channels.
    @render.ui
    def algo_ctrl():
        """Render the scoring selector and expose CSM only when ZCM exists."""
        # Resolve the recording selected elsewhere in the application.
        raw = get_active(input, rv_mode, rv_single, rv_batch)

        # CSM is valid only for recordings containing zero-crossing-mode data.
        has_zcm = _zcm_series(raw) is not None

        # Offer algorithms that use the standard activity channel by default.
        choices = [
            "Cole-Kripke",
            "Sadeh",
            "Scripps",
            "Oakley",
            "Roenneberg",
            "Crespo",
        ]

        # Add CSM only when its required input channel is available.
        if has_zcm:
            choices.append("CSM")

        # Use Cole-Kripke as the initial scoring algorithm.
        sel = ui.input_select(
            "algo",
            "Algorithm",
            choices=choices,
            selected="Cole-Kripke",
        )

        # Explain why CSM is absent when the recording lacks ZCM data.
        if not has_zcm:
            return ui.div(
                sel,
                ui.help_text("CSM hidden: no ZCM channel in this recording."),
            )

        # Avoid adding explanatory text when every algorithm is available.
        return sel

    def _raw():
        """Return the recording currently selected in the application."""
        return get_active(input, rv_mode, rv_single, rv_batch)

    def _threshold():
        """Return the Cole-Kripke threshold or its default value."""
        try:
            return float(input.threshold())

        # The conditional input may not exist when another algorithm is selected.
        except Exception:
            return 1.0

    def _rescoring():
        """Return the Webster-rescoring choice or its enabled default."""
        try:
            return bool(input.rescoring())

        # The conditional input may not exist when another algorithm is selected.
        except Exception:
            return True

    # ---------------------------------------------------------------------
    # Optional sleep-diary upload
    # ---------------------------------------------------------------------

    # Re-run this effect only when the user uploads a new diary file.
    @reactive.effect
    @reactive.event(input.diary)
    def _on_diary():
        """Attach an uploaded sleep diary to the active Raw recording."""
        # Read the Shiny upload metadata and resolve the active recording.
        finfo = input.diary()
        raw = _raw()

        # Do nothing if either the diary or its target recording is unavailable.
        if not finfo or raw is None:
            return

        # Use the first uploaded file because this input accepts one diary.
        src = finfo[0]

        # Preserve the filename and extension in a temporary directory.
        tmp = Path(tempfile.mkdtemp()) / src["name"]
        shutil.copy(src["datapath"], tmp)

        # Delegate format parsing and diary attachment to the Raw object.
        try:
            raw.read_sleep_diary(str(tmp))
            ui.notification_show("Sleep diary attached.", type="message")

        # Report parsing or attachment failures without interrupting the app.
        except Exception as exc:
            ui.notification_show(f"Diary error: {exc}", type="error")

    # Re-run scoring only when the user explicitly clicks the Run button.
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        """Score the active recording and store a shared analysis snapshot."""
        # Resolve the recording selected at the time the analysis is requested.
        raw = _raw()

        # Prevent scoring before a recording has been loaded.
        if raw is None:
            ui.notification_show("Load a recording first.", type="warning")
            return

        # Read the algorithm selection, using the UI default if unavailable.
        try:
            algo = input.algo()
        except Exception:
            algo = "Cole-Kripke"

        # Calculate the binary sleep/wake series using the selected settings.
        try:
            s = _score(algo, raw, _threshold(), _rescoring())

        # Leave the previous valid state untouched if the new scoring run fails.
        except Exception as exc:
            ui.notification_show(
                f"Scoring failed: {exc}",
                type="error",
                duration=8,
            )
            return

        # Store the recording and aligned inputs used by all result components.
        _state.set(
            {
                "raw": raw,
                "algo": algo,
                "sleep": s,
                "act": activity_series(raw),
            }
        )

        # Confirm that the reactive results are ready for inspection.
        ui.notification_show("Scoring complete.", type="message")

    # ---------------------------------------------------------------------
    # Activity and sleep/wake overlay
    # ---------------------------------------------------------------------

    # Render the Plotly output whenever the stored scoring state changes.
    @render_widget
    def overlay_plot():
        """Plot activity counts with epochs classified as sleep shaded below."""
        # Read the latest complete scoring snapshot.
        st = _state.get()

        # Display guidance until the first successful run is available.
        if not st:
            return empty_fig("Load a recording and click 'Run'.")

        # Retrieve the activity and binary sleep series from the same run.
        act, s = st["act"], st["sleep"]

        # Scale the sleep shading to the maximum observed activity value.
        ymax = float(np.nanmax(act.values)) if len(act) else 1.0

        # Align sleep epochs to activity timestamps using the nearest score.
        sleep = s.reindex(act.index, method="nearest")

        # Fill the full y-range during sleep and zero height during wake.
        shade = np.where(sleep.values == 1, ymax, 0.0)

        # Create the plot and add the sleep layer before the activity trace.
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=act.index.astype(str),
                y=shade,
                fill="tozeroy",
                mode="none",
                fillcolor="rgba(100,149,237,0.25)",
                name="Sleep",
            )
        )

        # Draw activity above the translucent sleep intervals.
        fig.add_trace(
            go.Scatter(
                x=act.index.astype(str),
                y=act.values,
                line=dict(width=0.8, color="#333"),
                name="Activity",
            )
        )

        # Label the axes and place the legend horizontally to save vertical space.
        fig.update_layout(
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis_title="DateTime",
            yaxis_title="Activity",
            legend=dict(orientation="h"),
        )
        return fig

    # ---------------------------------------------------------------------
    # Bout-level summaries
    # ---------------------------------------------------------------------

    def _alg():
        """Return an algorithm compatible with the package summary helpers."""
        st = _state.get()
        return _summary_algo(st["algo"]) if st else "Roenneberg"

    # Render a table of contiguous sleep intervals.
    @render.data_frame
    def sleep_bouts_tbl():
        """Report the start, end, and duration of detected sleep bouts."""
        st = _state.get()

        # Keep the table informative before scoring has been run.
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})

        # Apply the optional duration limits using the compatible summary algorithm.
        try:
            b = sleep_bouts(
                st["act"],
                duration_min=_dur_opt_num("dur_min", input),
                duration_max=_dur_opt_num("dur_max", input),
                algo=_alg(),
            )
            return render.DataGrid(_bouts_to_df(b), width="100%")

        # Show calculation failures inside the output rather than hiding the panel.
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    # Render a table of contiguous wake/activity intervals.
    @render.data_frame
    def active_bouts_tbl():
        """Report the start, end, and duration of detected active bouts."""
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})

        # Apply the same duration filters used for the sleep-bout table.
        try:
            b = active_bouts(
                st["act"],
                duration_min=_dur_opt_num("dur_min", input),
                duration_max=_dur_opt_num("dur_max", input),
                algo=_alg(),
            )
            return render.DataGrid(_bouts_to_df(b), width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    # Render the principal sleep interval detected for each day.
    @render.data_frame
    def main_bouts_tbl():
        """Report the daily main sleep bout and its duration in minutes."""
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})

        try:
            # Pass RAW activity: main_sleep_bouts runs Roenneberg internally,
            # which expects raw counts. Passing the pre-scored 0/1 series
            # (st["sleep"]) double-scores and inverts sleep/wake, yielding the
            # active period instead of the main sleep bout.
            df, _mean = main_sleep_bouts(st["act"], report="major")

            # Copy the result before adding display-specific columns.
            out = df.copy()

            # Convert timedeltas to minutes for direct interpretation in the table.
            out["duration_min"] = (out["duration"] / pd.Timedelta("1min")).round(1)

            # Mark every returned row as a major bout because ``report='major'``.
            out["is_major"] = True

            # Retain and order only columns useful to the user.
            out = out[["date", "start_time", "stop_time", "duration_min", "is_major"]]

            # Convert temporal values to strings for reliable DataGrid display.
            out = out.astype({"date": str, "start_time": str, "stop_time": str})
            return render.DataGrid(out, width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    def _dur_hist(fn):
        """Build a histogram for sleep- or active-bout durations."""
        # Use the same stored analysis state as the associated bout table.
        st = _state.get()
        if not st:
            return empty_fig("Run scoring first.")

        # Calculate durations with the current minimum and maximum filters.
        try:
            durs = _durations_minutes(
                fn(
                    st["act"],
                    duration_min=_dur_opt_num("dur_min", input),
                    duration_max=_dur_opt_num("dur_max", input),
                    algo=_alg(),
                )
            )
        except Exception as exc:
            return empty_fig(f"Error: {exc}")

        # Explain when filtering or scoring produced no reportable bouts.
        if not durs:
            return empty_fig("No durations.")

        # Plot the distribution using a fixed maximum of 30 histogram bins.
        fig = go.Figure(go.Histogram(x=durs, nbinsx=30))

        # Summarise the distribution in the plot title.
        mean, sd = float(np.mean(durs)), float(np.std(durs))
        fig.update_layout(
            xaxis_title="Duration (min)",
            yaxis_title="Count",
            title=f"mean={mean:.1f}  sd={sd:.1f}  n={len(durs)}",
            margin=dict(l=40, r=20, t=40, b=30),
        )
        return fig

    # Render the sleep-bout duration distribution.
    @render_widget
    def sleep_dur_hist():
        """Display the distribution of detected sleep-bout durations."""
        return _dur_hist(sleep_durations)

    # Render the active-bout duration distribution.
    @render_widget
    def active_dur_hist():
        """Display the distribution of detected active-bout durations."""
        return _dur_hist(active_durations)

    # ---------------------------------------------------------------------
    # Daily sleep summary
    # ---------------------------------------------------------------------

    # Render the sleep regularity index as a compact value-box result.
    @render.text
    def vb_sri():
        """Report the sleep regularity index for the active recording."""
        st = _state.get()

        # Use an em dash to show that scoring has not yet been run.
        if not st:
            return "—"

        # Format a valid index to two decimal places.
        try:
            return f"{float(SleepRegularityIndex(st['act'], algo=_alg())):.2f}"

        # Distinguish calculation failure from the pre-analysis state.
        except Exception:
            return "n/a"

    # Render mean sleep onset as a compact value-box result.
    @render.text
    def vb_sod():
        """Report the mean sleep-onset metric across available days."""
        st = _state.get()
        if not st:
            return "—"

        try:
            # Calculate sleep onset using an algorithm supported by the helper.
            v = SoD(st["act"], algo=_alg())

            # Average vector outputs while preserving scalar outputs directly.
            if hasattr(v, "__len__"):
                v = np.nanmean(np.asarray(v, dtype=float))
            else:
                v = float(v)

            return f"{v:.2f}"
        except Exception:
            return "n/a"

    # Render mean functional sleep onset as a compact value-box result.
    @render.text
    def vb_fsod():
        """Report the mean functional sleep-onset metric across days."""
        st = _state.get()
        if not st:
            return "—"

        try:
            # Calculate functional sleep onset with the compatible algorithm.
            v = fSoD(st["act"], algo=_alg())

            # Average vector outputs while preserving scalar outputs directly.
            if hasattr(v, "__len__"):
                v = np.nanmean(np.asarray(v, dtype=float))
            else:
                v = float(v)

            return f"{v:.3f}"
        except Exception:
            return "n/a"

    def _waso_pair():
        """Return daily and mean wake after sleep onset (WASO) values."""
        # WASO is unavailable until a complete scoring state exists.
        st = _state.get()
        if not st:
            return None

        # Request Cole-Kripke WASO using the activity epoch duration.
        try:
            return waso(
                st["act"],
                frequency=pd.Timedelta(st["act"].index.freq),
                algo="Cole-Kripke",
                settings="mean",
            )

        # Represent any unavailable WASO calculation with a single sentinel.
        except Exception:
            return None

    # Render the mean WASO value in minutes.
    @render.text
    def vb_waso():
        """Report mean wake after sleep onset in minutes."""
        wp = _waso_pair()

        # Keep the value box usable if WASO could not be calculated.
        if wp is None:
            return "n/a"

        return f"{float(wp[1]):.1f}"

    # Render daily or mean sleep-midpoint values as a table.
    @render.data_frame
    def midpoint_tbl():
        """Report sleep midpoint as clock time for each available day."""
        st = _state.get()
        if not st:
            return pd.DataFrame({"info": ["Run scoring first."]})

        try:
            # Request timedelta output so clock times can be formatted consistently.
            smp = SleepMidPoint(st["act"], to_td=True, algo=_alg())

            # Create one row per day when the metric returns a Series.
            if isinstance(smp, pd.Series):
                df = pd.DataFrame(
                    {
                        "day": [str(i) for i in smp.index],
                        "midpoint": [_hhmm(v) for v in smp.values],
                    }
                )

            # Otherwise report the single summary value returned by the function.
            else:
                df = pd.DataFrame(
                    {
                        "metric": ["Sleep midpoint (mean)"],
                        "value": [_hhmm(smp)],
                    }
                )

            return render.DataGrid(df, width="100%")
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    # Render the average probability of sleep across the 24-hour cycle.
    @render_widget
    def sleep_profile_plot():
        """Plot the sleep probability profile by time of day."""
        st = _state.get()
        if not st:
            return empty_fig("Run scoring first.")

        try:
            # Aggregate the active recording into its 24-hour sleep profile.
            sp = SleepProfile(st["act"], algo=_alg())

            # Plot time-of-day bins against their estimated sleep probability.
            fig = go.Figure(
                go.Scatter(
                    x=[str(i) for i in sp.index],
                    y=sp.values,
                    mode="lines",
                )
            )
            fig.update_layout(
                xaxis_title="Time of day",
                yaxis_title="Sleep probability",
                margin=dict(l=40, r=20, t=20, b=30),
            )
            return fig
        except Exception as exc:
            return empty_fig(f"Error: {exc}")

    # Render the daily WASO values in minutes.
    @render.data_frame
    def waso_tbl():
        """Report wake after sleep onset separately for each day."""
        wp = _waso_pair()

        # Explain when the WASO calculation did not return a usable result.
        if wp is None:
            return pd.DataFrame({"info": ["WASO unavailable."]})

        # The first tuple element contains the daily WASO Series.
        series = wp[0]

        # Convert dates to strings and round daily values to tenths of a minute.
        df = pd.DataFrame(
            {
                "day": [str(i) for i in series.index],
                "WASO_min": [round(float(v), 1) for v in series.values],
            }
        )
        return render.DataGrid(df, width="100%")

    # ---------------------------------------------------------------------
    # Activity onset and offset
    # ---------------------------------------------------------------------

    # Render the general activity-onset estimate as clock time.
    @render.text
    def vb_aont():
        """Report activity onset using the selected half-window size."""
        st = _state.get()
        if not st:
            return "—"

        try:
            # Use 12 points when the numeric input is missing or zero-like.
            return _hhmm(AonT(st["act"], whs=int(input.whs() or 12)))
        except Exception:
            return "n/a"

    # Render the general activity-offset estimate as clock time.
    @render.text
    def vb_aofft():
        """Report activity offset using the selected half-window size."""
        st = _state.get()
        if not st:
            return "—"

        try:
            return _hhmm(AoffT(st["act"], whs=int(input.whs() or 12)))
        except Exception:
            return "n/a"

    # Render algorithm-specific onset and offset values where implemented.
    @render.ui
    def aot_section():
        """Report Roenneberg- or Crespo-specific onset and offset estimates."""
        st = _state.get()

        # Return an empty container before scoring to preserve the page layout.
        if not st:
            return ui.div()

        # Use the actual overlay-scoring selection for algorithm-specific outputs.
        algo = st["algo"]

        try:
            # Roenneberg estimates onset and offset directly from activity.
            if algo == "Roenneberg":
                aont, aofft = Roenneberg_AoT(st["act"])

            # Crespo additionally requires the activity epoch duration.
            elif algo == "Crespo":
                aont, aofft = Crespo_AoT(
                    st["act"],
                    frequency=pd.Timedelta(st["act"].index.freq),
                )

            # Explain that other algorithms do not expose a specialised metric.
            else:
                return ui.help_text(
                    "Algorithm-specific onset/offset is available for "
                    "Roenneberg and Crespo.")

        # Convert calculation errors into explanatory interface text.
        except Exception as exc:
            return ui.help_text(f"Onset/offset error: {exc}")

        def _fmt(x):
            """Format one value or preview the first eight daily values."""
            # Daily Series are shortened to keep the interface table compact.
            if isinstance(x, pd.Series):
                return ", ".join(_hhmm(v) for v in x.values[:8])

            # Scalar results require only the standard clock-time formatter.
            return _hhmm(x)

        # Display onset and offset together in a compact two-row table.
        return ui.tags.table(
            ui.tags.tr(ui.tags.th(f"{algo} onset"), ui.tags.td(_fmt(aont))),
            ui.tags.tr(ui.tags.th(f"{algo} offset"), ui.tags.td(_fmt(aofft))),
            class_="table table-sm",
        )

    # ---------------------------------------------------------------------
    # Comprehensive metric table and CSV export
    # ---------------------------------------------------------------------

    # Cache the complete table and recalculate it when any dependency changes.
    @reactive.calc
    def _all_metrics_df():
        """Compile the principal sleep metrics into one labelled table."""
        # Use the same scoring snapshot as all other output panels.
        st = _state.get()

        # Preserve the final table schema before results are available.
        if not st:
            return pd.DataFrame(columns=["Metric", "Value", "Unit"])

        # Unpack the activity, selected sleep scores, and summary algorithm.
        act, s, alg = st["act"], st["sleep"], _alg()

        # Express the sampling epoch as a fraction of one minute.
        freq = pd.Timedelta(act.index.freq)
        epoch_min = freq / pd.Timedelta("1min")

        # Accumulate every output as a metric, value, and unit tuple.
        rows = []

        # Estimate total sleep time and efficiency from the selected sleep scores.
        try:
            # Read the binary scores and locate every epoch classified as sleep.
            vals = s.values
            idx = np.where(vals == 1)[0]

            # Convert the number of sleep epochs into total minutes.
            tst = int(np.sum(vals == 1)) * epoch_min

            # Normalise by recording length, with at least one day as denominator.
            n_days = max(1, (act.index[-1] - act.index[0]) / pd.Timedelta("1D"))

            # Convert mean daily total sleep time from minutes to hours.
            rows.append(
                (
                    "Total sleep time (mean/day)",
                    round(tst / n_days / 60, 2),
                    "hours",
                )
            )

            # Measure sleep efficiency within the span from first to last sleep epoch.
            if len(idx):
                span = vals[idx[0] : idx[-1] + 1]
                eff = 100.0 * np.sum(span == 1) / len(span)

            # Define efficiency as zero when no epoch was classified as sleep.
            else:
                eff = 0.0

            rows.append(("Sleep efficiency", round(eff, 1), "%"))

        # Preserve both metric rows if either calculation cannot be completed.
        except Exception:
            rows.append(("Total sleep time (mean/day)", "n/a", "hours"))
            rows.append(("Sleep efficiency", "n/a", "%"))

        def _try(label, fn, unit, fmt=None):
            """Calculate one metric and append a consistent fallback on failure."""
            try:
                # Defer the calculation until it is protected by this error boundary.
                v = fn()

                # Use a metric-specific formatter for clock times or complex outputs.
                if fmt:
                    v = fmt(v)

                # Round scalar numeric values uniformly to two decimal places.
                elif isinstance(v, (int, float, np.floating)):
                    v = round(float(v), 2)

                rows.append((label, v, unit))

            # Keep one failed metric from suppressing the remainder of the report.
            except Exception:
                rows.append((label, "n/a", unit))

        # Reuse the shared WASO result to keep the table and value box consistent.
        wp = _waso_pair()

        # Report the mean WASO in minutes when calculation succeeded.
        if wp:
            waso_mean = round(float(wp[1]), 1)
        else:
            waso_mean = "n/a"
        rows.append(("WASO (mean)", waso_mean, "minutes"))

        # Normalise the number of detected sleep bouts by recording duration.
        _try(
            "Number of sleep bouts (mean/day)",
            lambda: len(sleep_bouts(act, algo=alg))
            / max(1, (act.index[-1] - act.index[0]) / pd.Timedelta("1D")),
            "—",
        )

        # Average daily midpoint values before formatting them as clock time.
        _try(
            "Sleep midpoint (mean)",
            lambda: SleepMidPoint(act, to_td=True, algo=alg),
            "HH:MM",
            fmt=lambda v: _hhmm(
                v.mean() if isinstance(v, pd.Series) else v
            ),
        )

        # Add the recording-level sleep regularity index.
        _try(
            "Sleep regularity index",
            lambda: SleepRegularityIndex(act, algo=alg),
            "—",
        )

        # Average daily sleep-onset values before converting to clock time.
        _try(
            "Sleep onset (SoD, mean)",
            lambda: SoD(act, algo=alg),
            "HH:MM",
            fmt=lambda v: _hhmm(
                np.nanmean(np.asarray(v, float))
                if hasattr(v, "__len__")
                else v
            ),
        )

        # Average daily functional sleep-onset values before clock formatting.
        _try(
            "Functional SoD (mean)",
            lambda: fSoD(act, algo=alg),
            "HH:MM",
            fmt=lambda v: _hhmm(
                np.nanmean(np.asarray(v, float))
                if hasattr(v, "__len__")
                else v
            ),
        )

        # Add activity onset using the configured half-window size.
        _try(
            "Activity onset (AonT)",
            lambda: AonT(act, whs=int(input.whs() or 12)),
            "HH:MM",
            fmt=_hhmm,
        )

        # Add activity offset using the same half-window convention.
        _try(
            "Activity offset (AoffT)",
            lambda: AoffT(act, whs=int(input.whs() or 12)),
            "HH:MM",
            fmt=_hhmm,
        )

        # Convert the accumulated metric tuples into the displayed table schema.
        return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])

    # Render the comprehensive metrics table in the final tab panel.
    @render.data_frame
    def all_metrics_tbl():
        """Display the consolidated sleep-metric report."""
        st = _state.get()

        # Prompt the user to run scoring before displaying the empty schema.
        if not st:
            return pd.DataFrame({"info": ["Run scoring to populate this table."]})

        return render.DataGrid(_all_metrics_df(), width="100%")

    # Stream the current comprehensive table as a CSV download.
    @render.download(filename="sleep_metrics.csv")
    def download_summary():
        """Download the consolidated sleep metrics as ``sleep_metrics.csv``."""
        # Build the CSV in memory so no persistent server-side file is required.
        buf = io.StringIO()
        _all_metrics_df().to_csv(buf, index=False)

        # Yield the text payload expected by Shiny's download handler.
        yield buf.getvalue()


def _dur_opt_num(name, input):
    """Convert an optional numeric duration input to ``'<n>min'`` syntax."""
    # Resolve the named Shiny input dynamically for minimum or maximum duration.
    try:
        v = getattr(input, name)()

    # Treat a missing or unavailable input as no duration restriction.
    except Exception:
        return None

    # Preserve blank inputs as ``None`` for the sleep-analysis functions.
    if v is None or v == "":
        return None

    # Normalise numeric strings and floats to an integer number of minutes.
    try:
        return f"{int(float(v))}min"

    # Ignore invalid values rather than applying a malformed duration filter.
    except Exception:
        return None