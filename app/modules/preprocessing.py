"""Tab 1 - Preprocessing (non-wear detection, masking, filtering).

This module provides the user interface and server logic for the preprocessing
page of circStudio. It supports:
- automatic non-wear detection with the Choi (2011) or Troiano (2008) method;
- manual mask-period entry using start and stop datetimes;
- manual mask selection by drawing a rectangle on the activity plot;
- mask import from CSV, XLS, XLSX, or ODS files;
- resampling, binarization, mask application, and missing-value imputation;
- an interactive Plotly activity trace with non-wear and NaN overlays;
- export of the current mask as a CSV file;
- single-recording and batch-recording workflows.

The tab always edits a local deep copy of the active ``Raw`` object. Therefore,
preprocessing changes remain isolated from the other analysis tabs until the
user explicitly clicks "Apply & Export to Analysis Tabs".
"""
# Standard library imports
import copy
import io
import tempfile
import shutil
from pathlib import Path

# Third-party scientific and plotting libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Shiny components used to define reactive UI and server behavior
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

# Add the application root to Python's import path so local shared modules are
# resolved before packages with the same name elsewhere.
from import_paths import _add_import_path
app_dir = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)
_add_import_path(app_dir)

# Import shared modules from the application root. These are not installed
# packages, so we need to add the app root to sys.path first
from modules._common import (  # noqa: E402
    activity_series,
    empty_fig,
    factor_filter_controls,
    get_active,
    subject_controls,
    selected_subject,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ``preprocessing_server`` helper function
def _copy_or_none(value):
    """
    Return a deep copy of ``value``, or ``None`` when no value is present.

    Deep copies are important here because ``Raw`` objects are mutable. Editing
    the copy prevents preprocessing operations from changing the shared object
    used by the other tabs before the user explicitly exports the result.

    Parameters
    ----------
    value:
        Any object that may be ``None``.
    
    Returns
    -------
    object or None
        A deep copy of ``value`` when it is not ``None``; otherwise, ``None``.
    """
    # Return a deep copy of value if value is not None; return None otherwise
    return copy.deepcopy(value) if value else None

# ``plot_main`` helper functions
def _safe_mask(raw):
    """
    Return the public mask associated with ``raw``, if it is available.
    
    Parameters
    ----------
    raw:
        Recording object that may contain a public ``mask`` attribute.

    Returns
    -------
    pandas.Series or None
        The recording mask when it can be accessed; otherwise, ``None``.
    """
    try:
        # Return the mask if raw is not None; return None otherwise
        return raw.mask if raw else None

    except Exception:  # noqa: BLE001
        # Return None if an exception occurs while accessing the mask
        return None

def _time_axis_details(activity):
    """
    Prepare numeric x positions and readable date labels for the activity plot.

    Parameters
    ----------
    activity:
        Activity time series whose index contains timestamps.

    Returns
    -------
    tuple
        Numeric x positions, timestamp labels, tick positions, and tick labels.
    """
    # Assign numeric x positions to each activity observation
    x_positions = np.arange(
        len(activity),
        dtype=float,
    )

    # Preserve standardized timestamps when the index is already a DatetimeIndex
    # Otherwise, retain the original labels and try to parse them.
    if isinstance(activity.index, pd.DatetimeIndex):
        # Convert each timestamp to a string in the format "YYYY-MM-DD HH:MM:SS"
        time_labels = (
            activity.index
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        # No need to parse the dates again since they are already in a DatetimeIndex
        parsed_dates = activity.index

    else:
        # Preserve non-datetime index values as strings
        time_labels = [
            str(value)
            for value in activity.index
        ]

        # Try to interpret the index labels as dates
        # Values that cannot be converted are replaced with NaT (Not a Time)
        parsed_dates = pd.to_datetime(
            time_labels,
            errors="coerce",
        )

    # Display at most seven evenly spaced ticks to avoid overcrowding.
    if len(activity) > 1:
        tick_indexes = np.unique(
            np.linspace(
                0,
                len(activity) - 1,
                num=min(7, len(activity)),
                dtype=int,
            )
        )

    else:
        # A single-value recording requires only one tick at position zero
        tick_indexes = np.array([0])

    # Convert the indexes to the numeric values used by the Plotly x-axis.
    tick_values = tick_indexes.astype(float).tolist()

    # Use formatted dates when more than 90% of the index can be parsed.
    if (
        len(parsed_dates) == len(activity)
        and pd.Series(
            parsed_dates
            ).notna().mean() > 0.9
    ):
        # Determine the first and last valid dates in the recording
        date_min = pd.to_datetime(
            parsed_dates
            ).min()

        date_max = pd.to_datetime(
            parsed_dates
            ).max()

        # Calculate the total recording duration in days
        span_days = (
            date_max - date_min
        ).total_seconds() / 86400 # 24 * 60 * 60

        # Long recordings show only the month and day
        if span_days > 2:
            tick_text = [
                pd.to_datetime(
                    parsed_dates[index]
                ).strftime("%b %d")
                for index in tick_indexes
            ]

        # Short recordings additionally show the time of day.
        else:
            tick_text = [
                pd.to_datetime(
                    parsed_dates[index]
                ).strftime("%b %d<br>%H:%M")
                for index in tick_indexes
            ]

    else:
        # Fall back to the original index labels when parsing is unreliable.
        tick_text = [
            time_labels[index]
            for index in tick_indexes
        ]
    
    # Return all x-axis information required by the plot_main function
    return (
        x_positions,
        time_labels,
        tick_values,
        tick_text,
    )

def _activity_y_range(y_values):
    """
    Calculate a padded y-axis range for the activity plot.

    Parameters
    ----------
    y_values:
        Numeric activity values, which may include NaN or infinite values.

    Returns
    -------
    tuple of float
        Lower and upper limits for the Plotly y-axis.
    """
    # Exclude NaN and infinite values from the range calculation.
    finite_values = y_values[
        np.isfinite(y_values)
    ]

    # Use a small default range when no finite observations are available
    if len(finite_values) == 0:
        return -1.0, 1.0

    # Find the minimum and maximum finite activity values
    y_min_data = float(
        np.nanmin(finite_values)
    )

    y_max_data = float(
        np.nanmax(finite_values)
    )

    # Include zero as the lower reference unless negative data extend below it
    y_lower_reference = min(0.0, y_min_data)

    # Use the maximum finite value as the upper reference
    y_upper_reference = y_max_data

    # Calculate the visible activity range
    y_span = y_upper_reference - y_lower_reference

    # Prevent a zero-width axis when all finite values are identical.
    if y_span <= 0:
        y_span = max(abs(y_upper_reference), 1.0)

    # Add five percent spacing below and above the activity range.
    y_padding = 0.05 * y_span

    # Calculate the padded y-axis limits for the Plotly activity plot
    y_min = y_lower_reference - y_padding
    y_max = y_upper_reference + y_padding

    # Return the padded y-axis limits for the Plotly activity plot
    return y_min, y_max

def _find_true_runs(flags):
    """
    Find contiguous ``True`` regions in a one-dimensional Boolean sequence.

    Parameters
    ----------
    flags:
        Boolean-like sequence in which ``True`` identifies observations that
        belong to a region of interest.

    Returns
    -------
    tuple of numpy.ndarray
        Two arrays containing the start positions and exclusive end positions
        of all contiguous ``True`` regions.

    Notes
    -----
    The flag sequence ``[False, True, True, False]`` will produce a start position 
    of ``1`` and an end position of ``3``.
    """
    # Convert the input to a Boolean NumPy array
    flags = np.asarray(flags, dtype=bool)

    # Calculate the differences between adjacent elements
    edges = np.diff(flags.astype(int), prepend=0, append=0)

    # Find all False-to-True transitions
    # A change from 0 to 1 marks the beginning of a True region
    starts = np.where(edges == 1)[0]

    # Find all True-to-False transitions
    # A change from 1 to 0 marks the exclusive end of a True region
    ends = np.where(edges == -1)[0]

    # Return the start and exclusive end positions of all contiguous
    # True regions
    return starts, ends

def _add_interval_bands(
    figure,
    flags,
    x_positions,
    *,
    y0,
    y1,
    fillcolor,
    linecolor,
):
    """
    Add one Plotly band for every contiguous ``True`` region.

    Parameters
    ----------
    figure:
        Plotly figure to which the bands should be added.

    flags:
        Boolean-like sequence identifying observations that belong to a band.

    x_positions:
        Numeric x-axis positions used by the activity trace.

    y0:
        Lower vertical position in Plotly paper coordinates.

    y1:
        Upper vertical position in Plotly paper coordinates.

    fillcolor:
        Plotly-compatible color used to fill the bands.

    linecolor:
        Plotly-compatible color used for the border line of the bands.

    Notes
    -----
    (1)``_find_true_runs()`` returns the end of each region as an exclusive 
    index. For example, a region covering indexes 1 and 2 is represented as
    ``start=1`` and ``end=3``.

    (2) Plotly paper coordinates range from 0 at the bottom of the plotting
    area to 1 at the top. Therefore, the vertical height of each band remains
    independent of the activity-count scale.

    (3) Paper coordinates are normalized positioning units (from 0 to 1). Paper 
    coordinates are also known as "normalized coordinates"  or "plot_fractions"
    """
    # Find every contiguous region marked as True.
    starts, ends = _find_true_runs(flags)

    # Draw one rectangle for every contiguous region.
    for start, end in zip(starts, ends):
        # The exclusive end may equal len(x_positions) when a region continues
        # through the final observation. Since that index does not exist,
        # limit it to the last valid x-position index.
        end_position = min(end, len(x_positions) - 1)

        # Add the interval as a horizontal band at the bottom of the plot
        figure.add_shape(
            type="rect", # Draw a rectangle
            xref="x", # Use data coordinates for horizontal placement
            x0=x_positions[start], # Left boundary of the rectangle
            x1=x_positions[end_position], # Right boundary of the rectangle
            yref="paper", # Use paper coordinates for vertical placement
            y0=y0, # Lower vertical boundary
            y1=y1, # Upper vertical boundary
            fillcolor=fillcolor, # Color used to fill the rectangle
            line=dict(color=linecolor, width=1), # Border line for the rectangle
            layer="above", # Draw the rectangle above the activity trace
        )

# ``btn_export_mask``helper function
def _mask_to_dataframe(mask):
    """
    Convert a binary mask into a table of non-wear start and stop times.

    Parameters
    ----------
    mask:
        Binary mask Series in which 0 represents non-wear and 1 represents wear.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing ``Start_time`` and ``Stop_time`` columns.
    """
    # Convert the mask convention into an indicator where 1 means non-wear
    in_nonwear = (mask == 0).astype(int)

    # Compute the difference between consecutive epochs to identify transitions
    edges = (in_nonwear.diff().fillna(in_nonwear.iloc[0]))

    # Positive changes identify starts
    starts = mask.index[edges == 1].tolist()

    # Negative changes identify stops
    ends = mask.index[edges == -1].tolist()
    
    # Include the first epoch when the recording begins in non-wear
    if in_nonwear.iloc[0] == 1:
        starts = [mask.index[0]] + starts

    # Include the final epoch when the recording ends in non-wear
    if in_nonwear.iloc[-1] == 1:
        ends = ends + [mask.index[-1]]

    # Pair each start with its corresponding stop
    rows = [
        {
            "Start_time": start,
            "Stop_time": stop,
        }
        for start, stop in zip(starts, ends)
    ]
    # Return a DataFrame containing the start and stop times of all non-wear periods
    return pd.DataFrame(rows, columns=["Start_time", "Stop_time"])

# ``_on_import_mask`` helper function
def _copy_uploaded_file(upload_info):
    """
    Copy a Shiny upload to a temporary path using its original filename.

    Parameters
    ----------
    upload_info:
        Upload metadata dictionary returned by a Shiny file input.

    Returns
    -------
    pathlib.Path
        Path to the copied temporary file.

    Notes
    -----
    Shiny stores uploaded files under temporary internal names. Restoring the
    original filename preserves the extension required by the mask-file reader.
    """
    # Retain only the filename and remove any unexpected directory components
    original_name = Path(upload_info["name"]).name

    # Create a dedicated temporary directory for the uploaded file
    temporary_directory = Path(tempfile.mkdtemp())

    # Reconstruct a temporary path containing the original filename
    temporary_path = temporary_directory / original_name

    # Copy the uploaded data to the reconstructed temporary path
    shutil.copy(upload_info["datapath"], temporary_path)

    # Return the path to the copied temporary file
    return temporary_path

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

# ---------------------------------------------------------------------
# Upload UI
# ---------------------------------------------------------------------
# Register this function as the UI component of a Shiny module
@module.ui
def preprocessing_ui():
    """
    Build and return the complete user interface for the preprocessing tab.

    The sidebar contains controls for automatic non-wear detection, manual
    mask creation, mask import, preprocessing filters, and batch subject
    selection. 

    The main panel displays status messages, the interactive activity plot,
    and a button for exporting the current non-wear mask.
    """
    # Combine the JavaScript listener and the visible interface into a single
    # collection of HTML elements (a tag list is a fragment of HTML)
    return ui.TagList(
        # Add the JavaScript listerner that converts rectangles drawn on the
        # Plotly activity plot into start and stop timestamps
        _SELECT_LISTENER_JS,

        # Arrange the preprocessing controls in a sidebar and the activity plot
        # in the main content area
        ui.layout_sidebar(
            # -----------------------------------------------------------------
            # Sidebar
            # -----------------------------------------------------------------
            ui.sidebar(
                # ---------------------------------------------------------------
                # Automatic non-wear detection
                # ---------------------------------------------------------------
                # Display the section heading
                ui.h6("Automatic Non-Wear Detection"),

                # Let the user select the algorithm used to identify non-wear
                ui.input_select(
                    "nw_method",
                    "Algorithm",
                    choices={
                        "choi": "Choi et al. (2011)",
                        "troiano": "Troiano et al. (2008)",
                    },
                    selected="choi", # Choi et al. (2011) is selected by default
                ),

                # Let the user define the minimum duration required for a period
                # of low activity to be classified as non-wear
                ui.input_text(
                    "nw_min_length",
                    "Minimum non-wear length",
                    value="90min",
                    placeholder="e.g. 60min, 2h",
                ),

                # Set the maximum number of isolated activity epochs tolerated
                # within an otherwise inactive non-wear period
                ui.input_numeric(
                    "nw_spike_tolerance",
                    "Spike tolerance (epochs)",
                    value=2,
                    min=0,
                    max=10,
                    step=1,
                ),

                # Set the maximum activity count that a tolerated spike may contain
                # witout interrupting a non-wear period
                ui.input_numeric(
                    "nw_spike_max_counts",
                    "Max spike counts",
                    value=100,
                    min=0,
                    step=10,
                ),

                # Show the neighbourhood-window option only when the Choi
                # non-wear algorithm is selected
                ui.panel_conditional(
                    "input.nw_method === 'choi'",

                    # Define the window examined before and after a possible
                    # activity spike when applying the Choi algorithm
                    ui.input_text(
                        "nw_window_size",
                        "Neighbourhood window size",
                        value="30min",
                        placeholder="e.g. 30min, 1h",
                    ),
                ),

                # Trigger automatic non-wear detection using the selected
                # algorithm parameter values
                ui.input_action_button(
                    "btn_detect",
                    "Detect Non-Wear",
                    class_="btn-primary btn-sm w-100",
                ),

                # Separate automatic detection from manual mask controls
                ui.hr(),
                # ---------------------------------------------------------------
                # Manual mask period
                # ---------------------------------------------------------------
                # Section heading
                ui.h6("Manual Mask Period"),

                # Let the user enter the beginning of a manual mask interval
                ui.input_text(
                    "mask_start",
                    "Start (YYYY-MM-DD HH:MM:SS)",
                    placeholder="2024-01-01 00:00:00",
                ),

                # Let the user enter the end of a manual mask interval
                ui.input_text(
                    "mask_stop",
                    "Stop (YYYY-MM-DD HH:MM:SS)",
                    placeholder="2024-01-01 08:00:00",
                ),

                # Button to add the manually entered mask period to the current mask
                ui.input_action_button(
                    "btn_add_period",
                    "Add Period",
                    class_="btn-outline-secondary btn-sm w-100",
                ),

                # Explain that the start and stop fields can also be populated
                # by drawing a rectangle directly on the activity plot
                ui.p(
                    "Or draw a rectangle on the chart — Start/Stop will be "
                    "filled in automatically.",
                    class_="text-muted small mt-1",
                ),

                # Separate manual mask controls from mask file import controls
                ui.hr(),

                # ---------------------------------------------------------------
                # Import mask from file
                # ---------------------------------------------------------------
                # Header title
                ui.h6("Import Mask from File"),

                # Let the user upload a mask file in CSV, XLS, XLSX, or ODS format
                ui.input_file(
                    "mask_file",
                    "Upload mask log (.csv / .xlsx / .ods)",
                    accept=[".csv", ".xlsx", ".xls", ".ods"],
                    multiple=False,
                ),

                # Button to import the selected mask file
                ui.input_action_button(
                    "btn_import_mask",
                    "Import Mask",
                    class_="btn-outline-secondary btn-sm w-100",
                ),

                # Separate import mask from file from applying filters section
                ui.hr(),

                # ---------------------------------------------------------------
                #  Preprocessing filters
                # ---------------------------------------------------------------
                # Header title
                ui.h6("Apply Filters"),

                # Let the user define a new sampling interval. Leaving the field
                # blank preserves the current recording frequency
                ui.input_text(
                    "filt_resample",
                    "Resample epoch (blank = no change)",
                    placeholder="e.g. 1min, 30s",
                    value="",
                ),

                # Let the user enable or disable activity binarization
                ui.input_checkbox(
                    "filt_binarize",
                    "Binarize",
                    value=False
                    ),
                
                # Show the binarization threshold only when binarization is enabled
                ui.panel_conditional(
                    "input.filt_binarize",

                    # Values above the threshold are converted to one
                    # Values below the threshold are converted to zero
                    ui.input_numeric(
                        "filt_threshold",
                        "Binarize threshold",
                        value=0,
                        min=0,
                    ),
                ),

                # Let the user enable or disable missing value imputation
                ui.input_checkbox(
                    "filt_impute",
                    "Impute NaN",
                    value=False
                    ),
                
                # Show the imputation method only when imputation is enabled
                ui.panel_conditional(
                    "input.filt_impute",

                    # Let the user select the method used to fill in missing values
                    ui.input_select(
                        "filt_impute_method",
                        "Imputation method",
                        choices={"mean": "Mean", "median": "Median"},
                        selected="mean",
                    ),
                ),

                # Place the Apply and reset buttons side by side in two equal-width columns
                ui.layout_columns(
                    # Apply button triggers the selected preprocessing filters on the current recording
                    ui.input_action_button(
                        "btn_apply",
                        "Apply",
                        class_="btn-success btn-sm",
                    ),
                    # Reset button discards all preprocessing changes and restores the original recording
                    ui.input_action_button(
                        "btn_reset",
                        "Reset",
                        class_="btn-outline-danger btn-sm",
                    ),
                    col_widths=[6, 6],
                ),
                # Separate local preprocessing from export to other tabs
                ui.hr(),
                
                # Export to analysis tab button
                ui.input_action_button(
                    "btn_export_to_tabs",
                    "Export to Analysis Tabs",
                    class_="btn-primary btn-sm w-100",
                ),
                # Explain that the export button pushes the preprocessed recording to all other tabs
                ui.p(
                    "Pushes the preprocessed recording to all other tabs.",
                    class_="text-muted small mt-1",
                ),

                # Separate preprocessing controls from batch selectors
                ui.hr(),
                # ---------------------------------------------------------------
                # Batch mode selectors
                # ---------------------------------------------------------------
                # Dynamically display factor and subject selections when
                # circStudio is operating in batch mode
                ui.output_ui("batch_controls"),

                # Set the sidebar width in pixels
                width=320,
            ),
            # --------------------------------------------------------------- 
            # Main panel
            # ---------------------------------------------------------------
            # Display the latest success or error message generated by the
            # preprocessing server
            ui.output_ui("status_msg"),

            # Activity plot heading
            ui.h5("Activity trace"),

            # Explain how to create manual mask periods from the plot and how
            # the colored bands should be interpreted
            ui.p(
                "Draw a rectangle on the chart to populate the Start/Stop "
                "fields, then click Add Period. Use the toolbar (top-right) "
                "to zoom or pan. Red band = non-wear; grey band = NaN after "
                "filter.",
                class_="text-muted small",
            ),

            # Display the interactive Plotly activity plot
            output_widget(
                "plot_main",
                height="430px"
                ),
            
            # Separate the plot from the  mask export control
            ui.hr(),

            # Export mask to CSV button
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
# Register this function as the server component of a Shiny module
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
    """
    Define the reactive server logic for the preprocessing tab.

    The preprocessing tab operates on a local deep copy of the active recording.
    Consequently, non-wear detection, masking, resampling, binarization, and
    imputation do not affect the shared recording until the user explicitly exports
    the processed object to the remaining analysis tabs.

    Parameters
    -----------
    input:
        Namespaced Shiny input object containing the values of the controls
        defined in `preprocessing_ui()``.
    
    output:
        Namespaced Shiny output object used to register dynamically rendered UI
        components.
    
    session:
        Namespaced Shiny session object associated with this module.
    
    rv_single:
        Shared reactive value containing the recording loaded in single file mode
    
    rv_batch:
        Shared reactive value containing the collection of recordings loaded in
        batch mode.
    
    rv_mode:
        Shared reactive value identifying whether circStudio is operating in single
        file or batch mode
    
    rv_active_subject
        Shared reactive value identifying the currently active subject mode
    """
    # ------------------------------------------------------------------------
    # Local reactive state
    # ------------------------------------------------------------------------
    # Deep copy of the active Raw so we never mutate the shared reactive value
    _raw_local = reactive.Value(None)

    # Status message to surface to the user
    _status = reactive.Value("")
    
    # ------------------------------------------------------------------------
    # Keep the local copy in sync with the active subject
    # ------------------------------------------------------------------------
    @reactive.effect
    def _sync_raw():
        """
        Replace the local recording when the active recording changes.

        A deep copy is stored locally so that preprocessing operations  remain
        isolated from the shared application state.
        """
        # Retrieve the currently active recording according to the selected
        # single file or batch workflow
        raw = get_active(input, rv_mode, rv_single, rv_batch)

        # Store an independent copy of the recording, or None when no record
        # is available
        _raw_local.set(_copy_or_none(raw))

        # Remove status messages associated with the previously active recording
        _status.set("")
    # ------------------------------------------------------------------------
    # Batch mode selectors
    # ------------------------------------------------------------------------
    @render.ui
    def batch_controls():
        """Display factor and subject selectors when batch mode is active"""
        # Read the current application mode
        mode = rv_mode()

        # Batch controls are unnecessary when operating in single file mode
        if mode != "batch":
            return ui.div()

        # Build the optional factor filter control from the current batch
        factor_ctrl = factor_filter_controls(rv_mode, rv_batch)

        # Build the subject selector and use the active subject as the default
        subj_ctrl = subject_controls(
            input,
            rv_mode,
            rv_batch,
            default=rv_active_subject(),
        )

        # Collect only the controls that are currently available
        controls = []

        # Add the factor filter control if it was successfully constructed
        if factor_ctrl:
            controls.append(factor_ctrl)
        
        # Add the subject selector control if it was successfully constructed
        if subj_ctrl:
            controls.append(subj_ctrl)

        # Return the batch controls if any were successfully constructed
        if controls:
            return ui.div(ui.hr(), ui.h6("Batch"), *controls)

        # Return an empty container when no batch controls can be constructed
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
        """
        Import mask periods from the file selected by the user.

        The uploaded file is first copied to a temporary path that preserves its
        original extension. The current local recording then imports the mask
        periods from that copied file.
        """
        # Access the editable local recording.
        raw = _raw_local()

        if raw is None:
            _status.set(
                "Error: No recording loaded."
            )
            return

        # Read the file metadata returned by the Shiny upload control.
        finfo = input.mask_file()

        if not finfo:
            _status.set(
                "Error: Please select a mask log file first."
            )
            return

        # Only one file is permitted by the UI.
        src = finfo[0]

        # Copy the upload while preserving its original filename and extension.
        tmp = _copy_uploaded_file(src)

        try:
            # Add all mask periods contained in the uploaded file.
            raw.add_mask_periods(
                str(tmp)
            )

            # Store a new object reference so dependent outputs re-render.
            _raw_local.set(
                copy.deepcopy(raw)
            )

            _status.set(
                f"Mask imported from '{src['name']}'."
            )

        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error importing mask: {exc}"
            )

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

        # Prepare the numeric x-axis and its corresponding timestamp labels.
        (
            x_pos,
            time_labels,
            tick_vals,
            tick_text,
        ) = _time_axis_details(act)

        # Convert activity to plain numeric values. This avoids silent rendering
        # problems when the Series uses pandas nullable/object dtypes.
        y_plot = pd.to_numeric(act, errors="coerce").to_numpy(dtype=float)

        color = ("#2ca02c"
        if np.isnan(y_plot).any()
        else "#1f77b4"
        )

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
        mask = _safe_mask(raw)

        if mask is not None:
            # Align the mask with the displayed activity epochs.
            aligned = mask.reindex(act.index)

            # The Raw mask convention uses 0 to identify non-wear.
            is_nw = (aligned == 0).values

            # Draw a red band for every contiguous non-wear period.
            _add_interval_bands(
                fig,
                is_nw,
                x_pos,
                y0=0,
                y1=1,
                fillcolor="rgba(255, 0, 0, 0.3)",
                linecolor="red",
            )

        # --- NaN regions: grey band just above red --------------------------
        if np.isnan(y_plot).any():
            # Identify activity epochs that remain missing after filtering.
            is_nan = np.isnan(y_plot)

            # Draw a grey band above the red non-wear band for every NaN region.
            _add_interval_bands(
                fig,
                is_nan,
                x_pos,
                y0=0,
                y1=1,
                fillcolor="rgba(127,127,127,0.6)",
                linecolor="grey"
            )

        y_min, y_max = _activity_y_range(y_plot)

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
    @render.download(
        filename="mask_export.csv"
        )
    def btn_export_mask():
        """
        Export the current non-wear mask as a CSV file.

        The downloaded file contains one row per exported non-wear interval and
        provides ``Start_time`` and ``Stop_time`` columns. When no mask exists,
        an empty CSV containing only those column headers is returned.
        """
        # Access the currently edited local recording.
        raw = _raw_local()

        # Assume that no mask is available until one is retrieved successfully.
        mask = None

        if raw is not None:
            try:
                mask = raw.mask

            except Exception:
                pass

        # Return an empty table when no mask is available.
        if mask is None:
            df = pd.DataFrame(
                columns=[
                    "Start_time",
                    "Stop_time",
                ]
            )

        # Convert the available binary mask to start/stop intervals.
        else:
            df = _mask_to_dataframe(mask)

        # Write the DataFrame to an in-memory text buffer.
        buf = io.StringIO()

        df.to_csv(
            buf,
            index=False,
        )

        # Yield the CSV text to Shiny's download handler.
        yield buf.getvalue()