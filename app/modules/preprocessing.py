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
    # ------------------------------------------------------------------------
    # Status message
    # ------------------------------------------------------------------------
    # Render the latest preprocessing outcome as a Bootstrap alert
    @render.ui
    def status_msg():
        """Display the most recent success or error message to the user"""
        # Read the current status message
        msg = _status()

        # Return an empty container when no message is available
        if not msg:
            return ui.div()

        # Show errors in and successful operations in green
        colour = "danger" if msg.startswith("Error") else "success"

        # Wrap the message in a Bootstrap alert with compact spacing
        return ui.div(
            ui.tags.div(
                msg,
                class_=f"alert alert-{colour} alert-dismissible py-2 px-3",
                role="alert",
            ),
            class_="mb-2",
        )
    # ------------------------------------------------------------------------
    # Automatic non-wear detection
    # ------------------------------------------------------------------------
    # Run this effect only when the user clicks the Detect Non-Wear button
    @reactive.effect
    @reactive.event(input.btn_detect)
    def _on_detect():
        """Detect non-wear periods and add them to the current local mask"""
        # Access the editable local copy of the active recording
        raw = _raw_local()

        # Stop if no recording has been loaded into the preprocessing tab
        if raw is None:
            _status.set(
                "Error: No recording loaded. Please upload a file first."
                )
            return None

        # Read the selected non-wear detection algorithm
        method = input.nw_method()

        # Read the minimum non-wear duration (90 minutes by default) and remove any leading/trailing whitespace
        min_length = (
            input.nw_min_length()
            or "90min"
        ).strip()

        # Convert the permitted number of isolated spikes to an integer
        spike_tol = int(
            input.nw_spike_tolerance()
            )
        
        # Convert the maximum permitted spike count to an integer
        spike_max = int(
            input.nw_spike_max_counts()
            )

        # Start with no optional arguments for an algorithm
        detection_options = {}

        # The Choi algorithm requires a neighbourhood window size to examine 
        # before and after a spike
        if method == "choi":
            detection_options["window_size"] = (
                input.nw_window_size()
                or "30min"
            ).strip()

        try:
            # Preserve the existing mask periods before automatic detection
            if raw._mask is not None:
                existing_mask = raw._mask.copy()
            else:
                existing_mask = None

            # Detect non-wear using the algorithm and parameters selected by the user
            raw.detect_nonwear(
                method=method,
                min_length=min_length,
                spike_tolerance=spike_tol,
                spike_max_counts=spike_max,
                **detection_options,
            )

            # Combine automatic detection with any previously defined mask periods
            if existing_mask is not None and raw._mask is not None:
                # Align the previous mask with the epochs in the newly detected mask
                aligned_mask = existing_mask.reindex(
                    raw._mask.index,
                    fill_value=1
                )

                # Use the minimum because 0 means non-wear and must take priority
                raw._mask = raw._mask.combine(
                    aligned_mask,
                    min
                )

            # Access the combined mask through the public Raw interface
            mask = raw.mask

            # Count all epochs classified as non-wear (mask value == 0)
            if mask is not None:
                nw_epochs = int(
                    (mask == 0).sum()
                )
            else:
                nw_epochs = 0
            
            # Use the recording epoch duration when it supports time conversion
            if hasattr(raw.frequency, "total_seconds"):
                epoch_seconds = raw.frequency.total_seconds()
            else:
                epoch_seconds = 60

            # Convert the number of non-wear epochs into hours for reporting
            nonwear_hours = round(
                nw_epochs * epoch_seconds / 3600,
                1
            )

            # Assume no distinct non-wear intervals were detected
            nonwear_periods = 0

            # Count transitions from wear (1) to nonwear (0) as interval starts
            if mask is not None and nw_epochs > 0:
                interval_starts = (
                    mask.diff()
                    .fillna(mask.iloc[0] - 1)
                    < 0
                )
                nonwear_periods = int(
                    interval_starts.sum()
                )

            # Store a new object reference so dependent Shiny outputs re-render
            # and reflect the updated mask
            _raw_local.set(copy.deepcopy(raw))

            # Update the status message to summarize the detection results
            _status.set(
                f"Non-wear detection complete ({method.capitalize()}). "
                f"Flagged {nw_epochs} epochs ({nonwear_hours} h) "
                f"across {nonwear_periods} separate period(s). "
                f"Zoom in on the plot to inspect individual periods."
            )
        
        # Convert any detection failure into a user-visible error message
        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error during detection: {exc}"
            )
    # -------------------------------------------------------------------------
    #  manual add mask period
    # -------------------------------------------------------------------------
    # Run this effect only when the user clicks the Add Period button
    @reactive.effect
    @reactive.event(input.btn_add_period)
    def _on_add_period():
        """Add a nonwear interval, specified by the user, to the local mask"""
        # Access the editable local copy of the active recording
        raw = _raw_local()

        # Stop if no recording has been loaded
        if raw is None:
            _status.set("Error: No recording loaded.")
            return None

        # Read and clean the manually entered start datetime
        start = (
            input.mask_start() 
            or ""
        ).strip()

        # Read and clean the manually entered stop date
        stop = (
            input.mask_stop()
            or ""
        ).strip()
        
        # Require both boundaries before creating a mask interval
        if not start or not stop:
            _status.set("Error: Please enter both a start and a stop time.")
            return None

        try:
            # Add the requested interval to the Raw object's mask
            raw.add_mask_period(
                start,
                stop
            )

            # Store a deep copy so Shiny recognizes that the recording changed
            _raw_local.set(copy.deepcopy(raw))

            # Confirm which interval was added
            _status.set(
                f"Mask period added: {start} → {stop}."
            )
        
        # Report invalid dates or mask-update failures without stopping the app
        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error adding mask period: {exc}"
            )

    # -------------------------------------------------------------------------
    # Rectangle selection from the activity plot
    # -------------------------------------------------------------------------
    # Run this effect when JavaScript sends a newly drawn Plotly rectangle
    @reactive.effect
    @reactive.event(input.selected_range)
    def _on_drawn_rect():
        """Copy a drawn plot interval into the manual start and stop fields"""
        # Read the x-axis limits captured by the Plottly rectangle
        selected_range = input.selected_range()

        # Ignore empty selection events
        if not selected_range:
            return None

        # Convert the first rectangle boundaery to a string and remove any
        # leading/trailing whitespace
        x0 = str(
            selected_range.get("x0", "")
        ).strip()

        # Convert the second rectangle boundary to a string and remove any
        # leading/trailing whitespace
        x1 = str(
            selected_range.get("x1", "")
        ).strip()
        
        # Ignore incomplete rectangle events
        if not x0 or not x1:
            return None

        try:
            # Parse both boundaries as pandas timestamps
            start = pd.to_datetime(x0)
            stop = pd.to_datetime(x1)

            # Restore chronological order when the rectangle was drawn
            # right-to-left 
            if start > stop:
                start, stop = stop, start

            # Format the start timestamp for the manual-entry control
            start_string = start.strftime("%Y-%m-%d %H:%M:%S")
            stop_string = stop.strftime("%Y-%m-%d %H:%M:%S")

        # Preserve unusual Plotly date strings when pandas cannot parse them
        except Exception:
            start_string, stop_string = sorted([x0, x1])
        
        # Populate the manual mask start field with the selected start time
        ui.update_text(
            "mask_start",
            value=start_string
        )

        # Populate the manual mask stop field with the selected stop time
        ui.update_text(
            "mask_stop",
            value=stop_string
        )

        # Tell the user that the interval still requires explicit confirmation
        _status.set(
            f"Selection: {start_string} → {stop_string}. Click 'Add Period' to confirm."
        )

    # -------------------------------------------------------------------------
    # Mask import from file
    # -------------------------------------------------------------------------
    # Run this effect only when the user clicks the Import Mask button
    @reactive.effect
    @reactive.event(input.btn_import_mask)
    def _on_import_mask():
        """
        Import mask periods from the file selected by the user.

        The uploaded file is first copied to a temporary path that preserves its
        original extension. The current local recording then imports the mask
        periods from that copied file.
        """
        # Access the editable local recording
        raw = _raw_local()

        # Stop if no recording has been loaded
        if raw is None:
            _status.set(
                "Error: No recording loaded."
            )
            return None

        # Read the file metadata returned by the Shiny upload control.
        uploaded_files = input.mask_file()

        # Require the user to select a mask log before importing
        if not uploaded_files:
            _status.set(
                "Error: Please select a mask log file first."
            )
            return None

        # Only one file is permitted by the UI
        src = uploaded_files[0]

        # Copy the upload while preserving its original filename and extension
        tmp = _copy_uploaded_file(src)

        try:
            # Add all mask periods contained in the uploaded file
            raw.add_mask_periods(
                str(tmp)
            )

            # Store a new object reference so dependent outputs re-render
            _raw_local.set(
                copy.deepcopy(raw)
            )
            
            # Confirm the source filename used for the import
            _status.set(
                f"Mask imported from '{src['name']}'."
            )
        # Report file format, parsing, or mask-update errors to the user
        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error importing mask: {exc}"
            )
    # -------------------------------------------------------------------------
    # Preprocessing filters
    # -------------------------------------------------------------------------
    # Run this effect only when the user clicks the Apply button
    @reactive.effect
    @reactive.event(input.btn_apply)
    def _on_apply():
        """Apply the selected filters to the editable local recording"""
        # Access the editable local copy of the active recording
        raw = _raw_local()

        # Stop when no recording has been loaded
        if raw is None:
            _status.set("Error: No recording loaded.")
            return None

        # --- Resampling ----------------------------------------------------------
        # Read the request frequency; None preserves the current frequency
        new_freq = (
            input.filt_resample()
            or ""
        ).strip() or None

        # --- Binarization -------------------------------------------------------
        # Record whether activity counts should be binarized
        binarize = bool(
            input.filt_binarize()
        )

        # Use the user-specified threshold when binarization is enabled
        if binarize:
            threshold = float(
                input.filt_threshold()
            )
        else:
            threshold = 0
        
        # --- Imputation ----------------------------------------------------------
        # Record whether missing values should be imputed        
        impute = bool(
            input.filt_impute()
        )
        
        # Use the selected imputation method only when imputation is enabled
        if impute:
            imp_method = input.filt_impute_method()
        else:
            imp_method = "mean"
        
        # --- Apply filters -------------------------------------------------------
        # Apply the mask only when the recording currently contains one
        has_mask = raw._mask is not None

        try:
            # Run all selected preprocessing operations in the Raw pipeline
            raw.apply_filters(
                new_freq=new_freq,
                binarize=binarize,
                threshold=threshold,
                apply_mask=has_mask,
                impute_nan=impute,
                imputation_method=imp_method,
            )

            # Store a deep copy so reactive plots update the filtered data
            _raw_local.set(copy.deepcopy(raw))

            # Confirm that the preprocessing pipeline completed
            _status.set("Filters applied.")

        # Report invalid parameters/processing failures to the user
        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error applying filters: {exc}"
            )

    # -------------------------------------------------------------------------
    # Reset local preprocessing
    # -------------------------------------------------------------------------
    # Run this effect only when the user clicks the Reset button
    @reactive.effect
    @reactive.event(input.btn_reset)
    def _on_reset():
        """Restore the local copy of the unprocessed active recording"""
        # Retrieve the original recording from the shared application state
        original_raw = get_active(input, rv_mode, rv_single, rv_batch)

        # Replace the edited copy only when an active recording exists
        if original_raw is not None:
            _raw_local.set(copy.deepcopy(original_raw))

        # Confirm the local preprocessing changes were discarded
        _status.set("Reset to original recording.")
    # -------------------------------------------------------------------------
    # Interactive activity plot
    # -------------------------------------------------------------------------
    # Render the activity plot whenever its reactive inputs change
    @render_widget
    def plot_main():
        """Plot activity with overlays for non-wear and missing-value periods"""
        # Access the current editable recording
        raw = _raw_local()

        # Display message when no recording has been loaded
        if raw is None:
            return empty_fig(
                "Upload a file on the 'Data Upload' tab first."
            )

        # Extract the activity time series from the Raw object
        activity = activity_series(raw)

        # Informative placeholder when activity data are unavailable
        if activity is None:
            return empty_fig(
                "No activity data available."
            )

        # Create an empty Plotly figure to receive the trace and interval bounds
        fig = go.Figure()

        # Prepare numeric x positions, hover labels, and readable axis ticks
        (
            x_pos,
            time_labels,
            tick_vals,
            tick_text,
        ) = _time_axis_details(
            activity
        )

        # Convert activity to plain numeric values. This avoids silent rendering
        # problems when the Series uses pandas nullable/object dtypes.
        y_plot = pd.to_numeric(
            activity,
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        # Check if recording contains missing/nowear periods
        contains_missing_values = np.isnan(
            y_plot
        ).any()

        # Use green after NaNs appear; use the standard blue trace otherwise
        color = ("#2ca02c" if contains_missing_values else "#1f77b4")

        # Add the continuous activity-count trace to the figure
        # The hover template shows the date/time and activity counts for each point
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
        # Access the current mask without allowing mask errors to break the plot
        mask = _safe_mask(raw)

        # Add red bands when a mask is available
        if mask is not None:
            # Align the mask with the activity epochs currently being displayed
            aligned_mask = mask.reindex(
                activity.index
            )

            # Mark epochs whose mask value of 0 identifies nonwear
            is_nw = (
                aligned_mask == 0
            ).values

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
        
        # Calculate the padded vertical range from the finite activity values
        y_min, y_max = _activity_y_range(y_plot)
        
        # Configure the axes, margins, drawing mode, and rectangle appearance
        fig.update_layout(
            xaxis={
                "title": "Date/Time",
                "type": "linear",
                "tickmode": "array",
                "tickvals": tick_vals,
                "ticktext": tick_text,
            },
            yaxis={
                "title": "Activity counts",
                "range": [y_min, y_max],
                "zeroline": True,
            },
            margin={
                "l": 40,
                "r": 20,
                "t": 30,
                "b": 45
            },
            height=430,
            autosize=True,
            showlegend=False,
            dragmode="drawrect",
            newshape={
                "line": {
                    "color": "rgba(214,39,40,0.7)",
                    "width": 1.5
                },
                "fillcolor": "rgba(214,39,40,0.15)",
            },
        )

        # Return the completed interactive figure to the Shiny output
        return fig

    # -------------------------------------------------------------------------
    # Export preprocessed recording to the analysis tabs
    # -------------------------------------------------------------------------
    # Run this effect only when the user clicks Export to Analysis Tabs
    @reactive.effect
    @reactive.event(input.btn_export_to_tabs)
    def _on_export_to_tabs():
        """Commit the locally preprocessed recording to shared app state"""
    
        # BUG(BUG-1): After sharing the data with the other tabs, it is not possible
        # to revert to the original recording in the preprocessing tab, because
        # rv_single.set() / batch[subj] overwrite the shared state with no backup of
        # the prior value. The user must re-upload the file to start over.
        
        # Access the current editable recording
        raw = _raw_local()

        # Stop if there is no recording to export
        if raw is None:
            _status.set(
                "Error: No recording loaded."
            )
            return None

        try:
            # Read whether the app is currently operating in single file or batch mode
            mode = rv_mode()

            # Replace the shared reactive value with a deep copy of the local recording
            if mode == "single":
                rv_single.set(copy.deepcopy(raw))
            
            # Replace only the selected recording in the batch
            else:
                # Access the shared batch collection
                batch = rv_batch()

                # Identify the subject selected in the preprocessing controls
                subj = selected_subject(input)

                # Update the batch only when both collection and subject exist
                if batch is not None and subj:
                    batch[subj] = copy.deepcopy(raw)

                    # Store a new batch reference so all analysis tabs update
                    rv_batch.set(copy.deepcopy(batch))
            
            # Confirm that the processed recording entered the shared app state
            _status.set("Recording exported to all analysis tabs.")
        
        # Report failures while updating shared single or batch mode
        except Exception as exc:  # noqa: BLE001
            _status.set(
                f"Error exporting: {exc}"
            )

    # -------------------------------------------------------------------------
    # Export non-wear mask to CSV
    # -------------------------------------------------------------------------
    # Register a download handler with a fixed CSV name
    @render.download(filename="mask_export.csv")
    def btn_export_mask():
        """
        Export the current non-wear mask as a CSV file.

        The downloaded file contains one row per exported non-wear interval and
        provides ``Start_time`` and ``Stop_time`` columns. When no mask exists,
        an empty CSV containing only those column headers is returned.
        """
        # Access the currently edited local recording
        raw = _raw_local()

        # Assume that no mask is available until one is retrieved successfully
        mask = None

        # Try to access the public mask when one exists
        if raw is not None:
            try:
                mask = raw.mask

            # Leave mask as None if the Raw object cannot provide it
            except Exception:
                pass

        # Return an empty dataframe when no mask is available
        if mask is None:
            df = pd.DataFrame(
                columns=[
                    "Start_time",
                    "Stop_time",
                ]
            )

        # Convert the available binary mask to start/stop intervals
        else:
            df = _mask_to_dataframe(mask)

        # Write the DataFrame to an in-memory text buffer
        buf = io.StringIO()

        # Write the mask dataframe without the pandas row index
        df.to_csv(
            buf,
            index=False,
        )

        # Yield the CSV text to Shiny's download handler
        yield buf.getvalue()