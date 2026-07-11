"""
Tab 0 - Data upload

This module allows the user to load actigraphy data in two ways:

    1. Single file mode:
    Load one recording, inspect metadata and plot activity/light channels.

    2. Batch mode:
    Load a directory containing several recordings, inspect metadata, and 
    show a quick activity overview for the first subjects.

The loaded data are stored in shared reactive (rv) values created in ``app.py``:

``rv_single``
    The currently loaded single Raw object.

``rv_batch``
    The currently loaded batch collection.

``rv_mode``
    The current loading mode: either ``"single"`` or ``"batch"``.

Other analysis tabs read from these same reactive values, so loading data here
defines what the rest of the app will analyse.
"""
# Standard library imports
import asyncio
import shutil
import sys
import tempfile
import subprocess
from pathlib import Path
import textwrap

# Third-party scientific and plotting libraries
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shiny components used to define reactive UI and server behavior
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget
from import_paths import _add_import_path

# ---------------------------------------------------------------------------
# Project structure and import paths
# ---------------------------------------------------------------------------
# circStudio/
# ├── app/
# │   ├── app.py
# │   ├── modules/
# │   │   └── upload.py      <- this file
# │   └── utils.py
# └── src/
#     └── circstudio/
#
# ``upload.py`` lives inside app/modules/.
# Therefore:
#
#   this_file = .../circStudio/app/modules/upload.py
#   module_dir = .../circStudio/app/modules
#   app_dir = .../circStudio/app
#
# Adding app_dir to sys.path allows imports such as:
#
#   from utils import ...
#   from modules._common import ...
# ---------------------------------------------------------------------------

# Update the import path
_this_file = Path(__file__).resolve()
_module_dir = _this_file.parent
app_dir = _module_dir.parent
_add_import_path(app_dir)

# Import shared utility functions and constants
from utils import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    dispatch_reader,
    raw_has_light,
    raw_summary_row,
    scan_batch_directory,
)
from modules._common import activity_series, empty_fig  # noqa: E402

# ---------------------------------------------------------------------------
# Optional bundled example data
# ---------------------------------------------------------------------------
try:
    from importlib.resources import files as _res_files
except Exception:
    _res_files = None

# ---------------------------------------------------------------------------
# Supported file formats
# ---------------------------------------------------------------------------
_format_choices = {
    "auto": "Auto-detect (by extension)",
    "awd": "AWD (Actiwatch)",
    "agd": "AGD (ActiGraph)",
    "rpx": "RPX / Respironics (.csv)",
    "dqt": "DQT (.csv)",
    "mesa": "MESA (.csv)",
    "tal": "TAL / MotionWare",
    "atr": "ATR (.txt)",
}

# ---------------------------------------------------------------------------
# ``upload_ui()`` helpers
# ---------------------------------------------------------------------------
def _accepted_file_extensions():
    """Return accepted file extensions, including upper-case variants."""
    # Create list to store uppercase extensions
    uppercase_extensions = []

    # Uppercase all extensions in the supported extensions list
    for extension in SUPPORTED_EXTENSIONS:
        uppercase_extensions.append(extension.upper())
    
    # Return the combined list of supported extensions and uppercase extensions
    return sorted(SUPPORTED_EXTENSIONS) + sorted(uppercase_extensions)

def _single_file_inputs(accept):
    """Inputs shown when the user selects single-file loading."""
    return ui.panel_conditional(
        "input.load_mode === 'single'",
        ui.input_file(
            "file",
            "Upload an actigraphy file",
            accept=accept,
            multiple=False,
        ),
        ui.input_action_button(
            "use_example",
            "Use example file",
            class_="btn-outline-secondary btn-sm",
        ),
    )

def _batch_inputs():
    """
    Inputs shown when the user selects batch-directory loading.
    """
    return ui.panel_conditional(
        "input.load_mode === 'batch'",
        ui.input_action_button(
            "browse_btn",
            "Browse…",
            class_="btn-outline-secondary btn-sm",
        ),
        ui.input_text(
            "batch_dir",
            "Directory path",
            placeholder="/path/to/root",
            width="100%",
        ),
        ui.input_action_button(
            "load_batch",
            "Load batch",
            class_="btn-primary btn-sm",
        ),
    )

def _upload_intro_text():
    """
    Short explanation shown at the top of the upload tab.
    """
    return ui.div(
        ui.p(
            "Both single-file and batch (multi-subject) modes are supported "
            "— use the radio buttons in the sidebar to switch between them. "
            "All subsequent analysis tabs operate on whichever recording is "
            "currently active."
        ),
        class_="text-muted small mb-3",
    )

def _single_file_outputs():
    """
    Outputs shown after loading one recording.
    """
    return ui.panel_conditional(
        "input.load_mode === 'single'",
        ui.h4("Recording summary"),
        ui.output_data_frame("single_summary"),
        ui.hr(),
        ui.input_radio_buttons(
            "plot_channel",
            "Channel",
            {"activity": "Activity", "light": "Light"},
            selected="activity",
            inline=True,
        ),
        output_widget("single_plot"),
    )

def _batch_outputs():
    """
    Outputs shown after loading a batch directory.
    """
    return ui.panel_conditional(
        "input.load_mode === 'batch'",
        ui.layout_columns(
            ui.value_box("Files loaded", ui.output_text("vb_loaded")),
            ui.value_box("Failed", ui.output_text("vb_failed")),
            ui.value_box("Factors", ui.output_text("vb_factors")),
            col_widths=[4, 4, 4],
        ),
        ui.h4("Loaded recordings"),
        ui.output_data_frame("batch_meta"),
        ui.hr(),
        ui.h4("Activity overview (first subjects)"),
        output_widget("batch_overview_plot"),
    )

# ---------------------------------------------------------------------------
# ``upload_server()`` helpers
# ---------------------------------------------------------------------------
def _check_format(selected_format):
    """Detect whether the user selected "auto" or a specific format."""
    # If "auto", return None to let the reader auto-detect the format
    if selected_format == "auto":
        return None

    # Otherwise, return the selected format as-is
    else:
        return selected_format

def _copy_upload_to_temp_file(upload_info):
    """Copy a Shiny upload to a temporary file with the original extension."""
    # Get the original file name, including extension
    original_name = Path(upload_info["name"])

    # Create a temporary folder to store a copy of the file
    temporary_folder = tempfile.mkdtemp()

    # Build a temporary file path that keeps the original name (stem + extension)
    temporary_file_path = Path(temporary_folder) / original_name.name

    # Copy Shiny's uploaded file into that temporary filepath
    shutil.copy(upload_info["datapath"], temporary_file_path)
    
    # Return the copied file path so the app can read it
    return temporary_file_path

def _loading_spinner(what: str):
    """Small loading message shown while a file or batch is being read."""
    return ui.div(
        ui.tags.div(
            class_="spinner-border spinner-border-sm text-primary",
            role="status",
        ),
        ui.tags.strong(
            f"  Loading {what}…  ",
            style="margin-left:10px;",
        ),
        ui.tags.span("please wait", class_="text-muted"),
        class_="d-flex align-items-center my-2",
    )

def _single_summary_dataframe(raw) -> pd.DataFrame:
    """Create the metadata table shown for one loaded recording."""
    s = raw_summary_row(raw)

    return pd.DataFrame(
        {
            "property": [
                "Recording duration",
                "Sampling frequency",
                "Number of epochs",
                "Start timestamp",
                "End timestamp",
                "Light channel present",
            ],
            "value": [
                s["duration"],
                s["sampling_freq"],
                s["n_epochs"],
                s["start_time"],
                s["end_time"],
                s["has_light"],
            ],
        }
    )

def _plotly_palette():
    """Return the active Plotly color palette, with a safe fallback."""
    # Retrieve the trace colors defined by the active Plotly template
    palette = go.Figure().layout.template.layout.colorway

    # If no colors are defined, return a hard-coded fallback palette
    if not palette:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # Return trace colors if available (Plotly), otherwise use fallback palette
    return palette

def _factor_color_map(levels, palette):
    """Map factor levels to plot colors."""
    # Create a dictionary to map each factor level to a color
    level_to_color = {}

    # Iterate over the factor levels and assign colors from the palette
    for i, level in enumerate(levels):
        # Reuse colors from the start if the palette runs out
        color = palette[i % len(palette)]

        # Map level to color
        level_to_color[level] = color

    # Return mapping
    return level_to_color

def _batch_subplot_titles(entries):
    """Create subplot titles using subject IDs and optional factor labels."""
    subplot_titles = []
    for entry in entries:
        # Start the title with the subject ID
        title = str(entry.subject_id)

        # Add factor labels in brackets when they are available
        if entry.factor_levels:
            factor_label = " / ".join(entry.factor_levels)
            title += f" [{factor_label}]"
        
        # Collect subplot titles for each entry
        subplot_titles.append(title)

    # Return list of subplot titles
    return subplot_titles

def _add_subject_activity_trace(fig, entry, row, color_map, palette):
    """Add one subject's activity trace to the batch overview plot."""
    activity = activity_series(entry.raw)

    if activity is None:
        return None

    # Use the first factor level to decide the trace color
    if entry.factor_levels:
        level = entry.factor_levels[0]
    else:
        level = None
    
    # Get the color for this level or use the first palette color as fallback
    color = color_map.get(level, palette[0])

    # Add this subject's activity trace to the requested subplot
    fig.add_trace(
        go.Scatter(
            x=activity.index.astype(str),
            y=activity.values,
            line=dict(width=0.7, color=color),
            name=entry.subject_id,
            showlegend=False,
        ),
        row=row,
        col=1,
    )

def _batch_overview_figure(batch, max_rows = 12):
    """Create the batch activity overview plot."""
    # Keep only the first subjects to prevent creating a large figure
    entries = batch.entries[:max_rows]

    # Get the levels of the first factor (if any) to assign colors to traces
    if batch.factor_names:
        levels = batch.levels_for_factor(0)
    else:
        levels = []

    # Prepare the color palette and map each factor level to one color
    palette = _plotly_palette()
    color_map = _factor_color_map(levels, palette)

    # Create one subplot title per subject
    titles = _batch_subplot_titles(entries)

    # Make a vertical plot with one row per subject
    fig = make_subplots(
        rows=len(entries),
        cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=max(0.02, 0.12 / max(1, len(entries))),
    )

    # Add each subject's activity trace to the corresponding subplot
    for row, entry in enumerate(entries, start=1):
        _add_subject_activity_trace(fig, entry, row, color_map, palette)

    # Increase the figure height according to the number of displayed subjects
    figure_height = max(220, 130 * len(entries))

    # Show a title only when the plot is displaying a subset of the batch
    if len(batch) > max_rows:
        title = f"Showing first {max_rows} of {len(batch)} subjects"
    else:
        title = None

    # Apply the final layout settings
    fig.update_layout(
        height=figure_height,
        margin=dict(l=40, r=20, t=30, b=30),
        title=title,
    )

    return fig
# ---------------------------------------------------------------------
# Batch loading helpers
# ---------------------------------------------------------------------
def _pick_folder_via_dialog():
    """Open a native OS folder picker and return the selected path."""
    # Standalone script used to open the native folder picker
    _dialog_script = textwrap.dedent(
        """
        import tkinter as tk
        from tkinter import filedialog

        r = tk.Tk()
        r.withdraw()
        r.wm_attributes('-topmost', True)

        p = filedialog.askdirectory(title='Select batch directory')
        r.destroy()
        print(p or '')
        """
    )

    try:
        # Run a small separate Python script that opens the folder picker
        proc = subprocess.run(
            [sys.executable, "-c", _dialog_script],
            capture_output=True,
            text=True,
            timeout=600,
        )

        # Read the selected folder path printed by the dialog script
        selected_path = (proc.stdout or "").strip()

        # Return the path if one was selected; otherwise, return None
        if selected_path:
            return selected_path

        return None

    except Exception:
        # Return None if the dialog fails, times out, or cannot open
        return None

# ---------------------------------------------------------------------
# Upload UI
# ---------------------------------------------------------------------
@module.ui
def upload_ui():
    """Create and upload page UI"""
    # Accept both lower-case and upper-case file extensions
    accepted_extensions = _accepted_file_extensions()

    return ui.layout_sidebar(
        # Choose whether to load one file or a full batch directory
        ui.sidebar(
            ui.input_radio_buttons(
                "load_mode",
                "Loading mode",
                {"single": "Single file", "batch": "Batch directory"},
                selected="single",
            ),

            # Choose the file format, or let the app auto-detect it by extension
            ui.input_select(
                "fmt",
                "File format",
                choices=_format_choices,
                selected="auto",
            ),

            # Inputs shown when loading a single file
            _single_file_inputs(accepted_extensions),

            # Inputs shown when loading a batch directory
            _batch_inputs(),
            width=320,
        ),

        # Introductory help text for the upload page
        _upload_intro_text(),

        # Status message shown while files are being loaded
        ui.output_ui("load_status"),

        # Outputs shown after loading a single file
        _single_file_outputs(),

        # Outputs shown after loading a batch directory
        _batch_outputs()
    )

# ---------------------------------------------------------------------
# Upload server
# ---------------------------------------------------------------------
# Register the upload page's server-side logic
@module.server
def upload_server(input, output, session, rv_single, rv_batch, rv_mode):
    """
    Define the server logic for the upload page.

    This function handles single file uploads, example file loading, batch
    directory loading, loading status messages, summary tables, and preview
    plots. Loaded single file and batch objects are stored in shared reactive
    values so other modules can access them.
    """
    # Store the loaded single-file Raw object for display
    _single_raw = reactive.Value(None)

    # Store the latest load error so it remains visible in the UI
    _load_error: reactive.Value[str | None] = reactive.Value(None)

    # Name of the file currently being loaded
    _pending_name = reactive.Value("")

    # Keep the selected loading mode synchronized with the shared app state
    @reactive.effect
    def _sync_mode():
        """Update the active loading mode and clear the inactive data store"""
        # Read the active loading mode from the UI
        mode = input.load_mode()

        # Store the selected mode so other modules can react to it
        rv_mode.set(mode)

        # Clear the inactive data store when switching modes
        if mode == "single":
            rv_batch.set(None)
        
        # Clear single file data when switching to batch mode
        else:
            rv_single.set(None)
            _single_raw.set(None)

    # ---------------------------------------------------------------------
    # Single-file loading
    # ---------------------------------------------------------------------
    # Decorator: mark this function as a long-running Shiny task
    @reactive.extended_task
    async def _single_task(path: str, fmt):
        """
        Read a single actigraphy file in a background thread and return the 
        loaded Raw object.
        """
        # Typecast the file path string to a Path object
        file_path = Path(path)

        # Read the file in a worker thread so the UI remains responsive
        raw = await asyncio.to_thread(lambda: dispatch_reader(
            file_path,
            fmt=fmt)
            )

        # Return the loaded Raw object
        return raw

    # 1st decorator: register this function as a reactive side effect
    # 2nd decorator: run this event only when the uploaded file input changes
    @reactive.effect
    @reactive.event(input.file)
    def _on_file():
        """Load the uploaded file when the user selects one from the file input."""
        # Read the current uploaded file information from the file input
        file_info = input.file()
        
        # Stop if no file has been selected
        if not file_info:
            return None

        # Shiny stores uploads in a list, even when only one file is allowed
        selected_file = file_info[0]

        # Copy the upload to a temporary file with its original extension
        temporary_file_path = _copy_upload_to_temp_file(selected_file)

        # Convert the UI format choice into the reader format
        fmt = _check_format(input.fmt())

        # Store the filename for status messages
        _pending_name.set(selected_file["name"])

        # Start the background file-loading task
        _single_task(str(temporary_file_path), fmt)

    # 1st decorator: register this function as a reactive side effect
    # 2nd decorator: run this event only when the user clicks "Use example file"
    @reactive.effect
    @reactive.event(input.use_example)
    def _on_example():
        """
        Load the example file included in the circStudio package when the user
        clicks the "Use example file" button.
        """
        # Stop if package example files are unavailable and display a warning
        if _res_files is None:
            ui.notification_show(
                "Example data not available in this environment.",
                type="warning",
            )
            return None
        
        # Locate the example file included in the circStudio package
        example = _res_files("circstudio.data") / "example_01.AWD"

        # Store the example filename for status messages
        _pending_name.set("example_01.AWD")

        # Force AWD so we never mis-detect the example file.
        _single_task(str(example), "awd")

    # 1st decorator: register this function as a reactive side effect
    @reactive.effect
    def _single_done():
        """
        Load the single-file Raw object into the shared reactive value and show
        a notification when loading succeeds or fails.
        """
        # Get the current status of the single-file loading task
        status = _single_task.status()

        # Do nothing while no task has run or while the task is still running
        if status in ("initial", "running"):
            return None
        
        # Use the pending filename or a generic fallback
        if not _pending_name():
            # Generic fallback name for unknown files (e.g., example data)
            name = "file"
        else:
            # Use the filename of the file currently being loaded
            name = _pending_name()
        
        # Handle file-loading errors
        if status == "error":
            try:
                # Try to retrieve the task result to see if it raises an exception
                _single_task.result()

                # Fallback message in the unlikely case that no exception was raised
                error = "unknown error"
            except Exception as exception:  # noqa: BLE001
                # Store the actual exception raised during file loading
                error = exception
            
            # Build a user-facing error message with a practical format-selection tip
            message = (
                f"Could not read '{name}'. {error}\n"
                "Tip: if this is a .csv/.txt/.mtn file, pick the exact format "
                "in the 'File format' dropdown instead of Auto-detect."
            )

            # Store the error so it remains visible in the summary table
            _load_error.set(message)

            # Show the error as a Shiny notification
            ui.notification_show(message, type="error", duration=12)

            # print the error to the console for debugging purposes
            print("[upload] " + message)

            # Stop here because loading failed
            return None

        # Retrieve the successfully loaded Raw object from the task
        raw = _single_task.result()

        # Clear any previous load error
        _load_error.set(None)

        # Store the Raw object in the shared single-file state
        rv_single.set(raw)

        # Store the Raw object for the single-file summary and plot outputs
        _single_raw.set(raw)

        # Notify the user that loading succeeded
        ui.notification_show(f"Loaded {name}.", type="message", duration=4)
    
    # Decorator: render this function's return value as a UI element
    @render.ui
    def load_status():
        """
        Show a loading spinner while a file or batch is being read
        """
        # Get the current status of the single-file loading task
        single = _single_task.status()

        # Get the current status of the batch-loading task
        batch = _batch_task.status()

        # Show a spinner while a single file is loading
        if single == "running":
            return _loading_spinner("file")

        # Show a spinner while a batch directory is loading
        if batch == "running":
            return _loading_spinner("batch directory")

        # Otherwise, show nothing
        return ui.div()

    # Render this function's return value as a data-frame
    @render.data_frame
    def single_summary():
        """
        Show a summary of the currently loaded single-file recording, or an error
        message if loading failed, or a placeholder if no file has been loaded yet.
        """
        # Get the currently loaded single-file Raw object
        raw = _single_raw.get()

        # Check whether no file has been successfully loaded yet
        if raw is None:
            # Get the latest loading error, if any
            error = _load_error.get()
            
            # Show the loading error as a small table
            if error:
                return pd.DataFrame({"load error": error.split("\n")})

            # Show a placeholder message if no file has been loaded yet
            return pd.DataFrame({"info": ["No file loaded yet."]})
        
        # Build the metadata summary for the loaded file
        df = _single_summary_dataframe(raw)

        # Render the summary table as an interactive data grid
        return render.DataGrid(df, width="100%")

    # Decorator: render this function's return value as a widget (Plotly figure)
    # shown on the upload page
    @render_widget
    def single_plot():
        """Show the selected recording channel or a placeholder"""
        
        # Get the currently loaded single file Raw object
        raw = _single_raw.get()

        # Show a placeholder before a file is loaded
        if raw is None:
            return empty_fig("Upload a file or click 'Use example file'.")
        
        # Read the selected plotting channel from the UI
        channel = input.plot_channel()

        # Show a placeholder if the recording has no light channel
        if channel == "light" and not raw_has_light(raw):
            return empty_fig("This recording has no light channel.")
        
        # Try to plot the selected channel
        try:
            return raw.plot(mode=channel)
        
        # Show a placeholder if plotting fails
        except Exception as exception:
            return empty_fig(f"Could not plot: {exception}")

    # ---------------------------------------------------------------------
    # Batch loading
    # ---------------------------------------------------------------------
    # Decorator: mark this function as a long-running Shiny task
    @reactive.extended_task
    async def _browse_task():
        # Open the folder picker in a worker thread so the UI stays
        # responsive
        return await asyncio.to_thread(_pick_folder_via_dialog)

    # 1st decorator: register this function as a reactive side effect
    # 2nd decorator: run this even only when the browse button is clicked
    @reactive.effect
    @reactive.event(input.browse_btn)
    def _open_folder_dialog():
        """
        Start the native OS folder picker when the user clicks 'Browse...'
        """
        _browse_task()

    # 1st decorator: register this function as a reactive side effect
    @reactive.effect
    def _browse_done():
        """Handle the result of the folder-picker task"""
        # Get the current status of the folder-picker task
        status = _browse_task.status()

        # Do nothing while the task has not started or is still running
        if status in ("initial", "running"):
            return None

        # Show a warning if the folder picker failed to open
        if status == "error":
            ui.notification_show(
                "Folder dialog unavailable. Please type or paste the path.",
                type="warning",
                duration=6,
            )
            return None
        
        # Read the folder selected by the user
        path = _browse_task.result()

        # Pre-fill the editable path field with the chosen folder.
        if path:
            ui.update_text("batch_dir", value=path)
        
        # Show a warning if the user closed the dialog without selecting a folder
        else:
            ui.notification_show(
                "No folder selected. Tip: you can also type the path manually.",
                type="warning",
                duration=5,
            )
    
    # Decorator: mark this function as a long-running Shiny task
    @reactive.extended_task
    async def _batch_task(path, fmt):
        """Scan the batch directory without blocking the Shiny UI"""
        # Typecast the directory path string to a Path object
        batch_directory = Path(path)

        # Define the blocking batch scan that should run outside the event loop
        def _scan_directory():
            return scan_batch_directory(batch_directory, fmt=fmt)
        
        # Run the batch scan in a worker thread so the UI stays responsive
        return await asyncio.to_thread(_scan_directory)
    
    # 1st decorator: register this function as a reactive side effect
    # 2nd decorator: run this even only when the user clicks "Load batch"
    @reactive.effect
    @reactive.event(input.load_batch)
    def _on_load_batch():
        """
        Start batch loading after the user provides a directory path 
        and clicks 'Load batch'
        """
        # Read and clean the directory path from the UI
        directory_path = (input.batch_dir() or "").strip()

        # Stop if no directory path was provided and display a warning
        if not directory_path:
            ui.notification_show(
                "Please select or enter a directory path.", type="warning"
            )
            return
        # Convert the UI format choice into the appropriate reader format
        fmt = _check_format(input.fmt())

        # Start the background batch-loading task
        _batch_task(directory_path, fmt)

    # 1st decorator: register this function as a reactive side effect
    @reactive.effect
    def _batch_done():
        """
        Handle the result of the batch-loading task.
        """
        # Get the current status of the batch-loading task
        status = _batch_task.status()

        # Do nothing when the task has not started or is still running
        if status in ("initial", "running"):
            return None

        # Handle batch-loading errors
        if status == "error":
            try:
                # Placeholder error message if no exception was raised
                _batch_task.result()
                exception = "unknown error"
            except Exception as e:  # noqa: BLE001
                # Retrieve the actual exception raised during batch loading
                exception = e
            
            # Show the error as a Shiny notification
            ui.notification_show(
                f"Batch load failed: {exception}", type="error", duration=8
            )
            return None

        # Retrieve the successfully loaded batch collection from the task
        collection = _batch_task.result()

        # Store the batch collection in the shared reactive value so that it
        # can be used by other tabs
        rv_batch.set(collection)

        # Show a notification summarizing the batch loading results
        msg = (
            f"Loaded {len(collection)} file(s); "
            f"{len(collection.errors)} issue(s)."
        )
        # Show a preview of the first few errors if any occurred during loading
        if collection.errors:
            # Construct a preview of the first few errors in the notification
            preview = "\n".join(collection.errors[:6])

            # Notify the user
            ui.notification_show(msg + "\n" + preview, type="warning", duration=10)
        else:
            # Notify the user that loading succeeded without errors
            ui.notification_show(msg, type="message", duration=5)

    def _current_batch():
        """Return the currently loaded batch collection, if any"""
        # Get the currently loaded batch collection from shared state
        batch = rv_batch()

        # Return None if no batch has been loaded yet
        if batch is None:
            return None
        
        # Return the loaded batch collection
        return batch

    # Render the number of successfully loaded batch files as text.
    @render.text
    def vb_loaded():
        """"Show the number of loaded batch files"""
        # Get the currently loaded batch collection
        batch = _current_batch()

        # Show the number of successfully loaded files
        # Show zero if no batch has been loaded yet
        return str(len(batch)) if batch else "0"

    # Render the number of failed batch files as text
    @render.text
    def vb_failed():
        """Show the number of batch-loading issues."""
        # Get the currently loaded batch collection
        batch = _current_batch()
        
        # Show the number of files that raised loading issues
        # Show zero before any batch has been loaded
        return str(len(batch.errors)) if batch else "0"

    # Render the detected batch factor names as text
    @render.text
    def vb_factors():
        """Show the factor names detected in the batch."""
        # Get the currently loaded batch collection
        batch = rv_batch()

        # Show "none" if no batch or no factor names are available
        if batch is None or not batch.factor_names:
            return "none"

        # Show the detected factor names as a comma-separated list
        return ", ".join(batch.factor_names)

    # Decorator: render the metadata table for the loaded batch
    @render.data_frame
    def batch_meta():
        """Show the metadata table for the loaded batch"""
        # Get the currently loaded batch collection
        batch = rv_batch()

        # Show a placeholder before any batch has been loaded
        if batch is None or len(batch) == 0:
            return pd.DataFrame({"info": ["No batch loaded yet."]})
        
        # Render the batch metadata as an interactive data grid
        return render.DataGrid(batch.to_metadata_df(), width="100%")

    # Decorator: render the batch activity overview plot as a widget (Plotly figure)
    #  shown on the upload page
    @render_widget
    def batch_overview_plot():
        # Get the currently loaded batch collection
        batch = rv_batch()

        # Show a placeholder before any batch has been loaded
        if batch is None or len(batch) == 0:
            return empty_fig("Load a batch directory to see the overview.")
        
        # Build and show the batch overview figure
        return _batch_overview_figure(batch)