"""
circStudio Shiny web app
-------------------------
Each analysis tab is a Shiny module in its own file in the modules package. This file 
assembles them into a single Shiny app and defines the per-session reactive state passed
to every module.

Each analysis page has two parts:
    1. A ``*_ui``function, which defines what the user sees in the browser.
    2. A ``*_server``function, which defines what happens when the user uploads data, clicks
    buttons, changes parameters, or requests plots.

Launch with::

    cd /path/to/circStudio
    pip install -e src/
    pip install -r app/requirements.txt
    shiny run app/app.py

The three top-level reactive values (``rv_single``, ``rv_batch``, ``rv_mode``)
plus a shared ``rv_active_subject`` are created per session and threaded into
each module so the analysis tabs always operate on the active recording.

Data flow
----------
The app keeps four shared reactive (rv) values per user session:

``rv_single``
    Holds one uploaded actigraphy file, i.e., a ``Raw``object.

``rv_batch``
    Holds a batch collection when several a directory with several recordings
    is specified.

``rv_mode``
    Either "single" or "batch", depending on whether the user uploaded a single
    recording or specified a directory with several recordings.

``rv_active_subject```
    In batch mode, stores which subject is currently selected for analysis.

These values are passed into the modlules so that all tabsanalyse the same currently
active recording or subject.
"""
# ---------------------------------------------------------------------------
# Project directory structure
# ---------------------------------------------------------------------------
# circStudio/
# ├── app/
# │   ├── app.py
# │   ├── modules/
# │   └── www/
# └── src/
#     └── circstudio/
#
# The code below constructs this structure from the location of app.py
# ---------------------------------------------------------------------------

from pathlib import Path
from import_paths import _add_import_path

# Absolute path to this app.py file (Path(__file__) == path to this file)
_this_file = Path(__file__).resolve()

# Directory containing this app.py file (app/)
app_dir = _this_file.parent

# Project (circStudio/) directory
_project_dir = app_dir.parent

# Source code directory containing the circStudio package
src_dir = _project_dir / "src"

# Put app/ and src/ on the import path so `modules`, `utils`, and `circstudio` resolve
_add_import_path(app_dir)
_add_import_path(src_dir)

# HTMLDependency is used to include a custom CSS theme for the app
from htmltools import HTMLDependency  # noqa: E402

# Shiny core imports:
# - App: application object
# - reactive: reactive values/effects/calculations
# - ui: user interface constructors
from shiny import App, reactive, ui  # noqa: E402

# Import modules for each analysis tab. Each module has a *_ui and *_server function
from modules.upload import upload_ui, upload_server  # noqa: E402
from modules.preprocessing import preprocessing_ui, preprocessing_server  # noqa: E402
from modules.batch_overview import batch_overview_ui, batch_overview_server # noqa: E402
from modules.daily_profile import daily_profile_ui, daily_profile_server # noqa: E402
from modules.sleep import sleep_ui, sleep_server  # noqa: E402
from modules.cosinor import cosinor_ui, cosinor_server  # noqa: E402
from modules.lids import lids_ui, lids_server  # noqa: E402
from modules.fractal import fractal_ui, fractal_server  # noqa: E402
from modules.flm import flm_ui, flm_server  # noqa: E402
from modules.ssa import ssa_ui, ssa_server  # noqa: E402
from modules.circ_models import circ_models_ui, circ_models_server # noqa : E402
from modules.llm_chat import llm_chat_ui, llm_chat_server  # noqa: E402

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
# This links the custom CSS file located in app/www/theme.css.
_THEME_DEP = HTMLDependency(
    name="circstudio-theme",
    version="1.0.0",
    head='<link rel="stylesheet" href="theme.css">',
)

# Build the main app navigation bar, with one tab for each analysis module
app_ui = ui.page_navbar(
    ui.nav_panel("Data Upload", upload_ui("upload")),
    ui.nav_panel("Preprocessing", preprocessing_ui("preprocess")),
    ui.nav_panel("Batch Overview", batch_overview_ui("batch")),
    ui.nav_panel("Daily Profile", daily_profile_ui("daily")),
    ui.nav_panel("Sleep Scoring", sleep_ui("sleep")),
    ui.nav_panel("Cosinor", cosinor_ui("cosinor")),
    ui.nav_panel("LIDS", lids_ui("lids")),
    ui.nav_panel("Fractal / MFDFA", fractal_ui("fractal")),
    ui.nav_panel("FLM", flm_ui("flm")),
    ui.nav_panel("SSA", ssa_ui("ssa")),
    ui.nav_panel("Circadian Models", circ_models_ui("circ")),
    ui.nav_panel("Assistant", llm_chat_ui("chat")),
    title="circStudio",
    id="main_nav",
    header=_THEME_DEP,
)

def server(input, output, session):
    # -----------------------------------------------------------------------
    # Per-session reactive state
    # -----------------------------------------------------------------------
    # Reactive values are containers that notify downstream code when their
    # contents change. For example, when a user uploads a new recording, tabs
    # that depend on that recording can automatically update

    # Holds a single ``Raw``object
    rv_single = reactive.Value(None)

    # Holds a batch collection of recordings
    rv_batch = reactive.Value(None)

    # Either "single" or "batch"
    rv_mode = reactive.Value("single")

    # Current subject in batch mode
    rv_active_subject = reactive.Value(None)

    # -----------------------------------------------------------------------
    # Connect each tab's server to the shared reactive state. The following
    # four variables constitute the app's shared memory:
    #
    #     rv_single: single loaded actigraphy recording (single mode)
    #     rv_batch: batch collection of recordings (batch mode)
    #     rv_mode: "single" or "batch"
    #     rv_active_subject: current subject in batch mode
    #
    # When one tab updates one, the others will react automatically.
    # -----------------------------------------------------------------------

    # The loading module does not neeed information about the selected subject
    upload_server(
        "upload",
        rv_single,
        rv_batch,
        rv_mode
        )
    
    # The preprocessing module needs information on the selected subject
    preprocessing_server(
        "preprocess",
        rv_single,
        rv_batch,
        rv_mode,
        rv_active_subject
        )
    
    # The batch module onlyb requires information about the collection
    # And selected subject
    batch_overview_server(
        "batch",
        rv_batch,
        rv_active_subject
        )

    # All remaining modules take the same four values, set them up in a loop
    for tab_id, server_func in (
        ("daily", daily_profile_server),
        ("sleep", sleep_server),
        ("cosinor", cosinor_server),
        ("lids", lids_server),
        ("fractal", fractal_server),
        ("flm", flm_server),
        ("ssa", ssa_server),
        ("circ", circ_models_server),
        ("chat", llm_chat_server)
    ):
        server_func(tab_id, rv_single, rv_batch, rv_mode, rv_active_subject)

# Assemble the Shiny app from the UI layout and server logic
# Serve static files (CSS) from app/www
app = App(app_ui, server, static_assets=app_dir / "www")
