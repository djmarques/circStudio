"""circStudio Shiny app - interactive actigraphy analysis.

Entry point that assembles every analysis tab as a Shiny *module*. Launch with::

    cd /path/to/circStudio
    pip install -e src/
    pip install -r app/requirements.txt
    shiny run app/app.py

The three top-level reactive values (``rv_single``, ``rv_batch``, ``rv_mode``)
plus a shared ``rv_active_subject`` are created per session and threaded into
each module so the analysis tabs always operate on the active recording.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``utils`` and the ``modules`` package importable regardless of CWD.
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# Also make the bundled ``circstudio`` package importable straight from
# ``src/`` so the app runs even when circStudio was not pip-installed.
_SRC_DIR = APP_DIR.parent / "src"
if _SRC_DIR.is_dir():
    sys.path.insert(0, str(_SRC_DIR))

from htmltools import HTMLDependency  # noqa: E402
from shiny import App, reactive, ui  # noqa: E402

from modules.upload import upload_ui, upload_server  # noqa: E402
from modules.preprocessing import preprocessing_ui, preprocessing_server  # noqa: E402
from modules.batch_overview import (  # noqa: E402
    batch_overview_ui,
    batch_overview_server,
)
from modules.daily_profile import (  # noqa: E402
    daily_profile_ui,
    daily_profile_server,
)
from modules.sleep import sleep_ui, sleep_server  # noqa: E402
from modules.cosinor import cosinor_ui, cosinor_server  # noqa: E402
from modules.lids import lids_ui, lids_server  # noqa: E402
from modules.fractal import fractal_ui, fractal_server  # noqa: E402
from modules.flm import flm_ui, flm_server  # noqa: E402
from modules.ssa import ssa_ui, ssa_server  # noqa: E402
from modules.circ_models import (  # noqa: E402
    circ_models_ui,
    circ_models_server,
)
from modules.llm_chat import llm_chat_ui, llm_chat_server  # noqa: E402

_THEME_DEP = HTMLDependency(
    name="circstudio-theme",
    version="1.0.0",
    head='<link rel="stylesheet" href="theme.css">',
)

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
    # Per-session reactive state.
    rv_single = reactive.Value(None)  # holds a single Raw object
    rv_batch = reactive.Value(None)  # holds a BatchCollection
    rv_mode = reactive.Value("single")  # "single" | "batch"
    rv_active_subject = reactive.Value(None)  # global subject (batch mode)

    upload_server("upload", rv_single, rv_batch, rv_mode)
    preprocessing_server("preprocess", rv_single, rv_batch, rv_mode, rv_active_subject)
    batch_overview_server("batch", rv_batch, rv_active_subject)

    for sid, srv in (
        ("daily", daily_profile_server),
        ("sleep", sleep_server),
        ("cosinor", cosinor_server),
        ("lids", lids_server),
        ("fractal", fractal_server),
        ("flm", flm_server),
        ("ssa", ssa_server),
    ):
        srv(sid, rv_single, rv_batch, rv_mode, rv_active_subject)

    circ_models_server("circ", rv_single, rv_batch, rv_mode, rv_active_subject)
    llm_chat_server("chat", rv_single, rv_batch, rv_mode, rv_active_subject)


app = App(app_ui, server, static_assets=APP_DIR / "www")
