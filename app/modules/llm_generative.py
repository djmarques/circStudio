"""Generative UI extension for llm_chat.py.

This module adds tool-calling to the LLM assistant so it can render analysis
components directly inside the chat panel — alongside (not instead of) the
existing tabs and nav links.

Overview
--------
- GENERATIVE_TOOLS         Tool definitions passed as ``tools=`` in the API call.
- GENERATIVE_PROMPT_SECTION  System-prompt section that is appended to the base
                             prompt when generative mode is active.
- render_tool_call()        Shiny dispatcher: maps a tool-call name + args to a
                             ``ui.TagList`` containing the rendered component.

Integration checklist (all changes are in llm_chat.py)
-------------------------------------------------------
1.  Import::

        from modules.llm_generative import (
            GENERATIVE_TOOLS,
            GENERATIVE_PROMPT_SECTION,
            render_tool_call,
        )

2.  Add a toggle to the chat sidebar (generative mode on/off)::

        ui.input_switch("generative_mode", "✨ Generative UI", value=False)

3.  Extend ``_build_system_prompt`` to append the section when the toggle is on::

        def _build_system_prompt(session_context, generative=False):
            base = _SYSTEM_PROMPT
            if generative:
                base += GENERATIVE_PROMPT_SECTION
            return base + f"\\n## Current session\\n{session_context}\\n"

4.  Pass tools to the API call when the toggle is on::

        tools_arg = GENERATIVE_TOOLS if input.generative_mode() else None
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=_build_api_messages(system, rv_history()),
            **({"tools": tools_arg, "tool_choice": "auto"} if tools_arg else {}),
        )

5.  Parse the response. A response can contain *both* a text block and tool
    calls. Collect each separately::

        msg     = response.choices[0].message
        text    = (msg.content or "").strip()
        calls   = getattr(msg, "tool_calls", None) or []

6.  Append to history (keep existing text bubble logic unchanged; add rendered
    widgets as a special ``"tool_result"`` role that the chat thread knows how
    to render)::

        new_entries = []
        if text:
            new_entries.append({"role": "assistant", "content": text})
        for tc in calls:
            import json
            args = json.loads(tc.function.arguments)
            widget = render_tool_call(
                tc.function.name, args, rv_single, rv_batch, rv_mode
            )
            new_entries.append({
                "role": "tool_result",
                "content": "",          # not sent back to the model
                "widget": widget,       # ui.TagList consumed by chat_thread
                "tool_name": tc.function.name,
            })
        history += new_entries

7.  Render tool_result entries in ``chat_thread``::

        for msg in history:
            if msg["role"] == "tool_result":
                bubbles.append(
                    ui.div(msg["widget"], style=_COMPONENT_BUBBLE)
                )

   Add a style constant for the component bubble::

        _COMPONENT_BUBBLE = (
            "align-self:flex-start;background:#fff;"
            "border:1px solid #dee2e6;border-radius:12px;"
            "padding:14px;max-width:95%;overflow:hidden;"
        )

Notes
-----
- ``render_tool_call`` works by running the circstudio analysis functions
  directly (same code path as the tabs), converting the Plotly figure to
  self-contained HTML via ``plotly.io.to_html``, and wrapping everything in
  a ``ui.TagList``.  No new Shiny output IDs are registered — the widget is
  just static HTML injected into the chat bubble.
- Tool calls are *not* added to the message history sent to the model (only
  the assistant text is), keeping history management simple and avoiding
  serialisation of large HTML blobs.
- Both Ollama (llama3.1+) and Groq (llama-3.3-70b-versatile,
  llama3-groq-70b-8192-tool-use-preview) support the OpenAI tool-calling
  schema used here.
"""

from __future__ import annotations

import json
import sys
import uuid as _uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio
from shiny import ui

APP_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = APP_DIR.parent / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

from modules._common import activity_series, empty_fig, get_active, light_series  # noqa: E402


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
# Each entry is an OpenAI-schema tool definition.  The ``description`` field
# is the most important part: it is what the model reads to decide whether to
# call the tool, which parameters to fill, and when to prefer a nav link
# instead.  Keep descriptions precise and grounded in the actual analysis.

GENERATIVE_TOOLS: list[dict] = [

    # ------------------------------------------------------------------
    # Daily Profile
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_daily_profile",
            "description": (
                "Render the average 24-hour activity (or light) profile directly "
                "in the chat, together with the full NPCRA metrics table "
                "(IS, IV, RA, M10, L5, AonT, AoffT, ADAT, kRA, kAR, etc.).\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'plot', 'visualise', or 'display' the daily profile\n"
                "  • asks about the 'average daily profile', 'daily activity profile', 'mean daily profile', or 'typical daily pattern'\n"
                "  • asks 'what is the daily profile' or 'what does the daily activity look like'\n"
                "  • asks for the value of any NPCRA metric (IS, IV, RA, M10, L5 …)\n"
                "  • wants a quick overview of the recording's circadian structure\n\n"
                "DO NOT use this tool (prefer a nav link instead) when:\n"
                "  • the user wants to change parameters interactively (whs, LMX window …)\n"
                "  • the user is asking a conceptual question about what these metrics mean\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "string",
                        "enum": ["activity", "light"],
                        "default": "activity",
                        "description": (
                            "Which channel to plot. Use 'light' only if the user "
                            "explicitly asks for the light profile or a light metric."
                        ),
                    },
                    "cyclic": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, the profile wraps (first hour = last hour). "
                            "Use only when the user explicitly requests a cyclic plot."
                        ),
                    },
                    "log_scale": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Apply a log scale to the y-axis. "
                            "ALWAYS default to false. "
                            "Set to true ONLY when the user explicitly asks for a log scale or log-transformed axis."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Sleep Scoring
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_sleep_scoring",
            "description": (
                "Render a sleep/wake hypnogram and a summary statistics table "
                "(TST, SE, WASO, SRI, SoD) directly in the chat.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show' or 'plot' sleep results, the hypnogram, or sleep metrics\n"
                "  • asks for the value of TST, SE, WASO, SRI, SoD, or fSoD\n"
                "  • wants to compare sleep algorithms side-by-side in chat\n\n"
                "DO NOT use this tool when:\n"
                "  • the user wants to adjust rescoring thresholds or epoch settings interactively\n"
                "  • the user is asking conceptually what TST or SE means\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["Cole-Kripke", "Roenneberg", "Sadeh", "Scripps", "Oakley"],
                        "default": "Cole-Kripke",
                        "description": (
                            "Sleep-scoring algorithm to apply. "
                            "Default to Cole-Kripke for adult wrist data. "
                            "Use Roenneberg if SRI or SoD are the main outcomes, "
                            "or if the user explicitly requests it. "
                            "Use Sadeh for adolescent data."
                        ),
                    },
                    "show_hypnogram": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include the epoch-by-epoch sleep/wake time series plot.",
                    },
                    "show_summary": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include the summary statistics table (TST, SE, WASO …).",
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Cosinor
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_cosinor",
            "description": (
                "Fit a cosine model to the activity time series and render the "
                "fitted curve overlaid on the data, together with the parameter "
                "table (mesor, amplitude, acrophase, period, R²).\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'fit', or 'run' Cosinor analysis\n"
                "  • asks for mesor, amplitude, acrophase, period, or R²\n"
                "  • wants to see how well a cosine fits their data\n\n"
                "DO NOT use this tool when:\n"
                "  • the user asks about the 'daily profile', 'average profile', 'daily activity profile', or 'typical daily pattern' — use render_daily_profile instead\n"
                "  • the user wants a general overview of their activity rhythm — use render_daily_profile instead\n"
                "  • the user wants to adjust initial parameter guesses interactively\n"
                "  • the user asks conceptually what acrophase means\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period_h": {
                        "type": "number",
                        "default": 24,
                        "description": (
                            "Expected period in hours. Use 24 for standard circadian analysis. "
                            "Change only if the user suspects a non-24 h rhythm."
                        ),
                    },
                    "method": {
                        "type": "string",
                        "enum": ["leastsq", "differential_evolution"],
                        "default": "leastsq",
                        "description": (
                            "Fitting method. 'leastsq' is fast and suitable for most recordings. "
                            "Use 'differential_evolution' for very noisy or fragmented data "
                            "where leastsq tends to get stuck in local minima."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # LIDS
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_lids",
            "description": (
                "Apply the LIDS transform to the loaded recording and render the "
                "LIDS time series with its fitted oscillation, plus the MRI value.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'run', or 'plot' LIDS\n"
                "  • asks for the MRI (LIDS Rhythm Index)\n\n"
                "IMPORTANT CONSTRAINT: LIDS is only valid on night recordings (a single "
                "sleep episode, not a multi-day continuous recording). If the loaded "
                "recording spans multiple days, do NOT call this tool — instead tell "
                "the user that LIDS requires a night-only slice and suggest they use "
                "the LIDS tab where they can select the relevant window.\n\n"
                "DO NOT use this tool when:\n"
                "  • the loaded recording is clearly a multi-day continuous recording\n"
                "  • the user is asking what LIDS is or how to interpret MRI\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fit_func": {
                        "type": "string",
                        "enum": ["cosine", "gaussian"],
                        "default": "cosine",
                        "description": "Function to fit to the LIDS oscillation.",
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Fractal / MFDFA
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_fractal",
            "description": (
                "Run Multifractal DFA on the activity series and render the "
                "generalised Hurst exponent h(q) curve.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'run', or 'plot' fractal analysis or MFDFA\n"
                "  • asks for h(q), the Hurst exponent, or multifractality\n\n"
                "NOTE: MFDFA is computationally expensive on long recordings. "
                "For recordings longer than ~7 days, warn the user and still call "
                "the tool unless they decline.\n\n"
                "DO NOT use this tool when:\n"
                "  • the user is asking conceptually what MFDFA or h(q) means\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q_min": {
                        "type": "number",
                        "default": -3,
                        "description": "Minimum statistical moment q.",
                    },
                    "q_max": {
                        "type": "number",
                        "default": 3,
                        "description": "Maximum statistical moment q.",
                    },
                    "q_steps": {
                        "type": "integer",
                        "default": 7,
                        "description": "Number of q values between q_min and q_max.",
                    },
                    "detrending_order": {
                        "type": "integer",
                        "default": 1,
                        "description": (
                            "Polynomial detrending order. 1 (linear) is standard. "
                            "Increase to 2 only for clearly non-stationary data."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # FLM
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_flm",
            "description": (
                "Fit a Functional Linear Model to the daily activity profile "
                "and render the fitted curve overlaid on the empirical profile, "
                "plus the RMSE.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'fit', or 'run' FLM\n"
                "  • asks for the RMSE of the FLM fit\n"
                "  • wants to compare basis representations of the daily profile\n\n"
                "DO NOT use this tool when:\n"
                "  • the user is asking conceptually what FLM or basis functions are\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "basis": {
                        "type": "string",
                        "enum": ["fourier", "spline"],
                        "default": "fourier",
                        "description": (
                            "Basis function family. 'fourier' is best for rhythmic data "
                            "with a clear periodicity. 'spline' is more flexible and "
                            "handles irregular shapes better."
                        ),
                    },
                    "max_order": {
                        "type": "integer",
                        "default": 10,
                        "description": (
                            "Number of harmonics (Fourier) or knots (spline). "
                            "Higher values fit more detail but risk overfitting. "
                            "10 is a sensible default for 1-min epoch data."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # SSA
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_ssa",
            "description": (
                "Run Singular Spectrum Analysis on the activity series and render "
                "the variance-explained bar chart plus the reconstructed signal "
                "from the top components.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'run', or 'plot' SSA\n"
                "  • asks how much variance each SSA component explains\n"
                "  • asks to see the trend or rhythmic components extracted by SSA\n\n"
                "DO NOT use this tool when:\n"
                "  • the user asks conceptually what SSA or the W-correlation matrix means\n"
                "  • the user wants to interactively select components for reconstruction\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "string",
                        "default": "24h",
                        "description": (
                            "Embedding window length as a pandas offset string "
                            "(e.g. '24h', '12h'). Should match the expected cycle length. "
                            "Use '24h' for standard circadian analysis."
                        ),
                    },
                    "n_components": {
                        "type": "integer",
                        "default": 6,
                        "description": (
                            "Number of components to extract and show. "
                            "6 is usually enough to see trend + first two harmonics + noise. "
                            "Increase if the user asks for more detail."
                        ),
                    },
                    "resample": {
                        "type": "string",
                        "default": "10min",
                        "description": (
                            "Resampling frequency before SSA (for speed). "
                            "Does not affect interpretation for smooth signals."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Recording summary (preprocessing)
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_recording_summary",
            "description": (
                "Show a summary table of the loaded recording: start/end times, "
                "duration, epoch length, total epochs, missing data (NaN) %, "
                "non-wear %, and available channels.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks 'what recording is loaded?', 'what data do I have?'\n"
                "  • asks about duration, start/end time, epoch frequency, or file quality\n"
                "  • asks how much missing data or non-wear there is\n"
                "  • wants a quality overview before starting analysis\n\n"
                "DO NOT use this tool when:\n"
                "  • the user wants to see the raw trace — use render_activity_trace instead\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Activity trace (preprocessing)
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_activity_trace",
            "description": (
                "Plot the raw actigraphy time series directly in the chat, "
                "with non-wear periods highlighted as red shading when present.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'show', 'plot', or 'display' the raw activity data or actigram\n"
                "  • asks 'what does the recording look like?'\n"
                "  • wants to visually inspect the data before running any analysis\n\n"
                "DO NOT use this tool when:\n"
                "  • the user wants the average 24-h pattern — use render_daily_profile instead\n"
                "  • no recording is loaded"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "show_nonwear": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            "Highlight non-wear periods as red shading. "
                            "ALWAYS default to true when a mask is available."
                        ),
                    },
                    "log_scale": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Use logarithmic y-axis. "
                            "ALWAYS default to false. "
                            "Set true ONLY if the user explicitly asks for it."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    # ------------------------------------------------------------------
    # Batch comparison
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "render_batch_comparison",
            "description": (
                "Render a group-level comparison chart for a chosen circadian or "
                "sleep metric across all subjects in the loaded batch.\n\n"
                "USE THIS TOOL when the user:\n"
                "  • asks to 'compare', 'show', or 'plot' a metric across subjects\n"
                "  • is in batch mode and asks for group-level results\n"
                "  • asks 'which subject has the highest/lowest [metric]?'\n\n"
                "DO NOT use this tool when:\n"
                "  • the app is in single-subject mode (only one recording loaded)\n"
                "  • no batch is loaded\n"
                "  • the user wants to drill into one subject's trace"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["IS", "IV", "RA", "M10", "L5", "ADAT", "kRA", "kAR",
                                 "TST", "SE", "WASO", "SRI",
                                 "Mesor", "Amplitude", "Acrophase", "R2"],
                        "description": (
                            "The metric to compare across subjects. "
                            "Choose from the NPCRA, sleep, or Cosinor output metrics. "
                            "Infer from the user's question (e.g. 'who sleeps the most' → TST, "
                            "'who has the strongest rhythm' → IS or RA)."
                        ),
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["bar", "box", "violin"],
                        "default": "bar",
                        "description": (
                            "Chart style. Use 'bar' for a simple per-subject overview, "
                            "'box' or 'violin' when the user asks about distribution or spread."
                        ),
                    },
                },
                "required": ["metric"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System-prompt section (append to _SYSTEM_PROMPT when generative mode is on)
# ---------------------------------------------------------------------------

GENERATIVE_PROMPT_SECTION = """
---

## Generative UI mode  *(active)*

You now have access to tool calls that render analysis components directly
inside the chat panel.  This extends — it does not replace — your existing
guide behaviour.  You can mix free text, nav links, and tool calls in a
single response.

---

### Decision rule: when to render vs. navigate

**Render a component (call a tool) when the user wants to *see* a result:**
- "Show me my IS value."
- "Plot the daily profile."
- "What does my Cosinor fit look like?"
- "How fragmented is my rhythm?"
- "Compare IS across all subjects."

**Navigate (use a nav link) when the user wants to *interact*:**
- "How do I change the Cosinor period?"  → explain + nav link to Cosinor tab
- "I want to try a different algorithm." → explain options + nav link
- "I want to adjust the LMX window."    → nav link to Daily Profile tab

**Explain conceptually (no tool, no nav) when the user is asking *what* or *why*:**
- "What is IS?"
- "Why does a high IV matter?"
- "What's the difference between SSA and FLM?"

In ambiguous cases, do both: briefly answer the conceptual question and then
call the tool to show the actual result from their data.

---

### Tool call rules

1. **No recording loaded → do not call any render tool.**  Instead say:
   "Load a recording first — [Data Upload →](nav:Data Upload)" and stop.

2. **LIDS is only valid on night recordings.**  If the loaded recording spans
   more than ~48 hours, do not call `render_lids`.  Explain the constraint and
   point to the LIDS tab where the user can select the relevant window.

3. **Batch mode awareness.**  Only call `render_batch_comparison` when the
   session is in batch mode and a batch is loaded.  For single-subject mode,
   use per-subject render tools instead.

4. **Parameter defaults are good for most recordings.**  Only override a
   default parameter if:
   (a) the user explicitly specifies a value, or
   (b) a previous message establishes a reason (e.g., "I think my rhythm is
       non-24 h" → use a different period in `render_cosinor`).

5. **Multiple tools in one response are fine** when the user asks for several
   things at once (e.g., "show me the daily profile and the Cosinor fit").
   Call both tools; use text to tie the results together.

6. **After rendering, always add 1–2 sentences of interpretation** — what
   the result means for *this* recording, what to look for, and whether a
   follow-up analysis is warranted.  Do not merely describe what the plot
   shows; add clinical or analytical insight.

---

### Response format when using tools

Preferred structure:

> [Optional: one sentence of context or framing]
>
> [Tool call(s)]
>
> [1–2 sentences of interpretation of the rendered result]
>
> [Optional: nav link if interactive follow-up is natural]

Example (good):
> Here is your average daily profile for the loaded recording.
>
> *(renders daily_profile component)*
>
> Your IS of 0.71 indicates a well-entrained rhythm with consistent
> day-to-day timing.  The M10 onset around 09:00 and high RA suggest
> a clear contrast between active and rest periods — a healthy pattern.
> If you want to adjust the LMX window or run the light metrics, open
> the full tab: [Daily Profile →](nav:Daily Profile).

Example (bad — do not do this):
> I'll now show you the daily profile.  The daily profile is the mean
> activity waveform folded across 24 hours.  It was first described by...
> *(renders daily_profile component)*
> As you can see in the plot above, the activity rises in the morning...

---

### Style rules for generative responses

- Keep framing text **short** — one sentence before, 1–2 sentences after.
- Do not narrate what the component will show before rendering it.
- Do not re-describe what is visually obvious in the rendered chart.
- Use markdown tables and code blocks only when they add information not
  visible in the rendered component.
- If a tool call fails (component returns an error), tell the user what
  went wrong and suggest the equivalent tab as a fallback.
"""


# ---------------------------------------------------------------------------
# Render dispatcher
# ---------------------------------------------------------------------------

def render_tool_call(
    name: str,
    args: dict[str, Any],
    rv_single,
    rv_batch,
    rv_mode,
    rv_active_subject=None,
) -> ui.TagList:
    """Map an LLM tool-call to a Shiny TagList containing the rendered component.

    Each branch runs the relevant circstudio analysis directly (same functions
    used by the analysis tabs) and converts the Plotly figure to self-contained
    HTML via ``plotly.io.to_html``.  No new Shiny output IDs are registered —
    everything is static HTML injected into the chat bubble.

    Parameters
    ----------
    name:
        Tool name exactly as defined in GENERATIVE_TOOLS.
    args:
        Parsed JSON arguments from the tool call.
    rv_single, rv_batch, rv_mode:
        Reactive values passed from llm_chat_server.
    rv_active_subject:
        Optional; used in batch mode to resolve the active subject.
    """
    try:
        raw = _get_raw(rv_single, rv_batch, rv_mode, rv_active_subject)
    except Exception as exc:
        return _error_card(name, str(exc))

    dispatch: dict[str, Any] = {
        "render_daily_profile":    _render_daily_profile,
        "render_sleep_scoring":    _render_sleep_scoring,
        "render_cosinor":          _render_cosinor,
        "render_lids":             _render_lids,
        "render_fractal":          _render_fractal,
        "render_flm":              _render_flm,
        "render_ssa":              _render_ssa,
        "render_batch_comparison": _render_batch_comparison,
        "render_recording_summary": _render_recording_summary,
        "render_activity_trace":   _render_activity_trace,
    }

    fn = dispatch.get(name)
    if fn is None:
        return _error_card(name, f"Unknown tool '{name}'.")

    try:
        return fn(raw, args, rv_single, rv_batch, rv_mode)
    except Exception as exc:
        return _error_card(name, str(exc))


# ---------------------------------------------------------------------------
# Per-tool render functions
# ---------------------------------------------------------------------------

def _render_daily_profile(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import (  # local import keeps startup fast
        daily_profile, IS, IV, l5, m10, ra, adat, kRA, kAR,
        AonT, AoffT,
    )
    from circstudio.analysis.sleep.sleep import AonT as _AonT, AoffT as _AoffT

    signal   = args.get("signal", "activity")
    cyclic   = bool(args.get("cyclic", False))
    log      = bool(args.get("log_scale", False))

    if signal == "light":
        s = light_series(raw)
        if s is None:
            return _error_card("render_daily_profile", "No light channel in this recording.")
    else:
        s = activity_series(raw)
        if s is None:
            return _error_card("render_daily_profile", "No activity channel in this recording.")

    fig = daily_profile(s, cyclic=cyclic, plot=True, log=log)
    fig_html = _fig_to_html(fig)

    # Build metrics table (activity only)
    rows = []
    if signal == "activity":
        def _safe(fn, fmt=None):
            try:
                v = fn()
                if isinstance(v, tuple):
                    v = v[1]
                if fmt == "hhmm":
                    return _hhmm(v)
                return round(float(v), 4)
            except Exception:
                return "n/a"

        rows = [
            ("IS",    _safe(lambda: IS(s)),   ""),
            ("IV",    _safe(lambda: IV(s)),   ""),
            ("RA",    _safe(lambda: ra(s)),   ""),
            ("M10",   _safe(lambda: m10(s)[1]), ""),
            ("M10 onset", _safe(lambda: m10(s)[0], "hhmm"), "HH:MM"),
            ("L5",    _safe(lambda: l5(s)[1]),  ""),
            ("L5 onset",  _safe(lambda: l5(s)[0], "hhmm"), "HH:MM"),
            ("ADAT",  _safe(lambda: adat(s)), "counts"),
            ("kRA",   _safe(lambda: kRA(s)),  ""),
            ("kAR",   _safe(lambda: kAR(s)),  ""),
        ]

    df_html = _df_to_html(
        pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])
    ) if rows else ""

    toolbar = _rerun_toolbar("render_daily_profile", {"signal": signal, "cyclic": cyclic, "log_scale": log}, [
        {"type": "select",   "key": "signal",    "label": "Signal",
         "options": [("activity", "Activity"), ("light", "Light")]},
        {"type": "checkbox", "key": "cyclic",    "label": "Cyclic"},
        {"type": "checkbox", "key": "log_scale", "label": "Log scale"},
    ])

    return ui.TagList(
        _section_label("Daily Profile"),
        ui.HTML(fig_html),
        ui.HTML(df_html) if df_html else ui.span(),
        ui.HTML(toolbar),
    )


def _render_sleep_scoring(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis.sleep.sleep import (
        Cole_Kripke, Roenneberg, Sadeh, Scripps,
        SleepRegularityIndex, waso,
    )
    import numpy as np
    import plotly.graph_objects as go

    algorithm    = args.get("algorithm", "Cole-Kripke")
    show_hyp     = bool(args.get("show_hypnogram", True))
    show_summary = bool(args.get("show_summary", True))

    act = activity_series(raw)
    if act is None:
        return _error_card("render_sleep_scoring", "No activity channel in this recording.")

    algo_map = {
        "Cole-Kripke": lambda a: Cole_Kripke(a, settings="mean"),
        "Roenneberg":  lambda a: Roenneberg(a),
        "Sadeh":       lambda a: Sadeh(a),
        "Scripps":     lambda a: Scripps(a),
    }
    algo_fn = algo_map.get(algorithm, algo_map["Cole-Kripke"])
    sw = algo_fn(act)

    parts: list = [_section_label(f"Sleep Scoring — {algorithm}")]

    if show_hyp:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sw.index, y=sw.values,
            mode="lines", line=dict(width=0.8, color="#0d6efd"),
            fill="tozeroy", fillcolor="rgba(13,110,253,0.15)",
            name="Sleep/Wake",
        ))
        fig.update_layout(
            yaxis=dict(tickvals=[0, 1], ticktext=["Wake", "Sleep"], title=""),
            xaxis_title="Time",
            margin=dict(l=50, r=20, t=30, b=40),
            height=220,
        )
        parts.append(ui.HTML(_fig_to_html(fig)))

    if show_summary:
        rows = []
        try:
            n_ep   = len(sw)
            freq   = getattr(raw, "frequency", pd.Timedelta("1min"))
            sleep  = sw.sum()
            tst_h  = float(sleep * freq.total_seconds() / 3600)
            tib_h  = float(n_ep * freq.total_seconds() / 3600)
            se     = tst_h / tib_h * 100 if tib_h > 0 else float("nan")
            _, waso_mean = waso(act, frequency=freq)
            rows = [
                ("TST (Total Sleep Time)", f"{tst_h:.2f}", "h"),
                ("SE (Sleep Efficiency)",  f"{se:.1f}",    "%"),
                ("WASO",                   f"{float(waso_mean):.1f}", "min"),
            ]
            try:
                sri = SleepRegularityIndex(act, algo="Roenneberg")
                rows.append(("SRI", f"{float(sri):.2f}", "−100 → +100"))
            except Exception:
                pass
        except Exception as exc:
            rows = [("Error", str(exc), "")]
        parts.append(ui.HTML(_df_to_html(
            pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])
        )))

    toolbar = _rerun_toolbar("render_sleep_scoring",
        {"algorithm": algorithm, "show_hypnogram": show_hyp, "show_summary": show_summary}, [
        {"type": "select",   "key": "algorithm",     "label": "Algorithm",
         "options": [("Cole-Kripke", "Cole-Kripke"), ("Roenneberg", "Roenneberg"),
                     ("Sadeh", "Sadeh"), ("Scripps", "Scripps"), ("Oakley", "Oakley")]},
        {"type": "checkbox", "key": "show_hypnogram", "label": "Hypnogram"},
        {"type": "checkbox", "key": "show_summary",   "label": "Summary table"},
    ])
    parts.append(ui.HTML(toolbar))
    return ui.TagList(*parts)


def _render_cosinor(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import Cosinor
    from lmfit import Parameters
    import plotly.graph_objects as go
    import numpy as np

    period_h = float(args.get("period_h", 24))
    method   = args.get("method", "leastsq")

    act = activity_series(raw)
    if act is None:
        return _error_card("render_cosinor", "No activity channel in this recording.")

    p = Parameters()
    p.add("Amplitude", value=float(act.std()),    min=0)
    p.add("Acrophase", value=3.14,                min=0, max=6.28)
    p.add("Period",    value=period_h * 60,       min=0)
    p.add("Mesor",     value=float(act.mean()),   min=0)

    model  = Cosinor()
    result = model.fit(act, params=p, method=method)

    # Build fitted curve
    t    = np.arange(len(act))
    pv   = result.params.valuesdict()
    fit  = (pv["Mesor"]
            + pv["Amplitude"] * np.cos(2 * np.pi / pv["Period"] * t + pv["Acrophase"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=act.index, y=act.values,
                             mode="lines", name="Activity",
                             line=dict(color="#adb5bd", width=0.8)))
    fig.add_trace(go.Scatter(x=act.index, y=fit,
                             mode="lines", name="Cosinor fit",
                             line=dict(color="#e63946", width=2)))
    fig.update_layout(
        xaxis_title="Time", yaxis_title="Activity",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=20, t=30, b=40), height=280,
    )

    acro_h = (pv["Acrophase"] / (2 * np.pi) * pv["Period"] / 60) % 24
    r2     = 1 - result.residual.var() / act.values.var() if act.values.var() > 0 else float("nan")

    rows = [
        ("Mesor",      f"{pv['Mesor']:.2f}",         "counts"),
        ("Amplitude",  f"{pv['Amplitude']:.2f}",      "counts"),
        ("Acrophase",  _hhmm(acro_h * 3600),          "HH:MM"),
        ("Period",     f"{pv['Period']/60:.2f}",       "h"),
        ("R²",         f"{r2:.3f}",                    ""),
    ]

    toolbar = _rerun_toolbar("render_cosinor", {"period_h": period_h, "method": method}, [
        {"type": "number", "key": "period_h", "label": "Period (h)",
         "min": 12, "max": 48, "step": 0.5},
        {"type": "select", "key": "method",   "label": "Method",
         "options": [("leastsq", "Least squares"),
                     ("differential_evolution", "Diff. evolution")]},
    ])

    return ui.TagList(
        _section_label(f"Cosinor fit — {method}, period={period_h} h"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(_df_to_html(pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"]))),
        ui.HTML(toolbar),
    )


def _render_lids(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import LIDS

    fit_func = args.get("fit_func", "cosine")
    act = activity_series(raw)
    if act is None:
        return _error_card("render_lids", "No activity channel in this recording.")

    obj      = LIDS()
    lids_ts  = obj.lids_transform(act)
    result   = obj.lids_fit(lids_ts, fit_func=fit_func)
    mri      = obj.lids_mri(result)

    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lids_ts.index, y=lids_ts.values,
                             mode="lines", name="LIDS",
                             line=dict(color="#0d6efd", width=1)))
    fig.update_layout(
        xaxis_title="Time", yaxis_title="LIDS",
        margin=dict(l=50, r=20, t=30, b=40), height=250,
    )

    rows = [("MRI (LIDS Rhythm Index)", f"{float(mri):.4f}", "")]

    toolbar = _rerun_toolbar("render_lids", {"fit_func": fit_func}, [
        {"type": "select", "key": "fit_func", "label": "Fit function",
         "options": [("cosine", "Cosine"), ("gaussian", "Gaussian")]},
    ])

    return ui.TagList(
        _section_label(f"LIDS — {fit_func} fit"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(_df_to_html(pd.DataFrame(rows, columns=["Metric", "Value", "Unit"]))),
        ui.HTML(toolbar),
    )


def _render_fractal(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import Fractal
    import numpy as np
    import plotly.graph_objects as go

    q_min  = float(args.get("q_min", -3))
    q_max  = float(args.get("q_max",  3))
    steps  = int(args.get("q_steps", 7))
    deg    = int(args.get("detrending_order", 1))

    act = activity_series(raw)
    if act is None:
        return _error_card("render_fractal", "No activity channel in this recording.")

    q_arr = np.linspace(q_min, q_max, steps)
    q_arr = np.where(np.abs(q_arr) < 1e-6, 1e-6, q_arr)
    n_max = max(16, len(act) // 4)
    n_arr = np.unique(np.logspace(np.log10(16), np.log10(n_max), 30).astype(int))
    n_arr = n_arr[n_arr >= 4]

    Fm = Fractal.mfdfa(act, n_arr, q_arr, deg=deg)
    h_vals = [Fractal.generalized_hurst_exponent(Fm[:, j], n_arr)[0]
              for j in range(len(q_arr))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_arr, y=h_vals,
                             mode="lines+markers",
                             line=dict(color="#0d6efd", width=2)))
    fig.update_layout(
        xaxis_title="q", yaxis_title="h(q)",
        margin=dict(l=50, r=20, t=30, b=40), height=260,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#6c757d",
                  annotation_text="h=0.5 (random walk)")

    h2 = h_vals[len(h_vals) // 2]  # h at q≈0
    rows = [
        ("h(2) — long-range correlation",  f"{h_vals[steps // 2]:.3f}", ""),
        ("Δh — multifractal width",        f"{max(h_vals)-min(h_vals):.3f}", ""),
    ]

    toolbar = _rerun_toolbar("render_fractal",
        {"q_min": q_min, "q_max": q_max, "q_steps": steps, "detrending_order": deg}, [
        {"type": "number", "key": "q_min",            "label": "q min",    "step": 1},
        {"type": "number", "key": "q_max",            "label": "q max",    "step": 1},
        {"type": "number", "key": "q_steps",          "label": "q steps",  "min": 3, "max": 20, "step": 1},
        {"type": "number", "key": "detrending_order", "label": "Detrend",  "min": 1, "max": 3,  "step": 1},
    ])

    return ui.TagList(
        _section_label(f"Fractal / MFDFA — deg={deg}"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(_df_to_html(pd.DataFrame(rows, columns=["Metric", "Value", "Unit"]))),
        ui.HTML(toolbar),
    )


def _render_flm(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import FLM, daily_profile as dp
    import numpy as np
    import plotly.graph_objects as go

    basis     = args.get("basis", "fourier")
    max_order = int(args.get("max_order", 10))

    act = activity_series(raw)
    if act is None:
        return _error_card("render_flm", "No activity channel in this recording.")

    freq    = act.index.freq
    flm     = FLM(basis=basis, sampling_freq=freq, max_order=max_order)
    flm.fit(act)
    fitted  = np.asarray(flm.evaluate())
    profile = np.asarray(dp(act).values, dtype=float)
    n       = min(len(profile), len(fitted))
    rmse    = float(np.sqrt(np.nanmean((profile[:n] - fitted[:n]) ** 2)))

    x = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=profile[:n],
                             mode="lines", name="Daily profile",
                             line=dict(color="#adb5bd", width=1)))
    fig.add_trace(go.Scatter(x=x, y=fitted[:n],
                             mode="lines", name="FLM fit",
                             line=dict(color="#e63946", width=2)))
    fig.update_layout(
        xaxis_title="Epoch (within day)", yaxis_title="Activity",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=20, t=30, b=40), height=260,
    )

    rows = [
        ("Basis",      basis,            ""),
        ("Max order",  str(max_order),   ""),
        ("RMSE",       f"{rmse:.4f}",    "counts"),
    ]

    toolbar = _rerun_toolbar("render_flm", {"basis": basis, "max_order": max_order}, [
        {"type": "select", "key": "basis",     "label": "Basis",
         "options": [("fourier", "Fourier"), ("spline", "Spline")]},
        {"type": "number", "key": "max_order", "label": "Order", "min": 2, "max": 30, "step": 1},
    ])

    return ui.TagList(
        _section_label(f"FLM — {basis}, order={max_order}"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(_df_to_html(pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"]))),
        ui.HTML(toolbar),
    )


def _render_ssa(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    from circstudio.analysis import SSA
    import plotly.graph_objects as go
    import plotly.subplots as sp

    window       = args.get("window", "24h")
    n_components = int(args.get("n_components", 6))
    resample     = args.get("resample", "10min")

    act = activity_series(raw)
    if act is None:
        return _error_card("render_ssa", "No activity channel in this recording.")

    s = act.resample(resample).mean().dropna()
    s = s.asfreq(resample) if s.index.freq is None else s   # SSA requires freq != None
    s = s.interpolate()
    ssa = SSA(s, window_length=window)
    ssa.fit()

    var_exp = ssa.variance_explained[:n_components]
    x_comp  = list(range(1, len(var_exp) + 1))

    fig = go.Figure(go.Bar(
        x=x_comp, y=var_exp * 100,
        marker_color="#0d6efd",
    ))
    fig.update_layout(
        xaxis_title="Component", yaxis_title="Variance explained (%)",
        margin=dict(l=50, r=20, t=30, b=40), height=220,
    )

    reconstructed = ssa.reconstruct_signal(list(range(min(2, n_components))))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=s.index, y=s.values,
                              mode="lines", name="Original",
                              line=dict(color="#adb5bd", width=0.8)))
    fig2.add_trace(go.Scatter(x=s.index, y=reconstructed.values,
                              mode="lines", name="Reconstructed (comp. 1–2)",
                              line=dict(color="#e63946", width=1.5)))
    fig2.update_layout(
        xaxis_title="Time", yaxis_title="Activity",
        legend=dict(orientation="h"),
        margin=dict(l=50, r=20, t=30, b=40), height=240,
    )

    toolbar = _rerun_toolbar("render_ssa",
        {"window": window, "n_components": n_components, "resample": resample}, [
        {"type": "text",   "key": "window",       "label": "Window"},
        {"type": "number", "key": "n_components", "label": "Components", "min": 2, "max": 20, "step": 1},
        {"type": "text",   "key": "resample",     "label": "Resample"},
    ])

    return ui.TagList(
        _section_label(f"SSA — window={window}, top {n_components} components"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(_fig_to_html(fig2)),
        ui.HTML(toolbar),
    )


def _render_batch_comparison(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    """Compute a scalar metric for every subject in the batch and render a chart."""
    from circstudio.analysis import IS, IV, ra, m10, l5, adat, kRA, kAR
    import plotly.graph_objects as go

    metric    = args.get("metric", "IS")
    plot_type = args.get("plot_type", "bar")

    batch = rv_batch()
    if batch is None:
        return _error_card("render_batch_comparison", "No batch loaded.")

    _metric_fns = {
        "IS":   lambda a: float(IS(a)),
        "IV":   lambda a: float(IV(a)),
        "RA":   lambda a: float(ra(a)),
        "M10":  lambda a: float(m10(a)[1]),
        "L5":   lambda a: float(l5(a)[1]),
        "ADAT": lambda a: float(adat(a)),
        "kRA":  lambda a: float(kRA(a)),
        "kAR":  lambda a: float(kAR(a)),
    }

    fn = _metric_fns.get(metric)
    if fn is None:
        return _error_card(
            "render_batch_comparison",
            f"Metric '{metric}' is not directly computable in batch mode from here. "
            f"Open the relevant tab to compute it across subjects."
        )

    subjects, values = [], []
    for entry in batch.entries:
        try:
            act = activity_series(entry.raw)
            if act is not None:
                subjects.append(entry.subject_id)
                values.append(fn(act))
        except Exception:
            pass

    if not subjects:
        return _error_card("render_batch_comparison", "Could not compute metric for any subject.")

    if plot_type == "bar":
        fig = go.Figure(go.Bar(x=subjects, y=values, marker_color="#0d6efd"))
    elif plot_type == "box":
        fig = go.Figure(go.Box(y=values, name=metric, marker_color="#0d6efd"))
    else:  # violin
        fig = go.Figure(go.Violin(y=values, name=metric,
                                  box_visible=True, meanline_visible=True,
                                  fillcolor="#0d6efd", opacity=0.6))

    fig.update_layout(
        xaxis_title="Subject" if plot_type == "bar" else "",
        yaxis_title=metric,
        margin=dict(l=50, r=20, t=30, b=60),
        height=280,
    )

    toolbar = _rerun_toolbar("render_batch_comparison", {"metric": metric, "plot_type": plot_type}, [
        {"type": "select", "key": "metric",    "label": "Metric",
         "options": [(m, m) for m in ["IS", "IV", "RA", "M10", "L5", "ADAT", "kRA", "kAR",
                                       "TST", "SE", "WASO", "SRI", "Mesor", "Amplitude", "Acrophase", "R2"]]},
        {"type": "select", "key": "plot_type", "label": "Chart",
         "options": [("bar", "Bar"), ("box", "Box"), ("violin", "Violin")]},
    ])

    return ui.TagList(
        _section_label(f"Batch comparison — {metric}"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(toolbar),
    )


# ---------------------------------------------------------------------------
# New preprocessing render functions
# ---------------------------------------------------------------------------

def _render_recording_summary(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    act = activity_series(raw)
    if act is None:
        return _error_card("render_recording_summary", "No activity channel in this recording.")

    n_epochs = len(act)
    start    = act.index[0]  if n_epochs > 0 else None
    end      = act.index[-1] if n_epochs > 0 else None

    duration_str = "unknown"
    if start is not None and end is not None:
        duration_days = (end - start).total_seconds() / 86400
        duration_str  = f"{duration_days:.1f} days"

    freq = getattr(raw, "frequency", None)
    if freq is None and act.index.freq is not None:
        freq = act.index.freq
    freq_str = str(freq) if freq is not None else "unknown"

    n_nan   = int(act.isna().sum())
    pct_nan = 100 * n_nan / n_epochs if n_epochs > 0 else 0.0

    n_nonwear   = 0
    pct_nonwear = 0.0
    try:
        mask = raw.mask
        if mask is not None:
            aligned     = mask.reindex(act.index)
            n_nonwear   = int((aligned == 0).sum())
            pct_nonwear = 100 * n_nonwear / n_epochs if n_epochs > 0 else 0.0
    except Exception:
        pass

    has_light = light_series(raw) is not None

    rows = [
        ("Start",          start.strftime("%Y-%m-%d %H:%M") if start else "n/a", ""),
        ("End",            end.strftime("%Y-%m-%d %H:%M")   if end   else "n/a", ""),
        ("Duration",       duration_str,                                          ""),
        ("Epoch length",   freq_str,                                              ""),
        ("Total epochs",   str(n_epochs),                                         ""),
        ("Missing (NaN)",  f"{n_nan}  ({pct_nan:.1f} %)",                         ""),
        ("Non-wear",       f"{n_nonwear}  ({pct_nonwear:.1f} %)",                 ""),
        ("Light channel",  "yes" if has_light else "no",                          ""),
    ]

    return ui.TagList(
        _section_label("Recording summary"),
        ui.HTML(_df_to_html(pd.DataFrame(rows, columns=["Property", "Value", ""]))),
    )


def _render_activity_trace(raw, args, rv_single, rv_batch, rv_mode) -> ui.TagList:
    import numpy as np
    import plotly.graph_objects as go

    show_nonwear = bool(args.get("show_nonwear", True))
    log_scale    = bool(args.get("log_scale", False))

    act = activity_series(raw)
    if act is None:
        return _error_card("render_activity_trace", "No activity channel in this recording.")

    # Downsample for display speed on long recordings
    if len(act) > 8000:
        step     = len(act) // 8000 + 1
        act_plot = act.iloc[::step]
    else:
        act_plot = act

    y = pd.to_numeric(act_plot, errors="coerce").to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=act_plot.index, y=y,
        mode="lines", line=dict(width=0.8, color="#1f77b4"),
        name="Activity",
    ))

    if show_nonwear:
        try:
            mask = raw.mask
            if mask is not None:
                aligned = mask.reindex(act_plot.index)
                is_nw   = (aligned == 0).values
                edges   = np.diff(is_nw.astype(int), prepend=0, append=0)
                for s_i, e_i in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                    fig.add_vrect(
                        x0=act_plot.index[s_i],
                        x1=act_plot.index[min(e_i, len(act_plot) - 1)],
                        fillcolor="rgba(214,39,40,0.2)",
                        line_width=0,
                    )
        except Exception:
            pass

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Activity (log)" if log_scale else "Activity",
        yaxis_type="log" if log_scale else "linear",
        margin=dict(l=50, r=20, t=30, b=40),
        height=260,
        showlegend=False,
    )

    resolved = {"show_nonwear": show_nonwear, "log_scale": log_scale}
    toolbar  = _rerun_toolbar("render_activity_trace", resolved, [
        {"type": "checkbox", "key": "show_nonwear", "label": "Show non-wear"},
        {"type": "checkbox", "key": "log_scale",    "label": "Log scale"},
    ])

    return ui.TagList(
        _section_label("Activity trace"),
        ui.HTML(_fig_to_html(fig)),
        ui.HTML(toolbar),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rerun_toolbar(tool_name: str, current_args: dict, controls: list[dict]) -> str:
    """Return a compact HTML control bar that re-runs the tool when any control changes.

    Relies on the event-delegation script injected by llm_chat._TOOLBAR_JS, which
    listens for 'change' events on .gen-toolbar divs and calls
    Shiny.setInputValue('chat-gen_rerun', ...).  No inline scripts needed here.
    """
    args_attr = json.dumps(current_args).replace("'", "&#39;")

    parts: list[str] = []
    for ctrl in controls:
        k   = ctrl["key"]
        lbl = ctrl["label"]
        val = current_args.get(k)

        if ctrl["type"] == "select":
            opts = "".join(
                f'<option value="{v}"{" selected" if v == val else ""}>{olbl}</option>'
                for v, olbl in ctrl["options"]
            )
            parts.append(
                f'<label style="font-size:11px;display:flex;align-items:center;gap:3px;">'
                f'{lbl}:&nbsp;<select data-key="{k}" '
                f'style="font-size:11px;padding:1px 4px;border:1px solid #ced4da;'
                f'border-radius:3px;background:#fff;">{opts}</select></label>'
            )
        elif ctrl["type"] == "checkbox":
            checked = " checked" if val else ""
            parts.append(
                f'<label style="font-size:11px;display:flex;align-items:center;'
                f'gap:3px;cursor:pointer;">'
                f'<input type="checkbox" data-key="{k}"{checked}>&nbsp;{lbl}</label>'
            )
        elif ctrl["type"] == "number":
            mn_a   = f' min="{ctrl["min"]}"'   if "min"  in ctrl else ""
            mx_a   = f' max="{ctrl["max"]}"'   if "max"  in ctrl else ""
            step_a = f' step="{ctrl["step"]}"' if "step" in ctrl else ""
            parts.append(
                f'<label style="font-size:11px;display:flex;align-items:center;gap:3px;">'
                f'{lbl}:&nbsp;<input type="number" data-key="{k}" value="{val}"'
                f'{mn_a}{mx_a}{step_a} style="font-size:11px;width:68px;padding:1px 4px;'
                f'border:1px solid #ced4da;border-radius:3px;"></label>'
            )
        elif ctrl["type"] == "text":
            parts.append(
                f'<label style="font-size:11px;display:flex;align-items:center;gap:3px;">'
                f'{lbl}:&nbsp;<input type="text" data-key="{k}" value="{val}" '
                f'style="font-size:11px;width:68px;padding:1px 4px;'
                f'border:1px solid #ced4da;border-radius:3px;"></label>'
            )

    controls_html = "\n  ".join(parts)
    uid = _uuid.uuid4().hex[:8]  # not strictly needed but helps debugging
    return (
        f'<div class="gen-toolbar" data-tool="{tool_name}" data-args=\'{args_attr}\' '
        f'id="tb-{uid}" '
        f'style="margin-top:10px;padding:6px 10px;background:#f8f9fa;border-radius:6px;'
        f'border:1px solid #e9ecef;display:flex;flex-wrap:wrap;gap:8px;align-items:center;">\n'
        f'  <span style="font-size:10px;color:#6c757d;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:.04em;">Re-run:&nbsp;</span>\n'
        f'  {controls_html}\n'
        f'</div>'
    )


def _get_raw(rv_single, rv_batch, rv_mode, rv_active_subject):
    mode = rv_mode()
    if mode == "batch":
        batch = rv_batch()
        if batch is None:
            raise ValueError("No batch loaded.")
        subject = rv_active_subject() if rv_active_subject else None
        if subject:
            for e in batch.entries:
                if e.subject_id == subject:
                    return e.raw
        return batch.entries[0].raw
    raw = rv_single()
    if raw is None:
        raise ValueError("No recording loaded.")
    return raw


def _fig_to_html(fig) -> str:
    """Convert a Plotly figure to a self-contained HTML snippet."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displayModeBar": False},
    )


def _df_to_html(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a styled HTML table."""
    return (
        '<div style="overflow-x:auto;margin-top:10px;">'
        + df.to_html(index=False, border=0,
                     classes="table table-sm table-striped",
                     justify="left")
        + "</div>"
    )


def _section_label(title: str) -> ui.Tag:
    return ui.p(
        title,
        style=(
            "font-size:.78rem;font-weight:600;color:#6c757d;"
            "text-transform:uppercase;letter-spacing:.05em;"
            "margin-bottom:6px;"
        ),
    )


def _error_card(tool_name: str, message: str) -> ui.TagList:
    return ui.TagList(
        ui.div(
            ui.HTML(f"<b>⚠ {tool_name}</b>: {message}"),
            style=(
                "background:#fff3cd;border:1px solid #ffc107;"
                "border-radius:8px;padding:10px 14px;font-size:.88rem;"
            ),
        )
    )


def _hhmm(value) -> str:
    """Convert a numeric hour value or Timedelta to HH:MM string."""
    try:
        if isinstance(value, pd.Timedelta):
            total = int(value.total_seconds())
        elif isinstance(value, pd.Timestamp):
            return value.strftime("%H:%M")
        else:
            total = int(float(value) * 3600)
        total %= 86400
        return f"{total // 3600:02d}:{(total % 3600) // 60:02d}"
    except Exception:
        return "n/a"
