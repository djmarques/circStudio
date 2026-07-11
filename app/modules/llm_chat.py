"""Tab — Assistant (guide/tutor mode).

The assistant explains circStudio analyses, interprets results, and guides
users through the UI and Python API.  It never reports specific numeric values
from the loaded recording (those are shown directly in each tab); instead it
helps users understand what the numbers mean and what to do next.

Navigation links
----------------
When the model includes a link of the form  [label](nav:Tab Name)  in its
response, the markdown renderer converts it to a button that switches the
Shiny navbar to that tab via  Shiny.setInputValue('main_nav', ...).

Providers supported
-------------------
- Ollama  (local,  http://localhost:11434/v1,  no key required)
- Groq    (cloud,  https://api.groq.com/openai/v1,  free API key)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import json

import markdown as md_pkg
from openai import OpenAI, OpenAIError
from shiny import module, reactive, render, ui

from modules.llm_generative import (  # noqa: E402
    GENERATIVE_TOOLS,
    GENERATIVE_PROMPT_SECTION,
    render_tool_call,
)

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = APP_DIR.parent / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

from utils import active_raw  # noqa: E402

# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------
PROVIDERS: dict[str, dict] = {
    "ollama": {
        "label":         "🦙  Ollama  (local, no key needed)",
        "base_url":      "http://localhost:11434/v1",
        "needs_key":     False,
        "default_model": "llama3.1",
        "model_choices": [],
        "description": (
            "Runs entirely on your computer. No account or internet required. "
            "Requires Ollama to be installed and a model pulled "
            "(e.g. <code>ollama pull llama3.1</code>)."
        ),
    },
    "groq": {
        "label":         "⚡  Groq  (free cloud, larger models)",
        "base_url":      "https://api.groq.com/openai/v1",
        "needs_key":     True,
        "default_model": "llama-3.3-70b-versatile",
        "model_choices": [
            "llama-3.3-70b-versatile",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "description": (
            "Free account at console.groq.com — no credit card required. "
            "Runs Llama 3.3 70B in the cloud; much more capable than local 8B models."
        ),
    },
}

MAX_TOKENS    = 2048
HISTORY_LIMIT = 16

# ---------------------------------------------------------------------------
# Tab catalogue (matches app.py nav_panel titles exactly)
# ---------------------------------------------------------------------------
TABS = {
    "Data Upload":       "Load .awd, .csv, or .txt actigraphy files (single or batch).",
    "Batch Overview":    "Inspect and compare multiple recordings side by side.",
    "Daily Profile":     "Average 24-h activity/light pattern + scalar NPCRA metrics.",
    "Sleep Scoring":     "Epoch-by-epoch sleep/wake classification and summary statistics.",
    "Cosinor":           "Parametric cosine fit — mesor, amplitude, acrophase, period, R².",
    "LIDS":              "LIDS transform + oscillation fit (night recordings only).",
    "Fractal / MFDFA":   "Multifractal scaling analysis — generalised Hurst exponents h(q).",
    "FLM":               "Functional linear model fit to the daily profile (RMSE).",
    "SSA":               "Singular Spectrum Analysis — trend, rhythm, and noise decomposition.",
    "Circadian Models":  "Mathematical clock models (Two-Process, van der Pol, etc.).",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert guide and tutor for **circStudio**, \
a Shiny-for-Python actigraphy analysis application built on pyActigraphy.

## Your role
- Explain what each analysis does and when to use it.
- Help users choose parameters and interpret their results.
- Show equivalent Python API usage when asked.
- Suggest which tab to visit next based on the user's goal.
- Answer conceptual questions about actigraphy, circadian rhythms, and sleep.

## What you do NOT do
- Report specific numeric values from the user's recording — those are shown \
directly in each tab.  If a user asks "what is my IS?", redirect them: \
"Your IS value is shown in the Daily Profile tab under NPCRA metrics \
— [open it here](nav:Daily Profile)."
- Diagnose medical conditions or make clinical recommendations.

## Navigation links
When directing the user to a tab, embed a link using EXACTLY this format:
  [Open Tab Name →](nav:Tab Name)
where Tab Name is one of the exact strings listed in the tab reference below. \
These render as clickable buttons that switch the app to that tab.

---

## circStudio tab reference

### Data Upload  [Open →](nav:Data Upload)
Load a single recording (.awd, .csv, .txt) or a whole directory for batch analysis. \
The app auto-detects the file format. Once loaded, all other tabs become active.

**Python API:**
```python
import pyActigraphy
raw = pyActigraphy.io.read_raw_awd("recording.awd")
print(raw.activity)      # pandas Series with DatetimeIndex
print(raw.frequency)     # sampling interval, e.g. Timedelta('1 min')
```

---

### Daily Profile  [Open →](nav:Daily Profile)
Computes the mean 24-h activity (or light) pattern across all days. Also computes \
a comprehensive set of scalar NPCRA metrics.

**Key metrics:**
| Metric | Meaning | Typical range (healthy adults) |
|--------|---------|-------------------------------|
| IS (Interdaily Stability) | Day-to-day rhythm consistency | 0.6 – 0.8 |
| IV (Intradaily Variability) | Within-day fragmentation | 0.5 – 0.8 |
| RA (Relative Amplitude) | Active/rest contrast: (M10−L5)/(M10+L5) | > 0.8 |
| M10 | Mean activity in the most active 10 h | — |
| L5  | Mean activity in the least active 5 h | — |
| AonT / AoffT | Activity onset / offset time | — |
| ADAT | Average daily activity total | — |
| kRA / kAR | Non-parametric rhythm indices | — |

A **high IS** means the rhythm is very consistent day-to-day (strong social entrainment). \
A **high IV** means bursts of activity are interspersed with rest throughout the day \
(fragmented rhythm — common in older adults, dementia, shift workers). \
A **low RA** means the difference between active and rest periods is small (weaker rhythm).

**Python API:**
```python
from circstudio.analysis.metrics.metrics import IS, IV, l5, m10, ra, daily_profile
print(IS(raw.activity))
onset_l5, val_l5 = l5(raw.activity)
profile = daily_profile(raw.activity)   # Series with 1-cycle mean profile
```

---

### Sleep Scoring  [Open →](nav:Sleep Scoring)
Classifies each epoch as sleep (1) or wake (0) using validated actigraphy algorithms.

**Algorithms available:**
- **Cole-Kripke** (1992) — most widely validated for wrist actigraphy; requires a \
`settings` argument matching the epoch length (use "mean" for 1-min data).
- **Roenneberg** — suitable for long recordings; used internally by SRI and SoD.
- **Sadeh** — validated in adolescents.
- **Scripps** — alternative weighted regression method.
- **Oakley** — threshold-based method.

**Key output metrics:**
| Metric | Meaning | Normal range |
|--------|---------|-------------|
| TST | Total Sleep Time per night (h) | 7 – 9 h |
| SE  | Sleep Efficiency = TST / time-in-bed × 100 % | > 85 % |
| WASO | Wake After Sleep Onset (min) | < 30 min |
| SRI | Sleep Regularity Index (−100 → +100) | > 85 = regular |
| SoD | Sleep onset distribution (clock hour) | — |
| fSoD | Functional SoD — first sustained sleep | — |

**Choosing an algorithm:** Cole-Kripke is the default for adults with 1-min epoch data. \
Use Roenneberg for longer recordings or when SRI / SoD are the main outcomes.

**Python API:**
```python
from circstudio.analysis.sleep.sleep import (
    Cole_Kripke, Roenneberg, SleepRegularityIndex,
    SoD, waso, main_sleep_bouts,
)
sleep_wake = Cole_Kripke(raw.activity, settings="mean", threshold=1.0, rescoring=True)
sri = SleepRegularityIndex(raw.activity, algo="Roenneberg")
waso_series, waso_mean = waso(raw.activity, frequency=raw.frequency)
```

---

### Cosinor  [Open →](nav:Cosinor)
Fits a cosine function to the full activity time series to characterise the dominant \
24-h rhythm with four parameters.

**Parameters to set:**
- **Period** (default 1440 min = 24 h) — change only if you suspect a non-24 h rhythm.
- **Fitting method** — `leastsq` is fastest; `differential_evolution` is more robust \
for noisy or irregular data.
- **Initial guesses** for mesor, amplitude, acrophase — defaults work for most recordings; \
adjust if the fit fails.

**Output interpretation:**
| Parameter | Meaning | Typical value |
|-----------|---------|---------------|
| Mesor | Rhythm-adjusted mean activity | ≈ mean(activity) |
| Amplitude | Half the peak-to-trough range | proxy for rhythm strength |
| Acrophase | Clock time of the fitted peak (h) | 14:00 – 16:00 in healthy adults |
| Period | Fitted period (h) | ≈ 24 h |
| R² | Goodness of fit | > 0.5 = strong 24h component; < 0.3 = weak / irregular |

**Python API:**
```python
from circstudio.analysis import Cosinor
from lmfit import Parameters
p = Parameters()
p.add("Amplitude", value=50, min=0)
p.add("Acrophase", value=3.14, min=0, max=6.28)
p.add("Period",    value=1440, min=0)
p.add("Mesor",     value=float(raw.activity.mean()), min=0)
model = Cosinor()
result = model.fit(raw.activity, params=p, method="leastsq")
print(result.params)
```

---

### LIDS  [Open →](nav:LIDS)
⚠️ **Designed for night recordings only.** LIDS (LIDS = actigraphic inactivity → \
oscillation) applies a nonlinear transform to reveal ultradian oscillations within \
a single sleep episode. Do not use on full multi-day recordings.

**MRI (LIDS Rhythm Index):** higher values indicate stronger within-sleep oscillations \
(related to sleep architecture regularity).

**Python API:**
```python
from circstudio.analysis import LIDS
obj = LIDS()
lids_ts = obj.lids_transform(night_activity)   # pass a night-only slice
result  = obj.lids_fit(lids_ts, fit_func="cosine")
mri     = obj.lids_mri(result)
```

---

### Fractal / MFDFA  [Open →](nav:Fractal / MFDFA)
Multifractal Detrended Fluctuation Analysis quantifies how the statistical properties \
of the signal scale across time windows of different sizes.

**Parameters:**
- **q range** (default −3 to +3) — statistical moments to probe. Negative q emphasises \
small fluctuations; positive q emphasises large ones.
- **n min / n max** — scale range in epochs. nmax ≤ ~1/4 of recording length is a good rule.
- **Detrending order** — 1 (linear) is standard; increase for non-stationary data.

**Output — generalised Hurst exponent h(q):**
| h(2) value | Interpretation |
|-----------|----------------|
| ≈ 0.5 | Random walk (uncorrelated) |
| 0.8 – 1.0 | Healthy long-range correlations |
| > 1.0 | Non-stationary signal |

A **flat h(q) curve** across all q values indicates monofractality. \
**Variation in h(q)** (wider spread) indicates multifractality, reflecting more \
complex, heterogeneous scaling behaviour.

**Python API:**
```python
import numpy as np
from circstudio.analysis import Fractal

q_array = np.linspace(-3, 3, 7)
q_array = np.where(np.abs(q_array) < 1e-6, 1e-6, q_array)   # avoid q=0
n_array = np.unique(np.logspace(np.log10(16), np.log10(1440), 30).astype(int))
n_array = n_array[n_array >= 4]

Fm = Fractal.mfdfa(raw.activity, n_array, q_array, deg=1)
h_values = [Fractal.generalized_hurst_exponent(Fm[:, j], n_array) for j in range(len(q_array))]
# Each element of h_values is a (h, std_err) tuple
```

---

### FLM  [Open →](nav:FLM)
Functional Linear Modelling represents the daily activity profile as a smooth function \
built from basis functions (Fourier harmonics or spline knots).

**Parameters:**
- **Basis** — `fourier` is best for rhythmic data; `spline` for more flexible shapes.
- **Max order** — number of harmonics (Fourier) or knots (spline). \
Higher = finer detail but risk of overfitting. 10 is a good default.
- **Smooth only** — applies smoothing to the raw time series instead of fitting the daily profile.

**RMSE** (Root Mean Squared Error): lower = better fit of the basis to your daily profile. \
Compare across recordings or conditions to assess rhythm regularity.

**Python API:**
```python
from circstudio.analysis import FLM, daily_profile
import numpy as np

freq = raw.activity.index.freq
flm = FLM(basis="fourier", sampling_freq=freq, max_order=10)
flm.fit(raw.activity)
fitted  = np.asarray(flm.evaluate())
profile = np.asarray(daily_profile(raw.activity).values, dtype=float)
n    = min(len(profile), len(fitted))
rmse = float(np.sqrt(np.nanmean((profile[:n] - fitted[:n]) ** 2)))
```

---

### SSA  [Open →](nav:SSA)
Singular Spectrum Analysis decomposes the activity series into additive components \
— trend, oscillatory patterns, and noise — without assuming a fixed parametric model.

**Parameters:**
- **Window length** (default "24h") — the embedding dimension. Set to the expected \
cycle length. Larger windows resolve lower-frequency components.
- **Resample frequency** (default "10min") — resampling for speed; does not affect \
interpretation if the signal is smooth.
- **Number of components** — inspect the variance-explained bar chart to decide how many \
components capture meaningful structure vs. noise.

**Interpreting results:**
- Components with high variance explained and clear periodicity (visible in the reconstructed \
signal) represent real rhythm structure.
- The **W-correlation matrix** shows which components are related; components with high \
mutual correlation can be grouped for reconstruction.

**Python API:**
```python
from circstudio.analysis import SSA

s = raw.activity.resample("10min").mean().dropna().interpolate()
ssa = SSA(s, window_length="24h")
ssa.fit()
print(ssa.variance_explained[:10])          # top-10 components
reconstructed = ssa.reconstruct_signal([0, 1])   # combine first two components
```

---

### Circadian Models  [Open →](nav:Circadian Models)
Mathematical models of the circadian clock fitted to the activity data. Options include \
the Two-Process Model (Borbély), van der Pol oscillator, and others. Each model has \
specific parameters that require tuning to the individual recording.

---

## Example questions I can help with

- *"What is IS and how do I improve it?"*
- *"My R² from Cosinor is 0.2 — is that a problem?"*
- *"What's the difference between SSA and FLM?"*
- *"When should I use MFDFA instead of Cosinor?"*
- *"How do I run sleep scoring in Python?"*
- *"My SE is 60% — what does that suggest?"*
- *"What window length should I use in SSA?"*
- *"Can I run LIDS on a full week recording?"*
- *"How do I interpret the singularity spectrum from MFDFA?"*

---

## Style rules
- Be concise. Use markdown tables and code blocks where they help.
- When directing the user to a tab, always include a `(nav:...)` link.
- If a user asks for a specific value from their recording, tell them which tab shows it \
and include a nav link — do not try to state the value yourself.
- For Python examples, show working code using the `circstudio.analysis` imports.
"""


def _build_system_prompt(session_context: str, generative: bool = False) -> str:
    base = _SYSTEM_PROMPT
    if generative:
        base += GENERATIVE_PROMPT_SECTION
    return base + f"\n## Current session\n{session_context}\n"


def _session_context(rv_single, rv_batch, rv_mode) -> str:
    mode  = rv_mode()
    lines = [f"**Mode**: {mode}"]
    if mode == "single":
        from modules._common import activity_series, light_series  # local to avoid circular
        raw = rv_single()
        if raw is None:
            lines.append("**Recording**: No file loaded yet.")
        else:
            try:
                act   = getattr(raw, "activity", None)
                freq  = getattr(raw, "frequency", "unknown")
                start = str(getattr(raw, "start_time", "unknown"))
                stop  = str(getattr(raw, "stop_time",  "unknown"))
                n     = int(len(act)) if act is not None else "unknown"
                has_l = getattr(raw, "light", None) is not None
                lines += [
                    "**Recording**: single file loaded",
                    f"**Start**: {start}  |  **End**: {stop}",
                    f"**Epochs**: {n}  |  **Frequency**: {freq}",
                    f"**Light channel**: {'yes' if has_l else 'no'}",
                ]
            except Exception:
                lines.append("**Recording**: loaded (metadata unavailable)")
    else:
        batch = rv_batch()
        if batch is None:
            lines.append("**Batch**: No batch loaded yet.")
        else:
            try:
                lines.append(f"**Batch**: {len(batch.entries)} recordings loaded")
            except Exception:
                lines.append("**Batch**: loaded")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_client(provider: str, api_key: str = "") -> OpenAI:
    return OpenAI(base_url=PROVIDERS[provider]["base_url"], api_key=api_key or "local")


def _list_ollama_models() -> list[str]:
    try:
        return sorted(m.id for m in _make_client("ollama").models.list().data)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Markdown + nav-link rendering
# ---------------------------------------------------------------------------

_NAV_LINK_RE = re.compile(r'href="nav:([^"]+)"')

# Matches text-format tool calls emitted by models that don't use structured
# tool_calls (e.g. llama-3.3-70b-versatile on Groq).
# Handles both <function(name){...}> and <function(name){...} (no closing >).
_TEXT_TOOL_RE = re.compile(
    r"<function\((\w+)\)\s*(\{.*?\})\s*>?",
    re.DOTALL,
)

# Maps each tool's parameter names → tool name.  Used by _parse_text_tool_calls
# to identify which tool a schema-contaminated JSON object refers to.
_TOOL_PARAM_SETS: dict[str, frozenset] = {
    "render_daily_profile":     frozenset(["signal", "cyclic", "log_scale"]),
    "render_sleep_scoring":     frozenset(["algorithm", "show_hypnogram", "show_summary"]),
    "render_cosinor":           frozenset(["period_h", "method"]),
    "render_lids":              frozenset(["fit_func"]),
    "render_fractal":           frozenset(["q_min", "q_max", "q_steps", "detrending_order"]),
    "render_flm":               frozenset(["basis", "max_order"]),
    "render_ssa":               frozenset(["window", "n_components", "resample"]),
    "render_batch_comparison":  frozenset(["metric", "plot_type"]),
    "render_recording_summary": frozenset([]),
    "render_activity_trace":    frozenset(["show_nonwear", "log_scale"]),
}

# Groq model that reliably supports structured tool calling.
_GROQ_TOOL_MODEL = "llama3-groq-70b-8192-tool-use-preview"

# Injected once into the chat panel — defines window.csNavTo used by all buttons.
_NAV_JS = """
<script>
window.csNavTo = function(tabName) {
    // Shiny for Python / bslib page_navbar: nav links carry data-value attributes.
    var sel = '.navbar-nav [data-value="' + tabName + '"]';
    var el  = document.querySelector(sel);
    // Fallback: any element with a matching data-value (handles different DOM shapes)
    if (!el) el = document.querySelector('[data-value="' + tabName + '"]');
    // Last resort: find a nav-link whose visible text matches
    if (!el) {
        document.querySelectorAll('.navbar-nav a, .nav-link').forEach(function(a) {
            if (a.textContent.trim() === tabName) el = a;
        });
    }
    if (el) el.click();
};
</script>
"""


def _md(text: str) -> str:
    """Render markdown and convert nav: links to tab-switch buttons."""
    try:
        html = md_pkg.markdown(
            text, extensions=["tables", "fenced_code", "nl2br"]
        )
    except Exception:
        html = text.replace("\n", "<br>")

    # Replace href="nav:Tab Name" with an onclick that calls csNavTo
    def _nav_sub(m: re.Match) -> str:
        tab  = m.group(1)
        safe = tab.replace("'", "\\'").replace('"', "&quot;")
        return (
            f'onclick="window.csNavTo(\'{safe}\')" '
            f'style="background:#0d6efd;color:#fff;border:none;'
            f'padding:3px 10px;border-radius:4px;cursor:pointer;font-size:.85rem;"'
        )

    html = _NAV_LINK_RE.sub(_nav_sub, html)
    # Promote <a onclick=...> to <button onclick=...>
    html = re.sub(
        r'<a ([^>]*onclick[^>]*)>(.*?)</a>',
        r'<button \1>\2</button>',
        html,
        flags=re.DOTALL,
    )
    return html


def _parse_text_tool_calls(text: str) -> tuple[list, str]:
    """Fallback: extract tool calls from plain text.

    Handles three formats emitted by models that bypass the structured
    tool_calls field:

    1. ``<function(name){...}>``  — format used by some Ollama models.
    2. ``{"name": "...", "arguments": {...}}``  — OpenAI-like JSON in text.
    3. ``{"type": "object", "param": val, ...}``  — schema-contaminated JSON
       emitted by llama-3.3-70b-versatile on Groq; the tool is inferred from
       the parameter names via _TOOL_PARAM_SETS.

    Returns the call list and the content with function-call strings stripped.
    """
    import types

    def _make_call(name: str, args: dict):
        fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
        return types.SimpleNamespace(function=fn)

    # ── Format 1: <function(name){...}> ──────────────────────────────────
    matches = _TEXT_TOOL_RE.findall(text)
    if matches:
        calls = []
        for name, args_str in matches:
            try:
                args = json.loads(args_str)
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}
            calls.append(_make_call(name, args))
        return calls, _TEXT_TOOL_RE.sub("", text).strip()

    # ── Format 2: {"name": "...", "arguments": {...}} ─────────────────────
    _json_call_re = re.compile(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    )
    matches2 = _json_call_re.findall(text)
    if matches2:
        calls = []
        for name, args_str in matches2:
            try:
                args = json.loads(args_str)
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}
            calls.append(_make_call(name, args))
        return calls, _json_call_re.sub("", text).strip()

    # ── Format 3: schema-contaminated bare JSON ───────────────────────────
    # llama-3.3-70b-versatile on Groq sometimes outputs the tool schema's
    # top-level {"type": "object", ...} with actual argument values mixed in.
    # We try to parse the entire text as JSON and infer the tool from params.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            blob = json.loads(stripped)
            if isinstance(blob, dict):
                # Strip schema artefacts; keep only actual argument keys
                args = {k: v for k, v in blob.items()
                        if k not in ("type", "properties", "required", "description")}
                arg_keys = frozenset(args.keys())
                # Match against known parameter sets (best / most-overlap wins)
                best_tool, best_score = None, 0
                for tool_name, param_set in _TOOL_PARAM_SETS.items():
                    score = len(arg_keys & param_set)
                    if score > best_score:
                        best_tool, best_score = tool_name, score
                if best_tool and best_score > 0:
                    return [_make_call(best_tool, args)], ""
        except Exception:
            pass

    return [], text


def _build_api_messages(system: str, history: list[dict]) -> list[dict]:
    # Strip tool_result entries — they carry a Shiny widget and must not be
    # sent to the model.  Only user and assistant text messages are included.
    api_msgs = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ]
    return [{"role": "system", "content": system}] + api_msgs


# ---------------------------------------------------------------------------
# Styles & JS
# ---------------------------------------------------------------------------

_CARD = (
    "max-width:600px;margin:50px auto;padding:32px 36px;"
    "border:1px solid #dee2e6;border-radius:12px;"
    "background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.08);"
)
_CHAT_BOX = (
    "height:480px;overflow-y:auto;border:1px solid #dee2e6;"
    "border-radius:8px;padding:16px;background:#f8f9fa;"
    "display:flex;flex-direction:column;gap:10px;"
)
_USER = (
    "align-self:flex-end;background:#0d6efd;color:#fff;"
    "padding:10px 14px;border-radius:18px 18px 4px 18px;"
    "max-width:75%;font-size:.93rem;white-space:pre-wrap;word-break:break-word;"
)
_BOT = (
    "align-self:flex-start;background:#fff;border:1px solid #dee2e6;"
    "padding:10px 14px;border-radius:18px 18px 18px 4px;"
    "max-width:85%;font-size:.93rem;word-break:break-word;"
)
_THINK = (
    "align-self:flex-start;background:#fff;border:1px solid #dee2e6;"
    "padding:10px 14px;border-radius:18px 18px 18px 4px;"
    "display:flex;align-items:center;gap:4px;"
)
_COMPONENT_BUBBLE = (
    "width:100%;background:#fff;border:1px solid #dee2e6;"
    "border-radius:12px;padding:14px;overflow:hidden;box-sizing:border-box;"
)

_TYPING_CSS = """
<style>
@keyframes cs-bounce{0%,80%,100%{transform:translateY(0);opacity:.4}
40%{transform:translateY(-6px);opacity:1}}
.cs-dot{width:8px;height:8px;border-radius:50%;background:#6c757d;
display:inline-block;animation:cs-bounce 1.2s infinite ease-in-out}
.cs-dot:nth-child(2){animation-delay:.2s}
.cs-dot:nth-child(3){animation-delay:.4s}
</style>
"""

_ENTER_JS = """
(function(){
document.addEventListener('keydown',function(e){
  var ta=document.getElementById('chat-user_input');
  if(!ta||e.target!==ta)return;
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();
    var btn=document.getElementById('chat-send_btn');
    if(btn&&!btn.disabled)btn.click();
  }
});
})();
"""
_SCROLL_JS = """
(function(){
  var _o=new MutationObserver(function(){
    var el=document.getElementById('chat-thread-wrapper');
    if(!el)return;
    // Only auto-scroll when already near the bottom (< 120px away).
    // This prevents Plotly hover-tooltip DOM mutations from jumping the
    // view when the user is inspecting a plot mid-conversation.
    var distFromBottom=el.scrollHeight-el.scrollTop-el.clientHeight;
    if(distFromBottom<120)el.scrollTop=el.scrollHeight;
  });
  _o.observe(document.body,{childList:true,subtree:true});
})();
"""

# Event-delegation script for generative UI re-run toolbars.
# Attached to chat-thread-wrapper once so it covers all past and future toolbars.
# Fires Shiny.setInputValue('chat-gen_rerun', ...) whenever a toolbar control changes.
_TOOLBAR_JS = """
<script>
(function(){
  function setup(wrapper){
    wrapper.addEventListener('change',function(e){
      var t=e.target;
      var tb=t.closest('.gen-toolbar');
      if(!tb)return;
      var tool=tb.getAttribute('data-tool');
      var args;
      try{args=JSON.parse(tb.getAttribute('data-args'));}catch(err){args={};}
      var key=t.getAttribute('data-key');
      if(!key)return;
      var val;
      if(t.type==='checkbox')val=t.checked;
      else if(t.type==='number')val=Number(t.value);
      else val=t.value;
      args=Object.assign({},args);
      args[key]=val;
      tb.setAttribute('data-args',JSON.stringify(args));
      if(window.Shiny){
        Shiny.setInputValue('chat-gen_rerun',{tool:tool,args:args,nonce:Math.random()},{priority:'event'});
      }
    });
  }
  function trySetup(){
    var w=document.getElementById('chat-thread-wrapper');
    if(w){setup(w);return true;}
    return false;
  }
  if(!trySetup()){
    var obs=new MutationObserver(function(){if(trySetup())obs.disconnect();});
    obs.observe(document.body,{childList:true,subtree:true});
  }
})();
</script>
"""

# ---------------------------------------------------------------------------
# Shiny module
# ---------------------------------------------------------------------------

@module.ui
def llm_chat_ui():
    return ui.div(
        ui.tags.script(_ENTER_JS),
        ui.output_ui("main_panel"),
    )


@module.server
def llm_chat_server(
    input, output, session,
    rv_single, rv_batch, rv_mode, rv_active_subject,
):
    # ── Session state ─────────────────────────────────────────────────────
    rv_provider: reactive.Value[Optional[str]] = reactive.Value(None)
    rv_api_key:  reactive.Value[str]            = reactive.Value("")
    rv_model:    reactive.Value[Optional[str]]  = reactive.Value(None)
    rv_avail:    reactive.Value[list[str]]       = reactive.Value([])
    rv_history:  reactive.Value[list[dict]]      = reactive.Value([])
    rv_thinking: reactive.Value[bool]            = reactive.Value(False)

    # ── Top-level panel switcher ──────────────────────────────────────────
    @render.ui
    def main_panel():
        if rv_model() is None:
            return _setup_card()
        return _chat_panel()

    # ── Setup card ────────────────────────────────────────────────────────
    def _setup_card():
        provider = rv_provider() or "ollama"
        return ui.div(
            ui.div(
                ui.h4("✦  Assistant", style="margin-bottom:18px;"),
                ui.input_radio_buttons(
                    "provider_choice", "Choose your AI backend:",
                    choices={k: v["label"] for k, v in PROVIDERS.items()},
                    selected=provider, inline=False,
                ),
                ui.output_ui("provider_section"),
                ui.div(
                    ui.input_action_button(
                        "connect_btn", "Connect",
                        class_="btn-primary", style="min-width:110px;",
                    ),
                    style="margin-top:16px;",
                ),
                ui.p(
                    ui.tags.em(
                        "ⓘ  API keys are stored only in this session and "
                        "never written to disk."
                    ),
                    style="margin-top:14px;color:#6c757d;font-size:.85rem;",
                ),
                style=_CARD,
            )
        )

    @render.ui
    def provider_section():
        chosen = (
            input.provider_choice()
            if hasattr(input, "provider_choice") else None
        ) or "ollama"
        rv_provider.set(chosen)
        avail = rv_avail() if chosen == "ollama" else []

        if chosen == "ollama":
            model_widget = (
                ui.input_select("model_select", "Select model",
                                choices=avail, selected=avail[0])
                if avail else
                ui.input_text("model_name", "Model name",
                              value=PROVIDERS["ollama"]["default_model"],
                              placeholder="e.g. llama3.1, qwen2.5")
            )
            return ui.div(
                ui.p(ui.HTML(PROVIDERS["ollama"]["description"])),
                ui.tags.ol(
                    ui.tags.li(["Download Ollama from ",
                                ui.tags.a("ollama.com",
                                          href="https://ollama.com",
                                          target="_blank")]),
                    ui.tags.li([ui.tags.code("ollama pull llama3.1"),
                                " in a terminal"]),
                    ui.tags.li("Start Ollama, then click Connect."),
                ),
                ui.div(model_widget, style="margin-top:12px;"),
            )
        else:
            return ui.div(
                ui.p(ui.HTML(PROVIDERS["groq"]["description"])),
                ui.tags.ol(
                    ui.tags.li(["Sign up at ",
                                ui.tags.a("console.groq.com",
                                          href="https://console.groq.com",
                                          target="_blank"),
                                " (free, no credit card)."]),
                    ui.tags.li('Create an API key under "API Keys".'),
                    ui.tags.li("Paste it below and click Connect."),
                ),
                ui.input_password("groq_key", "Groq API key",
                                  placeholder="gsk_...", width="100%"),
                ui.input_select("groq_model", "Model",
                                choices=PROVIDERS["groq"]["model_choices"],
                                selected=PROVIDERS["groq"]["default_model"]),
            )

    @reactive.effect
    @reactive.event(input.connect_btn)
    def _on_connect():
        provider = rv_provider() or "ollama"
        if provider == "ollama":
            avail = _list_ollama_models()
            if not avail:
                ui.notification_show(
                    "Cannot reach Ollama at localhost:11434. "
                    "Make sure Ollama is installed and running.",
                    type="error", duration=8,
                )
                return
            try:
                chosen = input.model_select()
            except Exception:
                chosen = None
            if not chosen:
                try:
                    chosen = (input.model_name() or "").strip()
                except Exception:
                    chosen = ""
            if not chosen:
                chosen = avail[0]
            if chosen not in avail:
                matches = [
                    m for m in avail
                    if m.startswith(chosen) or m.split(":")[0] == chosen
                ]
                if matches:
                    chosen = matches[0]
                else:
                    ui.notification_show(
                        f"Model '{chosen}' not found. Available: {', '.join(avail)}",
                        type="error", duration=8,
                    )
                    return
            rv_avail.set(avail)
            rv_api_key.set("")
            rv_model.set(chosen)
        else:
            key = (input.groq_key() or "").strip()
            if not key:
                ui.notification_show("Please enter your Groq API key.",
                                     type="warning")
                return
            model = input.groq_model() or PROVIDERS["groq"]["default_model"]
            try:
                _make_client("groq", key).models.list()
            except Exception as e:
                ui.notification_show(f"Could not connect to Groq: {e}",
                                     type="error", duration=8)
                return
            rv_api_key.set(key)
            rv_model.set(model)

    # ── Chat panel ────────────────────────────────────────────────────────
    def _chat_panel():
        provider   = rv_provider() or "ollama"
        prov_label = PROVIDERS[provider]["label"]
        return ui.layout_sidebar(
            ui.sidebar(
                ui.h6("Session", style="font-weight:600;"),
                ui.output_ui("session_card"),
                ui.hr(),
                ui.tags.div(
                    ui.tags.small("Provider: ", style="color:#6c757d;"),
                    ui.tags.small(prov_label, style="color:#6c757d;"),
                    ui.tags.br(),
                    ui.tags.small("Model: ", style="color:#6c757d;"),
                    ui.tags.code(rv_model() or "", style="font-size:.8rem;"),
                ),
                ui.hr(),
                ui.input_action_button(
                    "clear_btn", "Clear conversation",
                    class_="btn-outline-secondary btn-sm w-100",
                ),
                ui.hr(),
                ui.input_switch("generative_mode", "✨ Generative UI", value=False),
                ui.tags.small(
                    "Render plots and metrics directly in chat.",
                    style="color:#6c757d;display:block;margin-top:-6px;",
                ),
                ui.hr(),
                ui.tags.small(
                    ui.tags.a(
                        "Change provider / model", href="#",
                        id="change_model_link", style="color:#6c757d;",
                    ),
                ),
                ui.tags.script(
                    "document.getElementById('change_model_link')"
                    "?.addEventListener('click',function(e){"
                    "e.preventDefault();"
                    "Shiny.setInputValue('chat-reset_model',Math.random());"
                    "});"
                ),
                width=270,
            ),
            ui.HTML(_NAV_JS),
            ui.HTML(_TOOLBAR_JS),
            ui.div(
                ui.output_ui("chat_thread"),
                id="chat-thread-wrapper",
                style=_CHAT_BOX,
            ),
            ui.tags.script(_SCROLL_JS),
            ui.div(
                ui.div(
                    ui.input_text_area(
                        "user_input", label=None, rows=2, resize="none",
                        placeholder=(
                            "Ask about any analysis… "
                            "(Enter to send, Shift+Enter for new line)"
                        ),
                        width="100%",
                    ),
                    style="flex:1;",
                ),
                ui.div(
                    ui.input_action_button(
                        "send_btn", "Send ↵",
                        class_="btn-primary btn-sm",
                        style="min-width:72px;",
                    ),
                    style="padding-left:8px;display:flex;align-items:flex-end;",
                ),
                style="display:flex;gap:0;margin-top:12px;align-items:flex-end;",
            ),
        )

    # ── Session card ──────────────────────────────────────────────────────
    @render.ui
    def session_card():
        ctx   = _session_context(rv_single, rv_batch, rv_mode)
        items = [
            ui.tags.div(l, style="font-size:.83rem;")
            for l in ctx.split("\n") if l.strip()
        ]
        return ui.div(
            *items,
            style="background:#f8f9fa;padding:10px;border-radius:6px;",
        )

    # ── Chat thread ───────────────────────────────────────────────────────
    @render.ui
    def chat_thread():
        history  = rv_history()
        thinking = rv_thinking()
        bubbles  = []

        for msg in history:
            role    = msg.get("role")
            content = msg.get("content") or ""
            if role == "user" and isinstance(content, str):
                bubbles.append(ui.div(content, style=_USER))
            elif role == "assistant" and isinstance(content, str) and content:
                bubbles.append(ui.div(ui.HTML(_md(content)), style=_BOT))
            elif role == "tool_result":
                bubbles.append(ui.div(msg["widget"], style=_COMPONENT_BUBBLE))

        if thinking:
            bubbles.append(
                ui.div(
                    ui.HTML(_TYPING_CSS),
                    ui.tags.span(class_="cs-dot"),
                    ui.tags.span(class_="cs-dot"),
                    ui.tags.span(class_="cs-dot"),
                    style=_THINK,
                )
            )

        if not bubbles:
            bubbles.append(
                ui.div(
                    ui.HTML(
                        "👋 <b>Hello!</b> I'm your circStudio guide. Ask me anything — "
                        "what an analysis does, how to interpret a result, which tab to "
                        "visit, or how to replicate something in Python."
                    ),
                    style=_BOT,
                )
            )

        return ui.div(
            *bubbles,
            style="display:flex;flex-direction:column;gap:10px;",
        )

    # ── Controls ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.reset_model)
    def _on_reset():
        rv_provider.set(None); rv_model.set(None)
        rv_api_key.set(""); rv_history.set([]); rv_avail.set([])

    @reactive.effect
    @reactive.event(input.clear_btn)
    def _on_clear():
        rv_history.set([])

    # ── Toolbar re-run (generative UI inline controls) ────────────────────
    @reactive.effect
    @reactive.event(input.gen_rerun)
    def _on_gen_rerun():
        data = input.gen_rerun()
        if not data:
            return
        tool_name = data.get("tool")
        args      = data.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if not tool_name:
            return
        widget = render_tool_call(
            tool_name, args, rv_single, rv_batch, rv_mode, rv_active_subject,
        )
        history = list(rv_history())
        history.append({
            "role":      "tool_result",
            "content":   "",
            "widget":    widget,
            "tool_name": tool_name,
        })
        rv_history.set(history[-HISTORY_LIMIT:])

    # ── Send handler ──────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.send_btn)
    def _on_send():
        user_text = (input.user_input() or "").strip()
        model     = rv_model()
        provider  = rv_provider() or "ollama"
        if not user_text or model is None:
            return

        history = list(rv_history())
        history.append({"role": "user", "content": user_text})
        rv_history.set(history[-HISTORY_LIMIT:])
        rv_thinking.set(True)

        generative = bool(input.generative_mode())

        # llama-3.3-70b-versatile doesn't reliably use structured tool_calls.
        # Switch to the dedicated tool-use model when generative mode is on.
        effective_model = model
        if generative and provider == "groq" and model != _GROQ_TOOL_MODEL:
            effective_model = _GROQ_TOOL_MODEL

        system = _build_system_prompt(
            _session_context(rv_single, rv_batch, rv_mode),
            generative=generative,
        )
        client = _make_client(provider, rv_api_key())

        try:
            call_kwargs: dict = dict(
                model=effective_model,
                max_tokens=MAX_TOKENS,
                messages=_build_api_messages(system, rv_history()),
            )
            if generative:
                call_kwargs["tools"]       = GENERATIVE_TOOLS
                call_kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**call_kwargs)
        except OpenAIError as e:
            ui.notification_show(f"API error: {e}", type="error")
            rv_thinking.set(False)
            return
        except Exception as e:
            ui.notification_show(f"Unexpected error: {e}", type="error")
            rv_thinking.set(False)
            return

        msg        = response.choices[0].message
        text       = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Fallback: some models output tool calls as text rather than structured
        # tool_calls.  Parse and strip them from the displayed text.
        if not tool_calls and generative and text:
            tool_calls, text = _parse_text_tool_calls(text)

        history = list(rv_history())
        if text:
            history.append({"role": "assistant", "content": text})
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                if not isinstance(args, dict):   # guard against json `null`
                    args = {}
            except Exception:
                args = {}
            widget = render_tool_call(
                tc.function.name, args, rv_single, rv_batch, rv_mode,
                rv_active_subject,
            )
            history.append({
                "role":      "tool_result",
                "content":   "",        # not sent to the model
                "widget":    widget,
                "tool_name": tc.function.name,
            })

        rv_history.set(history[-HISTORY_LIMIT:])
        rv_thinking.set(False)
        ui.update_text_area("user_input", value="")
