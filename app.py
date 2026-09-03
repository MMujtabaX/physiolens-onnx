"""
================================================================================
PhysioLens — Real-Time Exercise Form Correction (ONNX, TensorFlow-free demo)
================================================================================

Streamlit demo of the trained Side Leg Raise fault-detection model. Runs on
`onnxruntime` — NO TensorFlow, NO MediaPipe, NO OpenCV. Pose extraction
happens offline/locally (the recorder that produced the bundled JSONL
files); this app only consumes pre-extracted pose-landmark JSONL (schema v4)
and runs the model + rule-engine verdicts already recorded in it.

This is the same app as physiolens_app/app.py, with one change: the model
forward pass runs through onnx_classifier.py (onnxruntime) instead of
legraise_classifier (TensorFlow/Keras) — see convert_to_onnx.py's verified
max-difference report for numeric parity between the two. Dropping
TensorFlow entirely is what lets this deploy on hosts that don't have a
compatible TF wheel for their pinned Python version.

Run locally:
    streamlit run app.py
================================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from onnx_classifier import LegRaiseClassifier, aggregate_session  # noqa: E402

SAMPLES_DIR = APP_DIR / "samples"

# label -> (button text, sample filename)
SAMPLES = {
    "good": ("Analyze a good set", "good_reps.jsonl"),
    "knee_bent": ("Analyze a knee-bent set", "knee_bent_reps.jsonl"),
    "shallow": ("Analyze a shallow set", "shallow_reps.jsonl"),
}

# Minimal human-readable names for the rule-engine's raw issue codes — kept
# local rather than importing session_utils.py, so this app stays
# self-contained (no dependency outside onnx_classifier.py / feature_extract.py).
ISSUE_LABELS = {
    "KNEE_BENT": "Knee bent",
    "SHALLOW": "Shallow raise",
}

st.set_page_config(page_title="PhysioLens", page_icon="🦵", layout="wide")

# Colors used both in the CSS below and in the quality-cell/chart coloring
# further down -- kept as one set of constants so the palette stays
# consistent everywhere it's referenced.
COLOR_PRIMARY = "#2563EB"   # blue — brand / quality line
COLOR_ACCENT = "#F59E0B"    # amber — fault-frequency bars (contrast to blue)
COLOR_GOOD = "#c6f6d5"      # green — quality cell, >=70
COLOR_WARN = "#fef3c7"      # amber — quality cell, 40-69
COLOR_BAD = "#fecaca"       # red — quality cell, <40

CUSTOM_CSS = f"""
<style>
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}
.pl-header {{
    background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, #60A5FA 100%);
    padding: 1.75rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}}
.pl-header h1 {{
    color: white;
    margin: 0 0 0.4rem 0;
    font-size: 1.9rem;
}}
.pl-header p {{
    color: #EFF6FF;
    margin: 0;
    font-size: 1.02rem;
}}
.stButton > button {{
    background-color: {COLOR_PRIMARY};
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: background-color 0.15s ease;
}}
.stButton > button:hover {{
    background-color: #1D4ED8;
    color: white;
    border: none;
}}
[data-testid="stMetric"] {{
    background-color: #EFF6FF;
    border: 1px solid #DBEAFE;
    border-radius: 12px;
    padding: 0.9rem 0.75rem;
}}
[data-testid="stMetricLabel"] {{
    color: #1E3A8A;
}}
[data-testid="stMetricValue"] {{
    color: #1D4ED8;
}}
[data-testid="stSidebar"] {{
    background-color: #F0F7FF;
}}
</style>
"""


@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    return LegRaiseClassifier()


def parse_jsonl(text):
    """Two-pass read: collect all frame lines first, then rep lines — rep
    lines are written to the JSONL AFTER the frames they describe (a rep's
    final label is only known once it closes), so a single forward pass
    can't resolve a rep's frame window. Same approach as build_dataset.py /
    inspect_jsonl.py."""
    frames, reps = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rtype = rec.get("type")
        if rtype == "frame":
            frames[rec["frame"]] = rec
        elif rtype == "rep" and rec.get("completed"):
            reps.append(rec)
    return frames, reps


def rule_engine_verdict(rep):
    if rep["clean"]:
        return "Clean"
    return ", ".join(ISSUE_LABELS.get(code, code) for code in rep["issues"])


def score_recording(frames, reps, classifier):
    """Runs predict_rep() once per completed rep. Returns (table_rows,
    model_results) — model_results excludes skipped (degraded) reps, exactly
    matching what aggregate_session() expects."""
    rows = []
    model_results = []
    for rep in reps:
        rep_frames = [
            [frames[fn]["landmarks"][str(i)] for i in range(33)]
            for fn in range(rep["start_frame"], rep["end_frame"] + 1)
            if fn in frames
        ]
        verdict = rule_engine_verdict(rep)

        try:
            result = classifier.predict_rep(rep_frames)
        except ValueError:
            rows.append({
                "Rep": rep["rep_id"],
                "Rule engine": verdict,
                "Knee-bend likelihood": "—",
                "Shallow likelihood": "—",
                "Quality %": np.nan,
                "Model verdict": "skipped (degraded rep)",
            })
            continue

        model_results.append(result)
        rows.append({
            "Rep": rep["rep_id"],
            "Rule engine": verdict,
            "Knee-bend likelihood": f"{result['fault_percentages']['knee_bent']}%",
            "Shallow likelihood": f"{result['fault_percentages']['shallow']}%",
            "Quality %": result["rep_quality_percent"],
            "Model verdict": "Clean" if result["predicted_clean"] else "Fault detected",
        })
    return rows, model_results


def _quality_cell_color(val):
    if pd.isna(val):
        return ""
    if val >= 70:
        return f"background-color: {COLOR_GOOD}"
    if val >= 40:
        return f"background-color: {COLOR_WARN}"
    return f"background-color: {COLOR_BAD}"


@st.cache_data(show_spinner="Scoring reps...")
def analyze_bytes(data_bytes, _classifier):
    """Cached on the recording's raw bytes so re-clicking the same sample
    (or re-viewing an already-uploaded file) doesn't re-run the model.
    `_classifier` is excluded from the cache key (leading underscore) since
    Streamlit can't hash the loaded onnxruntime session."""
    text = data_bytes.decode("utf-8")
    frames, reps = parse_jsonl(text)
    return score_recording(frames, reps, _classifier)


def render_results(rows, model_results, label):
    st.subheader(f"📋 Results — {label}")

    if not rows:
        st.warning("No completed reps found in this recording.")
        return

    df = pd.DataFrame(rows)
    styled = df.style.map(_quality_cell_color, subset=["Quality %"]).format(
        {"Quality %": lambda v: "—" if pd.isna(v) else f"{v:.1f}%"}
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    summary = aggregate_session(model_results)

    st.markdown("##### 💙 Session summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reps scored", summary["total_reps"])
    c2.metric("Avg rep quality", f"{summary['avg_rep_quality_percent']}%")
    c3.metric("Flagged: knee-bend", summary["flagged_reps"]["knee_bent"])
    c4.metric("Flagged: shallow", summary["flagged_reps"]["shallow"])
    c5.metric("Predicted clean", summary["clean_reps"])

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.caption("📈 Per-rep quality % across the session")
        quality_series = pd.Series(
            [r["rep_quality_percent"] for r in model_results],
            index=[f"Rep {i + 1}" for i in range(len(model_results))],
            name="Quality %",
        )
        st.line_chart(quality_series, color=[COLOR_PRIMARY])
    with chart_col2:
        st.caption("📊 Fault frequency across the session")
        fault_counts = pd.Series(
            {"Knee-bend": summary["flagged_reps"]["knee_bent"],
             "Shallow": summary["flagged_reps"]["shallow"]},
            name="Flagged reps",
        )
        st.bar_chart(fault_counts, color=[COLOR_ACCENT])

    st.caption(
        "Percentages are the probability each fault is **present** (a likelihood), not a "
        "severity score — \"knee-bend likelihood 82%\" means the model is 82% confident this "
        "rep had a bent knee, not that the knee was bent by 82%."
    )
    st.caption(
        "Rule engine vs. model can disagree on a rep, by design: the rule engine is an exact, "
        "instant threshold check (what live coaching is built on, and what catches very subtle "
        "faults), while the model gives a graded quality score and session-level trend the rule "
        "engine can't produce. See the About panel for the full hybrid rationale."
    )


def render_about():
    st.header("💙 About PhysioLens")
    st.markdown(
        """
**What this is.** A hybrid rule-engine + deep-learning system for automated exercise form
analysis, currently covering the Side Leg Raise exercise.

**The hybrid design.**
- A deterministic **rule engine** (exact joint-angle thresholds) owns live, per-frame feedback,
  voice coaching, and the rep-flag count a user sees in the moment.
- A trained **CNN-LSTM model** scores each *completed* rep with a continuous 0–100% quality
  score and per-fault likelihoods — the graded, trackable metric a binary rule can't produce.

**Why this app has no MediaPipe (and no TensorFlow).** Live pose capture is inherently a
client-side concern (a webcam, a phone, a browser) and MediaPipe is heavy to install on free
cloud hosts. Model inference is a server-side concern — and this deployment of it runs on
`onnxruntime` rather than TensorFlow, converted from the same trained model (verified to match
the original TensorFlow model's output exactly), which keeps the server-side footprint small
enough to install cleanly on any Python version. Splitting live capture from inference is the
correct production architecture, not a workaround: pose extraction happens locally (the
recordings behind the sample buttons and any uploaded file were produced that way), and this
deployed app only ever consumes the resulting landmark data.

**Honest limitations:**
- The model under-detects *very subtle* knee-bends (bends only just past the straight-knee
  threshold) — it was trained mostly on clearer bends. The rule engine covers that gap live.
- Small dataset: 5 subjects, ~300 recorded reps total, 2 held-out test subjects.
- Single camera view (front), two exercises modeled so far (Side Leg Raise here; Scaption
  planned).
- Percentages are fault-*presence* likelihoods, not severity scores (see caption under results).
        """
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pl-header">
            <h1>🦵 PhysioLens — Exercise Form Feedback</h1>
            <p>A hybrid rule-engine + AI system that reviews your Side Leg Raise reps and gives
            clear, easy-to-read feedback on form and progress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        render_about()

    classifier = load_classifier()

    with st.container(border=True):
        st.subheader("🔍 Try it")
        st.write("No data of your own? Click a sample recording below — no upload needed.")

        cols = st.columns(3)
        if "source" not in st.session_state:
            st.session_state.source = None

        for col, key in zip(cols, SAMPLES.keys()):
            button_text = SAMPLES[key][0]
            if col.button(button_text, width="stretch"):
                st.session_state.source = ("sample", key)

        st.divider()
        uploaded = st.file_uploader(
            "...or upload your own schema-v4 Side Leg Raise JSONL", type=["jsonl"]
        )
        if uploaded is not None:
            st.session_state.source = ("upload", uploaded.name, uploaded.getvalue())

    source = st.session_state.source
    if source is None:
        st.info("Click a sample button above to see the model in action.")
        return

    if source[0] == "sample":
        key = source[1]
        label, fname = SAMPLES[key]
        data_bytes = (SAMPLES_DIR / fname).read_bytes()
    else:
        _, label, data_bytes = source

    rows, model_results = analyze_bytes(data_bytes, classifier)
    with st.container(border=True):
        render_results(rows, model_results, label)


if __name__ == "__main__":
    main()
