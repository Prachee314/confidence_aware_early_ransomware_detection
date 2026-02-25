# --------------------------------------------------
# Global UI Styling
# --------------------------------------------------



import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.markdown("""
<style>
.big-title {
    font-size: 36px;
    font-weight: 700;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Path setup
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import early_window
from src.features import execution_features
from src.decision import explanation_strength, decision

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Ransomware Detection System",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown('<div class="big-title">🔐 Confidence-Aware Early Ransomware Detection System</div>', unsafe_allow_html=True)

st.markdown("""
**Execution-level early ransomware detection** using behavioral signals,  
machine learning, and explainable AI.

- One row = **one execution (ProcessGuid)**
- Decisions: **🟢 BENIGN | 🟡 DEFER | 🔴 ALERT**
""")


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(PROJECT_ROOT, "models/lightgbm.pkl"))

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("📂 Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload FastText execution CSV",
    type=["csv"]
)

st.sidebar.markdown(
    """
    **Accepted input**
    - Single or multiple executions
    - Same schema as SILRAD FastText CSV
    """
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def build_feature_names(base_features):
    return (
        [f"{f}_mean" for f in base_features] +
        [f"{f}_std"  for f in base_features] +
        [f"{f}_max"  for f in base_features] +
        [f"{f}_diff" for f in base_features]
    )

def humanize_feature(f):
    if "event.code" in f:
        return "suspicious system event patterns"
    if "Image" in f:
        return "unusual executable image behavior"
    if "User" in f:
        return "unexpected user context"
    if "task" in f:
        return "abnormal task-level activity"
    if "Details" in f:
        return "low-level system modification indicators"
    return f.replace("_", " ")

def explain_execution(i, risk, label, shap_vals, feature_names):
    # Top contributing features
    top_idx = np.argsort(np.abs(shap_vals[i]))[::-1][:3]
    top_feats = [humanize_feature(feature_names[j]) for j in top_idx]

    # HTML-formatted explanation (IMPORTANT)
    return (
        f"<b>Decision:</b> {label}<br>"
        f"<b>Risk score:</b> {risk:.3f}<br>"
        f"<b>Key behavioral indicators:</b> {', '.join(top_feats)}"
    )


def decision_color(label):
    if label == "ALERT":
        return "🔴 ALERT"
    if label == "DEFER":
        return "🟡 DEFER"
    return "🟢 BENIGN"

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload an execution CSV file to start analysis.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

st.subheader("📊 Uploaded Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Early windowing
df = early_window(df)

# Identify base feature columns
ID_COLS = [
    "ProcessGuid", "ProcessId",
    "ParentProcessGuid", "ParentProcessId",
    "TargetProcessGUID", "TargetProcessId"
]
LABEL_COL = "class"

FEATURE_COLS = [
    c for c in df.columns
    if c not in ID_COLS + [LABEL_COL]
]

# Feature extraction (execution-level)
X, _ = execution_features(df)

# Prediction
risk_scores = model.predict_proba(X)[:, 1]

# SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

expl_strength = explanation_strength(shap_values)

# Thresholds
risk_th = 0.5
expl_th = np.percentile(expl_strength, 70)

decisions = [
    decision(r, e, expl_th)
    for r, e in zip(risk_scores, expl_strength)
]

st.markdown('<div class="section-title">Decision Summary</div>', unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:10px;
            padding:18px;
            text-align:center;">
            <div style="font-size:13px;color:#6b7280;">Total Executions</div>
            <div style="font-size:28px;font-weight:700;color:#111827;">
                {len(decisions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #fee2e2;
            border-radius:10px;
            padding:18px;
            text-align:center;">
            <div style="font-size:13px;color:#991b1b;">Alerts Raised</div>
            <div style="font-size:28px;font-weight:700;color:#b91c1c;">
                {sum(d == "ALERT" for d in decisions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #fef3c7;
            border-radius:10px;
            padding:18px;
            text-align:center;">
            <div style="font-size:13px;color:#92400e;">Deferred Decisions</div>
            <div style="font-size:28px;font-weight:700;color:#b45309;">
                {sum(d == "DEFER" for d in decisions)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# --------------------------------------------------
# Results table
# --------------------------------------------------
st.markdown('<div class="section-title">🚨 Detection Results</div>', unsafe_allow_html=True)

results_df = pd.DataFrame({
    "Execution ID": np.arange(len(decisions)),
    "Risk Score": np.round(risk_scores, 3),
    "Explanation Strength": np.round(expl_strength, 3),
    "Decision": [decision_color(d) for d in decisions]
})

st.dataframe(
    results_df.style.applymap(
        lambda x: "color:red;font-weight:bold" if "ALERT" in str(x)
        else "color:orange;font-weight:bold" if "DEFER" in str(x)
        else "color:green",
        subset=["Decision"]
    ),
    use_container_width=True
)


st.caption(
    "Each row corresponds to **one execution (ProcessGuid)** after early behavioral aggregation."
)

# --------------------------------------------------
# Detailed explanation section
# --------------------------------------------------

feature_names = build_feature_names(FEATURE_COLS)





# --------------------------------------------------
# Confidence-Aware Analyst Explanation (Color-coded)
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🧠 Analyst Explanation</div>',
    unsafe_allow_html=True
)

selected_idx = st.selectbox(
    "Select execution to inspect:",
    options=list(range(len(decisions))),
    key="analyst_execution_selectbox"
)


decision_label = decisions[selected_idx]

# Color logic based on decision
if decision_label == "ALERT":
    bg_color = "#FEF2F2"     # light red
    border_color = "#B91C1C" # red
elif decision_label == "DEFER":
    bg_color = "#FFFBEB"     # light amber
    border_color = "#B45309" # amber
else:  # BENIGN
    bg_color = "#F0FDF4"     # light green
    border_color = "#15803D" # green

# Generate explanation text
explanation_text = explain_execution(
    i=selected_idx,
    risk=risk_scores[selected_idx],
    label=decision_label,
    shap_vals=shap_values,
    feature_names=feature_names
)

# Render explanation box
st.markdown(
    f"""
    <div style="
        background:{bg_color};
        border-left:6px solid {border_color};
        padding:16px;
        border-radius:8px;
        font-size:15px;
        line-height:1.6;
        color:#111827;">
        {explanation_text}
    </div>
    """,
    unsafe_allow_html=True
)



# --------------------------------------------------
# SHAP visualization
# --------------------------------------------------
# --------------------------------------------------
# SHAP visualization (Compact & UI-friendly)
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Global Feature Attribution</div>',
    unsafe_allow_html=True
)

st.caption(
    "Top behavioral indicators contributing to ransomware risk "
    "(aggregated across executions)."
)

plt.figure(figsize=(6, 3), dpi=120)

shap.summary_plot(
    shap_values,
    X,
    feature_names=feature_names,
    max_display=8,
    show=False
)

plt.tight_layout()
st.pyplot(plt.gcf(), use_container_width=True)
plt.close()

# --------------------------------------------------
# Per-Execution Feature Impact (UI-grade)
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🔍 Execution-Level Risk Drivers</div>',
    unsafe_allow_html=True
)

# Get SHAP values for selected execution
local_shap = shap_values[selected_idx]
abs_local = np.abs(local_shap)

# Top-k features
k = 5
top_idx = np.argsort(abs_local)[-k:]

feature_labels = [humanize_feature(feature_names[i]) for i in top_idx]
impact_vals = abs_local[top_idx]

# Sort for better visual ordering
order = np.argsort(impact_vals)
feature_labels = np.array(feature_labels)[order]
impact_vals = impact_vals[order]

plt.figure(figsize=(6, 2.6), dpi=120)

bars = plt.barh(
    feature_labels,
    impact_vals,
    edgecolor="black",
    linewidth=0.3
)

# Color by relative impact
threshold = np.percentile(impact_vals, 60)
for bar, val in zip(bars, impact_vals):
    bar.set_color("#ef4444" if val >= threshold else "#60a5fa")

# Value labels at bar end
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01 * max(impact_vals),
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center",
        fontsize=9
    )

plt.xlabel("Impact on Risk Score", fontsize=10)
plt.title("Top Contributors for This Execution", fontsize=11, weight="bold")
plt.suptitle("Local Explanation (SHAP-based)", fontsize=9, y=0.93)

plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()

st.pyplot(plt.gcf(), use_container_width=True)
plt.close()



# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "**Note:** The system performs execution-level ransomware detection. "
    "Explanations are deterministic and rule-based for reproducibility."
)

