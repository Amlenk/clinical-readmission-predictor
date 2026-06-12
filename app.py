"""
Hospital Readmission Risk Dashboard
====================================
Streamlit front-end for the Keras 3 (PyTorch backend) readmission model.

Reuses the exact preprocessing logic from train_bayesian_pipeline.py
(engineered features, one-hot encoding, scaling) so training and
inference can never drift apart.

Run with:  streamlit run app.py
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import keras

from train_bayesian_pipeline import (
    engineer_features,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    BINARY_FLAG_COLS,
    DATA_PATH,
)

MODEL_PATH = "best_model_bayesian.keras"
SCALER_PATH = "scaler.joblib"
COLUMNS_PATH = "feature_columns.joblib"

# ----------------------------------------------------------------------
# Page config + premium glassmorphic theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Readmission Risk AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
}
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99, 102, 241, 0.22), transparent),
        radial-gradient(ellipse 60% 40% at 90% 10%, rgba(79, 70, 229, 0.14), transparent),
        #0F172A;
}
h1, h2, h3 { color: #F8FAFC !important; font-family: 'Outfit', sans-serif !important; }

.glass-card {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.45);
}
.section-title {
    color: #A5B4FC;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.4rem;
}
.app-title {
    text-align: center;
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #C7D2FE, #6366F1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.app-subtitle { text-align: center; color: #94A3B8; margin-bottom: 1.6rem; }

.risk-card {
    border-radius: 18px;
    padding: 1.6rem;
    text-align: center;
    border: 1px solid;
    margin-bottom: 1rem;
}
.risk-high   { background: rgba(239, 68, 68, 0.12);  border-color: rgba(239, 68, 68, 0.45); }
.risk-mod    { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }
.risk-low    { background: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.45); }
.risk-prob   { font-size: 3.2rem; font-weight: 800; line-height: 1.1; }
.risk-label  { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
.risk-note   { color: #CBD5E1; font-size: 0.88rem; margin-top: 0.5rem; }

.flag-chip {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    margin: 0.18rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.flag-on  { background: rgba(239, 68, 68, 0.16);  color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4); }
.flag-off { background: rgba(51, 65, 85, 0.45);   color: #94A3B8; border: 1px solid #334155; }

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45);
    transition: all 0.25s ease;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(99, 102, 241, 0.6); }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🏥 Readmission Risk AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">30-day hospital readmission prediction · Keras 3 (PyTorch) '
    "· Bayesian-tuned · SHAP-explainable</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Cached asset loading
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model assets...")
def load_assets():
    model = keras.saving.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, feature_columns


for path in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH):
    if not os.path.exists(path):
        st.error(f"Missing artefact: `{path}`. Run `python train_bayesian_pipeline.py` first.")
        st.stop()

model, scaler, feature_columns = load_assets()

# Guard against artefacts produced by an older pipeline version
scaler_cols = list(getattr(scaler, "feature_names_in_", []))
if scaler_cols != NUMERIC_COLS or not set(NUMERIC_COLS).issubset(feature_columns):
    st.error(
        "The saved scaler/feature columns do not match the current pipeline's "
        "feature set. Re-run `python train_bayesian_pipeline.py` to regenerate "
        "`best_model_bayesian.keras`, `scaler.joblib` and `feature_columns.joblib`."
    )
    st.stop()


# ----------------------------------------------------------------------
# Shared preprocessing (identical path for live input and SHAP background)
# ----------------------------------------------------------------------
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """raw patient rows -> scaled design matrix in the model's column order."""
    df = engineer_features(raw_df)
    X = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=float)
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    return X.astype("float32")


@st.cache_data(show_spinner="Loading background cohort...")
def load_background_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH).drop(columns=["readmitted"])


@st.cache_resource(show_spinner="Building SHAP explainer...")
def get_explainer(_model, _scaler, feature_columns):
    background = preprocess(load_background_raw().sample(100, random_state=42))

    def predict_fn(X):
        return _model.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()

    return shap.KernelExplainer(predict_fn, background.values)


explainer = get_explainer(model, scaler, feature_columns)

# ----------------------------------------------------------------------
# Layout: inputs (left) | prediction + explanation (right)
# ----------------------------------------------------------------------
left, right = st.columns([5, 4], gap="large")

with left:
    st.markdown('<div class="glass-card"><div class="section-title">Demographics</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        age = st.selectbox(
            "Age bracket",
            ["[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=3,
        )
    with d2:
        medical_specialty = st.selectbox(
            "Admitting specialty",
            ["Missing", "Other", "InternalMedicine", "Family/GeneralPractice",
             "Cardiology", "Surgery", "Emergency/Trauma"],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-title">Utilization History (previous year)</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        n_outpatient = st.number_input("Outpatient visits", 0, 40, 0)
    with h2:
        n_inpatient = st.number_input("Inpatient stays", 0, 20, 0)
    with h3:
        n_emergency = st.number_input("Emergency visits", 0, 70, 0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-title">Current Episode · Procedures & Medication</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 4)
        n_lab_procedures = st.slider("Lab procedures", 1, 120, 45)
        n_procedures = st.slider("Non-lab procedures", 0, 6, 1)
        n_medications = st.slider("Medications administered", 1, 80, 15)
    with p2:
        glucose_test = st.selectbox("Glucose test result", ["no", "normal", "high"])
        A1Ctest = st.selectbox("A1C test result", ["no", "normal", "high"])
        change = st.selectbox("Medication regimen changed", ["no", "yes"])
        diabetes_med = st.selectbox("On diabetes medication", ["yes", "no"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card"><div class="section-title">Diagnoses</div>', unsafe_allow_html=True)
    diag_options = ["Circulatory", "Respiratory", "Digestive", "Diabetes",
                    "Injury", "Musculoskeletal", "Missing", "Other"]
    g1, g2, g3 = st.columns(3)
    with g1:
        diag_1 = st.selectbox("Primary (diag_1)", diag_options, index=0)
    with g2:
        diag_2 = st.selectbox("Secondary (diag_2)", diag_options, index=7)
    with g3:
        diag_3 = st.selectbox("Tertiary (diag_3)", diag_options, index=7)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card"><div class="section-title">Prediction</div>', unsafe_allow_html=True)
    analyze = st.button("🔍 Analyze Readmission Risk")

    if analyze:
        raw_patient = pd.DataFrame([{
            "age": age,
            "time_in_hospital": time_in_hospital,
            "n_lab_procedures": n_lab_procedures,
            "n_procedures": n_procedures,
            "n_medications": n_medications,
            "n_outpatient": n_outpatient,
            "n_inpatient": n_inpatient,
            "n_emergency": n_emergency,
            "medical_specialty": medical_specialty,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "glucose_test": glucose_test,
            "A1Ctest": A1Ctest,
            "change": change,
            "diabetes_med": diabetes_med,
        }])

        engineered = engineer_features(raw_patient)
        X_patient = preprocess(raw_patient)
        prob = float(model.predict(X_patient.values, verbose=0).ravel()[0])

        if prob >= 0.55:
            css, color, label, note = (
                "risk-high", "#F87171", "High Risk",
                "Recommend intensive transition-of-care plan, 7-day follow-up "
                "appointment and medication reconciliation call.",
            )
        elif prob >= 0.45:
            css, color, label, note = (
                "risk-mod", "#FBBF24", "Moderate Risk",
                "Recommend standard follow-up with verification that medication "
                "changes were clearly explained at discharge.",
            )
        else:
            css, color, label, note = (
                "risk-low", "#34D399", "Low Risk",
                "Standard discharge procedures are appropriate.",
            )

        st.markdown(
            f"""
            <div class="risk-card {css}">
                <div class="risk-prob" style="color:{color}">{prob * 100:.1f}%</div>
                <div class="risk-label" style="color:{color}">{label}</div>
                <div class="risk-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- key engineered risk flags for this patient ---
        st.markdown('<div class="section-title">Key Risk Flags</div>', unsafe_allow_html=True)
        flag_labels = {
            "prior_inpatient_flag": "Prior inpatient stay",
            "med_intensification": "Diabetes therapy changed",
            "poor_glycemic_control": "Poor glycemic control",
            "polypharmacy_flag": "Polypharmacy (≥20 meds)",
            "diabetes_dx_flag": "Diabetes diagnosis",
            "diag_concordance": "Concentrated organ-system burden",
        }
        chips = "".join(
            f'<span class="flag-chip {"flag-on" if int(engineered[c].iloc[0]) else "flag-off"}">'
            f'{"⚠ " if int(engineered[c].iloc[0]) else ""}{flag_labels.get(c, c)}</span>'
            for c in BINARY_FLAG_COLS
        )
        chips += (
            f'<span class="flag-chip {"flag-on" if engineered["comorbidity_index"].iloc[0] >= 2 else "flag-off"}">'
            f'Comorbidity index: {int(engineered["comorbidity_index"].iloc[0])}/3</span>'
        )
        st.markdown(chips, unsafe_allow_html=True)

        # --- on-the-fly SHAP waterfall ---
        st.markdown('<div class="section-title" style="margin-top:1.2rem">Why this prediction? (SHAP)</div>', unsafe_allow_html=True)
        with st.spinner("Computing Shapley values..."):
            shap_vals = np.asarray(
                explainer.shap_values(X_patient.values[0], nsamples=256)
            ).reshape(-1)
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=float(np.ravel(explainer.expected_value)[0]),
                data=X_patient.values[0],
                feature_names=feature_columns,
            )

            plt.rcParams.update({
                "figure.facecolor": "#1E293B",
                "axes.facecolor": "#1E293B",
                "text.color": "#E2E8F0",
                "axes.labelcolor": "#E2E8F0",
                "xtick.color": "#94A3B8",
                "ytick.color": "#E2E8F0",
                "axes.edgecolor": "#334155",
                "font.size": 10,
            })
            fig = plt.figure(figsize=(8.5, 6))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(plt.gcf(), bbox_inches="tight")
            plt.close("all")

        st.caption(
            "Red bars push the prediction towards readmission; blue bars push it away. "
            "Feature values shown are standardized."
        )
    else:
        st.info("Set the patient parameters on the left, then click **Analyze Readmission Risk**.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;color:#475569;font-size:0.8rem;margin-top:1rem">'
    "MSc Data Analytics · X23176351 · Decision-support prototype, not a medical device"
    "</div>",
    unsafe_allow_html=True,
)
