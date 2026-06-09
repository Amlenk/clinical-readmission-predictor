# Upgraded Research Report: Predictive Analytics for Reducing Patient Readmission Rates in the Healthcare Sector

**Author:** Aman Lenka (Student ID: X23176351)  
**Program:** MSc Data Analytics, Winter Submission  

---

## 1. Abstract
Predicting 30-day hospital readmissions is a key quality and efficiency indicator in clinical healthcare. This project presents an upgraded, production-grade deep learning and machine learning pipeline to predict readmission risks. Implementing custom domain feature engineering, model tuning via Bayesian Optimization (Optuna) and Random Search (Keras Tuner), explainable AI (SHAP), and an interactive web application interface (Streamlit) provides a complete diagnostic solution for healthcare environments.

---

## 2. Methodology & System Architecture

### A. Preprocessing & Advanced Feature Engineering
In clinical datasets, raw counts do not always reflect underlying care intensity. To capture clinical severity relative to duration of stay, we engineered four domain-specific features *prior* to normalization:
1.  **Medication Intensity (`medications_per_day`)**: 
    $$\text{Medication Intensity} = \frac{n_{\text{medications}}}{\text{time\_in\_hospital} + 1}$$
    Identifies if a patient is receiving high volumes of pharmacological treatments per day, signaling acute medical episodes.
2.  **Laboratory Density (`lab_tests_per_day`)**: 
    $$\text{Lab Density} = \frac{n_{\text{lab\_procedures}}}{\text{time\_in\_hospital} + 1}$$
    Measures daily diagnostic tracking, reflecting clinical monitoring requirements.
3.  **Healthcare Utilization Index (`healthcare_utilization`)**: 
    $$\text{Utilization} = n_{\text{outpatient}} + n_{\text{inpatient}} + n_{\text{emergency}}$$
    A single aggregated index tracking the patient's utilization history in the preceding year.
4.  **Stay-to-Medication Ratio (`stay_to_medication_ratio`)**: 
    $$\text{Stay-to-Medication} = \frac{\text{time\_in\_hospital}}{n_{\text{medications}} + 1}$$
    Identifies instances of prolonged stays with minimal therapeutic escalation.

By performing this feature engineering **before** standard scaling, we preserved the mathematical meaning of the ratios and prevented zero-division or negative-ratio anomalies. All numerical features were subsequently normalized using a fitted `StandardScaler`.

### B. Modeling with Keras 3 and PyTorch Backend
To address the lack of pre-built TensorFlow libraries on newer Python installations (e.g., Python 3.14+), the architecture was migrated to **Keras 3 with a PyTorch backend** (`os.environ['KERAS_BACKEND'] = 'torch'`). This guarantees robust runtime execution, utilizing PyTorch tensors underneath Keras's familiar user-friendly Layer and Model APIs.

### C. Cross-Validated Hyperparameter Tuning
To ensure model validation scores are generalizable, both tuning notebooks execute search trials using **5-fold Stratified Cross-Validation**:
*   **Optuna (Bayesian Optimization)**: Explores a continuous parameter space for Dense layers (1–5), units (64–256), dropout (0.1–0.5), learning rates, and optimizers.
*   **Keras Tuner (Random Search)**: A custom `CVTuner` subclass overrides `run_trial` to calculate out-of-fold validation accuracy, preventing over-tuning on simple train/test splits.

---

## 3. Experimental Results

The models were evaluated on an independent 20% test subset (5,000 samples) extracted from the balanced dataset of 25,000 patient records. Below is the performance comparison across the updated architectures:

| Model & Optimization | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Test AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NN + Bayesian (Optuna)** | **61.74%** | **0.61** | 0.51 | 0.55 | **0.658** |
| **NN + Random Search (Keras Tuner)** | **61.74%** | 0.60 | **0.54** | **0.57** | 0.655 |
| **Logistic Regression (Baseline)** | 61.00% | **0.63** | 0.41 | 0.50 | 0.650 |
| **Random Forest (Baseline)** | 60.00% | 0.58 | 0.51 | 0.55 | 0.633 |
| **XGBoost (Baseline)** | 60.00% | 0.58 | 0.52 | 0.55 | 0.629 |

*Note: Neural networks optimized via Bayesian/Random Search achieved the highest test accuracy (61.74%) and AUC-ROC (up to 0.658), outperforming traditional machine learning ensembles.*

---

## 4. Model Interpretability (Explainable AI)
A key barrier to deep learning adoption in clinical medicine is the "black-box" nature of neural networks. To resolve this, the pipeline integrates **SHAP (SHapley Additive exPlanations)**. 

Using `shap.KernelExplainer`, Shapley values are calculated to estimate each feature's contribution to a patient's readmission risk score. A generated **SHAP Summary Plot** details:
*   The global importance of clinical variables (e.g. prior inpatient utilization and daily medication density are top risk drivers).
*   The direction of feature effects (e.g., higher healthcare utilization values drive the prediction towards "readmitted").

---

## 5. Web Application & Production Deployment
An interactive web application was designed in **Streamlit** (`app.py`). The application:
1.  Provides dropdowns and sliders for all clinical and demographic inputs.
2.  Dynamically computes engineered features (`healthcare_utilization`, `medications_per_day`, etc.).
3.  Loads the serialized Keras model (`best_model_bayesian.keras`) and scaler state (`scaler.joblib`).
4.  Outputs the patient's readmission probability alongside standardized risk classifications:
    *   **High Risk ($\ge$ 55%)**: Recommends clinical discharge transition care.
    *   **Moderate Risk (45%–55%)**: Advises review of treatment transitions.
    *   **Low Risk ($<$ 45%)**: Recommends standard discharge routines.
5.  Details patient-specific risk drivers (e.g., warning if prior inpatient stays or emergency visits exceed key risk thresholds).
