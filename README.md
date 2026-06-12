# Upgraded Research Report: Predictive Analytics for Reducing Patient Readmission Rates in the Healthcare Sector

**Author:** Aman Lenka (Student ID: X23176351)
**Program:** MSc Data Analytics, Winter Submission

---

## 1. Abstract
Predicting 30-day hospital readmissions is a key quality and efficiency indicator in clinical healthcare. This project presents an upgraded, production-grade deep learning and machine learning pipeline to predict readmission risks. Implementing custom domain feature engineering (13 engineered clinical features), model tuning via Bayesian Optimization (Optuna TPE) under Stratified 5-Fold Cross-Validation, explainable AI (per-patient SHAP waterfall explanations), and an interactive web application interface (Streamlit) provides a complete diagnostic solution for healthcare environments. The tuned network achieves an out-of-fold ROC-AUC of **0.6613** and accuracy of **62.28%** across all 25,000 patient records.

---

## 2. Methodology & System Architecture

### A. Preprocessing & Advanced Feature Engineering
In clinical datasets, raw counts do not always reflect underlying care intensity. To capture clinical severity relative to duration of stay, we engineered four intensity features *prior* to normalization:
1.  **Medication Intensity (`medications_per_day`)**:
    $$\text{Medication Intensity} = \frac{\text{Medications}}{\text{Time in Hospital} + 1}$$
    Identifies if a patient is receiving high volumes of pharmacological treatments per day, signaling acute medical episodes.
2.  **Laboratory Density (`lab_tests_per_day`)**:
    $$\text{Lab Density} = \frac{\text{Lab Procedures}}{\text{Time in Hospital} + 1}$$
    Measures daily diagnostic tracking, reflecting clinical monitoring requirements.
3.  **Healthcare Utilization Index (`healthcare_utilization`)**:
    $$\text{Utilization} = \text{Outpatient} + \text{Inpatient} + \text{Emergency}$$
    A single aggregated index tracking the patient's utilization history in the preceding year.
4.  **Stay-to-Medication Ratio (`stay_to_medication_ratio`)**:
    $$\text{Stay-to-Medication} = \frac{\text{Time in Hospital}}{\text{Medications} + 1}$$
    Identifies instances of prolonged stays with minimal therapeutic escalation.

These were extended with **nine advanced clinical interaction features** capturing comorbidity burden, glycemic instability, and acute-care patterns:

5.  **Comorbidity Index (`comorbidity_index`)**: the number of *distinct* informative disease systems coded across the three diagnosis slots (excluding "Missing"/"Other"). A crude Charlson-style proxy — multimorbid patients have less physiologic reserve and more fragmented post-discharge care.
6.  **Diagnostic Concordance (`diag_concordance`)**: flags when the same disease system appears in more than one diagnosis slot, distinguishing severity concentrated in a single organ system (e.g., heart failure plus arrhythmia) from scattered mild problems.
7.  **Diabetes Diagnosis Flag (`diabetes_dx_flag`)**: diabetes coded anywhere in the diagnoses, not only as primary (Strack et al., 2014).
8.  **Medication Intensification (`med_intensification`)**: the interaction of `change = yes` **and** `diabetes_med = yes` — a regimen altered *during* admission signals discharge on an unstabilized therapy, a classic readmission mechanism neither column captures alone.
9.  **Poor Glycemic Control (`poor_glycemic_control`)**: a high A1C or glucose test result, documenting metabolic decompensation at admission.
10. **Acute Care Ratio (`acute_care_ratio`)**: $(\text{Inpatient} + \text{Emergency}) / (\text{Utilization} + 1)$ — separates crisis-driven utilizers from patients engaged in planned chronic care.
11. **Prior Inpatient Flag (`prior_inpatient_flag`)**: any prior inpatient stay; consistently the strongest single readmission predictor in the literature, captured as a binary to model its non-linearity.
12. **Age-Weighted Acute Utilization (`age_x_acute_utilization`)**: age midpoint × acute visits — an explicit frailty interaction, since acute utilization is more ominous at 85 than at 45.
13. **Polypharmacy Flag (`polypharmacy_flag`)**: ≥ 20 medications, an independent geriatric risk marker for post-discharge adverse drug events.

By performing this feature engineering **before** standard scaling, we preserved the mathematical meaning of the ratios and prevented zero-division or negative-ratio anomalies. Categorical variables are one-hot encoded (68 total model inputs). Crucially, the `StandardScaler` is **fitted inside each cross-validation fold on training data only**, eliminating preprocessing leakage from validation metrics; the final scaler state is serialized for inference.

### B. Modeling with Keras 3 and PyTorch Backend
To address the lack of pre-built TensorFlow libraries on newer Python installations (e.g., Python 3.14+), the architecture was migrated to **Keras 3 with a PyTorch backend** (`os.environ['KERAS_BACKEND'] = 'torch'`). The network uses Dense → BatchNormalization → ReLU → Dropout blocks with L1L2 kernel regularization and a sigmoid output head, trained with early stopping on validation AUC (patience 10, best weights restored).

### C. Cross-Validated Bayesian Hyperparameter Tuning
The production pipeline (`train_bayesian_pipeline.py`) runs **50 Optuna trials** with the TPE (Tree-structured Parzen Estimator) sampler. Every trial is scored by **Stratified 5-Fold Cross-Validation**, where the objective is the **out-of-fold ROC-AUC computed across all 25,000 patients** — a single honest number per configuration rather than an average of fold scores. The search space covers:
*   Dense layers (1–5) and units per layer (64–256)
*   Per-layer dropout (0.1–0.5)
*   Log-uniform learning rate (1e-4 – 1e-2) and optimizer ∈ {Adam, AdamW, RMSprop}
*   Log-uniform L1 (1e-6 – 1e-3) and L2 (1e-6 – 1e-2) regularization

A **MedianPruner** terminates below-median configurations after 2 of 5 folds (21 of 50 trials were pruned, roughly halving search time), and the study persists to SQLite storage (`optuna_study.db`) so interrupted runs resume without losing completed trials. The complete trial history is exported to `tuning_results/optuna_trials.csv`.

---

## 3. Experimental Results

### A. Final Tuned Model (Stratified 5-Fold Cross-Validation, n = 25,000)

The completed 50-trial study selected a compact, strongly regularized architecture (best trial #39):

| Hyperparameter | Selected Value |
| :--- | :---: |
| Hidden layers | 2 (160 → 64 units) |
| Optimizer / learning rate | Adam / 3.4e-4 |
| Dropout | 0.33 / 0.40 |
| L1 / L2 regularization | 3.0e-5 / 5.2e-4 |
| Epochs (early-stopped) | ~30 |

| Metric (out-of-fold, all 25,000 patients) | Score |
| :--- | :---: |
| **ROC-AUC** | **0.6613** |
| **Accuracy** | **62.28%** |

Notably, the 29 completed trials clustered tightly between 0.654–0.661 AUC, and depth never paid off — the 5-layer candidates never beat the 2-layer winner. This indicates a robust, non-overfit optimum near the dataset's predictive ceiling, with regularization mattering more than capacity. The final model is refit on the full dataset at the early-stopped epoch budget and serialized as `best_model_bayesian.keras`.

### B. Baseline Comparison (independent 20% test subset, earlier iteration)

| Model & Optimization | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Test AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NN + Bayesian (Optuna)** | **61.74%** | **0.61** | 0.51 | 0.55 | **0.658** |
| **NN + Random Search (Keras Tuner)** | **61.74%** | 0.60 | **0.54** | **0.57** | 0.655 |
| **Logistic Regression (Baseline)** | 61.00% | **0.63** | 0.41 | 0.50 | 0.650 |
| **Random Forest (Baseline)** | 60.00% | 0.58 | 0.51 | 0.55 | 0.633 |
| **XGBoost (Baseline)** | 60.00% | 0.58 | 0.52 | 0.55 | 0.629 |

*Note: Neural networks outperformed traditional machine learning ensembles in both evaluation regimes, and the upgraded feature set plus cross-validated Bayesian tuning lifted out-of-fold AUC from 0.658 to 0.6613 with improved accuracy (62.28%).*

---

## 4. Model Interpretability (Explainable AI)
A key barrier to deep learning adoption in clinical medicine is the "black-box" nature of neural networks. To resolve this, the pipeline integrates **SHAP (SHapley Additive exPlanations)** at two levels:

*   **Global**: a SHAP summary plot details the overall importance of clinical variables (prior inpatient utilization, healthcare utilization, and laboratory volume emerge as top risk drivers) and the direction of their effects.
*   **Per-patient (live)**: the web application computes Shapley values **on demand** for each analyzed patient using a `shap.KernelExplainer` against a cached 100-patient background cohort, and renders a **waterfall plot of the top 10 contributing features** — showing the clinician exactly which factors pushed this individual's risk up or down.

---

## 5. Web Application & Production Deployment
An interactive web application was designed in **Streamlit** (`app.py`). To guarantee training and inference can never drift apart, the app **imports the feature-engineering and preprocessing logic directly from `train_bayesian_pipeline.py`** rather than duplicating it, and validates the loaded artefacts against the current pipeline's feature contract at startup. The application:
1.  Provides grouped clinical inputs (Demographics, Utilization History, Current Episode, Diagnoses) in a modern glassmorphic dark-mode interface.
2.  Reconstructs the full 68-column design matrix for the entered patient — engineered features, one-hot encoding, column reindexing, and scaling with the serialized `scaler.joblib`.
3.  Loads the serialized Keras model (`best_model_bayesian.keras`) and preprocessor state with Streamlit resource caching for sub-second reruns.
4.  Outputs the patient's readmission probability alongside standardized risk classifications:
    *   **High Risk ($\ge$ 55%)**: Recommends intensive transition-of-care planning with 7-day follow-up.
    *   **Moderate Risk (45%–55%)**: Advises review of treatment transitions and medication-change counselling.
    *   **Low Risk ($<$ 45%)**: Recommends standard discharge routines.
5.  Displays the patient's engineered clinical risk flags (prior inpatient stay, diabetes therapy intensification, poor glycemic control, polypharmacy, comorbidity index) and the live SHAP waterfall explaining the individual prediction.
