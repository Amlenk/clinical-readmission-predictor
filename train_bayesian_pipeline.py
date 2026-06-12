"""
30-Day Hospital Readmission Prediction
=======================================
Keras 3 (PyTorch backend) + Optuna Bayesian hyperparameter optimization
with Stratified 5-Fold Cross-Validation.

Outputs:
    best_model_bayesian.keras   - final model trained with best hyperparameters
    scaler.joblib               - fitted StandardScaler (numeric columns)
    feature_columns.joblib      - exact column order expected at inference time

Author: X23176351 (MSc Data Analytics)
"""

import os

# Must be set BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

import gc
import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import keras
from keras import layers, regularizers
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
N_TRIALS = 50          # Optuna trials; lower for a quick run
MAX_EPOCHS = 100
BATCH_SIZE = 256
DATA_PATH = "hospital_readmissions.csv"

keras.utils.set_random_seed(SEED)

# Ordinal midpoints for the age decade bins
AGE_MIDPOINTS = {
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
    "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}

# Diagnosis categories that represent a genuine disease-system code
# ("Missing"/"Other" carry no system information for comorbidity counting)
INFORMATIVE_DIAGS = {
    "Circulatory", "Respiratory", "Digestive",
    "Diabetes", "Injury", "Musculoskeletal",
}


# ----------------------------------------------------------------------
# 1. Feature engineering
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic + advanced clinical interaction features."""
    df = df.copy()
    diag_cols = ["diag_1", "diag_2", "diag_3"]

    # ----- basic features (existing) -----
    df["medications_per_day"] = df["n_medications"] / (df["time_in_hospital"] + 1)
    df["lab_tests_per_day"] = df["n_lab_procedures"] / (df["time_in_hospital"] + 1)
    df["healthcare_utilization"] = (
        df["n_outpatient"] + df["n_inpatient"] + df["n_emergency"]
    )
    df["stay_to_medication_ratio"] = df["time_in_hospital"] / (df["n_medications"] + 1)

    # ----- advanced features -----
    # F1: crude comorbidity index - number of distinct informative disease
    #     systems coded across the three diagnosis slots
    df["comorbidity_index"] = df[diag_cols].apply(
        lambda r: len({d for d in r if d in INFORMATIVE_DIAGS}), axis=1
    )

    # F2: diagnostic concordance - same system in more than one diagnosis slot
    #     (severity concentrated in a single organ system)
    df["diag_concordance"] = df[diag_cols].apply(
        lambda r: int(
            len([d for d in r if d in INFORMATIVE_DIAGS])
            > len({d for d in r if d in INFORMATIVE_DIAGS})
        ),
        axis=1,
    )

    # F3: diabetes coded anywhere in the diagnoses
    df["diabetes_dx_flag"] = (df[diag_cols] == "Diabetes").any(axis=1).astype(int)

    # F4: therapy instability - medication regimen changed for a patient
    #     already on diabetes medication
    df["med_intensification"] = (
        (df["change"] == "yes") & (df["diabetes_med"] == "yes")
    ).astype(int)

    # F5: documented poor glycemic control (either test high)
    df["poor_glycemic_control"] = (
        (df["A1Ctest"] == "high") | (df["glucose_test"] == "high")
    ).astype(int)

    # F6: share of prior utilization that was acute/unplanned
    df["acute_care_ratio"] = (df["n_inpatient"] + df["n_emergency"]) / (
        df["healthcare_utilization"] + 1
    )

    # F7: any prior inpatient admission (strongest single readmission predictor)
    df["prior_inpatient_flag"] = (df["n_inpatient"] > 0).astype(int)

    # F8: age-weighted utilization (frail elderly with frequent acute contacts)
    df["age_ordinal"] = df["age"].map(AGE_MIDPOINTS)
    df["age_x_acute_utilization"] = (
        df["age_ordinal"] / 100.0 * (df["n_inpatient"] + df["n_emergency"])
    )

    # F9: polypharmacy flag (>=20 medications, severe-case marker)
    df["polypharmacy_flag"] = (df["n_medications"] >= 20).astype(int)

    return df


# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
NUMERIC_COLS = [
    "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
    "n_outpatient", "n_inpatient", "n_emergency",
    "medications_per_day", "lab_tests_per_day", "healthcare_utilization",
    "stay_to_medication_ratio", "comorbidity_index", "acute_care_ratio",
    "age_ordinal", "age_x_acute_utilization",
]

CATEGORICAL_COLS = [
    "age", "medical_specialty", "diag_1", "diag_2", "diag_3",
    "glucose_test", "A1Ctest", "change", "diabetes_med",
]

BINARY_FLAG_COLS = [
    "diag_concordance", "diabetes_dx_flag", "med_intensification",
    "poor_glycemic_control", "prior_inpatient_flag", "polypharmacy_flag",
]


def build_design_matrix(df: pd.DataFrame):
    """One-hot encode categoricals; return X (unscaled), y, column order."""
    y = (df["readmitted"] == "yes").astype(int).values
    X = pd.get_dummies(
        df.drop(columns=["readmitted"]),
        columns=CATEGORICAL_COLS,
        drop_first=False,
        dtype=float,
    )
    feature_columns = X.columns.tolist()
    return X, y, feature_columns


# ----------------------------------------------------------------------
# 3. Model definition
# ----------------------------------------------------------------------
def build_model(input_dim: int, params: dict) -> keras.Model:
    reg = regularizers.L1L2(l1=params["l1"], l2=params["l2"])

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i in range(params["n_layers"]):
        x = layers.Dense(
            params[f"units_{i}"],
            kernel_regularizer=reg,
            use_bias=False,           # bias is redundant before BatchNorm
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(params[f"dropout_{i}"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    optimizer_name = params["optimizer"]
    lr = params["learning_rate"]
    if optimizer_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# 4. Stratified 5-fold CV evaluation of one hyperparameter set
# ----------------------------------------------------------------------
def cross_validate(X: pd.DataFrame, y: np.ndarray, params: dict,
                   trial: optuna.Trial | None = None):
    """Return (oof_auc, oof_accuracy, mean_best_epoch)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(y))
    best_epochs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Scaler fitted on the training fold ONLY (no leakage)
        scaler = StandardScaler()
        X_tr[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
        X_va[NUMERIC_COLS] = scaler.transform(X_va[NUMERIC_COLS])

        model = build_model(X.shape[1], params)
        es = keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=10,
            restore_best_weights=True, verbose=0,
        )
        model.fit(
            X_tr.values.astype("float32"), y_tr,
            validation_data=(X_va.values.astype("float32"), y_va),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[es], verbose=0,
        )
        best_epochs.append(es.best_epoch + 1 if es.best_epoch is not None
                           else MAX_EPOCHS)

        oof_preds[va_idx] = model.predict(
            X_va.values.astype("float32"), verbose=0
        ).ravel()

        # free backend graph/tensor memory between folds
        del model
        keras.utils.clear_session()
        gc.collect()

        # Optuna pruning on running OOF AUC
        if trial is not None:
            seen = oof_preds != 0
            interim_auc = roc_auc_score(y[seen], oof_preds[seen])
            trial.report(interim_auc, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    oof_auc = roc_auc_score(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds >= 0.5).astype(int))
    return oof_auc, oof_acc, int(np.mean(best_epochs))


# ----------------------------------------------------------------------
# 5. Optuna objective
# ----------------------------------------------------------------------
def make_objective(X: pd.DataFrame, y: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "optimizer": trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "rmsprop"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True),
            "l1": trial.suggest_float("l1", 1e-6, 1e-3, log=True),
            "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
        }
        for i in range(params["n_layers"]):
            params[f"units_{i}"] = trial.suggest_int(f"units_{i}", 64, 256, step=32)
            params[f"dropout_{i}"] = trial.suggest_float(f"dropout_{i}", 0.1, 0.5)

        oof_auc, oof_acc, mean_epochs = cross_validate(X, y, params, trial)
        trial.set_user_attr("oof_accuracy", oof_acc)
        trial.set_user_attr("mean_best_epochs", mean_epochs)
        print(f"  trial {trial.number:3d} | OOF AUC={oof_auc:.4f} "
              f"| OOF ACC={oof_acc:.4f} | epochs~{mean_epochs}")
        return oof_auc

    return objective


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    print(f"Keras backend: {keras.backend.backend()}")

    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    X, y, feature_columns = build_design_matrix(df)
    print(f"Design matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # --- Bayesian (TPE) hyperparameter search ---
    # SQLite storage so the study survives interruptions and resumes
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2),
        study_name="readmission_bayesian",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
    )
    finished = len([
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE,
                       optuna.trial.TrialState.PRUNED)
    ])
    remaining = max(0, N_TRIALS - finished)
    print(f"Resuming study: {finished} trials done, {remaining} to run")
    if remaining > 0:
        study.optimize(make_objective(X, y), n_trials=remaining,
                       show_progress_bar=True)

    best = study.best_trial
    print("\n================ BEST TRIAL ================")
    print(f"OOF ROC-AUC : {best.value:.4f}")
    print(f"OOF Accuracy: {best.user_attrs['oof_accuracy']:.4f}")
    print(json.dumps(best.params, indent=2))

    # --- Refit on the full dataset with the best hyperparameters ---
    best_params = dict(best.params)
    final_epochs = best.user_attrs["mean_best_epochs"]

    scaler = StandardScaler()
    X_full = X.copy()
    X_full[NUMERIC_COLS] = scaler.fit_transform(X_full[NUMERIC_COLS])

    final_model = build_model(X.shape[1], best_params)
    final_model.fit(
        X_full.values.astype("float32"), y,
        epochs=final_epochs, batch_size=BATCH_SIZE, verbose=2,
    )

    # --- Persist artefacts for inference (app.py) ---
    final_model.save("best_model_bayesian.keras")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")
    study.trials_dataframe().to_csv("tuning_results/optuna_trials.csv", index=False)

    print("\nSaved: best_model_bayesian.keras, scaler.joblib, "
          "feature_columns.joblib, tuning_results/optuna_trials.csv")


if __name__ == "__main__":
    main()
