"""
train_model.py — Agent 2 / Backend ML Engineer
Trains RF, GB, and Stacking models on the Bank Marketing dataset,
exports fitted preprocessor + 3 model .pkl files into models/ directory.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bank.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

NUMERIC_FEATURES = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]
TARGET = "deposit"
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data
    print("=" * 60)
    print("BANK MARKETING - MODEL TRAINING PIPELINE")
    print("=" * 60)
    data = pd.read_csv(DATA_PATH)
    print(f"\n[OK] Data loaded: {data.shape[0]} rows x {data.shape[1]} cols")

    # 2. Encode target
    data["deposit_num"] = data[TARGET].map({"yes": 1, "no": 0})
    print(f"   Target distribution:\n{data['deposit_num'].value_counts().to_string()}")

    # 3. Split
    X = data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = data["deposit_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 4. Preprocessing - fit on X_train ONLY
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)
    print(f"   Preprocessed shape: {X_train_pre.shape}")

    # 5. SMOTE - on preprocessed training data ONLY
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)
    print(f"   After SMOTE: {X_train_res.shape[0]} samples (was {X_train_pre.shape[0]})")

    # 6. Define models
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    stacking = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=lr,
    )

    models = {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Stacking": stacking,
    }

    # 7. Train & Evaluate
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION")
    print("=" * 60)

    model_filenames = {
        "Random Forest": "model_rf.pkl",
        "Gradient Boosting": "model_gb.pkl",
        "Stacking": "model_stacking.pkl",
    }

    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_pre)
        y_proba = model.predict_proba(X_test_pre)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"   ROC-AUC:   {auc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1 Score:  {f1:.4f}")

        # Export model
        model_path = os.path.join(MODEL_DIR, model_filenames[name])
        joblib.dump(model, model_path)
        print(f"   [SAVED] {model_path}")

    # 8. Export preprocessor
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n[SAVED] Preprocessor -> {preprocessor_path}")

    # 9. Export feature names for downstream use
    feature_names = list(preprocessor.get_feature_names_out())
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    joblib.dump(feature_names, feature_names_path)
    print(f"[SAVED] Feature names -> {feature_names_path}")

    print("\n" + "=" * 60)
    print("[OK] ALL MODELS TRAINED AND EXPORTED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
