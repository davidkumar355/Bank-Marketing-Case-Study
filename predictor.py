"""
predictor.py — Agent 2 / Backend ML Engineer
Utility module for loading models, preprocessing inputs, and making predictions.
Used by app.py (Streamlit frontend).
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# FEATURE DEFINITIONS  (must match train_model.py exactly)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome"
]

# Valid values for categorical inputs
VALID_VALUES = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["yes", "no"],
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    "poutcome": ["failure", "other", "success", "unknown"],
}

# Numeric ranges for validation
NUMERIC_RANGES = {
    "age": (18, 95),
    "balance": (-8000, 100000),
    "day": (1, 31),
    "duration": (0, 5000),
    "campaign": (1, 50),
    "pdays": (-1, 999),
    "previous": (0, 275),
}

# Model file mapping
MODEL_FILES = {
    "Random Forest": "model_rf.pkl",
    "Gradient Boosting": "model_gb.pkl",
    "Stacking": "model_stacking.pkl",
}

# ---------------------------------------------------------------------------
# MODEL LOADING (cached at module level for Streamlit)
# ---------------------------------------------------------------------------
_model_cache = {}
_preprocessor_cache = None
_feature_names_cache = None


def load_preprocessor():
    """Load the fitted ColumnTransformer."""
    global _preprocessor_cache
    if _preprocessor_cache is None:
        path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Preprocessor not found at {path}. Run train_model.py first."
            )
        _preprocessor_cache = joblib.load(path)
    return _preprocessor_cache


def load_feature_names():
    """Load saved feature names."""
    global _feature_names_cache
    if _feature_names_cache is None:
        path = os.path.join(MODEL_DIR, "feature_names.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature names not found at {path}. Run train_model.py first."
            )
        _feature_names_cache = joblib.load(path)
    return _feature_names_cache


def load_model(model_name: str):
    """
    Load a trained model by name.
    model_name: 'Random Forest' | 'Gradient Boosting' | 'Stacking'
    """
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_FILES.keys())}"
        )

    if model_name not in _model_cache:
        filename = MODEL_FILES[model_name]
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at {path}. Run train_model.py first."
            )
        _model_cache[model_name] = joblib.load(path)

    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------------------------

def validate_input(input_dict: dict) -> dict:
    """
    Validate and sanitize all 16 input fields.
    Returns a cleaned copy. Raises ValueError on invalid inputs.
    """
    cleaned = {}

    # --- Categorical validation ---
    for feat in CATEGORICAL_FEATURES:
        val = input_dict.get(feat)
        if val is None:
            raise ValueError(f"Missing required field: '{feat}'")
        val = str(val).strip().lower()
        if val not in VALID_VALUES[feat]:
            raise ValueError(
                f"Invalid value '{val}' for '{feat}'. "
                f"Valid: {VALID_VALUES[feat]}"
            )
        cleaned[feat] = val

    # --- Numeric validation ---
    for feat in NUMERIC_FEATURES:
        val = input_dict.get(feat)
        if val is None:
            raise ValueError(f"Missing required field: '{feat}'")
        try:
            val = int(float(val))
        except (ValueError, TypeError):
            raise ValueError(f"'{feat}' must be a number, got '{val}'")

        lo, hi = NUMERIC_RANGES[feat]
        val = max(lo, min(hi, val))  # clamp to valid range
        cleaned[feat] = val

    return cleaned


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess_input(input_dict: dict):
    """
    Convert a validated input dict into a preprocessed array
    ready for model.predict / model.predict_proba.
    """
    cleaned = validate_input(input_dict)
    df = pd.DataFrame([cleaned])
    # Ensure column order matches training
    df = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    preprocessor = load_preprocessor()
    return preprocessor.transform(df)


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def _risk_tier(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    elif probability >= 0.40:
        return "Medium"
    else:
        return "Low"


def _confidence_label(probability: float) -> str:
    distance_from_boundary = abs(probability - 0.50)
    if distance_from_boundary >= 0.25:
        return "High"
    elif distance_from_boundary >= 0.10:
        return "Moderate"
    else:
        return "Low"


def predict(input_dict: dict, model_name: str = "Gradient Boosting") -> dict:
    """
    Run a full prediction pipeline:
      validate → preprocess → predict_proba → enrich result.

    Returns:
      {
        "probability": float,       # P(subscribe)
        "prediction": str,          # "Will Subscribe" / "Will Not Subscribe"
        "risk_tier": str,           # "High" / "Medium" / "Low"
        "confidence": str,          # "High" / "Moderate" / "Low"
        "campaign_efficiency": float,  # duration / campaign
      }
    """
    cleaned = validate_input(input_dict)
    X = preprocess_input(input_dict)
    model = load_model(model_name)

    proba = model.predict_proba(X)[0, 1]  # P(deposit=1)

    campaign_val = max(cleaned["campaign"], 1)  # avoid division by zero
    efficiency = round(cleaned["duration"] / campaign_val, 1)

    return {
        "probability": round(float(proba), 4),
        "prediction": "Will Subscribe" if proba >= 0.50 else "Will Not Subscribe",
        "risk_tier": _risk_tier(proba),
        "confidence": _confidence_label(proba),
        "campaign_efficiency": efficiency,
    }


# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def get_feature_importance(model_name: str = "Gradient Boosting") -> dict:
    """
    Return top 10 feature importances as {feature_name: importance}.
    Works for RF and GB (tree-based models with feature_importances_).
    For Stacking, falls back to Gradient Boosting.
    """
    # Stacking doesn't have feature_importances_ directly
    effective_name = model_name
    if model_name == "Stacking":
        effective_name = "Gradient Boosting"

    model = load_model(effective_name)
    feature_names = load_feature_names()
    importances = model.feature_importances_

    imp_dict = dict(zip(feature_names, importances))
    # Sort descending, return top 10
    sorted_imp = dict(
        sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    )
    return sorted_imp


# ---------------------------------------------------------------------------
# SHAP EXPLANATIONS
# ---------------------------------------------------------------------------

def get_shap_waterfall(input_dict: dict, model_name: str = "Gradient Boosting"):
    """
    Compute SHAP values for a single input using TreeExplainer.
    Falls back to GB for Stacking model.

    Returns:
      shap.Explanation object for waterfall_plot
    """
    effective_name = model_name
    if model_name == "Stacking":
        effective_name = "Gradient Boosting"

    model = load_model(effective_name)
    feature_names = load_feature_names()
    X = preprocess_input(input_dict)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats across model types & SHAP versions
    if isinstance(shap_values, list):
        # Older SHAP returns list of arrays per class
        sv = shap_values[1][0]  # class 1 (subscribe)
        base = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value)
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        sv = shap_values[0, :, 1]  # class 1 (subscribe)
        base = float(explainer.expected_value[1])
    else:
        sv = shap_values[0]
        base = float(explainer.expected_value) if not hasattr(explainer.expected_value, '__len__') else float(explainer.expected_value[0])

    # Get data values for display
    if hasattr(X, "toarray"):
        data_row = X.toarray()[0]
    else:
        data_row = np.asarray(X[0]).flatten()

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=data_row,
        feature_names=feature_names,
    )
    return explanation


# ---------------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE = 500  # guard against very large uploads


def predict_batch(df: pd.DataFrame, model_name: str = "Gradient Boosting") -> pd.DataFrame:
    """
    Run predictions on a DataFrame of customer records.
    Returns the input DataFrame with added prediction columns.
    Limits to MAX_BATCH_SIZE rows to prevent memory issues.
    """
    if len(df) > MAX_BATCH_SIZE:
        df = df.head(MAX_BATCH_SIZE)

    results = []
    for _, row in df.iterrows():
        try:
            input_dict = row.to_dict()
            result = predict(input_dict, model_name)
            results.append(result)
        except (ValueError, KeyError) as e:
            results.append({
                "probability": None,
                "prediction": f"Error: {str(e)[:50]}",
                "risk_tier": "N/A",
                "confidence": "N/A",
                "campaign_efficiency": None,
            })

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)
