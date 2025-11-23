# modules/preprocess.py
# ---------------------------------------------------
# Industry-grade preprocessing for AutoML engine
# ---------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st


# =====================================================
# 1. HELPER: Datetime detection
# =====================================================

def _detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that are actual or likely datetime."""
    dt_cols: List[str] = []

    for col in df.columns:
        col_data = df[col]

        # already datetime
        if np.issubdtype(col_data.dtype, np.datetime64):
            dt_cols.append(col)
            continue

        # string-like
        if col_data.dtype == "object":
            sample = col_data.dropna().astype(str).head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="raise", infer_datetime_format=True)
                if parsed.notna().mean() > 0.8:
                    dt_cols.append(col)
            except Exception:
                pass

    # unique, preserve order
    return list(dict.fromkeys(dt_cols))


# =====================================================
# 2. RAW DATA CLEANING
# =====================================================

def preprocess_data(
    df_raw: pd.DataFrame,
    target_col: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    High-level cleaning BEFORE modeling & EDA.

    Steps:
      1) Drop duplicates
      2) Drop high-null columns (> 60% missing)
      3) Drop constant columns (zero variance)
      4) Detect & convert datetime columns
      5) Clip numeric outliers [1%, 99%]
      6) Compute skewness report
      7) For classification: simple class imbalance summary
    """

    df = df_raw.copy()

    # 1) Remove duplicates
    rows_before = len(df)
    df = df.drop_duplicates()
    rows_dropped_duplicates = rows_before - len(df)

    # 2) Drop high-null columns
    null_frac = df.isna().mean()
    high_null_cols = null_frac[null_frac > 0.60].index.tolist()
    df = df.drop(columns=high_null_cols, errors="ignore")

    # 3) Drop constant columns
    constant_cols = [
        c for c in df.columns
        if df[c].dropna().nunique() <= 1
    ]
    if constant_cols:
        df = df.drop(columns=constant_cols, errors="ignore")

    # 4) Datetime conversion
    dt_cols = _detect_datetime_columns(df)
    for c in dt_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        except Exception:
            # if fails, leave as original type
            pass

    # 5) Outlier clipping
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = df[col]
        if s.dropna().nunique() <= 1:
            continue
        df[col] = s.clip(s.quantile(0.01), s.quantile(0.99))

    # 6) Skewness report
    skewed_cols = []
    if num_cols:
        skew_vals = df[num_cols].skew(numeric_only=True)
        skewed_cols = skew_vals[skew_vals.abs() > 1].index.tolist()

    # 7) Simple class imbalance report
    class_imbalance = None
    if target_col and problem_type == "classification" and target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        ratios = (vc / len(df)).round(4).to_dict()
        min_ratio = float(vc.min() / len(df)) if len(df) > 0 else 0.0

        class_imbalance = {
            "target": target_col,
            "class_distribution": ratios,
            "min_class_ratio": min_ratio,
            "imbalance_flag": "severe" if min_ratio < 0.1 else "ok",
        }

    prep_report: Dict[str, Any] = {
        "final_shape": df.shape,
        "rows_dropped_duplicates": int(rows_dropped_duplicates),
        "dropped_high_null_cols": high_null_cols,
        "dropped_constant_cols": constant_cols,
        "outlier_clipping": f"Clipped numeric columns to [1%, 99%] for: {num_cols}",
        "skewed_numeric_cols_abs_gt_1": skewed_cols,
        "class_imbalance": class_imbalance,
    }

    return df, prep_report


# =====================================================
# 3. SUPERVISED PREPROCESSOR (Pipeline)
# =====================================================

def build_supervised_preprocessor(
    X: pd.DataFrame,
    use_feature_selection: bool = False,
    problem_type: Optional[str] = None,
) -> ColumnTransformer:
    """
    Build ColumnTransformer for supervised ML.

    - Datetime → numeric timestamps
    - Numeric: median impute + StandardScaler
    - Categorical: most_frequent impute + OneHotEncoder
    """

    X = X.copy()

    # Datetime → numeric
    dt_cols = _detect_datetime_columns(X)
    for c in dt_cols:
        try:
            X[c] = pd.to_datetime(X[c], errors="coerce", infer_datetime_format=True)
            X[c] = X[c].astype("int64") // 10**9
        except Exception:
            X[c] = X[c].astype("object")

    numeric_features = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    # NOTE: use_feature_selection can later be used by wrapping this in a Pipeline
    return preprocessor


# =====================================================
# 4. UNSUPERVISED MATRIX BUILDER
# =====================================================

def build_unsupervised_matrix(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build numeric feature matrix for unsupervised learning.
    """

    df = df.copy()
    if exclude_cols:
        df = df.drop(columns=exclude_cols, errors="ignore")

    dt_cols = _detect_datetime_columns(df)
    for c in dt_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = df[c].astype("int64") // 10**9
        except Exception:
            df[c] = df[c].astype("object")

    numeric_features = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    categorical_features = [c for c in df.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        n_jobs=-1,
    )

    X_unsup = col_transformer.fit_transform(df)

    try:
        feature_names = col_transformer.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_unsup.shape[1])]

    return X_unsup, feature_names


# =====================================================
# 5. TRANSFORMED FEATURE REPORT (for UI)
# =====================================================

def generate_feature_report(preprocessor) -> Optional[pd.DataFrame]:
    """Builds a structured DataFrame describing transformed features."""
    if preprocessor is None:
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = []

    # Raw groups from ColumnTransformer
    try:
        numeric_cols = preprocessor.transformers_[0][2]
        categorical_cols = preprocessor.transformers_[1][2]
    except Exception:
        return None

    rows = []

    # Numeric columns
    for col in numeric_cols:
        rows.append(
            {
                "Raw Column": col,
                "Raw Type": "Numeric",
                "Transformation": "MedianImputer + StandardScaler",
                "Output Columns": col,
            }
        )

    # Categorical columns (OneHot)
    for col in categorical_cols:
        ohe_out = [f for f in feature_names if f.startswith(col + "_")]
        rows.append(
            {
                "Raw Column": col,
                "Raw Type": "Categorical",
                "Transformation": "MostFreqImputer + OneHotEncoder",
                "Output Columns": ", ".join(ohe_out) if ohe_out else "(encoded)",
            }
        )

    return pd.DataFrame(rows)


def export_feature_report(df_report: Optional[pd.DataFrame], key_prefix: str = "feat"):
    """Provides download buttons for CSV + Excel (safe with unique keys)."""
    if df_report is None or df_report.empty:
        st.warning("Feature report is empty – nothing to download.")
        return

    # ---------- CSV ----------
    st.download_button(
        label="📥 Download Feature Report (CSV)",
        data=df_report.to_csv(index=False),
        file_name="feature_report.csv",
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )

    # ---------- EXCEL ----------
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_report.to_excel(writer, index=False, sheet_name="FeatureReport")

    st.download_button(
        label="📥 Download Feature Report (Excel)",
        data=buffer.getvalue(),
        file_name="feature_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"{key_prefix}_xlsx",
    )
