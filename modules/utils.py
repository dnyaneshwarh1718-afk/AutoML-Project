import streamlit as st
import pandas as pd
import numpy as np


def download_link_from_df(df: pd.DataFrame, filename: str = "data.csv"):
    """Create a Streamlit download button for a dataframe."""
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def detect_problem_type(y: pd.Series | None) -> str:
    """Auto-detect classification vs regression from target."""
    if y is None:
        return "unsupervised"
    if pd.api.types.is_numeric_dtype(y):
        return "regression" if y.nunique() > 20 else "classification"
    return "classification"


def detect_leakage(X: pd.DataFrame, y: pd.Series | None):
    """Simple leakage checks: high correlation & identical columns."""
    warnings = []

    if y is None:
        return warnings

    # Numeric correlation checks
    if pd.api.types.is_numeric_dtype(y):
        num_X = X.select_dtypes(include=["int64", "float64", "int32", "float32"])
        if not num_X.empty:
            corr_with_target = num_X.apply(lambda col: col.corr(y))
            suspicious = corr_with_target[abs(corr_with_target) > 0.98]
            if not suspicious.empty:
                warnings.append(
                    f"High correlation (>0.98) with target for columns: {list(suspicious.index)}"
                )

    # Exact equality checks
    for col in X.columns:
        try:
            if X[col].equals(y):
                warnings.append(f"Column '{col}' is identical to target – possible leakage.")
        except Exception:
            continue

    return warnings
