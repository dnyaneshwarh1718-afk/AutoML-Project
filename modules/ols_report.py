import streamlit as st
import pandas as pd
import statsmodels.api as sm

def show_ols_report(X, y):
    """
    Generates and displays a full OLS report using statsmodels.
    Works only for regression.
    """

    st.subheader("📘 Full OLS Regression Report")

    try:
        # Add constant term
        X_const = sm.add_constant(X)

        # Fit OLS model
        model = sm.OLS(y, X_const).fit()

        # Display summary
        st.text(model.summary())

    except Exception as e:
        st.error(f"OLS report failed: {e}")


import numpy as np


def run_ols_with_pipeline(best_model, X, y):
    """
    Runs OLS using the same preprocessing as the trained ML pipeline.
    Ensures all data is numeric and aligned with statsmodels requirements.
    """

    try:
        preprocessor = best_model.named_steps["preprocessor"]
    except:
        st.error("No preprocessor found in pipeline. OLS cannot run.")
        return

    # --- Transform X using preprocessor ---
    try:
        X_trans = preprocessor.fit_transform(X)
    except Exception as e:
        st.error(f"Preprocessor transform failed: {e}")
        return

    # --- Get transformed feature names ---
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    # --- Convert to DataFrame ---
    X_df = pd.DataFrame(X_trans, columns=feature_names)

    # --- Add constant ---
    X_df = sm.add_constant(X_df)

    # --- Fit OLS ---
    try:
        model = sm.OLS(y, X_df)
        results = model.fit()
        st.text(results.summary())
    except Exception as e:
        st.error(f"OLS failed: {e}")
