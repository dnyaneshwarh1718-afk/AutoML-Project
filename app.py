# app.py
# ---------------------------------------------------
# End-to-end AutoML Engine – Enterprise Grade
# ---------------------------------------------------

import io
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Global Plot Style
# -------------------------------
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

# -------------------------------
# Internal Modules
# -------------------------------
from modules.utils import (
    download_link_from_df,
    detect_problem_type,
    detect_leakage,
)
from modules.preprocess import (
    preprocess_data,
    generate_feature_report,
    export_feature_report,
)
from modules.eda import show_eda_tab
from modules.models_supervised import run_supervised_automl
from modules.models_unsupervised import run_unsupervised_automl
from modules.visualization_dashboard import show_visual_dashboard
from modules.diagnostics import show_supervised_diagnostics
from modules.drift import run_drift_check
from modules.ols_report import show_ols_report


# =========================================================
# MAIN APP
# =========================================================
def main():
    st.set_page_config(page_title="Full AutoML Engine (Pro)", layout="wide")

    st.title("🤖 End-to-End AutoML – From Problem to Production")

    st.write(
        """
        **End-to-end ML workflow this app follows:**

        1️⃣ Problem & business understanding  
        2️⃣ Data upload & RAW EDA (+ smart Power BI-style visuals)  
        3️⃣ Auto decision: **Do you have a target/label?**  
           - ✅ Yes → **Supervised** (Regression / Classification)  
           - ❌ No → **Unsupervised** (Clustering / Anomaly / Rules)  
        4️⃣ Cleaning & preprocessing  
        5️⃣ AutoML training & evaluation (multiple algorithms + CV)  
        6️⃣ Visualization & diagnostics (incl. OLS for regression)  
        7️⃣ Deployment & drift monitoring  
        """
    )

    # =====================================================
    # DATA UPLOAD
    # =====================================================
    st.sidebar.header("📂 Data Upload")
    file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if file is None:
        st.info("👈 Upload a dataset to start the AutoML workflow.")
        return

    df_raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    st.success(f"✅ Loaded dataset: **{df_raw.shape[0]} rows × {df_raw.shape[1]} columns**")

    # =====================================================
    # SUPERVISED vs UNSUPERVISED DECISION
    # =====================================================
    st.sidebar.header("🎯 Supervised or Unsupervised")

    mode = st.sidebar.radio(
        "Do you have a target/label to predict?",
        [
            "Yes – I have a target column (Supervised)",
            "No – Only patterns (Unsupervised only)",
        ],
        index=0,
    )

    target_col = None
    problem_type = None

    if mode.startswith("Yes"):
        target_col = st.sidebar.selectbox(
            "Select target column (y)",
            df_raw.columns,
            help="Dependent variable you want to predict.",
        )
        y_raw = df_raw[target_col]

        auto_detected = detect_problem_type(y_raw)
        choice = st.sidebar.radio(
            "Problem Type",
            ["Auto", "Classification", "Regression"],
            index=0,
            help=f"Auto-detected from target: **{auto_detected}**",
        )

        if choice == "Auto":
            problem_type = auto_detected
        else:
            problem_type = choice.lower()
    else:
        st.sidebar.info("Unsupervised mode selected – app will only run clustering/anomaly/rules.")
        y_raw = None
        problem_type = "unsupervised"

    st.sidebar.caption(f"Current learning mode: **{problem_type.upper()}**")

    # Train/validation options (used only for supervised)
    test_size = st.sidebar.slider("Validation Size (test split)", 0.1, 0.4, 0.2)
    random_state = st.sidebar.number_input("Random State", 0, 9999, 42)

    st.sidebar.header("⚙️ Advanced Options")
    tune_hyperparams = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True)
    use_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=False)

    # =====================================================
    # 1️⃣ PROBLEM / BUSINESS UNDERSTANDING (DOC SECTION)
    # =====================================================
    with st.expander("1️⃣ Problem Definition & Business Understanding", expanded=True):
        if target_col:
            if problem_type == "classification":
                detected_problem_display = "Classification"
            elif problem_type == "regression":
                detected_problem_display = "Regression"
            else:
                detected_problem_display = "Supervised (unspecified)"
        else:
            detected_problem_display = "Unsupervised (general)"

        problem_type_options = [
            "Classification",
            "Regression",
            "Clustering / Segmentation",
            "Anomaly Detection",
            "Unsupervised (general)",
        ]

        problem_type_display = st.selectbox(
            "Problem Type (auto-detected, editable for documentation)",
            options=problem_type_options,
            index=problem_type_options.index(detected_problem_display),
        )

        if problem_type_display == "Classification":
            metric_options = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Business KPI"]
            default_metric = "F1-score"
        elif problem_type_display == "Regression":
            metric_options = ["RMSE", "MAE", "R²", "MAPE", "Business KPI"]
            default_metric = "RMSE"
        else:
            metric_options = ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin", "Business KPI"]
            default_metric = "Silhouette Score"

        primary_metric = st.selectbox(
            "Primary success metric (for this project)",
            options=metric_options,
            index=metric_options.index(default_metric),
        )

        if problem_type_display in ["Classification", "Regression"]:
            default_constraints = (
                "Latency < 500 ms, explainable model, supports both batch and API predictions."
            )
        else:
            default_constraints = (
                "Must handle high dimensional data, robust to noise, interpretable segments/clusters."
            )

        constraints = st.text_area(
            "Constraints (data, latency, deployment, explainability, etc.)",
            value=default_constraints,
            height=80,
        )

        st.caption("This section is for documentation like a real DS project – it does not change training directly.")

    # =====================================================
    # 2️⃣–4️⃣ ENTERPRISE PREPROCESSING
    # =====================================================
    df_clean, prep_report = preprocess_data(
        df_raw,
        target_col=target_col,
        problem_type=problem_type if problem_type in ["classification", "regression"] else None,
    )

    # Keep cleaned data in session for things like OLS if needed later
    st.session_state["df_clean"] = df_clean
    st.session_state["target_col"] = target_col
    st.session_state["problem_type"] = problem_type

    # =====================================================
    # TABS (FULL WORKFLOW)
    # =====================================================
    tabs = st.tabs(
        [
            "📊 EDA + Smart Visuals",
            "🧹 Cleaning Summary",
            "🧠 Supervised AutoML",
            "🧩 Unsupervised AutoML",
            "🔬 Diagnostics",
            "🚀 Deployment & Drift",
        ]
    )

    # -----------------------------------------------------
    # TAB 1 – EDA + Smart Visuals
    # -----------------------------------------------------
    with tabs[0]:
        st.subheader("📊 Data Understanding – EDA + Smart Dashboard")

        st.markdown("### 📝 Raw Data Preview")
        st.dataframe(df_raw.head())

        if target_col:
            st.markdown("### 🎯 Feature / Target Roles")
            feat_cols = [c for c in df_raw.columns if c != target_col]
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Dependent variable (y)**")
                st.code(target_col)
                st.dataframe(df_raw[[target_col]].head())
            with c2:
                st.write("**Independent variables (X)**")
                st.write(feat_cols)

        st.markdown("### 📈 Classical EDA (Raw Data)")
        show_eda_tab(df_raw, target_col)

        st.markdown("---")
        st.markdown("### 📊 Smart Visualizer – Power BI Style (Cleaned Data)")
        st.caption(
            "These charts use the **cleaned dataset** that will be used for modeling. "
            "You can explore distributions, relationships, correlations and time-series."
        )
        show_visual_dashboard(df_clean, target_col)

    # -----------------------------------------------------
    # TAB 2 – Cleaning Summary
    # -----------------------------------------------------
    with tabs[1]:
        st.subheader("🧹 Enterprise-Grade Cleaning Summary")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Original shape**: {df_raw.shape}")
            st.write(f"**Cleaned shape**: {prep_report['final_shape']}")
            st.write(f"Duplicates removed: {prep_report['rows_dropped_duplicates']}")
        with c2:
            st.write("Dropped high-null columns (>60% missing):")
            st.write(prep_report["dropped_high_null_cols"] or "None")
            st.write("Dropped constant columns (zero variance):")
            st.write(prep_report["dropped_constant_cols"] or "None")

        st.markdown("### 📈 Skewed Numeric Columns (|skew| > 1)")
        skewed_cols = prep_report["skewed_numeric_cols_abs_gt_1"]
        if skewed_cols:
            st.write(skewed_cols)
        else:
            st.write("None above threshold.")

        if problem_type == "classification" and prep_report["class_imbalance"]:
            st.markdown("### ⚠️ Class Imbalance Summary")
            st.json(prep_report["class_imbalance"])
        else:
            st.info("No class imbalance reported or not a classification problem.")

        st.markdown("### 🧱 Cleaned Data Preview")
        st.dataframe(df_clean.head())

        if target_col:
            st.markdown("### 🕵️ Data Leakage Checks (Cleaned Data)")
            X_ = df_clean.drop(columns=[target_col])
            y_ = df_clean[target_col]
            leakage = detect_leakage(X_, y_)
            if leakage:
                for w in leakage:
                    st.error(f"🚨 Possible leakage: {w}")
            else:
                st.success("No obvious leakage patterns found (basic checks).")
        else:
            st.info("Leakage checks are only relevant for supervised problems.")

    # -----------------------------------------------------
    # TAB 3 – Supervised AutoML
    # -----------------------------------------------------
    with tabs[2]:
        st.subheader("🧠 Supervised AutoML – Training & Evaluation")

        if target_col is None:
            st.warning("Select a target column in the sidebar to enable supervised training.")
        elif problem_type not in ["classification", "regression"]:
            st.warning("Problem type must be classification or regression for supervised AutoML.")
        else:
            st.write(
                f"Task: **{problem_type.upper()}** "
                f"(target: `{target_col}` | features: {len(df_clean.columns) - 1} columns)"
            )

            if st.button("🚀 Run Supervised AutoML", key="run_automl"):
                (
                    best_model,
                    X_test,
                    y_test,
                    X_cols,
                    train_stats,
                    results_df,
                    best_row,
                ) = run_supervised_automl(
                    df_clean,
                    target_col,
                    problem_type,
                    test_size,
                    random_state,
                    tune_hyperparams,
                    use_feature_selection,
                )

                # Save to session for diagnostics & deployment
                st.session_state["best_model"] = best_model
                st.session_state["X_test"] = X_test
                st.session_state["y_test"] = y_test
                st.session_state["X_cols"] = X_cols
                st.session_state["train_stats"] = train_stats
                st.session_state["problem_type"] = problem_type
                st.session_state["target_col"] = target_col

                st.markdown("### 📊 Model Comparison Table")
                st.dataframe(results_df)

                # =============================
                # 🧬 Feature Transformation Report
                # =============================
                st.markdown("### 🧬 Feature Transformation Report (from Best Model Preprocessor)")
                try:
                    preprocessor = best_model.named_steps.get("preprocessor", None)
                    df_report = generate_feature_report(preprocessor)
                    if df_report is not None:
                        st.dataframe(df_report, use_container_width=True)
                        export_feature_report(df_report, key_prefix="featrep")
                    else:
                        st.info("Could not extract feature mapping from preprocessor.")
                except Exception as e:
                    st.warning(f"Feature report not available: {e}")

                from modules.ols_report import run_ols_with_pipeline

                if problem_type == "regression":
                    st.subheader("📘 Full OLS Regression Report (Statsmodels)")
                    with st.expander("Show OLS summary", expanded=False):

                        try:
                            X_ols = df_clean.drop(columns=[target_col])
                            y_ols = df_clean[target_col]
                            best_model = st.session_state["best_model"]

                            run_ols_with_pipeline(best_model, X_ols, y_ols)

                        except Exception as e:
                            st.error(f"OLS could not run: {e}")


    # -----------------------------------------------------
    # TAB 4 – Unsupervised AutoML
    # -----------------------------------------------------
    with tabs[3]:
        st.subheader("🧩 Unsupervised AutoML – Clustering / Anomaly / Rules")
        run_unsupervised_automl(df_clean, target_col)

    # -----------------------------------------------------
    # TAB 5 – Diagnostics
    # -----------------------------------------------------
    with tabs[4]:
        st.subheader("🔬 Diagnostics & Explainability (Best Supervised Model)")
        if "best_model" not in st.session_state:
            st.info("Train a supervised model first in the **Supervised AutoML** tab.")
        else:
            show_supervised_diagnostics()

    # -----------------------------------------------------
    # TAB 6 – Deployment & Drift
    # -----------------------------------------------------
    with tabs[5]:
        st.subheader("🚀 Deployment – Batch Prediction & Drift Monitoring")

        if "best_model" not in st.session_state:
            st.info("Train a supervised model first in the **Supervised AutoML** tab.")
        else:
            best_model = st.session_state["best_model"]
            train_cols = st.session_state["X_cols"]
            train_stats = st.session_state["train_stats"]

            st.markdown("### 📦 Batch Prediction on New Data")
            new_file = st.file_uploader(
                "Upload new data file (same structure as training features X)",
                type=["csv", "xlsx"],
                key="new_data_for_deploy",
            )

            if new_file is not None:
                new_df = (
                    pd.read_csv(new_file)
                    if new_file.name.endswith(".csv")
                    else pd.read_excel(new_file)
                )

                st.write("Preview of uploaded batch:")
                st.dataframe(new_df.head())

                missing_cols = [c for c in train_cols if c not in new_df.columns]
                extra_cols = [c for c in new_df.columns if c not in train_cols]

                if missing_cols:
                    st.error(f"Missing columns compared to training data: {missing_cols}")
                else:
                    if extra_cols:
                        st.warning(f"Extra columns will be ignored: {extra_cols}")
                        new_df = new_df[train_cols]

                    preds = best_model.predict(new_df)
                    pred_df = new_df.copy()
                    pred_df["Prediction"] = preds

                    st.subheader("Predictions (first 20 rows)")
                    st.dataframe(pred_df.head(20))
                    download_link_from_df(pred_df, filename="batch_predictions.csv")

                    # Drift check
                    if train_stats is not None:
                        st.markdown("### 🔁 Data Drift Check (Train vs New Batch)")
                        run_drift_check(train_stats, new_df)
                    else:
                        st.info("No stored training statistics found – drift check skipped.")

            st.markdown("---")
            st.markdown("### 💾 Download Trained Model")
            if st.button("Prepare model for download", key="prep_model_download"):
                buffer = io.BytesIO()
                joblib.dump(best_model, buffer)
                buffer.seek(0)
                st.download_button(
                    "📥 Download trained_model.pkl",
                    buffer,
                    "trained_model.pkl",
                    mime="application/octet-stream",
                    key="download_model_btn",
                )

    st.info("✅ Workflow complete: Problem → Data → EDA + Smart Visuals → Cleaning → AutoML → Diagnostics → Deployment → Drift.")


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    main()
