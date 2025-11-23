import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
def _get_feature_importances_from_pipeline(model, X_test):
    """Extract true feature names after preprocessing inside a pipeline."""

    try:
        preprocessor = model.named_steps["preprocessor"]
    except KeyError:
        return None, None

    # --- Get transformed feature names ---
    # For sklearn >= 1.0
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    # --- Get underlying estimator ---
    final_model = model.named_steps["model"]

    # Works only for models that expose feature_importances_
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
    elif hasattr(final_model, "coef_"):
        importances = np.abs(final_model.coef_).flatten()
    else:
        return None, None

    # Ensure same length
    if feature_names is None or len(feature_names) != len(importances):
        # fallback to generic index
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    return feature_names, importances



# =========================================================
# MASTER FUNCTION CALLED FROM APP
# =========================================================
def show_supervised_diagnostics():

    if "best_model" not in st.session_state:
        st.warning("Train a supervised model first.")
        return

    model = st.session_state["best_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    problem_type = st.session_state["problem_type"]

    y_pred = model.predict(X_test)

    st.subheader("🔍 Model Diagnostics & Explainability")

    if problem_type == "regression":
        _show_regression_diagnostics(y_test, y_pred, X_test, model)
    else:
        _show_classification_diagnostics(y_test, y_pred, X_test, model)



# =========================================================
# 1️⃣ REGRESSION DIAGNOSTICS
# =========================================================
def _show_regression_diagnostics(y_test, y_pred, X_test, model):

    st.markdown("## 📈 Regression Diagnostics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    with col2:
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
    with col3:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    # ----------------------------------------
    # Actual vs Predicted
    # ----------------------------------------
    st.markdown("### 🎯 Actual vs Predicted")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------------------
    # Residual Plot
    # ----------------------------------------
    st.markdown("### 📉 Residuals vs Fitted")

    residuals = y_test - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    st.pyplot(fig)

    # ----------------------------------------
    # Residual Distribution
    # ----------------------------------------
    st.markdown("### 📦 Residual Distribution")

    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residual error")
    st.pyplot(fig)

    # ----------------------------------------
    # Feature Importance
    # ----------------------------------------
    if hasattr(model.named_steps["model"], "feature_importances_"):
        st.markdown("### 🌳 Feature Importance (Tree Models)")

        importances = model.named_steps["model"].feature_importances_
        feature_names = X_test.columns

        feature_names, importances = _get_feature_importances_from_pipeline(model, X_test)

    if feature_names is not None:
        imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        st.write("### 🔥 Feature Importances")
        st.dataframe(imp.head(20))
    else:
        st.info("Feature importance not available for this model.")


        
        imp = imp.sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(y=imp["Feature"], x=imp["Importance"], ax=ax)
        st.pyplot(fig)

    # ----------------------------------------
    # Custom User-Defined Visual
    # ----------------------------------------
    st.markdown("### 🎨 Custom Relationship Explorer")

    x_col = st.selectbox("Choose X", X_test.columns)
    fig, ax = plt.subplots()
    ax.scatter(X_test[x_col], y_test, alpha=0.4, label="Actual")
    ax.scatter(X_test[x_col], y_pred, alpha=0.4, label="Predicted")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Target")
    ax.legend()
    st.pyplot(fig)



# =========================================================
# 2️⃣ CLASSIFICATION DIAGNOSTICS
# =========================================================
def _show_classification_diagnostics(y_test, y_pred, X_test, model):

    st.markdown("## 📊 Classification Diagnostics")

    st.text(classification_report(y_test, y_pred))

    # ---------------- Confusion Matrix -----------------
    st.markdown("### 🧱 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # ---------------- ROC Curve -----------------
    if hasattr(model.named_steps["model"], "predict_proba"):
        st.markdown("### 🧪 ROC Curve")
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    # ---------------- PR Curve -----------------
    st.markdown("### 📌 Precision–Recall Curve")
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        st.pyplot(fig)

    # ---------------- Feature Importance -----------------
    if hasattr(model.named_steps["model"], "feature_importances_"):
        st.markdown("### 🌳 Feature Importance (Tree Model)")
        importance = model.named_steps["model"].feature_importances_
        feat = pd.DataFrame({"Feature": X_test.columns, "Importance": importance})
        feat = feat.sort_values("Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=feat, ax=ax)
        st.pyplot(fig)
