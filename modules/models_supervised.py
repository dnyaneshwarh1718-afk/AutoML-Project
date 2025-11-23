# modules/models_supervised.py
# ---------------------------------------------------
# Supervised AutoML (Classification + Regression)
# ---------------------------------------------------

import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .preprocess import build_supervised_preprocessor
from .utils import download_link_from_df


# =====================================================
# Model zoos
# =====================================================

def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "KNN Classifier": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(probability=True, random_state=42),
    }


def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(),
        "Support Vector Regressor": SVR(),
    }


def get_classification_param_grids():
    return {
        "Random Forest Classifier": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting Classifier": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5],
        },
        "KNN Classifier": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
        "Support Vector Classifier": {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "poly"],
        },
        "Logistic Regression": {
            "model__C": [0.1, 1, 10],
            "model__penalty": ["l2"],
        },
    }


def get_regression_param_grids():
    return {
        "Random Forest Regressor": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting Regressor": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5],
        },
        "KNN Regressor": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
        "Support Vector Regressor": {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "poly"],
        },
        "Linear Regression": {},
    }


# =====================================================
# Evaluation helpers
# =====================================================

def evaluate_classification_models(
    models, X_train, X_test, y_train, y_test, preprocessor, tune_hyperparams=False
):
    rows = []
    best_model = None
    best_f1 = -1.0
    param_grids = get_classification_param_grids()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start = time.time()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        if tune_hyperparams and name in param_grids and param_grids[name]:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grids[name],
                n_iter=10,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            train_time = time.time() - start
            y_pred = best_est.predict(X_test)
            used_pipe = best_est
        else:
            pipe.fit(X_train, y_train)
            train_time = time.time() - start
            y_pred = pipe.predict(X_test)
            used_pipe = pipe

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        rows.append(
            {
                "Model": name,
                "CV F1 Mean": cv_mean,
                "CV F1 Std": cv_std,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Train Time (s)": train_time,
                "Pipeline": used_pipe,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = used_pipe

    results_df = pd.DataFrame(rows).sort_values(by="F1 Score", ascending=False)
    best_row = results_df.iloc[0]
    return results_df, best_model, best_row


def evaluate_regression_models(
    models, X_train, X_test, y_train, y_test, preprocessor, tune_hyperparams=False
):
    rows = []
    best_model = None
    best_r2 = -1e9
    param_grids = get_regression_param_grids()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        start = time.time()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        if tune_hyperparams and name in param_grids and param_grids[name]:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grids[name],
                n_iter=10,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            train_time = time.time() - start
            y_pred = best_est.predict(X_test)
            used_pipe = best_est
        else:
            pipe.fit(X_train, y_train)
            train_time = time.time() - start
            y_pred = pipe.predict(X_test)
            used_pipe = pipe

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rows.append(
            {
                "Model": name,
                "CV R2 Mean": cv_mean,
                "CV R2 Std": cv_std,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2,
                "Train Time (s)": train_time,
                "Pipeline": used_pipe,
            }
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model = used_pipe

    results_df = pd.DataFrame(rows).sort_values(by="R2 Score", ascending=False)
    best_row = results_df.iloc[0]
    return results_df, best_model, best_row


# =====================================================
# Main AutoML entrypoint
# =====================================================

def run_supervised_automl(
    df_model: pd.DataFrame,
    target_col: str,
    problem_type: str,
    test_size: float,
    random_state: int,
    tune_hyperparams: bool,
    use_feature_selection: bool,
):
    """
    Full supervised AutoML:
      - Train/validation split
      - Preprocessor construction
      - Multiple models + CV
      - Optional hyperparameter tuning
      - Best model selection
      - Sample predictions and export
    """

    y = df_model[target_col]
    X = df_model.drop(columns=[target_col])

    # Train/test split (stratify for classification)
    enable_stratify = problem_type == "classification" and y.value_counts().min() > 1
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if enable_stratify else None,
    )

    # Build preprocessor (ColumnTransformer)
    preprocessor = build_supervised_preprocessor(X, use_feature_selection, problem_type)

    # For drift check later
    train_stats = X_train.describe().T

    # Choose models based on problem type
    if problem_type == "classification":
        models = get_classification_models()
        results_df, best_model, best_row = evaluate_classification_models(
            models,
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            tune_hyperparams=tune_hyperparams,
        )
        st.write("### Model Comparison (Classification)")
        display_cols = [
            "Model",
            "CV F1 Mean",
            "CV F1 Std",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Train Time (s)",
        ]
    else:
        models = get_regression_models()
        results_df, best_model, best_row = evaluate_regression_models(
            models,
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
            tune_hyperparams=tune_hyperparams,
        )
        st.write("### Model Comparison (Regression)")
        display_cols = [
            "Model",
            "CV R2 Mean",
            "CV R2 Std",
            "MAE",
            "RMSE",
            "R2 Score",
            "Train Time (s)",
        ]

    # Success banner
    if problem_type == "classification":
        st.success(
            f"🏆 Best model: **{best_row['Model']}** "
            f"(F1 = {best_row['F1 Score']:.4f}, Acc = {best_row['Accuracy']:.4f}, "
            f"CV F1 = {best_row['CV F1 Mean']:.4f} ± {best_row['CV F1 Std']:.4f})"
        )
    else:
        st.success(
            f"🏆 Best model: **{best_row['Model']}** "
            f"(R² = {best_row['R2 Score']:.4f}, RMSE = {best_row['RMSE']:.4f}, "
            f"CV R² = {best_row['CV R2 Mean']:.4f} ± {best_row['CV R2 Std']:.4f})"
        )

    # Predictions from best model
    y_pred = best_model.predict(X_test)
    pred_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
    st.subheader("📈 Sample Predictions (Best Model)")
    st.dataframe(pred_df.head(20))
    download_link_from_df(pred_df, filename=f"predictions_{problem_type}.csv")

    return (
        best_model,
        X_test,
        y_test,
        list(X.columns),
        train_stats,
        results_df[display_cols],
        best_row,
    )
