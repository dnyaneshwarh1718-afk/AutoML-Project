
import matplotlib.pyplot as plt

def shap_summary_tree(best_model, X_trans, feature_names):
    """Return a SHAP summary plot figure for tree-based models."""
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed. Please install with 'pip install shap'."

    model_step = best_model.named_steps.get("model", None)
    if model_step is None or not hasattr(model_step, "feature_importances_"):
        return None, "SHAP summary is only supported for tree-based models in this helper."

    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_trans)

    fig = plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_trans, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    return fig, None
