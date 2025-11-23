# modules/drift.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def _compare_distributions(train_series, new_series, feature_name):
    """
    Compare two distributions visually + compute PSI (Population Stability Index)
    """

    def calculate_psi(expected, actual, buckets=10):
        """Population Stability Index."""
        expected = expected.dropna()
        actual = actual.dropna()

        quantiles = np.linspace(0, 1, buckets + 1)
        bins = np.unique(np.quantile(expected, quantiles))

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_perc = expected_counts / len(expected)
        actual_perc = actual_counts / len(actual)

        psi = np.sum(
            (actual_perc - expected_perc) * np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
        )
        return psi

    psi_value = calculate_psi(train_series, new_series)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(train_series, label="Train", linewidth=2, ax=ax)
    sns.kdeplot(new_series, label="New Batch", linewidth=2, ax=ax)
    ax.set_title(f"Distribution Drift — {feature_name}")
    ax.legend()

    return psi_value, fig


def run_drift_check(train_stats: pd.DataFrame, new_df: pd.DataFrame):
    """
    Main drift detection function.
    Compares training feature distributions vs. new batch data.
    """

    st.write("### 📉 Drift Monitoring — Train vs New Batch")

    # Convert index of train_stats = feature names
    features = train_stats.index.tolist()
    numeric_features = [
        f for f in features if np.issubdtype(train_stats.loc[f, "mean"].dtype, np.number)
        or "mean" in train_stats.columns
    ]

    drift_results = []

    for col in numeric_features:
        if col not in new_df.columns:
            continue

        train_series = train_stats.loc[col]
        new_series = new_df[col]

        # Full training distribution is not stored — only stats
        # So we approximate using mean ± std normal distribution
        if isinstance(train_series, pd.Series) and "mean" in train_stats.columns:
            synthetic_train = np.random.normal(
                loc=train_stats.loc[col, "mean"],
                scale=train_stats.loc[col, "std"] if train_stats.loc[col, "std"] > 0 else 1,
                size=len(new_series),
            )
        else:
            synthetic_train = new_series  # fallback

        psi_value, fig = _compare_distributions(
            pd.Series(synthetic_train),
            new_series,
            col,
        )

        drift_results.append({"Feature": col, "PSI": psi_value})

        st.pyplot(fig)

        if psi_value < 0.1:
            st.success(f"PSI {psi_value:.3f} — No drift detected for **{col}**")
        elif psi_value < 0.25:
            st.warning(f"PSI {psi_value:.3f} — Mild drift detected for **{col}**")
        else:
            st.error(f"PSI {psi_value:.3f} — ⚠ Significant drift detected for **{col}**")

        st.markdown("---")

    st.subheader("📄 Drift Summary (PSI Table)")
    st.dataframe(pd.DataFrame(drift_results))
