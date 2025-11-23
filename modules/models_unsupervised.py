import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .preprocess import build_unsupervised_matrix
from .utils import download_link_from_df


def run_unsupervised_automl(df_model: pd.DataFrame, target_col: str | None):
    st.subheader("🧩 Unsupervised AutoML – Clustering, Anomalies & Rules")

    exclude_cols = [target_col] if target_col else None
    X_unsup, feature_names = build_unsupervised_matrix(df_model, exclude_cols=exclude_cols)
    st.write(f"Using {X_unsup.shape[0]} samples × {X_unsup.shape[1]} features for unsupervised learning.")

    # -------- Clustering AutoML --------
    st.markdown("### 🧱 Clustering AutoML")
    max_k = st.slider("Max clusters (for KMeans / GMM / Agglomerative)", 3, 15, 8, 1)
    run_cluster = st.button("🚀 Run Clustering AutoML", key="run_cluster")

    if run_cluster:
        rows = []
        best_cluster_labels = None
        best_model_name = None
        best_sil = -1

        for algo in ["KMeans", "Agglomerative", "GMM"]:
            for k in range(2, max_k + 1):
                try:
                    if algo == "KMeans":
                        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
                        labels = model.fit_predict(X_unsup)
                    elif algo == "Agglomerative":
                        model = AgglomerativeClustering(n_clusters=k)
                        labels = model.fit_predict(X_unsup)
                    else:  # GMM
                        model = GaussianMixture(n_components=k, random_state=42)
                        labels = model.fit_predict(X_unsup)

                    sil = silhouette_score(X_unsup, labels)
                    ch = calinski_harabasz_score(X_unsup, labels)
                    db = davies_bouldin_score(X_unsup, labels)

                    rows.append(
                        {
                            "Algorithm": algo,
                            "k": k,
                            "Silhouette": sil,
                            "Calinski-Harabasz": ch,
                            "Davies-Bouldin": db,
                        }
                    )

                    if sil > best_sil:
                        best_sil = sil
                        best_cluster_labels = labels
                        best_model_name = f"{algo} (k={k})"
                except Exception:
                    continue

        # DBSCAN (no k)
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X_unsup)
            if len(set(labels)) > 1 and -1 in labels and len(set(labels)) > 2:
                sil = silhouette_score(X_unsup[labels != -1], labels[labels != -1])
                ch = calinski_harabasz_score(X_unsup[labels != -1], labels[labels != -1])
                db = davies_bouldin_score(X_unsup[labels != -1], labels[labels != -1])

                rows.append(
                    {
                        "Algorithm": "DBSCAN",
                        "k": None,
                        "Silhouette": sil,
                        "Calinski-Harabasz": ch,
                        "Davies-Bouldin": db,
                    }
                )
                if sil > best_sil:
                    best_sil = sil
                    best_cluster_labels = labels
                    best_model_name = "DBSCAN"
        except Exception:
            pass

        if not rows:
            st.error("Clustering failed – possibly due to too few samples or constant features.")
        else:
            cluster_results = pd.DataFrame(rows).sort_values(by="Silhouette", ascending=False)
            st.write("### Clustering Model Comparison (higher Silhouette is better)")
            st.dataframe(cluster_results)

            st.success(f"🏆 Best clustering: **{best_model_name}** (Silhouette = {best_sil:.4f})")

            st.session_state["cluster_labels"] = best_cluster_labels
            cluster_df = df_model.copy()
            cluster_df["Cluster"] = best_cluster_labels
            st.subheader("📂 Sample of Data with Cluster Assignments")
            st.dataframe(cluster_df.head(20))
            download_link_from_df(cluster_df, filename="clustered_data.csv")

            # Visualizations
            show_cluster_visualizations(X_unsup, best_cluster_labels, best_model_name)

    # -------- Anomaly Detection --------
    st.markdown("### 🚨 Anomaly Detection")
    run_anom = st.button("Run Anomaly Detection", key="run_anom")
    if run_anom:
        run_anomaly_detection(df_model, X_unsup)

    # -------- Association Rules --------
    st.markdown("### 🔗 Association Rules (Apriori)")
    with st.expander("Run Association Rules (for transactional / categorical data)", expanded=False):
        run_association_rules(df_model, target_col)


def show_cluster_visualizations(X_unsup, labels, best_model_name: str):
    st.markdown("### 📉 Cluster Visualizations (PCA / t-SNE)")

    vis_type = st.selectbox(
        "Visualization Type",
        ["PCA 2D", "PCA 3D", "t-SNE 2D", "t-SNE 3D"],
        key="cluster_vis",
    )

    if vis_type.startswith("PCA"):
        n_components = 2 if "2D" in vis_type else 3
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_unsup)

        if n_components == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.8)
            ax.set_title(f"PCA 2D – {best_model_name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(
                X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap="tab10", alpha=0.8
            )
            ax.set_title(f"PCA 3D – {best_model_name}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            fig.colorbar(p, ax=ax, label="Cluster")
            st.pyplot(fig)

    else:
        # t-SNE
        n_components = 2 if "2D" in vis_type else 3
        tsne = TSNE(
            n_components=n_components,
            learning_rate="auto",
            init="random",
            random_state=42,
            perplexity=min(30, max(5, X_unsup.shape[0] // 5)),
        )
        X_tsne = tsne.fit_transform(X_unsup)

        if n_components == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", alpha=0.8)
            ax.set_title(f"t-SNE 2D – {best_model_name}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(
                X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=labels, cmap="tab10", alpha=0.8
            )
            ax.set_title(f"t-SNE 3D – {best_model_name}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            fig.colorbar(p, ax=ax, label="Cluster")
            st.pyplot(fig)


def run_anomaly_detection(df_model: pd.DataFrame, X_unsup: np.ndarray):
    methods = {
        "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.05),
        "One-Class SVM": OneClassSVM(nu=0.05, kernel="rbf", gamma="scale"),
    }

    anom_results = {}
    for name, model in methods.items():
        try:
            if name == "Local Outlier Factor":
                labels = model.fit_predict(X_unsup)
                scores = -model.negative_outlier_factor_
            else:
                model.fit(X_unsup)
                labels = model.predict(X_unsup)
                scores = -model.score_samples(X_unsup) if hasattr(model, "score_samples") else None

            labels_norm = np.where(labels == -1, 1, 0)
            anom_results[name] = (labels_norm, scores)
        except Exception:
            continue

    if not anom_results:
        st.error("Anomaly detection failed.")
        return

    for name, (labels, scores) in anom_results.items():
        n_anom = labels.sum()
        st.write(f"#### {name}")
        st.write(f"Flagged anomalies: {n_anom} ({100 * n_anom / len(labels):.2f}% of data)")

    # Pick Isolation Forest if present for visualization
    if "Isolation Forest" in anom_results:
        labels, _ = anom_results["Isolation Forest"]
        vis_name = "Isolation Forest"
    else:
        vis_name, (labels, _) = next(iter(anom_results.items()))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_unsup)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="coolwarm", alpha=0.8
    )
    ax.set_title(f"Anomalies (red) vs Normal (blue) – {vis_name} + PCA 2D")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Anomaly Flag")
    st.pyplot(fig)

    anom_df = df_model.copy()
    anom_df["Anomaly"] = labels
    st.subheader("Sample of Data with Anomaly Flag")
    st.dataframe(anom_df.head(20))
    download_link_from_df(anom_df, filename="anomaly_flags.csv")


def run_association_rules(df_model: pd.DataFrame, target_col: str | None):
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        st.info("Install mlxtend to enable association rules: `pip install mlxtend`")
        return

    cat_cols = [
        c
        for c in df_model.columns
        if (df_model[c].dtype == "object" or df_model[c].nunique() < 15)
        and c != target_col
    ]

    if not cat_cols:
        st.warning("No suitable categorical/binary columns found for association rules.")
        return

    st.write(f"Using columns: {cat_cols}")
    df_cat = df_model[cat_cols].astype(str)
    df_encoded = pd.get_dummies(df_cat)

    min_support = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)

    if st.button("Run Apriori & Association Rules", key="run_rules"):
        freq = apriori(df_encoded, min_support=min_support, use_colnames=True)
        if freq.empty:
            st.warning("No frequent itemsets found with this support threshold.")
            return

        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        if rules.empty:
            st.warning("No rules satisfy the confidence threshold.")
            return

        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules = rules.sort_values(by="lift", ascending=False)
        st.write("Top Association Rules (sorted by lift)")
        st.dataframe(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(50)
        )
        download_link_from_df(rules, filename="association_rules.csv")
