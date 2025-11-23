import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def show_eda_tab(df: pd.DataFrame, target_col: str | None = None):
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # ---------------------------------------------------------------------
    # 1️⃣ DATA UNDERSTANDING
    # ---------------------------------------------------------------------
    st.markdown("## 1️⃣ Data Understanding")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Column overview
    st.markdown("### 🧱 Column Overview")
    info_df = (
        df.dtypes.to_frame("Dtype")
        .assign(
            **{
                "Missing %": (df.isna().mean() * 100).round(2),
                "Unique Values": df.nunique(),
            }
        )
        .reset_index(names="Column")
    )
    st.dataframe(info_df, use_container_width=True)

    # ---------------------------------------------------------------------
    # 2️⃣ DATA QUALITY
    # ---------------------------------------------------------------------
    st.markdown("## 2️⃣ Data Quality Check")

    st.write("### ⚠️ Missing Value Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    st.pyplot(fig)

    duplicates = df.duplicated().sum()
    st.write(f"**Duplicate Rows:** {duplicates}")

    # ---------------------------------------------------------------------
    # 3️⃣ UNIVARIATE ANALYSIS
    # ---------------------------------------------------------------------
    st.markdown("## 3️⃣ Univariate Analysis")

    with st.expander("📌 Summary Statistics", expanded=False):
        if numeric_cols:
            st.write("**Numeric Summary**")
            st.dataframe(df[numeric_cols].describe().T)
        if cat_cols:
            st.write("**Categorical Summary**")
            st.dataframe(df[cat_cols].describe().T)

    if numeric_cols:
        with st.expander("📦 Numeric Distribution"):
            col = st.selectbox("Select numeric column", numeric_cols, key="num_univariate")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    if cat_cols:
        with st.expander("🏷 Categorical Frequency"):
            col = st.selectbox("Select categorical column", cat_cols, key="cat_univariate")
            top_vals = df[col].value_counts().head(20)
            fig, ax = plt.subplots()
            sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax)
            ax.set_title(f"Top categories in {col}")
            st.pyplot(fig)

    # ---------------------------------------------------------------------
    # 4️⃣ BIVARIATE ANALYSIS
    # ---------------------------------------------------------------------
    st.markdown("## 4️⃣ Bivariate Analysis")

    if target_col and target_col in df.columns:
        if numeric_cols:
            with st.expander("📈 Numeric vs Target"):
                for col in numeric_cols:
                    if col != target_col:
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=df[col], y=df[target_col], ax=ax)
                        ax.set_title(f"{col} vs {target_col}")
                        st.pyplot(fig)

        if cat_cols:
            with st.expander("🗂 Categorical vs Target"):
                for col in cat_cols:
                    if col != target_col:
                        fig, ax = plt.subplots()
                        sns.boxplot(x=df[col], y=df[target_col], ax=ax)
                        ax.set_title(f"{col} impact on {target_col}")
                        st.pyplot(fig)

    # ---------------------------------------------------------------------
    # 5️⃣ OUTLIER ANALYSIS
    # ---------------------------------------------------------------------
    if numeric_cols:
        st.markdown("## 5️⃣ Outlier Detection (Boxplots)")
        with st.expander("📦 Boxplot for Outliers", expanded=False):
            col = st.selectbox("Select column for outlier check", numeric_cols, key="outlier_col")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outlier Detection for {col}")
            st.pyplot(fig)

    # ---------------------------------------------------------------------
    # 6️⃣ CORRELATION & MULTICOLLINEARITY
    # ---------------------------------------------------------------------
    if len(numeric_cols) >= 2:
        st.markdown("## 6️⃣ Correlation Analysis")

        with st.expander("🔥 Correlation Heatmap"):
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
            st.pyplot(fig)

    # ---------------------------------------------------------------------
    # 7️⃣ TARGET ANALYSIS
    # ---------------------------------------------------------------------
    if target_col and target_col in df.columns:
        st.markdown("## 7️⃣ Target Variable Analysis 🎯")

        y = df[target_col]

        if pd.api.types.is_numeric_dtype(y):
            fig, ax = plt.subplots()
            sns.histplot(y.dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of Target: {target_col}")
            st.pyplot(fig)
        else:
            freq = y.value_counts()
            st.write("Class distribution:")
            st.dataframe(freq.to_frame("count"))

