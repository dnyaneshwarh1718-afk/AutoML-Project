# =====================================================
# modules/smart_visualizer.py  (UPGRADED – DS LEVEL)
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# compact global styles
plt.rcParams["figure.figsize"] = (5.5, 3.0)
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7


# ------------------------------------------------------
# Smart Column Type Detection
# ------------------------------------------------------
def _detect_column_roles(df: pd.DataFrame, target_col: str | None = None):
    """Detect numeric, categorical, datetime columns using robust logic."""

    # numeric = int, float, bool
    num_cols = df.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()

    # detect datetime columns (dtype or name keyword)
    dt_cols = [
        c for c in df.columns
        if np.issubdtype(df[c].dtype, np.datetime64) or 
           any(k in c.lower() for k in ["date", "time", "year", "month"])
    ]

    # attempt to parse strings as datetime
    for col in df.columns:
        if col not in dt_cols and df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if parsed.notna().sum() > 0:
                    dt_cols.append(col)
            except:
                pass

    dt_cols = list(dict.fromkeys(dt_cols))

    # categoricals
    cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

    # put target first for convenience
    if target_col in num_cols:
        num_cols.insert(0, num_cols.pop(num_cols.index(target_col)))
    if target_col in cat_cols:
        cat_cols.insert(0, cat_cols.pop(cat_cols.index(target_col)))

    return num_cols, cat_cols, dt_cols


# ------------------------------------------------------
# Identify Prediction Column
# ------------------------------------------------------
def _find_prediction_column(df: pd.DataFrame, target_col: str | None):
    """Detect model predictions automatically."""

    if target_col and f"{target_col}_pred" in df.columns:
        return f"{target_col}_pred"

    for cand in ["Prediction", "prediction", "pred", "y_pred"]:
        if cand in df.columns:
            return cand

    return None


# ------------------------------------------------------
# MAIN POWER BI–STYLE DASHBOARD
# ------------------------------------------------------
def show_visual_dashboard(df: pd.DataFrame, target_col: str | None = None):

    st.subheader("📊 Smart Visualize — Power BI Dashboard")

    if df.empty:
        st.warning("Dataset is empty — no visualizations available.")
        return

    num_cols, cat_cols, dt_cols = _detect_column_roles(df, target_col)
    pred_col = _find_prediction_column(df, target_col)

    # ======================================================
    # TOP KPI CARDS (Overview)
    # ======================================================
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Rows", f"{len(df):,}")
    with k2: st.metric("Numeric Features", len(num_cols))
    with k3: st.metric("Categorical Features", len(cat_cols))

    # ======================================================
    # GLOBAL FILTERS
    # ======================================================
    with st.expander("🔍 Global Filters", expanded=False):
        df_vis = df.copy()

        # Categorical filter
        if cat_cols:
            fcol = st.selectbox("Filter by category", ["(none)"] + cat_cols)
            if fcol != "(none)":
                vals = sorted(df[fcol].dropna().unique())
                sel = st.multiselect("Values", vals, default=vals[:3])
                if sel:
                    df_vis = df_vis[df_vis[fcol].isin(sel)]

        # Numeric range filter
        if num_cols:
            fnum = st.selectbox("Numeric range filter", ["(none)"] + num_cols)
            if fnum != "(none)":
                mn, mx = df_vis[fnum].min(), df_vis[fnum].max()
                rmin, rmax = st.slider(f"{fnum} range", min_value=float(mn),
                                       max_value=float(mx),
                                       value=(float(mn), float(mx)))
                df_vis = df_vis[(df_vis[fnum] >= rmin) & (df_vis[fnum] <= rmax)]

    st.caption("All charts below adapt automatically based on your data + filters.")

    # ======================================================
    # ROW 1 — Distribution • Box/Violin • Category Breakdown
    # ======================================================
    c1, c2, c3 = st.columns(3)

    # 1️⃣ Distribution
    with c1:
        st.markdown("**1️⃣ Distribution Explorer**")
        if num_cols:
            col = st.selectbox("Column", num_cols, key="dist_col")
            plot_type = st.radio("Type", ["Histogram", "KDE"], horizontal=True)
            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                sns.histplot(df_vis[col], bins=25, ax=ax)
            else:
                sns.kdeplot(df_vis[col], fill=True, ax=ax)
            ax.set_title(f"Distribution: {col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns.")

    # 2️⃣ Box / Violin
    with c2:
        st.markdown("**2️⃣ Numeric vs Category**")
        if num_cols and cat_cols:
            num = st.selectbox("Numeric", num_cols, key="bx_num")
            cat = st.selectbox("Category", cat_cols, key="bx_cat")
            mode = st.radio("Chart", ["Boxplot", "Violin"], horizontal=True)
            fig, ax = plt.subplots()
            if mode == "Boxplot":
                sns.boxplot(x=df_vis[cat], y=df_vis[num], ax=ax)
            else:
                sns.violinplot(x=df_vis[cat], y=df_vis[num], ax=ax, inner="quartile")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Need numeric + categorical.")

    # 3️⃣ Category Breakdown
    with c3:
        st.markdown("**3️⃣ Category Breakdown**")
        if cat_cols:
            cat = st.selectbox("Category", cat_cols, key="br_cat")
            topN = st.slider("Top N", 3, 20, 10)
            vc = df_vis[cat].value_counts().head(topN)
            fig, ax = plt.subplots()
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            ax.set_title(f"Top {topN} categories: {cat}")
            st.pyplot(fig)
        else:
            st.info("No categorical columns.")

    # ======================================================
    # ROW 2 — Scatter • Correlation • Time / Stacked Summary
    # ======================================================
    c4, c5, c6 = st.columns(3)

    # 4️⃣ Scatter (with y_pred overlay)
    with c4:
        st.markdown("**4️⃣ Scatter + Model Comparison**")
        if len(num_cols) >= 2:
            x = st.selectbox("X-axis", num_cols, key="sc_x")
            y = st.selectbox("Y-axis", [c for c in num_cols if c != x], key="sc_y")
            reg = st.checkbox("Regression trendline")

            fig, ax = plt.subplots()
            sns.scatterplot(x=df_vis[x], y=df_vis[y], ax=ax, s=14, label="Actual")

            if reg:
                sns.regplot(x=df_vis[x], y=df_vis[y], scatter=False,
                            ax=ax, color="red", label="Trend")

            # Compare prediction if Y is target
            if pred_col and target_col == y:
                sns.scatterplot(x=df_vis[x], y=df_vis[pred_col],
                                ax=ax, s=16, color="orange", label="Predicted")

            ax.set_title(f"{y} vs {x}")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Need ≥ 2 numeric columns.")

    # 5️⃣ Correlation Heatmap
    with c5:
        st.markdown("**5️⃣ Correlation (Selected Features)**")
        if len(num_cols) >= 2:
            selected = st.multiselect("Select numeric features",
                                      num_cols, default=num_cols[:5])
            if len(selected) >= 2:
                corr = df_vis[selected].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
            else:
                st.info("Pick ≥ 2 columns.")
        else:
            st.info("Need numeric columns.")

    # 6️⃣ Time-series OR Stacked Summary
    with c6:
        st.markdown("**6️⃣ Time Series / Category Summary**")

        if dt_cols and num_cols:
            dt = st.selectbox("Datetime", dt_cols, key="ts_dt")
            val = st.selectbox("Value", num_cols, key="ts_val")

            df_ts = df_vis.copy()
            df_ts[dt] = pd.to_datetime(df_ts[dt], errors="coerce")
            df_ts = df_ts.dropna(subset=[dt])

            # auto resample if data is too dense
            df_ts = df_ts.sort_values(dt).set_index(dt)[val]
            if len(df_ts) > 150:
                df_ts = df_ts.resample("M").mean()

            fig, ax = plt.subplots()
            df_ts.plot(ax=ax)
            ax.set_title(f"{val} over time")
            st.pyplot(fig)

        elif cat_cols and num_cols:
            st.markdown("📊 *No datetime found → Showing category summary*")
            cat = st.selectbox("Category", cat_cols, key="stk_cat")
            val = st.selectbox("Numerical Value", num_cols, key="stk_val")

            df_grp = (
                df_vis.groupby(cat)[val]
                .agg(["mean", "count"])
                .sort_values("count", ascending=False)
                .head(12)
            )

            fig, ax1 = plt.subplots()
            df_grp["count"].plot(kind="bar", ax=ax1, color="steelblue")
            ax1.set_ylabel("Count")
            ax1.set_xticklabels(df_grp.index, rotation=45)

            ax2 = ax1.twinx()
            ax2.plot(df_grp["mean"], color="orange", marker="o")
            ax2.set_ylabel(f"Mean {val}")

            st.pyplot(fig)

        else:
            st.info("Not enough data for time-series or stacked summary.")
