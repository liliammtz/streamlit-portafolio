import streamlit as st

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

nav0, nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([0.12,0.21,0.16,0.22,0.21,0.25,0.14, 0.1])

nav0.page_link("Main.py", label="Home", icon="🏠")
nav1.page_link("pages/01_EDA_Toolkit.py", label="Data Exploration", icon="🔎")
nav2.page_link("pages/02_Forecasting.py", label="Forecasting", icon="📈")
nav3.page_link("pages/02_ML_Toolkit.py", label="Machine Learning", icon="🧠")
nav4.page_link("pages/04_LLM_Toolkit.py", label="LLM Applications", icon="🤖")
nav5.page_link("pages/03_MLOps_Toolkit.py", label="MLOps & Deployment", icon="⚙️")
nav6.page_link("pages/05_Responsable_AI.py", label="AI Safety", icon="🛡️")
nav7.page_link("pages/07_APIs.py", label="APIs", icon="🌐")
st.divider()

st.title("📊 Exploratory Data Analysis Toolkit")
#!st.markdown("**Contact**")
#!st.markdown("- GitHub: [@liliam-mtz](https://github.com/liliammtz)")
#!st.markdown("- LinkedIn: [Liliam Martínez](https://www.linkedin.com/in/liliammtz/)")
#!st.markdown("- Email: [liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")


# =========================
# Paste Archivo
# =========================

import io
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# 0) Datos de ejemplo o subida
# =========================
@st.cache_data
def sample_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "id": np.arange(1, n+1),
        "date": dates,
        "num_a": rng.normal(100, 20, n).round(2),
        "num_b": rng.gamma(2.0, 15.0, n).round(2),
        "cat_x": rng.choice(["A","B","C"], size=n, p=[0.5,0.3,0.2]),
        "cat_y": rng.choice(["Norte","Sur"], size=n),
        "flag": rng.choice([0,1], size=n, p=[0.7,0.3])
    })
    # Introducimos algunos NaNs y duplicados
    df.loc[rng.choice(n, size=10, replace=False), "num_a"] = np.nan
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)  # un duplicado
    return df

uploaded = st.file_uploader("Upload CSV/Excel for analysis", type=["csv","xlsx"])
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif uploaded.name.endswith(".xslx"):
        df = pd.read_excel(uploaded)
    else:
        st.warning("Only CSV or XLSX files are allowed")
else:
    df = sample_df()

st.write("**Preview**")
st.dataframe(df.head())



import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# Main tabs
# =========================
t0, t1, t3, t4, t5, t6 = st.tabs([
    "📊 Quality & structure",
    "📈 Univariate (Descriptive)",
    #"📦 Distributions (Skewness & Kurtosis)",
    "🔗 Multivariate relationships",
    "🚨 Outliers & Anomalies",
    "🧩 Categoricals & Class Balance",
    "⏳ Time series"
])

# =========================
# 📊 Quality & structure
# =========================
with t0:
    st.subheader("Basic information")
    c0, c1 = st.columns(2)
    c0.write(
        """
        ### 🔎 Step 1: Evaluate data quality  

        Before diving deeper into the analysis, it is essential to review the **dataset quality**.  
        Some key questions we should answer are:  

        - 📏 **What is the size of the dataset?** (rows and columns)  
        - 🧾 **What types of data does it contain?** (numeric, categorical, dates, text, etc.)  
        - 🏷️ **What do these variables mean?** (interpretation of the columns)  
        - ♻️ **Are there duplicate or inconsistent values?**  
        - ⚠️ **Are there missing (null) values that we need to handle?**  
        """
    )

    c1.info(
            '“No data is clean, but most is useful.”  \n'
            '— [Dean Abbott, Co-founder and Chief Data Scientist at SmarterHQ]'
        )
        
    with c1:
        sc1, sc2 = st.columns(2)
        sc1.metric("Rows", df.shape[0])
        sc1.code("""rows = df.shape[0]""", language="python")
        sc2.metric("Columns", df.shape[1])
        sc2.code("""columns = df.shape[1]""", language="python")
        
    
    st.divider()
    c0, c1 = st.columns(2)
    c1.subheader("Data types")
    c1.write(
        """
        In Python, the following data types exist:

        - **For text:** `str`  
        - **For numeric data:** `int`, `float`, `complex`  
        - **For sequences:** `list`, `tuple`, `range`  
        - **For mapping:** `dict`  
        - **For sets:** `set`, `frozenset`  
        - **For booleans (True/False):** `bool`  
        - **For binary:** `bytes`, `bytearray`, `memoryview`  
        - **For nulls:** `NoneType`
        """
    )

    c0.code("""pd.DataFrame(df.dtypes, columns=["dtype"])""", language="python")
    c0.write(pd.DataFrame(df.dtypes, columns=["dtype"]))
    st.divider()
    c0, c1 = st.columns(2)
    c0.subheader("Handling missing values")
    c0.write(
        """
        🔍 In a dataset, it is common to find missing values due to the nature of the data 
        or limitations in data collection. However, when the number of missing values 
        is significant, it is advisable to apply **imputation techniques** to avoid 
        losing valuable information.  

        Some common strategies include:  
        - 📊 **Mean or median** (for numeric variables).  
        - 📈 **Most frequent value (mode)** (for categorical variables).  
        - 🔄 **Forward/Backward fill** (in time series).  
        - 🤖 **Predictive models** (kNN, regressions, etc.).  

        The choice of technique depends on both the type of variable and the context 
        of the analysis.
        """
    )
    
    c1.code("""df.isna().sum().sum()""", language='python')
    nulos_total = df.isna().sum().sum()
    if nulos_total > 0:
        with c1:
            st.warning(f"⚠️ {nulos_total} missing values were found.")
            na = df.isna().sum().sort_values(ascending=False).to_frame("n_nulls")
            na["pct"] = (na["n_nulls"] / len(df)) * 100
            #!st.dataframe(na)
            fig_na = px.bar(
                na.reset_index(),
                x="index", y="n_nulls",
                title="Missing values by column",
                labels={"index": "Column", "n_nulls": "Number of missing values"}
            )
            st.plotly_chart(fig_na, use_container_width=True)
    else:
        with c1:
            st.success("✅ No missing values were found.")

    st.divider()
    c0, c1 = st.columns(2)
    c1.subheader("Duplicate records")
    c1.write(
        """
        🔁 In many datasets, **duplicate rows** may appear due to errors in 
        data entry, data integration processes, or repeated records from 
        different sources.  

        The presence of duplicates can bias analysis results, as it implies 
        counting the same information more than once.  

        Some common strategies to handle them are:  
        - ❌ **Remove exact duplicates** (`drop_duplicates`).  
        - 🔍 **Review partial duplicates**, keeping only the most recent or 
        most complete observation.  
        - 📊 **Group or consolidate records** when they represent the same entity.  

        The appropriate approach depends on the context and the importance 
        of each record within the analysis.
        """
    )

    duplicados_total = df.duplicated().sum()
    c0.code("df.duplicated().sum()",language='python')
    if duplicados_total > 0:
        with c0:
            st.warning(f"⚠️ {duplicados_total} duplicate records were found.")
            st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)))
    else:
        with c0:
            st.success("✅ No duplicates were found.")

    st.info(
        """
        ### 📚 Useful resources  

        - [Python Data Types – W3Schools](https://www.w3schools.com/python/python_datatypes.asp)  
        - [Data Imputation: A Comprehensive Guide (Medium)](https://medium.com/@ajayverma23/data-imputation-a-comprehensive-guide-to-handling-missing-values-b5c7d11c3488)  
        - [Imputation Methods – scikit-learn](https://scikit-learn.org/stable/modules/impute.html)  
        """
    )

# =========================
# 📈 Univariate (Descriptive)
# =========================
with t1:
    st.markdown(
        """
        ### 📈 Univariate analysis  
        Univariate analysis allows exploring each variable individually, 
        understanding its **distribution**, **variability**, and possible outliers.  
        Here, descriptive statistics and visualizations by data type are shown.
        """
    )

    # --- Descriptive statistics ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.subheader("📊 Descriptive statistics")
    c1, c2 = st.columns(2)
    if num_cols:
        with c1:
            st.markdown("**Numeric**")
            st.code("""df[num_cols].describe().T""",language='python')
            st.dataframe(df[num_cols].describe().T)
    if cat_cols:
        with c2:
            st.markdown("**Categorical**")
            st.code("""df[cat_cols].describe().T""",language='python')
            st.dataframe(df[cat_cols].describe().T)
    # --- Numeric (single combined chart) ---
    st.divider()
    st.subheader("📈 Distribution of numeric variables")
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.write(
            "Explore the distribution with **histogram** + **KDE** in a single view. "
            "**Tip:** Enable log scale if there are long tails."
        )

        if num_cols:
            col_num = st.selectbox("Numeric variable", num_cols, key="num_univar")
            bins = st.slider("Bins", 10, 100, 40, key="bins_univar")
            use_log = st.toggle("Log scale on Y-axis", value=False, key="log_univar")

            # Clean data
            serie = df[col_num].dropna().astype(float)

            # Combined chart: histogram + KDE (Figure Factory)
            # Note: ff.create_distplot already combines histogram + KDE curve
            fig_dist = ff.create_distplot(
                [serie.values], [col_num],
                bin_size=(serie.max() - serie.min()) / bins if bins else None,
                show_hist=True, show_rug=False
            )

            # Reference lines: mean and median
            mean_v = float(serie.mean())
            median_v = float(serie.median())
            fig_dist.add_vline(x=mean_v, line_dash="dash", annotation_text=f"mean={mean_v:.2f}")
            #!fig_dist.add_vline(x=median_v, line_dash="dot", annotation_text=f"median={median_v:.2f}")

            # Optional log scale on Y
            fig_dist.update_yaxes(type="log" if use_log else "linear")

            fig_dist.update_layout(title=f"Histogram + KDE: {col_num}", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    with c2:
        if num_cols:
            # Supporting stats (skew/kurtosis) for selected variable
            sel = df[col_num].dropna().astype(float)
            skew_v = float(sel.skew())
            kurt_v = float(sel.kurt())

            # Compact boxplot for context (same variable)
            st.markdown("**Boxplot (dispersion summary)**")
            box_fig = px.box(sel.to_frame(name=col_num), y=col_num)
            box_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(box_fig, use_container_width=True)

            sc1, sc2 = st.columns(2)
            sc1.metric("Skewness", f"{skew_v:.2f}")
            sc2.metric("Kurtosis", f"{kurt_v:.2f}")

            st.markdown(
                """
                **Interpretation tips**  
                - **Skewness > 0**: right tail (few very high values).  
                - **Skewness < 0**: left tail.  
                - High **Kurtosis**: heavy tails (possible outliers).  
                - Consider **median/IQR** if there is skewness; you can try **log** scaling on Y.
                """
            )

    st.divider()
    
    st.subheader("📊 Distribution of categorical variables")
         
    c1, c2 = st.columns(2) 
    with c2:    
        # --- Categorical ---
        if cat_cols:
            
            
                col_cat = st.selectbox("Select categorical variable", cat_cols)
                top_n = st.slider("Show Top N categories", 5, 20, 10)
                
                counts = df[col_cat].value_counts(dropna=False).reset_index()
                counts.columns = [col_cat, "count"]
                counts["pct"] = counts["count"] / counts["count"].sum() * 100
                counts = counts.head(top_n)

                st.dataframe(counts)

                fig_bar = px.bar(
                    counts, x=col_cat, y="count",
                    text="pct", title=f"Top {top_n} categories: {col_cat}",
                    labels={"count": "Frequency"},
                    color_discrete_sequence=["#EF553B"]
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)



    st.subheader("🔢 Unique values count")
    st.dataframe(df.nunique(dropna=True).to_frame("Unique values"))



# =========================
# 🔗 Multivariate relationships
# =========================
with t3:
    st.subheader("🔗 Multivariate relationships")
    st.write("""### 🔗 Step 3: Multivariate relationships  

In this step we aim to understand **how multiple variables relate to each other**.  
Some key questions this helps answer are:  

- 📈 **Which variables move together?**  
  (Pearson, Spearman, Kendall correlations to detect linear or monotonic associations).  

- 🔍 **Are there nonlinear relationships or hidden subgroups?**  
  (Scatter and density plots allow identifying heteroscedasticity, clusters or outliers).  

- 🧮 **How does a numeric variable vary across a categorical variable?**  
  (Boxplots or violin plots show distribution, medians and extreme values by category).  

- ⚠️ **Is there multicollinearity between numeric variables?**  
  (High correlations suggest redundancy and possible problems in models).  

- 🧭 **Which combinations of variables concentrate the highest variance in the dataset?**  
  (PCA identifies principal components and weights of each variable).  

Together, these tools allow **discovering dependencies, hidden patterns and global structure** of the dataset, to guide hypotheses and prepare more robust modeling.
""")
    st.divider()
    st.subheader("Correlation matrix")
    c0, c1 = st.columns(2)
    c0.write("""Measures association between two variables:
  - **Pearson (ρ)**: **linear** relationship; assumes continuity and is **sensitive to outliers**.
  - **Spearman (ρₛ)**: **rank correlation**; captures **monotonic relationships** (does not require linearity) and is more **robust to outliers**.
  - **Kendall (τ)**: based on **concordant/discordant pairs**; similar to Spearman, **more conservative** and stable in **small samples**.
  - **Magnitude guide** (indicative): |corr| < 0.3 weak, 0.3–0.6 moderate, > 0.6 strong.
  - **Warning**: correlation ≠ causation; many simultaneous tests ⇒ use **Bonferroni/FDR**.
""")
    
    # ---------- Global parameters ----------
    with c0.expander("Parameters", expanded=False):
        cA, cB = st.columns(2)
        corr_method = cA.selectbox("Correlation method", ["pearson","spearman","kendall"], index=0)
        cluster_heatmap = cB.checkbox("Cluster heatmap", True, help="Sort variables by similarity |corr|.")
        sample_on = cA.checkbox("Sampling for scatter", True, help="Speeds up plots with many points.")
        max_points = cB.slider("Max sample", 500, 50000, 5000, 500)

    df_plot = df.sample(n=min(max_points, len(df)), random_state=42) if sample_on else df

    @st.cache_data(show_spinner=False)
    def _corr_cached(df_num: pd.DataFrame, method: str) -> pd.DataFrame:
        return df_num.corr(method=method)

    def _cluster_corr(corr: pd.DataFrame) -> pd.DataFrame:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            dist = 1 - np.abs(corr.values)
            Z = linkage(dist, method="average")
            order = leaves_list(Z)
            cols = corr.columns[order]
            return corr.loc[cols, cols]
        except Exception:
            return corr

    # ---------- 1) Correlation matrix ----------

    if len(df.select_dtypes(np.number).columns) >= 2:
        num_cols = df.select_dtypes(np.number).columns.tolist()  # ensure consistency if changed before
        corr = _corr_cached(df[num_cols], corr_method)
        if cluster_heatmap:
            corr = _cluster_corr(corr)

        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            labels=dict(color="ρ"),
            title=f"Numeric correlation ({corr_method})"
        )
        c1.plotly_chart(fig_corr, use_container_width=True)
    else:
        c1.info("You need at least 2 numeric variables.")

    c0.code(
            """corr = df[num_cols].corr(method=corr_method)
corr = _cluster_corr(corr) if cluster_heatmap else corr
fig = px.imshow(corr, text_auto=True, labels=dict(color="ρ"))
st.plotly_chart(fig, use_container_width=True)""",
            language="python"
        )

    st.divider()

    # ---------- 2) Numeric vs numeric scatter ----------
    c0, c1 = st.columns(2)
    c1.subheader("Numeric vs numeric scatter")
    c1.write("""Useful to visualize **shape**, **non-linearity**, **heteroscedasticity**, **clusters** and **outliers**.  
    If there are many points, use **density (hex)** or **contours** and color by **category** (low cardinality).""")
    if len(num_cols) >= 2:
        with c1:
            sc1, sc2, sc3 = st.columns(3)
            x_col = sc1.selectbox("X", num_cols, index=0, key="scat_x")
            y_opts = [c for c in num_cols if c != x_col] or num_cols
            y_col = sc2.selectbox("Y", y_opts, index=0, key="scat_y")

            # Optional color by low-cardinality categorical
            cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
            small_cats = [c for c in cat_cols if df[c].nunique(dropna=True) <= 25]
            color_opt = sc3.selectbox("Color by (optional)", ["None"] + small_cats, index=0)
            color_arg = None if color_opt == "None" else color_opt

        kind = c1.radio("Type", ["Points","Density (hex)","Contours"], horizontal=True)

        if kind == "Points":
            fig_s = px.scatter(df_plot, x=x_col, y=y_col, color=color_arg, opacity=0.7, hover_data=df_plot.columns)
        elif kind == "Density (hex)":
            fig_s = px.density_heatmap(df_plot, x=x_col, y=y_col, nbinsx=40, nbinsy=40,
                                       marginal_x="histogram", marginal_y="histogram")
        else:
            fig_s = px.density_contour(df_plot, x=x_col, y=y_col, color=color_arg,
                                       contours_coloring="fill", nbinsx=40, nbinsy=40)

        c0.plotly_chart(fig_s, use_container_width=True)
    else:
        c0.info("Select at least two numeric columns.")

    c1.code(
            """fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_arg, opacity=0.7)
st.plotly_chart(fig, use_container_width=True)""",
            language="python"
        )

    st.divider()

    # ---------- 3) Numeric vs categorical (violin + box + Top-N) ----------
    c0, c1 = st.columns(2)
    c0.subheader("Numeric distribution by category")
    c0.write("**Violin/box** for distribution and outliers by group; limit to **Top-N** categories and add **summary** (count/mean/median/std).")

    if cat_cols and num_cols:
        with c0:
            sc1, sc2 = st.columns(2)
            cat_sel = sc1.selectbox("Category", cat_cols, key="cat_box")
            num_sel = sc2.selectbox("Numeric", num_cols, key="num_box")
        top_n = c0.slider("Top-N categories by frequency", 3, 30, 12)

        top_cats = df_plot[cat_sel].value_counts(dropna=False).head(top_n).index
        data_cat = df_plot[df_plot[cat_sel].isin(top_cats)].copy()

        fig_violin = px.violin(data_cat, x=cat_sel, y=num_sel, box=True, points="outliers",
                               title=f"{num_sel} by {cat_sel} (Top-{top_n})")
        c1.plotly_chart(fig_violin, use_container_width=True)

        summary = (
            data_cat.groupby(cat_sel, dropna=False)[num_sel]
            .agg(count="count", mean="mean", median="median", std="std")
            .sort_values("mean", ascending=False)
        )
        c1.dataframe(summary, use_container_width=True)
    else:
        c1.info("You need ≥1 categorical column and ≥1 numeric column.")

    c0.code(
            """top_cats = df_plot[cat_sel].value_counts().head(top_n).index
data_cat = df_plot[df_plot[cat_sel].isin(top_cats)]
fig = px.violin(data_cat, x=cat_sel, y=num_sel, box=True, points="outliers")
st.plotly_chart(fig, use_container_width=True)""",
            language="python"
        )

    st.divider()

    # ---------- 4) PCA ----------
    c0, c1 = st.columns(2)
    c1.subheader("PCA (2 components)")
    c1.markdown("""
**What does PCA answer?**  
- How many **latent dimensions** explain most of the variance?
- Which **variables** drive each component (loadings)?
- Are there **clusters** or **separation** between groups in 2D?

**How to read results**  
- **Explained variance**: PC1 explains x%, PC2 y% (sum ≈ information retained in 2D). 
- **Loadings**: high values (±) ⇒ most **influential variables**. Same sign across variables suggests a **common axis** (e.g., “size/income”).  
- **PC1 vs PC2 plot**: look for **groups**, **gradients** (color by category), **outliers**.

**Best practices**  
- Only **numeric** and **no NaNs**; apply **StandardScaler** (z-score).  
- Check **outliers** first: they can dominate components.  
- Low variance in PC1/PC2 ⇒ you may need **more PCs** or non-linear techniques (t-SNE/UMAP).

**Limitations**  
- It is **linear** and **unsupervised**: it does not maximize separation by target.  
- Components are combinations; interpretation depends on **loadings**.
""")
    c0.code(
                """X = df[num_cols].dropna()
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)
scores = pca.transform(Xs)
explained = pca.explained_variance_ratio_
pca_df = pd.DataFrame(scores, columns=["PC1","PC2"], index=X.index)
fig = px.scatter(pca_df, x="PC1", y="PC2", title=f"Var explained: {explained[:2].sum():.1%}")
st.plotly_chart(fig, use_container_width=True)""",
                language="python"
            )
    if len(num_cols) >= 2:
        X = df[num_cols].dropna(axis=0)
        idx = X.index

        # Optional color by low-cardinality categorical (aligned to index)
        small_cats_full = [c for c in cat_cols if df[c].nunique(dropna=True) <= 25] if cat_cols else []
        color_pca_opt = c0.selectbox("Color by (optional)", ["None"] + small_cats_full, index=0, key="pca_color")
        color_series = None if color_pca_opt == "None" else df.loc[idx, color_pca_opt]

        Xs = StandardScaler().fit_transform(X.values)
        pca = PCA(n_components=2).fit(Xs)
        scores = pca.transform(Xs)
        exp = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(scores, columns=["PC1","PC2"], index=idx)
        if color_series is not None:
            pca_df[color_pca_opt] = color_series.values

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            color=color_pca_opt if color_series is not None else None,
            title=f"PCA: PC1 {exp[0]:.1%} · PC2 {exp[1]:.1%} · Total {(exp[:2].sum()):.1%}"
        )
        c0.plotly_chart(fig_pca, use_container_width=True)
        
        loadings = pd.DataFrame(pca.components_.T, index=num_cols, columns=["PC1","PC2"])
        cL, cR = st.columns(2)
        cL.plotly_chart(px.bar(loadings["PC1"].abs().sort_values(ascending=False).head(12),
                               title="Top |loading| PC1", orientation="h"), use_container_width=True)
        cR.plotly_chart(px.bar(loadings["PC2"].abs().sort_values(ascending=False).head(12),
                               title="Top |loading| PC2", orientation="h"), use_container_width=True)

    else:
        c0.info("PCA requires ≥2 numeric columns.")


# =========================
# 🚨 Outliers & Anomalies
# =========================
with t4:
    st.subheader("🚨 Outliers & Anomalies")

    st.markdown("""
    **What does this step answer?**
    - How many and which observations fall outside what is expected by **dispersion**?
    - Are extremes isolated or systematic by subgroup?
    - Are there **multivariate anomalies** that are not visible in univariate analysis?

    **Quick theory**
    - **IQR (Tukey)**: flags outliers outside \\[Q1 − k·IQR, Q3 + k·IQR\\]. Simple and robust to non-normality.
    - **Robust z-score (MAD)**: uses median and **MAD** → less sensitive to outliers than classical z-score.
    - **Percentiles**: defines empirical thresholds (e.g., 1% and 99%) when the shape is irregular or multimodal.
    - **Multivariate** (IsolationForest): detects unusual points considering **all numeric variables together**.
    """)

    num_cols = df.select_dtypes(np.number).columns.tolist()
    if not num_cols:
        st.info("No numeric columns available.")
    else:
        # ---------- Parameters ----------
        c0, c1, c2 = st.columns(3)
        col_out = c0.selectbox("Numeric variable", num_cols, key="outlier_var")
        method = c1.radio("Method", ["IQR", "Robust Z (MAD)", "Percentiles"], horizontal=True, key="out_method")

        # Controls depending on method
        k = None; z_thr = None; q_lo = None; q_hi = None
        if method == "IQR":
            k = c2.slider("k (IQR)", 1.0, 5.0, 1.5, 0.1)
        elif method == "Robust Z (MAD)":
            z_thr = c2.slider("|z| robust", 2.0, 6.0, 3.5, 0.1)
        else:
            c2.empty()
            c3, c4 = st.columns(2)
            q_lo = c3.slider("Lower percentile", 0.0, 10.0, 1.0, 0.1)
            q_hi = c4.slider("Upper percentile", 90.0, 100.0, 99.0, 0.1)

        serie = df[col_out].astype(float)
        serie_clean = serie.dropna()
        n_valid = int(serie_clean.shape[0])

        # ---------- Outlier calculation ----------
        mask = pd.Series(False, index=df.index)
        lo = hi = None
        score = pd.Series(np.nan, index=df.index)  # "strength" of the outlier for sorting

        if n_valid == 0:
            st.warning("The selected variable only contains null values.")
        else:
            if method == "IQR":
                Q1, Q3 = serie_clean.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lo, hi = Q1 - k * IQR, Q3 + k * IQR
                mask = (serie < lo) | (serie > hi)
                # relative distance to IQR range for sorting
                med = float(serie_clean.median())
                score = (serie - med).abs() / (IQR if IQR != 0 else 1.0)

            elif method == "Robust Z (MAD)":
                med = float(serie_clean.median())
                mad = float((serie_clean - med).abs().median())
                if mad == 0:
                    st.info("MAD = 0; cannot compute robust z-score. Try IQR or percentiles.")
                    mask = pd.Series(False, index=df.index)
                else:
                    robust_z = (serie - med) / (1.4826 * mad)
                    score = robust_z.abs()
                    mask = score > z_thr

            else:  # Percentiles
                lo = float(np.percentile(serie_clean, q_lo))
                hi = float(np.percentile(serie_clean, q_hi))
                mask = (serie < lo) | (serie > hi)
                med = float(serie_clean.median())
                score = (serie - med).abs()  # absolute distance to median

            n_out = int(mask.fillna(False).sum())
            pct_out = (n_out / n_valid * 100) if n_valid > 0 else 0.0

            # ---------- Visualizations ----------
            st.subheader("Boxplot (quick view)")
            st.plotly_chart(px.box(df, y=col_out, points="outliers",
                                   title=f"Boxplot: {col_out}"),
                            use_container_width=True)

            st.subheader("Distribution and thresholds")
            fig_hist = px.histogram(serie_clean, nbins=50, opacity=0.85,
                                    title=f"Histogram: {col_out}")
            # Threshold lines
            try:
                if method in ["IQR", "Percentiles"]:
                    fig_hist.add_vline(x=lo, line_dash="dash", annotation_text=f"lo={lo:,.3g}")
                    fig_hist.add_vline(x=hi, line_dash="dash", annotation_text=f"hi={hi:,.3g}")
                elif method == "Robust Z (MAD)":
                    fig_hist.add_vline(x=med, line_dash="dot", annotation_text=f"median={med:,.3g}")
            except Exception:
                pass
            st.plotly_chart(fig_hist, use_container_width=True)

            # ---------- Metrics + table ----------
            cA, cB = st.columns(2)
            cA.metric("Detected outliers", f"{n_out:,}")
            cB.metric("% over valid", f"{pct_out:.2f}%")

            st.subheader("Outlier observations (Top-N by 'strength')")
            top_n = st.slider("Top-N to display", 10, 200, 50, 10)
            out_df = (
                df.loc[mask].assign(_out_score=score[mask])
                .sort_values("_out_score", ascending=False)
                .head(top_n)
            )
            st.dataframe(out_df, use_container_width=True)

            # Download CSV of outliers
            if n_out > 0:
                csv_buf = io.StringIO()
                df.loc[mask].assign(_out_score=score[mask]).to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Download outliers (CSV)",
                    data=csv_buf.getvalue(),
                    file_name=f"outliers_{col_out}_{method.replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # ---------- Code (expander) ----------
            with st.expander("View code (univariate detection)"):
                st.code(
                    """# IQR
Q1, Q3 = x.quantile([0.25, 0.75]); IQR = Q3 - Q1
lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
mask_iqr = (x < lo) | (x > hi)

# robust z (MAD)
med = x.median(); mad = (x - med).abs().median()
robust_z = (x - med) / (1.4826 * mad)
mask_mad = robust_z.abs() > 3.5

# percentiles
lo = np.percentile(x.dropna(), 1); hi = np.percentile(x.dropna(), 99)
mask_pct = (x < lo) | (x > hi)""",
                    language="python"
                )

        # ---------- Optional multivariate ----------
        st.divider()
        st.subheader("🔀 Multivariate detection (IsolationForest) — optional")
        st.caption("Useful when a point is not extreme in 1D, but is when combining multiple numeric variables.")

        enable_if = st.checkbox("Enable IsolationForest", value=False)
        if enable_if:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA

                feats = st.multiselect("Numeric variables to consider", num_cols,
                                       default=num_cols[:min(5, len(num_cols))])
                if len(feats) == 0:
                    st.info("Select at least 1 numeric variable.")
                else:
                    c1, c2 = st.columns(2)
                    contamination = c1.slider("Expected anomaly proportion", 0.01, 0.20, 0.05, 0.01)
                    n_estimators = c2.slider("Trees (n_estimators)", 50, 400, 200, 50)

                    X = df[feats].dropna()
                    if X.empty:
                        st.info("No complete rows for selected variables.")
                    else:
                        iso = IsolationForest(
                            contamination=contamination,
                            n_estimators=n_estimators,
                            random_state=42
                        ).fit(X)

                        pred = iso.predict(X)  # -1 = outlier
                        score_if = iso.decision_function(X)  # lower = more anomalous
                        res = df.loc[X.index].copy()
                        res["_anomaly_if"] = (pred == -1)
                        res["_if_score"] = score_if

                        n_anom = int(res["_anomaly_if"].sum())
                        pct_anom = n_anom / res.shape[0] * 100
                        cA, cB = st.columns(2)
                        cA.metric("Anomalies (IForest)", f"{n_anom:,}")
                        cB.metric("% over valid", f"{pct_anom:.2f}%")

                        # PCA 2D view for interpretation
                        Xs = StandardScaler().fit_transform(X)
                        p2 = PCA(n_components=2).fit_transform(Xs)
                        plot_df = pd.DataFrame({
                            "PC1": p2[:, 0],
                            "PC2": p2[:, 1],
                            "Anomaly": np.where(res["_anomaly_if"], "Yes", "No")
                        }, index=X.index)
                        fig_if = px.scatter(
                            plot_df, x="PC1", y="PC2", color="Anomaly",
                            title="IsolationForest in PCA space (2D)",
                            opacity=0.85
                        )
                        st.plotly_chart(fig_if, use_container_width=True)

                        st.subheader("Most anomalous samples (by score)")
                        show_n = st.slider("Show Top-N", 10, 200, 50, 10, key="if_topn")
                        st.dataframe(
                            res.sort_values("_if_score").head(show_n),
                            use_container_width=True
                        )

                        with st.expander("View code (IsolationForest)"):
                            st.code(
                                """from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42).fit(X)
pred = iso.predict(X)          # -1 = outlier
score = iso.decision_function(X)  # lower = more anomalous
mask_if = pred == -1""",
                                language="python"
                            )
            except Exception as e:
                st.info(f"Could not enable IsolationForest: {e}")

    # ---------- Treatment suggestions ----------
    st.divider()
    with st.expander("💊 Treatment: common options"):
        st.markdown("""
        - **Manual inspection** and correction if it is a data entry error.
        - **Winsorization (clip)** at percentiles (e.g., 1%–99%).
        - **Transformations** (log/sqrt/Box-Cox) if there is strong skewness.
        - **Robust modeling** (trees/boosting, regularization) if you prefer not to remove them.
        - **Remove** extreme cases only if their lack of representativeness is justified.

        **Winsorization snippet:**
        ```python
        lo, hi = x.quantile([0.01, 0.99])
        x_wins = x.clip(lo, hi)
        ```
        """)
        
# =========================
# 🧩 Categóricas & Balance de clases
# =========================
with t5:
    st.subheader("🧩 Categoricals & Class Balance")

    # Ensure column lists (recalculate in case df changed)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("""
    **What does this step answer?**  
    - Which categories are the most **frequent** (Top-N) and what do they represent (%)?  
    - Is there strong **class imbalance** that could affect modeling?  
    - How does each category contribute to a **total numeric value** (treemap)?  
    - How does a **numeric target** behave within each category (means/medians)?  
    """)

    st.divider()

    # =========================
    # 1) Treemap (Category -> Numeric value)
    # =========================
    if cat_cols and num_cols:
        c0, c1 = st.columns(2)

        c0.subheader("🌳 Treemap (Category → Value)")
        c0.write(
                "The **treemap** shows the contribution of each category to a **numeric total** "
                "(area proportional to the value). Useful to detect dominant categories."
            )
        c0.code(
                    """cat_t = st.selectbox("Category", cat_cols, key="t5_treemap_cat")
val_t = st.selectbox("Numeric value", num_cols, key="t5_treemap_val")
fig_tm = px.treemap(df, path=[cat_t], values=val_t, title=f"Treemap: {cat_t} → {val_t}")
st.plotly_chart(fig_tm, use_container_width=True)""",
                    language="python"
                )

        with c0:
            sc0, sc1 = st.columns(2)
            cat_t = sc0.selectbox("Category", cat_cols, key="t5_treemap_cat")
            val_t = sc1.selectbox("Numeric value", num_cols, key="t5_treemap_val")
            fig_tm = px.treemap(df, path=[cat_t], values=val_t, title=f"Treemap: {cat_t} → {val_t}")
        
        c1.plotly_chart(fig_tm, use_container_width=True)

    else:
        st.info("Treemap requires at least one categorical column and one numeric column.")

    st.divider()

    # =========================
    # 2) Category distribution (Top-N + 'Others') + robust %
    # =========================
    if cat_cols:
        c0, c1 = st.columns(2)

        c1.subheader("📊 Category distribution (Top-N + 'Others')")
        c1.write(
                "We show **Top-N** categories by frequency and group the rest into **'Others'** "
                "to maintain readability. We also compute the **%** of each category."
            )

        c1.code(
                    """col_cat = st.selectbox("Categorical variable", cat_cols, key="t5_cat_var")
top_n = st.slider("Top-N categories", 3, 30, 10, key="t5_topn")

# Robust counts
vc = df[col_cat].value_counts(dropna=False)          # Count series
top = vc.head(top_n)
others = vc.iloc[top_n:].sum()

plot_s = top.copy()
if others > 0:
    plot_s.loc["Others"] = int(others)

# Final dataframe without duplicates and numeric
counts_df = (
    plot_s.rename("count")
         .reset_index()
         .rename(columns={"index": col_cat})
)
counts_df["count"] = pd.to_numeric(counts_df["count"], errors="coerce").fillna(0)

total = counts_df["count"].sum(min_count=1)
counts_df["pct"] = (counts_df["count"] / (total if total else 1)) * 100

st.dataframe(counts_df, use_container_width=True)

fig_bar = px.bar(
    counts_df.sort_values("count", ascending=False),
    x=col_cat, y="count",
    text=counts_df["pct"].map(lambda x: f"{x:.1f}%"),
    title=f"Distribution: {col_cat}",
    labels={"count": "Frequency"}
)
fig_bar.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig_bar, use_container_width=True)""",
                    language="python"
                )

        with c0:
            # --- UI and robust calculation ---
            col_cat = st.selectbox("Categorical variable", cat_cols, key="t5_cat_var")
            top_n = st.slider("Top-N categories", 3, 30, 10, key="t5_topn")

            vc = df[col_cat].value_counts(dropna=False)   # Series
            top = vc.head(top_n)
            others = vc.iloc[top_n:].sum()

            plot_s = top.copy()
            if others > 0:
                plot_s.loc["Others"] = int(others)

            counts_df = (
                plot_s.rename("count")
                     .reset_index()
                     .rename(columns={"index": col_cat})
            )
            counts_df["count"] = pd.to_numeric(counts_df["count"], errors="coerce").fillna(0)

            total = counts_df["count"].sum(min_count=1)
            counts_df["pct"] = (counts_df["count"] / (total if total else 1)) * 100

            st.dataframe(counts_df, use_container_width=True)

            fig_bar = px.bar(
                counts_df.sort_values("count", ascending=False),
                x=col_cat, y="count",
                text=counts_df["pct"].map(lambda x: f"{x:.1f}%"),
                title=f"Distribution: {col_cat}",
                labels={"count": "Frequency"}
            )
            fig_bar.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ===== Quick class balance =====
        major_cat = vc.index[0] if len(vc) else None
        if major_cat is not None:
            major_count = int(vc.iloc[0])
            total_all = int(vc.sum())
            major_pct = (major_count / total_all * 100) if total_all else 0.0

            with c0:
                cA, cB, cC = st.columns(3)
                cA.metric("Majority class", str(major_cat))
                cB.metric("Majority frequency", f"{major_count:,}")
                cC.metric("Majority %", f"{major_pct:.1f}%")

            if major_pct >= 80:
                st.warning("Strong imbalance (≥80/20). Consider techniques: reweighting, undersampling/oversampling, class-based metrics.")

    else:
        st.info("No categorical columns available.")

    st.divider()

    # =========================
    # 3) Relationship with numeric target (mean/median by category)
    # =========================
    if cat_cols and num_cols:
        c0, c1 = st.columns(2)

        with c0:
            st.subheader("🎯 Numeric target by category")
            st.write(
                "Compare a numeric target across categories using **mean** or **median** "
                "(robust to outliers). For readability, it is limited to **Top-N** categories."
            )
            st.code(
                    """col_cat2 = st.selectbox("Category", cat_cols, key="t5_target_cat")
target = st.selectbox("Numeric target", num_cols, key="t5_target_val")
agg_fn = st.selectbox("Aggregation", ["mean", "median", "sum"], index=0, key="t5_target_agg")
top_n_target = st.slider("Top-N (by frequency)", 3, 30, 10, key="t5_target_topn")

vc2 = df[col_cat2].value_counts(dropna=False)
keep = vc2.head(top_n_target).index
data_f = df[df[col_cat2].isin(keep)]

agg = getattr(data_f.groupby(col_cat2, dropna=False)[target], agg_fn)().reset_index()
agg = agg.sort_values(by=agg.columns[1], ascending=False)

fig_t = px.bar(agg, x=col_cat2, y=agg.columns[1],
               title=f"{target} ({agg_fn}) by {col_cat2}",
               labels={agg.columns[1]: f"{target} ({agg_fn})"})
st.plotly_chart(fig_t, use_container_width=True)""",
                    language="python"
                )

        with c1:
            sc0, sc1, sc2 = st.columns(3)
            col_cat2 = sc0.selectbox("Category", cat_cols, key="t5_target_cat")
            target = sc1.selectbox("Numeric target", num_cols, key="t5_target_val")
            agg_fn = sc2.selectbox("Aggregation", ["mean", "median", "sum"], index=0, key="t5_target_agg")
            top_n_target = st.slider("Top-N (by frequency)", 3, 30, 10, key="t5_target_topn")

            vc2 = df[col_cat2].value_counts(dropna=False)
            keep = vc2.head(top_n_target).index
            data_f = df[df[col_cat2].isin(keep)]

            agg_series = getattr(data_f.groupby(col_cat2, dropna=False)[target], agg_fn)()
            agg = agg_series.reset_index().sort_values(by=target, ascending=False)

            fig_t = px.bar(
                agg, x=col_cat2, y=target,
                title=f"{target} ({agg_fn}) by {col_cat2}",
                labels={target: f"{target} ({agg_fn})"}
            )
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("This section requires at least one categorical and one numeric column.")

# =========================
# ⏳ Time series (t6 complete and robust)
# =========================
with t6:
    st.subheader("⏳ Time series")

    # ---------- Helpers ----------
    def _is_parseable_datetime(s: pd.Series, min_ok: float = 0.5) -> bool:
        """True if >= min_ok can be parsed to datetime."""
        if np.issubdtype(s.dtype, np.datetime64):
            return True
        try:
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            return parsed.notna().mean() >= min_ok
        except Exception:
            return False

    def _ensure_datetime_col(df_: pd.DataFrame, col: str) -> pd.Series:
        """Converts to datetime if it is not."""
        if np.issubdtype(df_[col].dtype, np.datetime64):
            return df_[col]
        return pd.to_datetime(df_[col], errors="coerce", infer_datetime_format=True)

    def _looks_like_id(name: str) -> bool:
        """Simple heuristic to exclude numeric identifiers."""
        n = str(name).lower()
        return (
            n == "id" or n == "index" or
            n.endswith("_id") or n.startswith("id_") or
            n.endswith(" id") or n.startswith("id ")
        )

    # ---------- Date column detection ----------
    date_candidates = [c for c in df.columns if _is_parseable_datetime(df[c])]

    if not date_candidates:
        st.info("No date columns were detected.")
    else:
        # ---------- 2-column layout ----------
        c0, c1 = st.columns(2)

        # ===== Left side: theory + parameters =====
        with c0:
            st.subheader("What it answers")
            st.write(
                "- How does the metric evolve over time (trend/seasonality)?\n"
                "- How do **frequency** and **aggregation** (sum/mean/…) affect it?\n"
                "- What is the **% change** period-over-period or **year-over-year (YoY)**?\n"
                "- Are there differences by **category** (multiple lines)?"
            )

            st.subheader("Parameters")
            date_col = st.selectbox("Date column", date_candidates, key="ts_date_col")

        # Build df_ts with valid date BEFORE choosing metric
        df_ts = df.copy()
        df_ts[date_col] = _ensure_datetime_col(df_ts, date_col)
        df_ts = df_ts.dropna(subset=[date_col]).sort_values(by=date_col)

        # Valid numeric columns (exclude identifiers)
        num_cols_ts = [
            c for c in df_ts.select_dtypes(np.number).columns
            if c != date_col and not _looks_like_id(c)
        ]

        # Low-cardinality categoricals
        cat_cols_all = df_ts.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        small_cats = [c for c in cat_cols_all if df_ts[c].nunique(dropna=True) <= 12]

        with c0:
            if not num_cols_ts:
                st.warning("No valid numeric columns for time series (excluding identifiers like 'id').")
            else:
                val_col = st.selectbox("Numeric variable", num_cols_ts, key="ts_val_col")
                split_col = st.selectbox("Split by category (optional)", ["None"] + small_cats, key="ts_split")

                cA, cB, cC = st.columns(3)
                freq_lbl = cA.selectbox("Frequency", ["D (day)", "W (week)", "M (month)", "Q (quarter)", "Y (year)"], index=2, key="ts_freq")
                agg_fn_name = cB.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], index=0, key="ts_agg")
                fill_mode = cC.selectbox("Missing fill", ["None", "Forward-fill", "Zero"], index=0, key="ts_fill")

                cD, cE = st.columns(2)
                pct_kind = cD.selectbox("% change", ["None", "Period-over-period", "Year-over-year (YoY)"], index=0, key="ts_pct")
                show_ma = cE.checkbox("Moving average", value=True, key="ts_ma")
                ma_window = st.slider("MA window (periods)", 2, 60, 12, key="ts_ma_win") if show_ma else None

                st.code(
                        """# Basic resample (single series)
ts = (df.set_index(date_col).sort_index()[val_col]
        .resample('M').sum()
        .reset_index())""",
                        language="python"
                    )

        # ===== Right side: results =====
        with c1:
            if not num_cols_ts:
                st.stop()

            freq_map = {"D (day)": "D", "W (week)": "W", "M (month)": "M", "Q (quarter)": "Q", "Y (year)": "Y"}
            f = freq_map[freq_lbl]

            if split_col == "None":
                # --- Single series ---
                s = (
                    df_ts.set_index(date_col)
                         .loc[:, val_col]
                         .resample(f)
                         .agg(agg_fn_name)
                )
                if fill_mode == "Forward-fill":
                    s = s.ffill()
                elif fill_mode == "Zero":
                    s = s.fillna(0)

                ts = s.reset_index().rename(columns={val_col: val_col})

                # Moving average
                ys = [val_col]
                if show_ma:
                    ts["MA"] = ts[val_col].rolling(int(ma_window)).mean()
                    ys.append("MA")

                fig_ts = px.line(
                    ts, x=date_col, y=ys,
                    title=f"{val_col} — {agg_fn_name} by {f}",
                    labels={date_col: "Date"}
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                # % change
                if pct_kind != "None":
                    if pct_kind == "Period-over-period":
                        periods = 1
                    else:
                        shift_map = {"M": 12, "Q": 4, "W": 52, "D": 365, "Y": 1}
                        periods = shift_map.get(f, 1)
                    ts["pct_change"] = ts[val_col].pct_change(periods=periods) * 100
                    fig_pct = px.line(
                        ts, x=date_col, y="pct_change",
                        title=("Period-over-period % change" if periods == 1 else "Year-over-year (YoY) % change"),
                        labels={"pct_change": "%"}
                    )
                    st.plotly_chart(fig_pct, use_container_width=True)

                st.dataframe(ts.tail(12), use_container_width=True)

                st.code(
                        """# % change
periods = 12  # monthly YoY example
ts["pct_change"] = ts[val_col].pct_change(periods=periods) * 100""",
                        language="python"
                    )

            else:
                # --- Multiple series by category ---
                grp = (
                    df_ts.groupby([pd.Grouper(key=date_col, freq=f), split_col])[val_col]
                         .agg(agg_fn_name)
                         .reset_index()
                )

                # Fill by group
                if fill_mode != "None":
                    def _fill(g):
                        if fill_mode == "Forward-fill":
                            g[val_col] = g[val_col].ffill()
                        elif fill_mode == "Zero":
                            g[val_col] = g[val_col].fillna(0)
                        return g
                    grp = grp.groupby(split_col, group_keys=False).apply(_fill)

                # Moving average by group
                if show_ma:
                    grp["MA"] = grp.groupby(split_col)[val_col].transform(lambda s: s.rolling(int(ma_window)).mean())

                fig_multi = px.line(
                    grp, x=date_col, y=val_col, color=split_col,
                    title=f"{val_col} — {agg_fn_name} by {f} (by {split_col})"
                )
                st.plotly_chart(fig_multi, use_container_width=True)

                if show_ma:
                    fig_ma = px.line(
                        grp, x=date_col, y="MA", color=split_col,
                        title=f"Moving average ({ma_window}) by {split_col}"
                    )
                    st.plotly_chart(fig_ma, use_container_width=True)

                # % change by group
                if pct_kind != "None":
                    if pct_kind == "Period-over-period":
                        periods = 1
                    else:
                        shift_map = {"M": 12, "Q": 4, "W": 52, "D": 365, "Y": 1}
                        periods = shift_map.get(f, 1)
                    grp["pct_change"] = grp.groupby(split_col)[val_col].pct_change(periods=periods) * 100
                    fig_pctm = px.line(
                        grp, x=date_col, y="pct_change", color=split_col,
                        title=("Period-over-period % change" if periods == 1 else "Year-over-year (YoY) % change"),
                        labels={"pct_change": "%"}
                    )
                    st.plotly_chart(fig_pctm, use_container_width=True)

                st.dataframe(
                    grp.sort_values(date_col).groupby(split_col).tail(6),
                    use_container_width=True
                )

                st.code(
                        """grp = (df
    .groupby([pd.Grouper(key=date_col, freq='M'), split_col])[val_col]
    .sum()
    .reset_index())
# Rolling by group
grp["MA"] = grp.groupby(split_col)[val_col].transform(lambda s: s.rolling(12).mean())""",
                        language="python"
                    )

        c0.write("💡 Tips & best practices")
        c0.markdown(
                "- If the date is not `datetime`, it is **parsed automatically**.\n"
                "- **Frequency** changes unit of analysis (D/W/M/Q/Y) and **aggregation** defines the summary.\n"
                "- **Forward-fill** is useful for rates; **Zero** for counts when periods are missing.\n"
                "- **YoY** requires at least one year of history at the same frequency.\n"
                "- For seasonal decomposition: `statsmodels.tsa.seasonal_decompose` (optional)."
            )