import streamlit as st

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 EDA Toolkit")

# ---------- SIDEBAR (always-visible contact + nav) ----------
with st.sidebar:
    st.markdown("### 👋 About me")
    st.write(
        "Data Scientist with a strong background in **forecasting**, **business intelligence**, and **ML-powered analytics**. "
        "I specialize in building **end-to-end data products** — from data pipelines and predictive models in **Snowflake/SQL** "
        "to polished **Streamlit apps** used daily by business teams. "
        "Passionate about turning raw data into clear, actionable insights that support **strategic decision-making**."
    )

    st.divider()

    st.page_link("Main.py", label="Home", icon="🏠")

    #!st.markdown("### 📊 Data Science")

    st.page_link("pages/01_EDA_Toolkit.py", label="EDA Toolkit", icon="🔎")
    st.page_link("pages/02_Forecasting.py", label="Forecasting Toolkit", icon="📈")

    #!st.markdown("### 🤖 Machine Learning")

    st.page_link("pages/02_ML_Toolkit.py", label="Machine Learning Toolkit", icon="🧠")
    st.page_link("pages/04_LLM_Toolkit.py", label="LLM Toolkit", icon="🤖")

    #!st.markdown("### ⚙️ AI Engineering")

    st.page_link("pages/03_MLOps_Toolkit.py", label="MLOps Toolkit", icon="⚙️")

    #!st.markdown("### 🛡️ AI Governance")

    st.page_link("pages/05_Responsable_AI.py", label="Responsible AI Toolkit", icon="🛡️")

    st.divider()
    st.markdown("**Contact**")
    st.markdown("- GitHub: [@liliam-mtz](https://github.com/liliammtz)")
    st.markdown("- LinkedIn: [Liliam Martínez](https://www.linkedin.com/in/liliammtz/)")
    st.markdown("- Email: [liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")




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

uploaded = st.file_uploader("Sube CSV/Excel para analisis", type=["csv","xlsx"])
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    elif uploaded.name.endswith(".xslx"):
        df = pd.read_excel(uploaded)
    else:
        st.warning("Solo estan permitidos archivos de CSV o XLSX")
else:
    df = sample_df()

st.write("**Vista previa**")
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
# Tabs principales
# =========================
t0, t1, t3, t4, t5, t6 = st.tabs([
    "📊 Calidad & estructura",
    "📈 Univariado (Descriptivas)",
    #"📦 Distribuciones (Skewness & Kurtosis)",
    "🔗 Relaciones multivariadas",
    "🚨 Outliers & Anomalías",
    "🧩 Categóricas & Balance de clases",
    "⏳ Series de tiempo"
])

# =========================
# 📊 Calidad & estructura
# =========================
with t0:
    st.subheader("Información básica")
    c0, c1 = st.columns(2)
    c0.write(
        """
        ### 🔎 Paso 1: Evaluar la calidad de los datos  

        Antes de profundizar en el análisis, es fundamental revisar la **calidad del dataset**.  
        Algunas preguntas clave que debemos responder son:  

        - 📏 **¿De qué tamaño es el dataset?** (filas y columnas)  
        - 🧾 **¿Qué tipos de datos contiene?** (numéricos, categóricos, fechas, texto, etc.)  
        - 🏷️ **¿Qué significan esas variables?** (interpretación de las columnas)  
        - ♻️ **¿Existen valores duplicados o inconsistentes?**  
        - ⚠️ **Hay valores faltantes (nulos) que debamos tratar?**  
        """
    )

    c1.info(
            '“No data is clean, but most is useful.”  \n'
            '— [Dean Abbott, Co-founder and Chief Data Scientist at SmarterHQ]'
        )
        
    with c1:
        sc1, sc2 = st.columns(2)
        sc1.metric("Filas", df.shape[0])
        sc1.code("""filas = df.shape[0]""", language="python")
        sc2.metric("Columnas", df.shape[1])
        sc2.code("""columnas = df.shape[1]""", language="python")
        
    
    st.divider()
    c0, c1 = st.columns(2)
    c1.subheader("Tipos de datos")
    c1.write(
        """
        En Python existen los siguientes tipos de datos:

        - **Para texto:** `str`  
        - **Para datos numéricos:** `int`, `float`, `complex`  
        - **Para secuencias:** `list`, `tuple`, `range`  
        - **Para mapping:** `dict`  
        - **Para sets:** `set`, `frozenset`  
        - **Para booleanos (True/False):** `bool`  
        - **Para binarios:** `bytes`, `bytearray`, `memoryview`  
        - **Para nulos:** `NoneType`
        """
    )

    c0.code("""pd.DataFrame(df.dtypes, columns=["dtype"])""", language="python")
    c0.write(pd.DataFrame(df.dtypes, columns=["dtype"]))
    st.divider()
    c0, c1 = st.columns(2)
    c0.subheader("Manejo de nulos")
    c0.write(
        """
        🔍 En un dataset es común encontrar valores nulos debido a la naturaleza de los datos 
        o a limitaciones en la recolección de la información. Sin embargo, cuando la cantidad 
        de valores faltantes es considerable, conviene aplicar técnicas de **imputación** para 
        no perder información valiosa.  

        Algunas estrategias habituales incluyen:  
        - 📊 **Media o mediana** (para variables numéricas).  
        - 📈 **Valor más frecuente (moda)** (para variables categóricas).  
        - 🔄 **Forward/Backward fill** (en series temporales).  
        - 🤖 **Modelos predictivos** (kNN, regresiones, etc.).  

        La elección de la técnica depende tanto del tipo de variable como del contexto 
        del análisis.
        """
    )
    
    c1.code("""df.isna().sum().sum()""", language='python')
    nulos_total = df.isna().sum().sum()
    if nulos_total > 0:
        with c1:
            st.warning(f"⚠️ Se encontraron {nulos_total} valores nulos.")
            na = df.isna().sum().sort_values(ascending=False).to_frame("n_nulls")
            na["pct"] = (na["n_nulls"] / len(df)) * 100
            #!st.dataframe(na)
            fig_na = px.bar(
                na.reset_index(),
                x="index", y="n_nulls",
                title="Nulos por columna",
                labels={"index": "Columna", "n_nulls": "Número de nulos"}
            )
            st.plotly_chart(fig_na, use_container_width=True)
    else:
        with c1:
            st.success("✅ No se encontraron valores nulos.")

    st.divider()
    c0, c1 = st.columns(2)
    c1.subheader("Registros duplicados")
    c1.write(
        """
        🔁 En muchos datasets pueden aparecer **filas duplicadas**, ya sea por errores en la 
        captura de información, procesos de integración de datos o registros repetidos en 
        distintas fuentes.  

        La presencia de duplicados puede sesgar los resultados del análisis, ya que implica 
        contar la misma información más de una vez.  

        Algunas estrategias comunes para tratarlos son:  
        - ❌ **Eliminar duplicados exactos** (`drop_duplicates`).  
        - 🔍 **Revisar duplicados parciales**, manteniendo solo la observación más reciente o 
        más completa.  
        - 📊 **Agrupar o consolidar registros** cuando representan la misma entidad.  

        El enfoque adecuado dependerá del contexto y de la importancia que tenga cada 
        registro dentro del análisis.
        """
    )

    duplicados_total = df.duplicated().sum()
    c0.code("df.duplicated().sum()",language='python')
    if duplicados_total > 0:
        with c0:
            st.warning(f"⚠️ Se encontraron {duplicados_total} registros duplicados.")
            st.dataframe(df[df.duplicated(keep=False)].sort_values(list(df.columns)))
    else:
        with c0:
            st.success("✅ No se encontraron duplicados.")

    st.info(
        """
        ### 📚 Recursos útiles  

        - [Python Data Types – W3Schools](https://www.w3schools.com/python/python_datatypes.asp)  
        - [Data Imputation: A Comprehensive Guide (Medium)](https://medium.com/@ajayverma23/data-imputation-a-comprehensive-guide-to-handling-missing-values-b5c7d11c3488)  
        - [Imputation Methods – scikit-learn](https://scikit-learn.org/stable/modules/impute.html)  
        """
    )

# =========================
# 📈 Univariado (Descriptivas)
# =========================
with t1:
    st.markdown(
        """
        ### 📈 Análisis univariado  
        El análisis univariado permite explorar cada variable de manera individual, 
        entendiendo su **distribución**, **variabilidad** y posibles valores atípicos.  
        Aquí se muestran estadísticas descriptivas y visualizaciones por tipo de dato.
        """
    )

    # --- Estadísticas descriptivas ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.subheader("📊 Estadísticas descriptivas")
    c1, c2 = st.columns(2)
    if num_cols:
        with c1:
            st.markdown("**Numéricas**")
            st.code("""df[num_cols].describe().T""",language='python')
            st.dataframe(df[num_cols].describe().T)
    if cat_cols:
        with c2:
            st.markdown("**Categóricas**")
            st.code("""df[cat_cols].describe().T""",language='python')
            st.dataframe(df[cat_cols].describe().T)
    # --- Numéricas (una sola gráfica combinada) ---
    st.divider()
    st.subheader("📈 Distribución de variables numéricas")
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.write(
            "Explora la distribución con **histograma** + **KDE** en una sola vista. "
            "**Tip:** Activa escala log si hay colas largas."
        )

        if num_cols:
            col_num = st.selectbox("Variable numérica", num_cols, key="num_univar")
            bins = st.slider("Bins", 10, 100, 40, key="bins_univar")
            use_log = st.toggle("Escala log en eje Y", value=False, key="log_univar")

            # Datos limpios
            serie = df[col_num].dropna().astype(float)

            # Gráfica combinada: histograma + KDE (Figure Factory)
            # Nota: ff.create_distplot ya combina histograma + curva KDE
            fig_dist = ff.create_distplot(
                [serie.values], [col_num],
                bin_size=(serie.max() - serie.min()) / bins if bins else None,
                show_hist=True, show_rug=False
            )

            # Líneas de referencia: media y mediana
            mean_v = float(serie.mean())
            median_v = float(serie.median())
            fig_dist.add_vline(x=mean_v, line_dash="dash", annotation_text=f"mean={mean_v:.2f}")
            #!fig_dist.add_vline(x=median_v, line_dash="dot", annotation_text=f"median={median_v:.2f}")

            # Escala log opcional en Y
            fig_dist.update_yaxes(type="log" if use_log else "linear")

            fig_dist.update_layout(title=f"Histograma + KDE: {col_num}", showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No hay columnas numéricas disponibles.")

    with c2:
        if num_cols:
            # Estadísticos de apoyo (skew/kurtosis) para la variable seleccionada
            sel = df[col_num].dropna().astype(float)
            skew_v = float(sel.skew())
            kurt_v = float(sel.kurt())

            # Boxplot compacto para contexto (misma variable)
            st.markdown("**Boxplot (resumen de dispersión)**")
            box_fig = px.box(sel.to_frame(name=col_num), y=col_num)
            box_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(box_fig, use_container_width=True)

            sc1, sc2 = st.columns(2)
            sc1.metric("Skewness", f"{skew_v:.2f}")
            sc2.metric("Kurtosis", f"{kurt_v:.2f}")

            st.markdown(
                """
                **Tips de interpretación**  
                - **Skewness > 0**: cola a la derecha (pocos valores muy altos).  
                - **Skewness < 0**: cola a la izquierda.  
                - **Kurtosis** alta: colas pesadas (posibles outliers).  
                - Considera **mediana/IQR** si hay sesgo; puedes probar **log** en Y.
                """
            )

    st.divider()
    
    st.subheader("📊 Distribución de variables categóricas")
         
    c1, c2 = st.columns(2) 
    with c2:    
        # --- Categóricas ---
        if cat_cols:
            
            
                col_cat = st.selectbox("Selecciona variable categórica", cat_cols)
                top_n = st.slider("Mostrar Top N categorías", 5, 20, 10)
                
                counts = df[col_cat].value_counts(dropna=False).reset_index()
                counts.columns = [col_cat, "count"]
                counts["pct"] = counts["count"] / counts["count"].sum() * 100
                counts = counts.head(top_n)

                st.dataframe(counts)

                fig_bar = px.bar(
                    counts, x=col_cat, y="count",
                    text="pct", title=f"Top {top_n} categorías: {col_cat}",
                    labels={"count": "Frecuencia"},
                    color_discrete_sequence=["#EF553B"]
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)



    st.subheader("🔢 Conteo de valores únicos")
    st.dataframe(df.nunique(dropna=True).to_frame("Valores únicos"))




# =========================
# 🔗 Relaciones multivariadas
# =========================
with t3:
    st.subheader("🔗 Relaciones multivariadas")
    st.write("""### 🔗 Paso 3: Relaciones multivariadas  

En este paso buscamos entender **cómo se relacionan varias variables entre sí**.  
Algunas preguntas clave que nos ayuda a responder son:  

- 📈 **¿Qué variables se mueven juntas?**  
  (Correlaciones Pearson, Spearman, Kendall para detectar asociaciones lineales o monótonas).  

- 🔍 **¿Existen relaciones no lineales o subgrupos escondidos?**  
  (Gráficos de dispersión y densidad permiten ver heterocedasticidad, clusters o outliers).  

- 🧮 **¿Cómo varía una numérica según una categórica?**  
  (Boxplots o violinplots muestran distribución, medianas y valores extremos por categoría).  

- ⚠️ **¿Hay multicolinealidad entre variables numéricas?**  
  (Correlaciones altas sugieren redundancia y posibles problemas en modelos).  

- 🧭 **¿Qué combinaciones de variables concentran la mayor varianza del dataset?**  
  (PCA identifica componentes principales y pesos de cada variable).  

En conjunto, estas herramientas permiten **descubrir dependencias, patrones ocultos y estructura global** del dataset, para guiar hipótesis y preparar un modelado más sólido.
""")
    st.divider()
    st.subheader("Matriz de correlación")
    c0, c1 = st.columns(2)
    c0.write("""Mide asociación entre dos variables:
  - **Pearson (ρ)**: relación **lineal**; asume continuidad y es **sensible a outliers**.
  - **Spearman (ρₛ)**: correlación de **rangos**; captura relaciones **monótonas** (no requiere linealidad) y es más **robusta a outliers**.
  - **Kendall (τ)**: basada en **pares concordantes/discordantes**; similar a Spearman, **más conservadora** y estable en **muestras pequeñas**.
  - **Guía de magnitudes** (orientativa): |corr| < 0.3 débil, 0.3–0.6 moderada, > 0.6 fuerte.
  - **Cuidado**: correlación ≠ causalidad; muchas pruebas simultáneas ⇒ usa **Bonferroni/FDR**.
""")
    
    # ---------- Parámetros globales ----------
    with c0.expander("Parámetros", expanded=False):
        cA, cB = st.columns(2)
        corr_method = cA.selectbox("Método correlación", ["pearson","spearman","kendall"], index=0)
        cluster_heatmap = cB.checkbox("Clusterizar heatmap", True, help="Ordena variables por similitud |corr|.")
        sample_on = cA.checkbox("Muestrear para dispersión", True, help="Acelera gráficos con muchos puntos.")
        max_points = cB.slider("Muestra máx.", 500, 50000, 5000, 500)

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

    # ---------- 1) Matriz de correlación ----------

    if len(df.select_dtypes(np.number).columns) >= 2:
        num_cols = df.select_dtypes(np.number).columns.tolist()  # asegura consistencia si cambió antes
        corr = _corr_cached(df[num_cols], corr_method)
        if cluster_heatmap:
            corr = _cluster_corr(corr)

        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            labels=dict(color="ρ"),
            title=f"Correlación numérica ({corr_method})"
        )
        c1.plotly_chart(fig_corr, use_container_width=True)
    else:
        c1.info("Necesitas al menos 2 variables numéricas.")

    c0.code(
            """corr = df[num_cols].corr(method=corr_method)
corr = _cluster_corr(corr) if cluster_heatmap else corr
fig = px.imshow(corr, text_auto=True, labels=dict(color="ρ"))
st.plotly_chart(fig, use_container_width=True)""",
            language="python"
        )

    st.divider()

    # ---------- 2) Dispersión numérica vs numérica ----------
    c0, c1 = st.columns(2)
    c1.subheader("Dispersión numérica vs numérica")
    c1.write("""Útil para ver **forma**, **no linealidad**, **heterocedasticidad**, **clusters** y **outliers**.  
    Si hay muchos puntos, usa **densidad (hex)** o **contornos** y colorea por **categoría** (baja cardinalidad).""")
    if len(num_cols) >= 2:
        with c1:
            sc1, sc2, sc3 = st.columns(3)
            x_col = sc1.selectbox("X", num_cols, index=0, key="scat_x")
            y_opts = [c for c in num_cols if c != x_col] or num_cols
            y_col = sc2.selectbox("Y", y_opts, index=0, key="scat_y")

            # Color opcional por categórica de baja cardinalidad
            cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
            small_cats = [c for c in cat_cols if df[c].nunique(dropna=True) <= 25]
            color_opt = sc3.selectbox("Color por (opcional)", ["Ninguno"] + small_cats, index=0)
            color_arg = None if color_opt == "Ninguno" else color_opt

        kind = c1.radio("Tipo", ["Puntos","Densidad (hex)","Contornos"], horizontal=True)

        if kind == "Puntos":
            fig_s = px.scatter(df_plot, x=x_col, y=y_col, color=color_arg, opacity=0.7, hover_data=df_plot.columns)
        elif kind == "Densidad (hex)":
            fig_s = px.density_heatmap(df_plot, x=x_col, y=y_col, nbinsx=40, nbinsy=40,
                                       marginal_x="histogram", marginal_y="histogram")
        else:
            fig_s = px.density_contour(df_plot, x=x_col, y=y_col, color=color_arg,
                                       contours_coloring="fill", nbinsx=40, nbinsy=40)

        c0.plotly_chart(fig_s, use_container_width=True)
    else:
        c0.info("Selecciona al menos dos columnas numéricas.")

    c1.code(
            """fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_arg, opacity=0.7)
st.plotly_chart(fig, use_container_width=True)""",
            language="python"
        )

    st.divider()

    # ---------- 3) Numérica vs categórica (violin + box + Top-N) ----------
    c0, c1 = st.columns(2)
    c0.subheader("Distribución numérica por categoría")
    c0.write("**Violin/box** para distribución y outliers por grupo; limita a **Top-N** categorías y añade **resumen** (count/mean/median/std).")

    if cat_cols and num_cols:
        with c0:
            sc1, sc2 = st.columns(2)
            cat_sel = sc1.selectbox("Categoría", cat_cols, key="cat_box")
            num_sel = sc2.selectbox("Numérica", num_cols, key="num_box")
        top_n = c0.slider("Top-N categorías por frecuencia", 3, 30, 12)

        top_cats = df_plot[cat_sel].value_counts(dropna=False).head(top_n).index
        data_cat = df_plot[df_plot[cat_sel].isin(top_cats)].copy()

        fig_violin = px.violin(data_cat, x=cat_sel, y=num_sel, box=True, points="outliers",
                               title=f"{num_sel} por {cat_sel} (Top-{top_n})")
        c1.plotly_chart(fig_violin, use_container_width=True)

        summary = (
            data_cat.groupby(cat_sel, dropna=False)[num_sel]
            .agg(count="count", mean="mean", median="median", std="std")
            .sort_values("mean", ascending=False)
        )
        c1.dataframe(summary, use_container_width=True)
    else:
        c1.info("Necesitas ≥1 columna categórica y ≥1 numérica.")

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
    c1.subheader("PCA (2 componentes)")
    c1.markdown("""
**¿Qué responde PCA?**  
- ¿Cuántas **dimensiones latentes** explican la mayor parte de la variación?
- ¿Qué **variables** impulsan cada componente (loadings)?
- ¿Existen **clústeres** o **separación** entre grupos en 2D?

**Cómo leer los resultados**  
- **Varianza explicada**: PC1 explica x%, PC2 y% (suma ≈ información retenida en 2D). 
- **Loadings**: valores altos (±) ⇒ variables **más influyentes**. Un mismo signo en varias variables sugiere un **eje común** (p. ej., “tamaño/ingreso”).  
- **Gráfico PC1 vs PC2**: busca **grupos**, **gradientes** (color por categoría), **outliers**.

**Buenas prácticas**  
- Solo **numéricas** y **sin NaNs**; aplicar **StandardScaler** (z-score).  
- Revisa **outliers** antes: pueden dominar componentes.  
- PC1/PC2 con muy baja varianza ⇒ quizá requieras **más PCs** o técnicas no lineales (t-SNE/UMAP).

**Limitaciones**  
- Es **lineal** y **no supervisado**: no maximiza separación por target.  
- Componentes son combinaciones; la **interpretación** depende de los **loadings**.
""")
    c0.code(
                """X = df[num_cols].dropna()
Xs = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(Xs)
scores = pca.transform(Xs)
explained = pca.explained_variance_ratio_
pca_df = pd.DataFrame(scores, columns=["PC1","PC2"], index=X.index)
fig = px.scatter(pca_df, x="PC1", y="PC2", title=f"Var explicada: {explained[:2].sum():.1%}")
st.plotly_chart(fig, use_container_width=True)""",
                language="python"
            )
    if len(num_cols) >= 2:
        X = df[num_cols].dropna(axis=0)
        idx = X.index

        # Color opcional por categórica de baja cardinalidad (alineada al índice)
        small_cats_full = [c for c in cat_cols if df[c].nunique(dropna=True) <= 25] if cat_cols else []
        color_pca_opt = c0.selectbox("Color por (opcional)", ["Ninguno"] + small_cats_full, index=0, key="pca_color")
        color_series = None if color_pca_opt == "Ninguno" else df.loc[idx, color_pca_opt]

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
        c0.info("PCA requiere ≥2 columnas numéricas.")


# =========================
# 🚨 Outliers & Anomalías
# =========================
with t4:
    st.subheader("🚨 Outliers & Anomalías")

    st.markdown("""
    **¿Qué responde este paso?**
    - ¿Cuántas y cuáles observaciones caen fuera de lo esperado por **dispersión**?
    - ¿Los extremos son puntuales o sistemáticos por subgrupo?
    - ¿Existen **anomalías multivariantes** que no se ven univariadamente?

    **Teoría rápida**
    - **IQR (Tukey)**: marca outliers fuera de \\[Q1 − k·IQR, Q3 + k·IQR\\]. Sencillo y robusto a no-normalidad.
    - **z-score robusto (MAD)**: usa mediana y **MAD** → menos sensible a outliers que el z-score clásico.
    - **Percentiles**: define umbrales empíricos (p.ej., 1% y 99%) cuando la forma es muy rara o multimodal.
    - **Multivariante** (IsolationForest): detecta puntos raros considerando **todas las numéricas a la vez**.
    """)

    num_cols = df.select_dtypes(np.number).columns.tolist()
    if not num_cols:
        st.info("No hay columnas numéricas disponibles.")
    else:
        # ---------- Parámetros ----------
        c0, c1, c2 = st.columns(3)
        col_out = c0.selectbox("Variable numérica", num_cols, key="outlier_var")
        method = c1.radio("Método", ["IQR", "Z robusto (MAD)", "Percentiles"], horizontal=True, key="out_method")

        # Controles dependientes del método
        k = None; z_thr = None; q_lo = None; q_hi = None
        if method == "IQR":
            k = c2.slider("k (IQR)", 1.0, 5.0, 1.5, 0.1)
        elif method == "Z robusto (MAD)":
            z_thr = c2.slider("|z| robusto", 2.0, 6.0, 3.5, 0.1)
        else:
            c2.empty()
            c3, c4 = st.columns(2)
            q_lo = c3.slider("Percentil inferior", 0.0, 10.0, 1.0, 0.1)
            q_hi = c4.slider("Percentil superior", 90.0, 100.0, 99.0, 0.1)

        serie = df[col_out].astype(float)
        serie_clean = serie.dropna()
        n_valid = int(serie_clean.shape[0])

        # ---------- Cálculo de outliers ----------
        mask = pd.Series(False, index=df.index)
        lo = hi = None
        score = pd.Series(np.nan, index=df.index)  # "fuerza" del outlier para ordenar

        if n_valid == 0:
            st.warning("La variable seleccionada solo contiene nulos.")
        else:
            if method == "IQR":
                Q1, Q3 = serie_clean.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lo, hi = Q1 - k * IQR, Q3 + k * IQR
                mask = (serie < lo) | (serie > hi)
                # distancia relativa al rango IQR para ordenar
                med = float(serie_clean.median())
                score = (serie - med).abs() / (IQR if IQR != 0 else 1.0)

            elif method == "Z robusto (MAD)":
                med = float(serie_clean.median())
                mad = float((serie_clean - med).abs().median())
                if mad == 0:
                    st.info("MAD = 0; no se puede calcular z robusto. Prueba IQR o percentiles.")
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
                score = (serie - med).abs()  # distancia absoluta a la mediana

            n_out = int(mask.fillna(False).sum())
            pct_out = (n_out / n_valid * 100) if n_valid > 0 else 0.0

            # ---------- Visualizaciones ----------
            st.subheader("Boxplot (vista rápida)")
            st.plotly_chart(px.box(df, y=col_out, points="outliers",
                                   title=f"Boxplot: {col_out}"),
                            use_container_width=True)

            st.subheader("Distribución y umbrales")
            fig_hist = px.histogram(serie_clean, nbins=50, opacity=0.85,
                                    title=f"Histograma: {col_out}")
            # Líneas de umbral
            try:
                if method in ["IQR", "Percentiles"]:
                    fig_hist.add_vline(x=lo, line_dash="dash", annotation_text=f"lo={lo:,.3g}")
                    fig_hist.add_vline(x=hi, line_dash="dash", annotation_text=f"hi={hi:,.3g}")
                elif method == "Z robusto (MAD)":
                    fig_hist.add_vline(x=med, line_dash="dot", annotation_text=f"mediana={med:,.3g}")
            except Exception:
                pass
            st.plotly_chart(fig_hist, use_container_width=True)

            # ---------- Métricas + tabla ----------
            cA, cB = st.columns(2)
            cA.metric("Outliers detectados", f"{n_out:,}")
            cB.metric("% sobre válidos", f"{pct_out:.2f}%")

            st.subheader("Observaciones atípicas (Top-N por 'fuerza')")
            top_n = st.slider("Top-N a mostrar", 10, 200, 50, 10)
            out_df = (
                df.loc[mask].assign(_out_score=score[mask])
                .sort_values("_out_score", ascending=False)
                .head(top_n)
            )
            st.dataframe(out_df, use_container_width=True)

            # Descargar CSV de outliers
            if n_out > 0:
                csv_buf = io.StringIO()
                df.loc[mask].assign(_out_score=score[mask]).to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Descargar outliers (CSV)",
                    data=csv_buf.getvalue(),
                    file_name=f"outliers_{col_out}_{method.replace(' ','_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # ---------- Código (expander) ----------
            with st.expander("Ver código (detección univariante)"):
                st.code(
                    """# IQR
Q1, Q3 = x.quantile([0.25, 0.75]); IQR = Q3 - Q1
lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
mask_iqr = (x < lo) | (x > hi)

# z robusto (MAD)
med = x.median(); mad = (x - med).abs().median()
robust_z = (x - med) / (1.4826 * mad)
mask_mad = robust_z.abs() > 3.5

# percentiles
lo = np.percentile(x.dropna(), 1); hi = np.percentile(x.dropna(), 99)
mask_pct = (x < lo) | (x > hi)""",
                    language="python"
                )

        # ---------- Multivariante opcional ----------
        st.divider()
        st.subheader("🔀 Detección multivariante (IsolationForest) — opcional")
        st.caption("Útil cuando un punto no es extremo en 1D, pero sí al combinar varias numéricas.")

        enable_if = st.checkbox("Activar IsolationForest", value=False)
        if enable_if:
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA

                feats = st.multiselect("Variables numéricas a considerar", num_cols,
                                       default=num_cols[:min(5, len(num_cols))])
                if len(feats) == 0:
                    st.info("Selecciona al menos 1 variable numérica.")
                else:
                    c1, c2 = st.columns(2)
                    contamination = c1.slider("Proporción esperada de anomalías", 0.01, 0.20, 0.05, 0.01)
                    n_estimators = c2.slider("Árboles (n_estimators)", 50, 400, 200, 50)

                    X = df[feats].dropna()
                    if X.empty:
                        st.info("No hay filas completas para las variables seleccionadas.")
                    else:
                        iso = IsolationForest(
                            contamination=contamination,
                            n_estimators=n_estimators,
                            random_state=42
                        ).fit(X)

                        pred = iso.predict(X)  # -1 = outlier
                        score_if = iso.decision_function(X)  # menor = más raro
                        res = df.loc[X.index].copy()
                        res["_anomaly_if"] = (pred == -1)
                        res["_if_score"] = score_if

                        n_anom = int(res["_anomaly_if"].sum())
                        pct_anom = n_anom / res.shape[0] * 100
                        cA, cB = st.columns(2)
                        cA.metric("Anomalías (IForest)", f"{n_anom:,}")
                        cB.metric("% sobre válidos", f"{pct_anom:.2f}%")

                        # Vista en PCA 2D para interpretar
                        Xs = StandardScaler().fit_transform(X)
                        p2 = PCA(n_components=2).fit_transform(Xs)
                        plot_df = pd.DataFrame({
                            "PC1": p2[:, 0],
                            "PC2": p2[:, 1],
                            "Anomalía": np.where(res["_anomaly_if"], "Sí", "No")
                        }, index=X.index)
                        fig_if = px.scatter(
                            plot_df, x="PC1", y="PC2", color="Anomalía",
                            title="IsolationForest en espacio PCA (2D)",
                            opacity=0.85
                        )
                        st.plotly_chart(fig_if, use_container_width=True)

                        st.subheader("Muestras más anómalas (según score)")
                        show_n = st.slider("Mostrar Top-N", 10, 200, 50, 10, key="if_topn")
                        st.dataframe(
                            res.sort_values("_if_score").head(show_n),
                            use_container_width=True
                        )

                        with st.expander("Ver código (IsolationForest)"):
                            st.code(
                                """from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42).fit(X)
pred = iso.predict(X)          # -1 = outlier
score = iso.decision_function(X)  # menor = más raro
mask_if = pred == -1""",
                                language="python"
                            )
            except Exception as e:
                st.info(f"No se pudo activar IsolationForest: {e}")

    # ---------- Sugerencias de tratamiento ----------
    st.divider()
    with st.expander("💊 Tratamiento: opciones habituales"):
        st.markdown("""
        - **Inspección manual** y corrección si es error de captura.
        - **Winsorización (clip)** en percentiles (p.ej., 1%–99%).
        - **Transformaciones** (log/sqrt/Box-Cox) si hay sesgo fuerte.
        - **Modelado robusto** (árboles/boosting, regularización) si no quieres remover.
        - **Eliminar** casos extremos solo si justificas su falta de representatividad.

        **Snippet winsorización:**
        ```python
        lo, hi = x.quantile([0.01, 0.99])
        x_wins = x.clip(lo, hi)
        ```
        """)
# =========================
# 🧩 Categóricas & Balance de clases
# =========================
with t5:
    st.subheader("🧩 Categóricas & Balance de clases")

    # Asegura listas de columnas (recalcula por si cambió df)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.markdown("""
    **¿Qué responde este paso?**  
    - ¿Cuáles son las categorías más **frecuentes** (Top-N) y cuánto representan (%)?  
    - ¿Hay **desbalance** fuerte entre clases que pueda afectar el modelado?  
    - ¿Cómo contribuye cada categoría a un **valor numérico total** (treemap)?  
    - ¿Cómo se comporta un **objetivo numérico** dentro de cada categoría (medias/medianas)?  
    """)

    st.divider()

    # =========================
    # 1) Treemap (Categoría -> Valor numérico)
    # =========================
    if cat_cols and num_cols:
        c0, c1 = st.columns(2)

        c0.subheader("🌳 Treemap (Categoría → Valor)")
        c0.write(
                "El **treemap** muestra la contribución de cada categoría a un **total numérico** "
                "(área proporcional al valor). Útil para detectar categorías dominantes."
            )
        c0.code(
                    """cat_t = st.selectbox("Categoría", cat_cols, key="t5_treemap_cat")
val_t = st.selectbox("Valor numérico", num_cols, key="t5_treemap_val")
fig_tm = px.treemap(df, path=[cat_t], values=val_t, title=f"Treemap: {cat_t} → {val_t}")
st.plotly_chart(fig_tm, use_container_width=True)""",
                    language="python"
                )

        with c0:
            sc0, sc1 = st.columns(2)
            cat_t = sc0.selectbox("Categoría", cat_cols, key="t5_treemap_cat")
            val_t = sc1.selectbox("Valor numérico", num_cols, key="t5_treemap_val")
            fig_tm = px.treemap(df, path=[cat_t], values=val_t, title=f"Treemap: {cat_t} → {val_t}")
        
        c1.plotly_chart(fig_tm, use_container_width=True)

    else:
        st.info("Para el Treemap se requiere al menos una columna categórica y una numérica.")

    st.divider()

    # =========================
    # 2) Distribución de categorías (Top-N + 'Otros') + % robusto
    # =========================
    if cat_cols:
        c0, c1 = st.columns(2)

        c1.subheader("📊 Distribución de categorías (Top-N + 'Otros')")
        c1.write(
                "Mostramos **Top-N** categorías por frecuencia y agrupamos el resto en **'Otros'** "
                "para mantener legibilidad. Además, calculamos el **%** de cada categoría."
            )

        c1.code(
                    """col_cat = st.selectbox("Variable categórica", cat_cols, key="t5_cat_var")
top_n = st.slider("Top-N categorías", 3, 30, 10, key="t5_topn")

# Conteos robustos
vc = df[col_cat].value_counts(dropna=False)          # Serie de conteos
top = vc.head(top_n)
others = vc.iloc[top_n:].sum()

plot_s = top.copy()
if others > 0:
    plot_s.loc["Otros"] = int(others)

# DataFrame final sin duplicados y con numéricos
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
    title=f"Distribución: {col_cat}",
    labels={"count": "Frecuencia"}
)
fig_bar.update_traces(textposition="outside", cliponaxis=False)
st.plotly_chart(fig_bar, use_container_width=True)""",
                    language="python"
                )

        with c0:
            # --- UI y cálculo robusto ---
            col_cat = st.selectbox("Variable categórica", cat_cols, key="t5_cat_var")
            top_n = st.slider("Top-N categorías", 3, 30, 10, key="t5_topn")

            vc = df[col_cat].value_counts(dropna=False)   # Serie
            top = vc.head(top_n)
            others = vc.iloc[top_n:].sum()

            plot_s = top.copy()
            if others > 0:
                plot_s.loc["Otros"] = int(others)

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
                title=f"Distribución: {col_cat}",
                labels={"count": "Frecuencia"}
            )
            fig_bar.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ===== Balance rápido de clases =====
        # Basado en los conteos "puros" (sin 'Otros') para evaluar desbalance
        major_cat = vc.index[0] if len(vc) else None
        if major_cat is not None:
            major_count = int(vc.iloc[0])
            total_all = int(vc.sum())
            major_pct = (major_count / total_all * 100) if total_all else 0.0

            with c0:
                cA, cB, cC = st.columns(3)
                cA.metric("Categoría mayoritaria", str(major_cat))
                cB.metric("Frecuencia mayoritaria", f"{major_count:,}")
                cC.metric("% mayoritario", f"{major_pct:.1f}%")

            if major_pct >= 80:
                st.warning("Desbalance fuerte (≥80/20). Considera técnicas: reponderación, undersampling/oversampling, métricas por clase.")

    else:
        st.info("No hay columnas categóricas disponibles.")

    st.divider()

    # =========================
    # 3) Relación con objetivo numérico (media/mediana por categoría)
    # =========================
    if cat_cols and num_cols:
        c0, c1 = st.columns(2)

        with c0:
            st.subheader("🎯 Objetivo numérico por categoría")
            st.write(
                "Compara un objetivo numérico entre categorías usando **media** o **mediana** "
                "(robusta a outliers). Para legibilidad, se limita a las **Top-N** categorías."
            )
            st.code(
                    """col_cat2 = st.selectbox("Categoría", cat_cols, key="t5_target_cat")
target = st.selectbox("Objetivo numérico", num_cols, key="t5_target_val")
agg_fn = st.selectbox("Agregación", ["mean", "median", "sum"], index=0, key="t5_target_agg")
top_n_target = st.slider("Top-N (por frecuencia)", 3, 30, 10, key="t5_target_topn")

vc2 = df[col_cat2].value_counts(dropna=False)
keep = vc2.head(top_n_target).index
data_f = df[df[col_cat2].isin(keep)]

agg = getattr(data_f.groupby(col_cat2, dropna=False)[target], agg_fn)().reset_index()
agg = agg.sort_values(by=agg.columns[1], ascending=False)

fig_t = px.bar(agg, x=col_cat2, y=agg.columns[1],
               title=f"{target} ({agg_fn}) por {col_cat2}",
               labels={agg.columns[1]: f"{target} ({agg_fn})"})
st.plotly_chart(fig_t, use_container_width=True)""",
                    language="python"
                )

        with c1:
            sc0, sc1, sc2 = st.columns(3)
            col_cat2 = sc0.selectbox("Categoría", cat_cols, key="t5_target_cat")
            target = sc1.selectbox("Objetivo numérico", num_cols, key="t5_target_val")
            agg_fn = sc2.selectbox("Agregación", ["mean", "median", "sum"], index=0, key="t5_target_agg")
            top_n_target = st.slider("Top-N (por frecuencia)", 3, 30, 10, key="t5_target_topn")

            vc2 = df[col_cat2].value_counts(dropna=False)
            keep = vc2.head(top_n_target).index
            data_f = df[df[col_cat2].isin(keep)]

            agg_series = getattr(data_f.groupby(col_cat2, dropna=False)[target], agg_fn)()
            agg = agg_series.reset_index().sort_values(by=target, ascending=False)

            fig_t = px.bar(
                agg, x=col_cat2, y=target,
                title=f"{target} ({agg_fn}) por {col_cat2}",
                labels={target: f"{target} ({agg_fn})"}
            )
            st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Para esta sección se requiere al menos una categórica y una numérica.")

# =========================
# ⏳ Series de tiempo (t6 completo y robusto)
# =========================
with t6:
    st.subheader("⏳ Series de tiempo")

    # ---------- Helpers ----------
    def _is_parseable_datetime(s: pd.Series, min_ok: float = 0.5) -> bool:
        """True si >= min_ok se puede parsear a datetime."""
        if np.issubdtype(s.dtype, np.datetime64):
            return True
        try:
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            return parsed.notna().mean() >= min_ok
        except Exception:
            return False

    def _ensure_datetime_col(df_: pd.DataFrame, col: str) -> pd.Series:
        """Convierte a datetime si no lo es."""
        if np.issubdtype(df_[col].dtype, np.datetime64):
            return df_[col]
        return pd.to_datetime(df_[col], errors="coerce", infer_datetime_format=True)

    def _looks_like_id(name: str) -> bool:
        """Heurística simple para excluir identificadores numéricos."""
        n = str(name).lower()
        return (
            n == "id" or n == "index" or
            n.endswith("_id") or n.startswith("id_") or
            n.endswith(" id") or n.startswith("id ")
        )

    # ---------- Detección de columnas de fecha ----------
    date_candidates = [c for c in df.columns if _is_parseable_datetime(df[c])]

    if not date_candidates:
        st.info("No se detectaron columnas de fecha.")
    else:
        # ---------- Layout 2 columnas ----------
        c0, c1 = st.columns(2)

        # ===== Lado izquierdo: teoría + parámetros =====
        with c0:
            st.subheader("Qué responde")
            st.write(
                "- ¿Cómo evoluciona la métrica en el tiempo (tendencia/estacionalidad)?\n"
                "- ¿Cómo afectan la **frecuencia** y la **agregación** (sum/mean/…)?\n"
                "- ¿Cuál es el **cambio %** período a período o **interanual (YoY)**?\n"
                "- ¿Hay diferencias por **categoría** (líneas múltiples)?"
            )

            st.subheader("Parámetros")
            date_col = st.selectbox("Columna de fecha", date_candidates, key="ts_date_col")

        # Construir df_ts con fecha válida ANTES de elegir la métrica
        df_ts = df.copy()
        df_ts[date_col] = _ensure_datetime_col(df_ts, date_col)
        df_ts = df_ts.dropna(subset=[date_col]).sort_values(by=date_col)

        # Numéricas válidas (excluye identificadores)
        num_cols_ts = [
            c for c in df_ts.select_dtypes(np.number).columns
            if c != date_col and not _looks_like_id(c)
        ]

        # Categóricas de baja cardinalidad
        cat_cols_all = df_ts.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        small_cats = [c for c in cat_cols_all if df_ts[c].nunique(dropna=True) <= 12]

        with c0:
            if not num_cols_ts:
                st.warning("No hay columnas numéricas válidas para series de tiempo (excluyendo identificadores como 'id').")
            else:
                val_col = st.selectbox("Variable numérica", num_cols_ts, key="ts_val_col")
                split_col = st.selectbox("Dividir por categoría (opcional)", ["Ninguno"] + small_cats, key="ts_split")

                cA, cB, cC = st.columns(3)
                freq_lbl = cA.selectbox("Frecuencia", ["D (día)", "W (semana)", "M (mes)", "Q (trimestre)", "Y (año)"], index=2, key="ts_freq")
                agg_fn_name = cB.selectbox("Agregación", ["sum", "mean", "median", "max", "min"], index=0, key="ts_agg")
                fill_mode = cC.selectbox("Relleno faltantes", ["Ninguno", "Forward-fill", "Cero"], index=0, key="ts_fill")

                cD, cE = st.columns(2)
                pct_kind = cD.selectbox("% cambio", ["Ninguno", "Período a período", "Interanual (YoY)"], index=0, key="ts_pct")
                show_ma = cE.checkbox("Media móvil", value=True, key="ts_ma")
                ma_window = st.slider("Ventana MA (periodos)", 2, 60, 12, key="ts_ma_win") if show_ma else None

                st.code(
                        """# Resample básico (una serie)
ts = (df.set_index(date_col).sort_index()[val_col]
        .resample('M').sum()
        .reset_index())""",
                        language="python"
                    )

        # ===== Lado derecho: resultados =====
        with c1:
            if not num_cols_ts:
                st.stop()

            freq_map = {"D (día)": "D", "W (semana)": "W", "M (mes)": "M", "Q (trimestre)": "Q", "Y (año)": "Y"}
            f = freq_map[freq_lbl]

            if split_col == "Ninguno":
                # --- Serie única ---
                s = (
                    df_ts.set_index(date_col)
                         .loc[:, val_col]
                         .resample(f)
                         .agg(agg_fn_name)
                )
                if fill_mode == "Forward-fill":
                    s = s.ffill()
                elif fill_mode == "Cero":
                    s = s.fillna(0)

                ts = s.reset_index().rename(columns={val_col: val_col})

                # Media móvil
                ys = [val_col]
                if show_ma:
                    ts["MA"] = ts[val_col].rolling(int(ma_window)).mean()
                    ys.append("MA")

                fig_ts = px.line(
                    ts, x=date_col, y=ys,
                    title=f"{val_col} — {agg_fn_name} por {f}",
                    labels={date_col: "Fecha"}
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                # % cambio
                if pct_kind != "Ninguno":
                    if pct_kind == "Período a período":
                        periods = 1
                    else:
                        shift_map = {"M": 12, "Q": 4, "W": 52, "D": 365, "Y": 1}
                        periods = shift_map.get(f, 1)
                    ts["pct_change"] = ts[val_col].pct_change(periods=periods) * 100
                    fig_pct = px.line(
                        ts, x=date_col, y="pct_change",
                        title=("Cambio % período a período" if periods == 1 else "Cambio % interanual (YoY)"),
                        labels={"pct_change": "%"}
                    )
                    st.plotly_chart(fig_pct, use_container_width=True)

                st.dataframe(ts.tail(12), use_container_width=True)

                st.code(
                        """# % cambio
periods = 12  # ejemplo mensual YoY
ts["pct_change"] = ts[val_col].pct_change(periods=periods) * 100""",
                        language="python"
                    )

            else:
                # --- Múltiples series por categoría ---
                grp = (
                    df_ts.groupby([pd.Grouper(key=date_col, freq=f), split_col])[val_col]
                         .agg(agg_fn_name)
                         .reset_index()
                )

                # Relleno por grupo (no crea nuevas fechas, solo rellena NaN existentes)
                if fill_mode != "Ninguno":
                    def _fill(g):
                        if fill_mode == "Forward-fill":
                            g[val_col] = g[val_col].ffill()
                        elif fill_mode == "Cero":
                            g[val_col] = g[val_col].fillna(0)
                        return g
                    grp = grp.groupby(split_col, group_keys=False).apply(_fill)

                # Media móvil por grupo
                if show_ma:
                    grp["MA"] = grp.groupby(split_col)[val_col].transform(lambda s: s.rolling(int(ma_window)).mean())

                fig_multi = px.line(
                    grp, x=date_col, y=val_col, color=split_col,
                    title=f"{val_col} — {agg_fn_name} por {f} (por {split_col})"
                )
                st.plotly_chart(fig_multi, use_container_width=True)

                if show_ma:
                    fig_ma = px.line(
                        grp, x=date_col, y="MA", color=split_col,
                        title=f"Media móvil ({ma_window}) por {split_col}"
                    )
                    st.plotly_chart(fig_ma, use_container_width=True)

                # % cambio por grupo
                if pct_kind != "Ninguno":
                    if pct_kind == "Período a período":
                        periods = 1
                    else:
                        shift_map = {"M": 12, "Q": 4, "W": 52, "D": 365, "Y": 1}
                        periods = shift_map.get(f, 1)
                    grp["pct_change"] = grp.groupby(split_col)[val_col].pct_change(periods=periods) * 100
                    fig_pctm = px.line(
                        grp, x=date_col, y="pct_change", color=split_col,
                        title=("Cambio % período a período" if periods == 1 else "Cambio % interanual (YoY)"),
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
# Rolling por grupo
grp["MA"] = grp.groupby(split_col)[val_col].transform(lambda s: s.rolling(12).mean())""",
                        language="python"
                    )

        c0.write("💡 Tips & buenas prácticas")
        c0.markdown(
                "- Si la fecha no es `datetime`, se **parsea automáticamente**.\n"
                "- **Frecuencia** cambia unidad de análisis (D/W/M/Q/Y) y **agregación** define el resumen.\n"
                "- **Forward-fill** es útil para tasas; **Cero** para conteos cuando faltan periodos.\n"
                "- **YoY** requiere al menos un año de historial a la misma frecuencia.\n"
                "- Para descomposición estacional: `statsmodels.tsa.seasonal_decompose` (opcional)."
            )


