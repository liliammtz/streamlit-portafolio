# app_bees_eda_nosidebar.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np 
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, kruskal, chi2_contingency, fisher_exact
import plotly.graph_objects as go


# =========================
# Config & Theme (bee style)
# =========================
st.set_page_config(page_title="Save the Bees 🐝", layout="wide")

BEE_YELLOW = "#fdae61"
BEE_HONEY  = "#f5deb3"
BEE_COMB   = "#fff8dc"
BEE_BROWN  = "#5a381e"

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
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_data(path_or_buffer) -> pd.DataFrame:
    if hasattr(path_or_buffer, "read"):  # file_uploader
        df = pd.read_csv(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer)
    # Normalizaciones ligeras
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "quarter" in df.columns:
        qcat = pd.api.types.CategoricalDtype(categories=["Q1","Q2","Q3","Q4"], ordered=True)
        df["quarter"] = df["quarter"].astype(str).str.upper().str.strip()
        df["quarter"] = df["quarter"].astype(qcat)
    return df

def compute_desc_stats(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["range"] = desc["max"] - desc["min"]
    desc["var"]   = df[num_cols].var(numeric_only=True)
    desc["IQR"]   = desc["75%"] - desc["25%"]
    modes = []
    for col in num_cols:
        m = df[col].mode(dropna=True)
        modes.append(m.iloc[0] if not m.empty else None)
    desc["mode"] = modes
    desc = desc[["mean","50%","mode","min","max","range","std","var","25%","75%","IQR"]]
    desc = desc.rename(columns={"50%":"median"})
    return desc

def compute_shape_stats(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({
        "skewness": df[num_cols].skew(numeric_only=True),
        "kurtosis": df[num_cols].kurt(numeric_only=True)
    }).sort_values(by=["skewness","kurtosis"], ascending=False)
    return out

# =========================
# UI (sin sidebar)
# =========================
st.title("Save the Bees 🐝 – EDA")

# Cargar datos
df = load_data("data/save_the_bees.csv")

# Aplicar filtros
df_f = df.copy()

# ======================
# Helpers
# ======================
def missing_summary(df):
    """% de valores faltantes por columna"""
    return df.isna().mean().mul(100).round(2).to_frame("% NA")

def detect_outliers_iqr(series: pd.Series):
    """Outliers según regla IQR"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask

def detect_outliers_mad(series: pd.Series):
    """Outliers según Z mod (basado en MAD)"""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    z_mod = 0.6745 * (series - median) / mad
    mask = np.abs(z_mod) > 3.5
    return mask

def winsorize_series(series: pd.Series, lower_q=0.01, upper_q=0.99):
    """Winsorización: cap en percentiles inferior y superior"""
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower, upper)

# =========================
# About / Diccionario
# =========================
about_md = """
**About the columns**

- **state**: estado dentro de EE. UU. *Nota:* “other” agrupa varios estados por privacidad; “United States” es el promedio nacional.
- **num_colonies**: número de colmenas.
- **max_colonies**: máximo de colmenas en el trimestre.
- **lost_colonies**: colmenas perdidas en el trimestre.
- **percent_lost**: % de colmenas perdidas en el trimestre.
- **renovated_colonies**: colmenas requeened o repobladas.
- **percent_renovated**: % de colmenas renovadas.
- **quarter**: Q1=Ene–Mar, Q2=Abr–Jun, Q3=Jul–Sep, Q4=Oct–Dic.
- **year**: 2015–2022.
- **varroa_mites**: % afectado por varroa.
- **other_pests_and_parasites**: % afectado por otras plagas/parásitos.
- **diseases**: % afectado por enfermedades.
- **pesticides**: % afectado por pesticidas.
- **other**: % afectado por otra causa no listada.
- **unknown**: % afectado por causa desconocida.
"""

with st.expander("📖 Ver diccionario de datos", expanded=False):
    st.markdown(about_md)

with st.expander("See Exploratory data analysis"):
    # =========================
    # Estadística descriptiva
    # =========================
    t0, t1, t2, t3, t4 = st.tabs(["data quality", "📊 Estadística descriptiva", "🌀 Distribución: Skewness & Kurtosis", "Relaciones Bivariadas", "Análisis de outiers"])

    with t0:
        st.dataframe(missing_summary(df))
        st.caption("Porcentaje de valores nulos (NA) en cada columna.")
        
        st.write("Validar que no haya duplicados")
        #st.dataframe(df.value_counts(['year','quarter','state']).reset_index().sort_values('count'))
    with t1:
        num_cols = [
            "num_colonies", "max_colonies", "lost_colonies", "percent_lost",
            "renovated_colonies", "percent_renovated",
            "varroa_mites", "other_pests_and_parasites", "diseases",
            "pesticides", "other", "unknown"
        ]
        num_cols = [c for c in num_cols if c in df_f.columns]

        desc_stats = compute_desc_stats(df_f, num_cols)

        summary_text = """
        **Resumen ejecutivo**  
        - Los valores **absolutos** están sesgados por unos pocos estados muy grandes → compara con **porcentajes**.  
        - **% perdido** es relativamente estable (medianas ~10%).  
        - **% renovado** es muy heterogéneo (1%–77%).  
        - **Varroa** es el factor **más constante**; los demás son más **episódicos**.
        """
        
        st.markdown(summary_text)
        st.dataframe(desc_stats.round(2), use_container_width=True)

    with t2:
        # =========================
        # Skewness & Kurtosis
        # =========================
        shape_stats = compute_shape_stats(df_f, num_cols)
        c1, c2 = st.columns([0.1,0.1])
        with c1:
            st.dataframe(shape_stats.round(3), use_container_width=True)

        with c2:
            st.markdown(
                """
        **Lectura rápida**  
        - **Skewness > 0** → cola a la derecha (muchos bajos, pocos muy altos).  
        - **Kurtosis alta** → colas pesadas/outliers; la media poco representa.  
        - **Colonias absolutas** son ultra-sesgadas → usar **mediana/IQR** y considerar **log** si vas a modelar.  
        - **Varroa** ≈ distribución más estable (mejor para comparaciones).
                """
            )
        
        @st.fragment
        def get_boxplot(df):
            options = ["percent_lost", "percent_renovated", "South", "West"]
            #selection = st.segmented_control("Directions", options, selection_mode="single")
            selection = options[0]
            fig_boxplot = px.box( df, x="year", y=selection, points="all", # muestra también los outliers como puntos 
                     title=f"Boxplot: {selection} by Year" )
            st.plotly_chart(fig_boxplot)
            
        get_boxplot(df)
            
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr

    with t3:
        def run_stat_test(df, col_x, col_y):
            result_text = f"### 🔎 Prueba estadística entre `{col_x}` y `{col_y}`\n\n"

            # Detectar tipos
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Caso 1: Numérica vs Numérica
            if col_x in num_cols and col_y in num_cols:
                x = df[col_x].dropna()
                y = df[col_y].dropna()
                xy = pd.concat([x, y], axis=1).dropna()
                x, y = xy[col_x], xy[col_y]

                pearson_r, pearson_p = pearsonr(x, y)
                spearman_r, spearman_p = spearmanr(x, y)

                result_text += f"""
        **Numérica vs Numérica**

        - Pearson r = {pearson_r:.3f} (p = {pearson_p:.3g})  
        - Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3g})  

        📌 Interpretación:  
        - Pearson detecta correlaciones lineales si los datos son normales y sin outliers.  
        - Spearman es robusto a outliers y detecta relaciones monótonas.  
        - Si los valores son similares → relación estable y lineal.  
        """
                return result_text

            # Caso 2: Categórica vs Numérica
            elif (col_x in cat_cols and col_y in num_cols) or (col_y in cat_cols and col_x in num_cols):
                cat_var = col_x if col_x in cat_cols else col_y
                num_var = col_y if col_x in cat_cols else col_x

                groups = [vals[num_var].dropna().values for _, vals in df.groupby(cat_var)]

                if len(groups) == 2:
                    # t-test clásico (no verificamos supuestos aquí, simplificado)
                    t_stat, p_val = ttest_ind(groups[0], groups[1], equal_var=False)
                    result_text += f"""
        **Categórica (2 grupos) vs Numérica**

        - Welch t-test (permite varianzas distintas)  
        - Estadístico t = {t_stat:.3f}, p = {p_val:.3g}  

        📌 Interpretación:  
        - Si p < 0.05, hay diferencia significativa entre los grupos de `{cat_var}` respecto a `{num_var}`.
        """
                elif len(groups) > 2:
                    f_stat, p_val = f_oneway(*groups)
                    result_text += f"""
        **Categórica (>2 grupos) vs Numérica**

        - ANOVA (prueba de igualdad de medias entre múltiples grupos)  
        - Estadístico F = {f_stat:.3f}, p = {p_val:.3g}  

        📌 Interpretación:  
        - Si p < 0.05, al menos un grupo difiere en la media de `{num_var}`.  
        - Para saber cuáles, deberías aplicar un post-hoc (Tukey HSD o Dunn).
        """
                return result_text

            # Caso 3: Categórica vs Categórica
            elif col_x in cat_cols and col_y in cat_cols:
                contingency = pd.crosstab(df[col_x], df[col_y])
                chi2, p_val, dof, expected = chi2_contingency(contingency)
                result_text += f"""
        **Categórica vs Categórica**

        - Chi² de independencia  
        - Chi² = {chi2:.3f}, p = {p_val:.3g}, dof = {dof}  

        📌 Interpretación:  
        - Si p < 0.05, hay dependencia entre `{col_x}` y `{col_y}`.
        """
                return result_text

            else:
                return "❌ No se pudo determinar el tipo de variables."

        # =======================
        # Streamlit App
        # =======================
        st.title("🔎 Explorador de correlaciones")

        # Detectar columnas numéricas y categóricas
        num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

        with st.form("correlation_form"):
            st.write("Selecciona dos variables numéricas para calcular la correlación:")

            c1, c2 = st.columns(2)
            with c1:
                var_x = st.selectbox("Variable X (numérica)", options=df.columns, index=0)
            with c2:
                var_y = st.selectbox("Variable Y (numérica)", options=df.columns, index=1)

            submitted = st.form_submit_button("Calcular correlación")

        if submitted:
            result = run_stat_test(df, var_x, var_y)
            st.markdown(result)
            

# 🎨 Paleta abeja
BEE_COLORS = {"percent_lost": "#a6611a", "percent_renovated": "#fdae61"}
BEE_THEME = dict(
    plot_bgcolor="white",#"#fff8dc",
    paper_bgcolor="white",#"#fff8dc",
    font=dict(color="#5a381e", size=13),
    hoverlabel=dict(bgcolor="white", font_size=12, font_color="#5a381e")
)

# ---- FIGURA 1: líneas por año ----
df_year = df.groupby("year", as_index=False)[["percent_lost", "percent_renovated"]].mean()
fig_year = px.line(
    df_year,
    x="year", y=["percent_lost", "percent_renovated"],
    markers=True,
    color_discrete_map=BEE_COLORS,
    title="🐝 Evolución anual: % Lost vs % Renovated"
)
fig_year.update_traces(mode="lines+markers")
fig_year.update_layout(**BEE_THEME, legend_title_text="Variable")

# ---- FIGURA 2: barras por trimestre ----
df_quarter = df.groupby("quarter", as_index=False)[["percent_lost", "percent_renovated"]].mean()
fig_quarter = px.bar(
    df_quarter,
    x="quarter", y=["percent_lost","percent_renovated"],
    barmode="group",
    color_discrete_map=BEE_COLORS,
    title="🍯 Estacionalidad trimestral: % Lost vs % Renovated"
)
fig_quarter.update_layout(**BEE_THEME, legend_title_text="Variable")

c1, c2 = st.columns(2)
c1.plotly_chart(fig_year, use_container_width=True)
c2.plotly_chart(fig_quarter, use_container_width=True)

# ---- FIGURA 3: trayectoria % renovado vs % perdido ----
df_avg = df.groupby("year", as_index=False)[["percent_renovated", "percent_lost"]].mean()

fig_traj = px.line(
    df_avg,
    x="percent_renovated", y="percent_lost",
    text="year",
    markers=True,
    title="📈 Trayectoria anual: % Renovated vs % Lost"
)
fig_traj.update_traces(mode="lines+markers+text", textposition="top center", line=dict(color="#5a381e"))

# Línea de referencia x=y
min_val = min(df_avg["percent_renovated"].min(), df_avg["percent_lost"].min())
max_val = max(df_avg["percent_renovated"].max(), df_avg["percent_lost"].max())
fig_traj.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color="red", dash="dash"),
    name="x=y"
)
fig_traj.update_layout(
    xaxis_title="Mean % Renovated",
    yaxis_title="Mean % Lost",
    hovermode="closest",
    **BEE_THEME
)
st.plotly_chart(fig_traj, use_container_width=True)

# ---- FIGURA 4: mapa coroplético ----
@st.fragment()
def chrolopleth(df):
    relevant_cols = [
        "percent_lost", "percent_renovated",
        "varroa_mites", "other_pests_and_parasites",
        "diseases", "pesticides", "other", "unknown"
    ]

    col = st.segmented_control(
        "Selecciona una variable", 
        options=relevant_cols, 
        default="percent_lost",
        selection_mode="single"
    )
    
    df_state = df.groupby(["year", "quarter", "state_code"], as_index=False)[col].sum() 

    # Normalizar por estado (máximo = 1) 
    df_state["normalized_value"] = df_state.groupby("state_code")[col].transform( lambda x: x / x.max() ) 

    # Crear frame año-trimestre 
    df_state["period"] = df_state["year"].astype(str) + " " + df_state["quarter"].astype(str)

    # Mapeo numérico para ordenar trimestres
    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df_state["quarter_num"] = df_state["quarter"].map(quarter_map)

    # Choropleth con facetas y animación
    fig_map = px.choropleth(
        df_state,
        locations="state_code",
        locationmode="USA-states",
        color="normalized_value",
        color_continuous_scale="YlOrBr",
        scope="usa",
        facet_col="quarter",
        facet_col_wrap=4,
        animation_frame="year",
        labels={"normalized_value": "% of state maximum"},
        hover_data={"state_code": True, col: ":,", "normalized_value": ":.2f"}
    )

    # Leyenda abajo, horizontal, con bins de 10%
    fig_map.update_layout(
        title=f"🌎 {col} by Quarter (Animation over Years)",
        geo=dict(showcoastlines=False),
        margin=dict(l=20, r=20, t=50, b=50),
        coloraxis=dict(
            cmin=0, cmax=1,                          # escala normalizada
            colorbar=dict(
                orientation="h",                     # horizontal
                x=0.5, xanchor="center", y=-0.2,     # posición debajo
                tickvals=np.linspace(0, 1, 11),      # 0.0 → 1.0 en pasos de 0.1
                ticktext=[f"{int(v*100)}%" for v in np.linspace(0, 1, 11)],  # etiquetas %
                title="% of state maximum"
            )
        )
    )

    st.plotly_chart(fig_map, use_container_width=True)

chrolopleth(df)


