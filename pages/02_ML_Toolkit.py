import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF

import matplotlib.pyplot as plt
from io import BytesIO

# =========================
# Layout & Sidebar
# =========================
st.set_page_config(page_title="Machine Learning Toolkit", layout="wide")
st.title("Machine Learning Toolkit")
#st.caption("Teoría → Código → Mini-demo ejecutable. Basado en scikit-learn, pensado para aprender y reutilizar.")

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
    st.markdown("**Tools**")
    st.page_link("pages/01_EDA_Toolkit.py", label="EDA Toolkit", icon="📊")
    st.page_link("pages/02_ML_Toolkit.py", label="Machine Learning Toolkit", icon="📈")
    st.page_link("pages/03_MLOps_Toolkit.py", label="MLOps Toolkit", icon="🧰")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")

    st.divider()
    st.markdown("**Contact**")
    st.markdown("- GitHub: [@liliam-mtz](https://github.com/liliammtz)")
    st.markdown("- LinkedIn: [Liliam Martínez](https://www.linkedin.com/in/liliammtz/)")
    st.markdown("- Email: [liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")

st.info("work in progress")

seed = 42

# Helpers
@st.cache_data
def load_data_classification():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

@st.cache_data
def load_data_regression():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

@st.cache_data
def load_data_clustering():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # solo para inspección
    return X, y

def render_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    st.pyplot(fig)

def download_bytes(content: bytes, filename: str, label: str):
    st.download_button(label, data=content, file_name=filename)

# =========================
# Tabs principales (mantengo estructura)
# =========================
t0, t1, t2 = st.tabs([
    "Supervised Learning",
    "Unsupervised Learning"
])

# ============================================================
# 1) SUPERVISED
# ============================================================
with t0:
    st.subheader("🎯 Supervised Learning")
    st.caption("Predicción con variable objetivo etiquetada (y). Clasificación (y discreta) y regresión (y continua).")

    colL, colR = st.columns([1.1, 1])
    with colL:
        st.markdown("""
**Flujo conceptual**
1) Define objetivo (**target**) y **métrica primaria** (éxito ≠ 100% accuracy).  
2) **Train/Test** (+ **CV** si aplica).  
3) Preprocesa con **Pipeline**/**ColumnTransformer** (num: escalado; cat: one-hot).  
4) Entrena **modelos base** con la misma métrica y CV.  
5) Elige finalista, **tunea hiperparámetros**, evalúa en **test**.  
6) **Interpreta y documenta**: errores, supuestos, límites, próximos pasos.
""")
        st.markdown("—")

        st.markdown("### 🧩 Clasificación (teoría)")
        st.markdown("""
- **Modelos**: Logistic Regression, SVM, kNN, Árboles, Random Forest, Gradient Boosting.  
- **Métricas**: Accuracy (cuidado con desbalance), Precision/Recall/F1, ROC-AUC (binario), PR-AUC (desbalance extremo).  
- **Umbral**: Ajusta threshold según costo FP/FN; **calibración** si necesitas probabilidades reales.  
- **Desbalance**: `class_weight='balanced'`, split estratificado, o re-muestreo (SMOTE).
""")

        st.code(
                """# Pipeline de clasificación + CV (esqueleto)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("pre", pre),
    ("model", LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"accuracy":"accuracy", "f1":"f1", "auc":"roc_auc"}
res = cross_validate(pipe, X, y, scoring=scoring, cv=cv, n_jobs=-1)
""", language="python")

        st.code(
                """# GridSearchCV — tuning rápido
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"]
}
gs = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
# gs.fit(X, y); gs.best_params_, gs.best_score_
""", language="python")

    with colR:
        st.markdown("### ▶️ Mini-demo: Clasificación (Breast Cancer)")
        run_cls = st.toggle("Ejecutar demo de clasificación", value=False, key="run_cls")
        if run_cls:
            X, y = load_data_classification()
            num_cols = X.columns.tolist()
            pre = ColumnTransformer([("num", StandardScaler(), num_cols)])
            pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)[:, 1]
            ypred = (proba >= 0.5).astype(int)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Accuracy", f"{accuracy_score(yte, ypred):.3f}")
            with c2: st.metric("F1", f"{f1_score(yte, ypred):.3f}")
            with c3: st.metric("ROC-AUC", f"{roc_auc_score(yte, proba):.3f}")

            st.markdown("**Matriz de confusión**")
            render_confusion_matrix(yte, ypred)

    st.markdown("---")
    colL2, colR2 = st.columns([1.1, 1])

    with colL2:
        st.markdown("### 📈 Regresión (teoría)")
        st.markdown("""
- **Modelos**: Linear/Ridge/Lasso/ElasticNet, Random Forest Regressor, Gradient Boosting.  
- **Métricas**: MAE (robusto), RMSE (penaliza grandes errores), R² (ojo con sobreajuste).  
- **Residuos**: inspecciona no linealidad y heterocedasticidad.  
- **Regularización**: estandariza antes de L1/L2.
""")
        st.code(
                """# RidgeCV + preprocesamiento
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate

pre = ColumnTransformer([("num", StandardScaler(), num_cols)])
ridge = Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-3,3,30)))])
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"MAE":"neg_mean_absolute_error", "RMSE":"neg_root_mean_squared_error", "R2":"r2"}
# res = cross_validate(ridge, X, y, scoring=scoring, cv=cv, n_jobs=-1)
""", language="python")

    with colR2:
        st.markdown("### ▶️ Mini-demo: Regresión (California Housing)")
        run_reg = st.toggle("Ejecutar demo de regresión", value=False, key="run_reg")
        if run_reg:
            X, y = load_data_regression()
            num_cols = X.columns.tolist()
            pre = ColumnTransformer([("num", StandardScaler(), num_cols)])
            ridge = Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-3,3,30)))])

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            ridge.fit(Xtr, ytr)
            yhat = ridge.predict(Xte)

            mae = mean_absolute_error(yte, yhat)
            rmse = np.sqrt(mean_squared_error(yte, yhat))
            r2 = r2_score(yte, yhat)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("MAE", f"{mae:.3f}")
            with c2: st.metric("RMSE", f"{rmse:.3f}")
            with c3: st.metric("R²", f"{r2:.3f}")

            fig, ax = plt.subplots()
            ax.scatter(yte, yhat, s=8)
            ax.plot([yte.min(), yte.max()], [yte.min(), yte.max()])
            ax.set_xlabel("y true"); ax.set_ylabel("y pred")
            ax.set_title("Predicho vs Real")
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("📋 Checklist final (antes de producción)")
    st.markdown("""
- Target claro, métrica primaria y costo de errores.  
- Pipeline reproducible (sin leakage) + CV apropiada.  
- Umbral ajustado al negocio; calibración si necesitas probabilidades.  
- Desbalance tratado y métricas por clase.  
- Regularización / early stopping según modelo.  
- Registro de semillas, versiones y parámetros.  
- Documentación: cómo usar, límites y próximos pasos.
""")

# ============================================================
# 2) UNSUPERVISED
# ============================================================
with t1:
    st.subheader("🧭 Unsupervised Learning")
    st.caption("Estructura sin etiquetas: agrupación, reducción, visualización, partes interpretables.")

    colUL, colUR = st.columns([1.1, 1])

    with colUL:
        st.markdown("### 1) Clustering (teoría)")
        st.markdown("""
- **KMeans**: minimiza distancia a centroides (escalar numéricas).  
- Elegir **k**: Codo, **Silhouette**, estabilidad.  
- Métricas internas: Silhouette (↑ mejor), Davies-Bouldin (↓), Calinski-Harabasz (↑).  
- Inspección: tamaños, centroides, perfiles medios por variable.
""")
        st.code(
                """# KMeans + escalado + selección de k (silhouette)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X_scaled = StandardScaler().fit_transform(X)
scores = {k: silhouette_score(X_scaled, KMeans(k, n_init="auto", random_state=42).fit_predict(X_scaled))
          for k in range(2, 11)}
k_star = max(scores, key=scores.get)
labels = KMeans(k_star, n_init="auto", random_state=42).fit_predict(X_scaled)
""", language="python")

        st.markdown("### 2) Reducción de Dimensión (PCA)")
        st.markdown("""
- **PCA** para decorrelacionar y comprimir manteniendo varianza.  
- Decide PCs por varianza acumulada (p.ej., 90–95%).  
- Útil antes de clustering/visualización.
""")
        st.code(
                """# PCA conservando 95% varianza
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
X_p = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.cumsum()
""", language="python")

        st.markdown("### 3) Partes interpretables (NMF)")
        st.markdown("""
- **NMF** en datos no negativos: X ≈ W·H → **temas/partes**.  
- En texto: TF-IDF + NMF para tópicos interpretables.
""")
        st.code(
                """# NMF sobre matriz no negativa (ej. TF-IDF)
from sklearn.decomposition import NMF
nmf = NMF(n_components=10, init="nndsvda", random_state=42, max_iter=1000)
W = nmf.fit_transform(X_nonneg)
H = nmf.components_
# top términos por componente: np.argsort(-H[i])[:10]
""", language="python")

    with colUR:
        st.markdown("### ▶️ Mini-demo: Clustering (Iris) + PCA 2D")
        run_cluster = st.toggle("Ejecutar demo de clustering", value=False, key="run_cluster")
        if run_cluster:
            X, y = load_data_clustering()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            k = st.slider("k (clusters)", 2, 8, 3, 1)
            km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
            labels = km.fit_predict(Xs)
            sil = silhouette_score(Xs, labels)

            pca = PCA(n_components=2, random_state=seed)
            X2 = pca.fit_transform(Xs)
            df_plot = pd.DataFrame({"PC1": X2[:,0], "PC2": X2[:,1], "cluster": labels})

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Silhouette", f"{sil:.3f}")
                st.write("Centroides (en espacio estandarizado):")
                st.dataframe(pd.DataFrame(km.cluster_centers_, columns=X.columns))
            with c2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["cluster"])
                ax.set_title("Clusters en PCA 2D")
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                st.pyplot(fig)

