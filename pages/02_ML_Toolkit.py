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
    
    
st.title("Machine Learning Toolkit")
#st.caption("Teoría → Código → Mini-demo ejecutable. Basado en scikit-learn, pensado para aprender y reutilizar.")

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
t0, t1, t2, t3, t4 = st.tabs([
    "🧠 ML Fundamentals",
    "🎯 Supervised Learning",
    "🧭 Unsupervised Learning",
    "⚙️ Model Evaluation & Tuning",
    "🚀 From Model to Production"
    ])

with t0:
    st.subheader("🧠 ML Fundamentals")
    st.caption("Core concepts to understand how machine learning models learn, generalize, and fail.")

    # =========================
    # What is ML
    # =========================
    st.markdown("""
    ### 📌 What is Machine Learning?

    **Machine Learning (ML)** is a way to build systems that learn patterns from data  
    instead of being explicitly programmed with rules.

    **Traditional Programming**
    - Rules + Data → Output  

    **Machine Learning**
    - Data + Output → Learns Rules  

    The goal is to learn a function:

    f(x) → y  

    Where:
    - x = input features  
    - y = target variable  
    """)

    st.divider()

    # =========================
    # Types of ML
    # =========================
    st.markdown("""
    ### 🧩 Types of Machine Learning

    **🎯 Supervised Learning**
    - Data includes labels (target variable y)  
    - Goal: predict outputs  

    Examples:
    - Classification → spam vs not spam  
    - Regression → price prediction  

    ---
    **🧭 Unsupervised Learning**
    - No labels  
    - Goal: find structure or patterns  

    Examples:
    - Clustering → customer segmentation  
    - Dimensionality reduction → PCA  

    ---
    **🔄 Other paradigms (high-level)**
    - Semi-supervised learning  
    - Reinforcement learning  
    """)

    st.divider()

    # =========================
    # Bias vs Variance
    # =========================
    st.markdown("""
    ### ⚖️ Bias vs Variance Tradeoff

    **Bias**
    - Error due to overly simple assumptions  
    - Model underfits  
    - Misses real patterns  

    **Variance**
    - Error due to sensitivity to data  
    - Model overfits  
    - Learns noise instead of signal  

    ---
    **Goal:** balance both to minimize total error

    Error = Bias² + Variance + Noise  

    - High Bias → too simple  
    - High Variance → too complex  
    """)

    st.divider()

    # =========================
    # Overfitting vs Underfitting
    # =========================
    st.markdown("""
    ### 📉 Overfitting vs Underfitting

    **Underfitting**
    - Model too simple  
    - Poor performance on both train and test  

    **Overfitting**
    - Model memorizes training data  
    - Good train performance, poor test performance  

    ---
    **How to detect**
    - Train ↓ and Test ↑ → overfitting  
    - Both high → underfitting  

    ---
    **How to fix**

    Underfitting:
    - Increase model complexity  
    - Add better features  

    Overfitting:
    - Regularization  
    - More data  
    - Cross-validation  
    """)

    st.divider()

    # =========================
    # Generalization
    # =========================
    st.markdown("""
    ### 🧠 Generalization

    The ultimate goal of ML is:

    → Perform well on **unseen data**

    ---
    **Train / Validation / Test**
    - Train → learn patterns  
    - Validation → tune model  
    - Test → final evaluation  

    ---
    **Cross-validation**
    - More robust performance estimate  
    - Reduces variance in evaluation  

    ---
    **⚠️ Data Leakage**
    - Using future or hidden information  
    - Leads to unrealistic performance  

    ---
    **Best practices**
    - Always separate datasets properly  
    - Use pipelines  
    - Track experiments  
    """)
# ============================================================
# 1) SUPERVISED
# ============================================================
with t1:
    st.subheader("🎯 Supervised Learning")
    st.caption("Predictive modeling with labeled data. Focus on model selection, evaluation, and real-world decision making.")

    # =========================
    # Tabs
    # =========================
    s1, s2, s3, s4 = st.tabs([
        "🧠 How to Choose a Model",
        "🧩 Classification",
        "📈 Regression",
        "🧪 Live Demo"
    ])

    # =========================
    # 🧠 MODEL SELECTION
    # =========================
    with s1:
        st.markdown("""
        ### 🧠 How to Choose the Right Model

        Choosing a model is NOT about using the most complex algorithm.  
        It’s about aligning the model with:
        - the **data**
        - the **problem**
        - the **business constraints**
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Step 1 — Identify the problem type

        - **Classification** → predict categories  
        - **Regression** → predict continuous values  
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Step 2 — Understand your data

        Ask yourself:

        - How many rows do I have?
        - Are relationships linear or complex?
        - Do I have many features?
        - Are there missing values?
        - Is the dataset imbalanced?

        ---
        **Rules of thumb:**

        - Small dataset → simpler models (Logistic, Linear)  
        - Large dataset → tree-based / boosting  
        - Many features → regularization (Lasso/Ridge)  
        - Non-linear patterns → trees / boosting  
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Step 3 — Start simple (baseline first)

        Always begin with:

        - Logistic Regression (classification)
        - Linear / Ridge (regression)

        Why?
        - Fast
        - Interpretable
        - Strong baseline
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Step 4 — Increase complexity if needed

        Move to more powerful models when:

        - Performance is low
        - Relationships are non-linear

        Options:
        - Random Forest
        - Gradient Boosting (XGBoost, LightGBM)
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Step 5 — Consider constraints (THIS IS SENIOR LEVEL)

        Choose based on:

        - ⚡ Speed (real-time vs batch)
        - 🔍 Interpretability (business needs)
        - 📦 Deployment complexity
        - 💰 Cost of errors

        ---
        Example:

        - Credit risk → interpretable (Logistic)  
        - Recommendations → complex (Boosting)  
        """)

        st.divider()

        st.markdown("""
        ### 🧭 Quick Decision Guide

        | Situation | Recommended Model |
        |----------|------------------|
        | Simple baseline | Logistic / Linear |
        | Tabular data (most cases) | Random Forest / XGBoost |
        | Small dataset | Logistic / Ridge |
        | High interpretability needed | Linear / Logistic |
        | Complex patterns | Gradient Boosting |
        """)

    # =========================
    # 🧩 CLASSIFICATION
    # =========================
    with s2:
        st.markdown("""
        ### 🧩 Classification

        Predict discrete labels (0/1, categories).

        ---
        ### 🔑 Common Models

        - Logistic Regression → baseline, interpretable  
        - Random Forest → robust, handles non-linearity  
        - Gradient Boosting → best performance in tabular data  
        """)

        st.divider()

        st.markdown("""
        ### 📏 Metrics

        - Accuracy → only if balanced  
        - Precision → minimize false positives  
        - Recall → minimize false negatives  
        - F1-score → balance  
        - ROC-AUC → ranking quality  
        """)

        st.divider()

        st.markdown("""
        ### ⚠️ Important Concepts

        - Threshold tuning (not always 0.5)  
        - Class imbalance handling  
        - Probability calibration  
        """)

    # =========================
    # 📈 REGRESSION
    # =========================
    with s3:
        st.markdown("""
        ### 📈 Regression

        Predict continuous values.

        ---
        ### 🔑 Common Models

        - Linear Regression → simple baseline  
        - Ridge / Lasso → regularized models  
        - Random Forest → non-linear patterns  
        - Gradient Boosting → high performance  
        """)

        st.divider()

        st.markdown("""
        ### 📏 Metrics

        - MAE → robust to outliers  
        - RMSE → penalizes large errors  
        - R² → explanatory power  
        """)

        st.divider()

        st.markdown("""
        ### ⚠️ Diagnostics

        - Residual analysis  
        - Non-linearity  
        - Heteroscedasticity  
        """)

    # =========================
    # 🧪 DEMO
    # =========================
    with s4:
        st.markdown("### 🧪 Live Demo")

        run_demo = st.toggle("Run classification demo", value=False)

        if run_demo:
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            X, y = load_breast_cancer(return_X_y=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            st.metric("Accuracy", round(accuracy_score(y_test, preds), 3))
            
# ============================================================
# 2) UNSUPERVISED
# ============================================================
with t2:
    st.subheader("🧭 Unsupervised Learning")
    st.caption("Discover patterns, structure, and hidden relationships in data without labeled outcomes.")

    # =========================
    # Tabs
    # =========================
    u1, u2, u3, u4 = st.tabs([
        "🧠 When to Use It",
        "🔍 Clustering",
        "📉 Dimensionality Reduction",
        "🧪 Live Demo"
    ])

    # =========================
    # 🧠 WHEN TO USE IT
    # =========================
    with u1:
        st.markdown("""
        ### 🧠 When to Use Unsupervised Learning

        Use unsupervised learning when:
        - You **don’t have labels**
        - You want to **explore structure**
        - You need **segmentation or grouping**
        """)

        st.divider()

        st.markdown("""
        ### 🔑 Common Use Cases

        - Customer segmentation  
        - Fraud / anomaly detection  
        - Feature engineering  
        - Data compression  
        """)

        st.divider()

        st.markdown("""
        ### ⚠️ Key Difference vs Supervised

        - No “correct answer”  
        - Evaluation is **harder and more subjective**  
        - Interpretation is critical  
        """)

    # =========================
    # 🔍 CLUSTERING
    # =========================
    with u2:
        st.markdown("""
        ### 🔍 Clustering

        Group similar observations together.

        ---
        ### 🔑 Most Common Model

        **KMeans**
        - Groups data into k clusters  
        - Minimizes distance to centroids  

        ---
        ### ⚠️ Important Requirements

        - Scale your data (VERY important)  
        - Choose the right number of clusters (k)  
        """)

        st.divider()

        st.markdown("""
        ### 🧠 How to Choose K

        - Elbow Method → diminishing returns  
        - Silhouette Score → cluster quality  
        - Business interpretability (MOST IMPORTANT)  
        """)

        st.divider()

        st.markdown("""
        ### 📏 Evaluation

        - Silhouette Score → higher is better  
        - Cluster size balance  
        - Separation vs overlap  
        """)

        st.divider()

        st.markdown("""
        ### 🎯 Interpretation (THIS IS WHAT MATTERS)

        After clustering, always ask:

        - What defines each cluster?  
        - How are they different?  
        - Are they useful for the business?  

        Example:
        - Cluster 1 → high spend customers  
        - Cluster 2 → low engagement  
        """)

    # =========================
    # 📉 DIMENSIONALITY REDUCTION
    # =========================
    with u3:
        st.markdown("""
        ### 📉 Dimensionality Reduction

        Reduce the number of features while preserving information.

        ---
        ### 🔑 PCA (Principal Component Analysis)

        - Transforms data into new components  
        - Each component captures variance  
        """)

        st.divider()

        st.markdown("""
        ### 🧠 Why Use PCA?

        - Visualization (2D / 3D plots)  
        - Remove noise  
        - Speed up models  
        """)

        st.divider()

        st.markdown("""
        ### ⚠️ Key Concepts

        - PC1 explains the most variance  
        - PC2 explains the second most  
        - Components are linear combinations of features  
        """)

        st.divider()

        st.markdown("""
        ### 📏 Interpretation

        - Look at explained variance  
        - Analyze feature loadings  
        - Identify clusters visually  
        """)

    # =========================
    # 🧪 DEMO
    # =========================
    with u4:
        st.markdown("### 🧪 Live Demo — Clustering + PCA")

        run_demo = st.toggle("Run clustering demo", value=False)

        if run_demo:
            from sklearn.datasets import load_iris
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            import matplotlib.pyplot as plt

            X, _ = load_iris(return_X_y=True)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.slider("Number of clusters (k)", 2, 8, 3)

            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(X_scaled)

            score = silhouette_score(X_scaled, labels)
            st.metric("Silhouette Score", round(score, 3))

            # PCA visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            fig, ax = plt.subplots()
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
            ax.set_title("Clusters in PCA space")
            st.pyplot(fig)
            
with t3:
    st.subheader("⚙️ Model Evaluation & Tuning")
    st.caption("Evaluate models correctly, avoid common pitfalls, and optimize performance for real-world scenarios.")

    # =========================
    # Tabs
    # =========================
    e1, e2, e3, e4 = st.tabs([
        "📏 Metrics",
        "🔁 Cross-Validation",
        "🎯 Threshold & Imbalance",
        "🔧 Hyperparameter Tuning"
    ])

    # =========================
    # 📏 METRICS
    # =========================
    with e1:
        st.markdown("""
        ### 📏 Choosing the Right Metric

        The metric defines what “good” means.

        ---
        ### 🎯 Classification Metrics

        - **Accuracy**
            - Good only if classes are balanced  

        - **Precision**
            - When false positives are costly  

        - **Recall**
            - When false negatives are costly  

        - **F1-score**
            - Balance between precision & recall  

        - **ROC-AUC**
            - Measures ranking quality  

        ---
        ### 📈 Regression Metrics

        - **MAE**
            - Robust to outliers  

        - **RMSE**
            - Penalizes large errors  

        - **R²**
            - Explains variance (use carefully)  

        ---
        ### ⚠️ Key Insight

        The best model depends on the **business objective**, not the highest score.
        """)

    # =========================
    # 🔁 CROSS VALIDATION
    # =========================
    with e2:
        st.markdown("""
        ### 🔁 Cross-Validation

        Single train/test splits are unstable.

        ---
        ### 🧠 Why use Cross-Validation?

        - Reduces variance in evaluation  
        - Uses data more efficiently  
        - Provides more reliable estimates  

        ---
        ### 🔑 Types

        - **K-Fold**
        - **Stratified K-Fold** (classification)
        - **Time-based splits** (time series)

        ---
        ### ⚠️ Common Mistake

        Data leakage:
        - Scaling BEFORE split ❌  
        - Using future data ❌  

        Always use:
        - Pipelines  
        """)

        st.code("""
from sklearn.model_selection import cross_validate, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1"])
""", language="python")

    # =========================
    # 🎯 THRESHOLD & IMBALANCE
    # =========================
    with e3:
        st.markdown("""
        ### 🎯 Threshold & Class Imbalance

        Default threshold = 0.5 → NOT always optimal

        ---
        ### 🔑 Threshold Tuning

        Adjust threshold depending on:

        - Business cost  
        - Risk tolerance  

        Example:
        - Fraud detection → lower threshold (catch more cases)  

        ---
        ### ⚠️ Class Imbalance

        Problem:
        - Model predicts majority class only  

        Solutions:
        - `class_weight='balanced'`  
        - Oversampling (SMOTE)  
        - Undersampling  
        - Use better metrics (F1, ROC-AUC)  
        """)

    # =========================
    # 🔧 HYPERPARAMETER TUNING
    # =========================
    with e4:
        st.markdown("""
        ### 🔧 Hyperparameter Tuning

        Models have parameters that must be optimized.

        ---
        ### 🔑 Methods

        - **Grid Search**
            - Exhaustive  
            - Expensive  

        - **Random Search**
            - Faster  
            - Often good enough  

        ---
        ### 🧠 Best Practice

        - Start simple  
        - Tune only important parameters  
        - Use cross-validation  

        ---
        ### ⚠️ Common Mistakes

        - Over-tuning → overfitting  
        - Not using CV  
        - Ignoring runtime cost  
        """)

        st.code("""
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__max_depth": [3, 5, 10],
    "model__n_estimators": [100, 200]
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X, y)
""", language="python")
        
with t4:
    st.subheader("🚀 From Model to Production")
    st.caption("How models move from experimentation to real-world systems.")

    st.markdown("""
    ### 🧠 The Reality of Machine Learning

    Building a model is only ~20% of the work.

    The real challenge is:
    - Making it reproducible  
    - Deploying it  
    - Monitoring it  
    """)

    st.divider()

    st.markdown("""
    ### 🔄 ML Lifecycle

    1. Data collection (APIs, databases)  
    2. Data preprocessing  
    3. Model training  
    4. Evaluation & validation  
    5. Deployment (API / batch)  
    6. Monitoring (performance, drift)  
    """)

    st.divider()

    st.markdown("""
    ### ⚠️ Common Production Challenges

    - Data drift  
    - Concept drift  
    - Latency constraints  
    - Model degradation over time  
    """)

    st.divider()

    st.markdown("""
    ### 🔑 Best Practices

    - Use pipelines (no leakage)  
    - Version everything (data, model, code)  
    - Monitor metrics in production  
    - Retrain periodically  
    """)

    st.divider()

    st.info("""
    👉 For full implementation (deployment, pipelines, monitoring),
    see the **MLOps & Deployment** section.
    """)