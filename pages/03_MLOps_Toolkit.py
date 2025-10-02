import streamlit as st

st.set_page_config(page_title="MLOps Toolkit", layout="wide")

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
    st.page_link("pages/04_LLM_Toolkit.py", label="LLM Theory Toolkit", icon="🤖")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")

    st.divider()
    st.markdown("**Contact**")
    st.markdown("- GitHub: [@liliam-mtz](https://github.com/liliammtz)")
    st.markdown("- LinkedIn: [Liliam Martínez](https://www.linkedin.com/in/liliammtz/)")
    st.markdown("- Email: [liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")
st.title("⚙️ MLOps Toolkit")
st.caption("An interactive guide to learn from scratch what MLOps is, its phases, roles, and tools.")

# =========================
# Main Tabs
# =========================
t_introduction, t_phases, t_design, t_development, t_deploy, t_roles, t_tools = st.tabs([
    "💡 Introduction",
    "🔄 MLOps Phases",
    "🎨 Design",
    "💻 Development",
    "🚀 Deployment & Monitoring",
    "👥 Roles & Maturity",
    "🔧 Tools"
])

# =========================
# Tab - Intro
# =========================
with t_introduction:
    st.header("📌 What is MLOps?")
    st.write("""
    **MLOps** (Machine Learning Operations) is a set of practices that combine *Machine Learning* and *DevOps* 
    to design, deploy, and maintain ML models in production **continuously, reliably, and efficiently**.

    Training a model is like building a prototype in a lab.  
    But for a company to actually use it, the model must be:  
    - Integrated into existing applications.  
    - Monitored constantly to check if it’s still accurate.  
    - Updated when data or business requirements change.  

    That’s what MLOps ensures: models don’t just work in a Jupyter Notebook, but become **production-ready systems**.
    """)

    st.subheader("🔑 Where does MLOps come from?")
    st.write("""
    - Born from **DevOps**, which aimed to improve collaboration between developers and operations.  
    - DevOps focused on speeding up software delivery; MLOps extends these principles to ML projects.  
    - In ML, we don’t just manage code: we also manage **data, models, experiments, and continuous monitoring**.  
    """)

    st.subheader("🚀 Why use MLOps?")
    st.write("""
    - **Better collaboration** between business, data science, and engineering teams.  
    - **Automation** reduces human error and accelerates deployment.  
    - **Continuous monitoring** of models ensures performance doesn’t degrade.  
    - Builds **trust with stakeholders** through structured, measurable processes.  
    """)

# =========================
# Tab - Phases
# =========================
with t_phases:
    st.header("📂 Main Phases in MLOps")
    st.write("MLOps is not a single step—it’s a lifecycle that spans from design to monitoring.")
    
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎨 Design")
            st.markdown("""
            1️⃣ **Problem definition and requirements**  
            - Define the business objective.  
            - Identify constraints, compliance, and key metrics.  

            2️⃣ **Exploratory Data Analysis (EDA)**  
            - Inspect datasets.  
            - Identify missing values, outliers, patterns.  

            3️⃣ **Implementation design**  
            - Plan solution architecture.  
            - Choose algorithms and infrastructure.  
            """)

        with col2:
            st.subheader("💻 Development")
            st.markdown("""
            1️⃣ **Feature engineering**  
            - Select and create features.  
            - Ensure data quality and transformations.  

            2️⃣ **Experiment design**  
            - Define hypotheses.  
            - Track hyperparameters, datasets, results.  

            3️⃣ **Model training & evaluation**  
            - Train multiple models.  
            - Evaluate against business and technical metrics.  
            """)

        with col3:
            st.subheader("🚀 Deployment")
            st.markdown("""
            1️⃣ **Setup CI/CD pipeline**  
            - Automate testing, integration, deployment.  

            2️⃣ **Deploy model**  
            - Expose via APIs or microservices.  
            - Containerize for scalability.  

            3️⃣ **Monitoring**  
            - Track model accuracy (statistical).  
            - Monitor latency, usage, and drift.  
            """)

    # =========================
    # Visual Diagram (Graphviz)
    # =========================
    st.header("🔗 Visual Flow of MLOps")
    st.graphviz_chart('''
        digraph MLOps {
            rankdir=LR;
            node [shape=polygon, sides=4, skew=0.5, style=filled, fontsize=11, fontname="Helvetica", width=2.5, height=1];

            // Colors for phases
            subgraph cluster_design {
                label="Design";
                color=lightblue;
                "Problem definition & requirements" [fillcolor=lightblue, style=filled];
                "Exploratory data analysis" [fillcolor=lightblue, style=filled];
                "Implementation design" [fillcolor=lightblue, style=filled];
            }

            subgraph cluster_dev {
                label="Development";
                color=lightgreen;
                "Feature engineering" [fillcolor=lightgreen, style=filled];
                "Experiment design" [fillcolor=lightgreen, style=filled];
                "Model training & evaluation" [fillcolor=lightgreen, style=filled];
            }

            subgraph cluster_deploy {
                label="Deployment";
                color=lightpink;
                "Setup CI/CD pipeline" [fillcolor=lightpink, style=filled];
                "Deploy model" [fillcolor=lightpink, style=filled];
                "Monitoring" [fillcolor=lightpink, style=filled];
            }

            // Sequence of steps
            "Problem definition & requirements" -> "Exploratory data analysis" -> "Implementation design";
            "Implementation design" -> "Feature engineering" -> "Experiment design" -> "Model training & evaluation";
            "Model training & evaluation" -> "Setup CI/CD pipeline" -> "Deploy model" -> "Monitoring";
        }
        ''')

# =========================
# Tab - design 
# =========================
with t_design:
    st.markdown("""
        The **design phase** decides if an ML project makes sense.  
        Before training any model we must:  

        **1. Define the problem and requirements:**  
        - What are we predicting?  
        - How often do we need predictions (daily, hourly, real-time)?  
        - What accuracy level is acceptable?  
        - Who will be the end user? 
        - How much transparency is needed in the model according to regulations?
        - What is the available budget and team size? 

        **2. Assess added value of ML:**  
        - Is ML solving a problem worth solving? What is the estimated expected value?
        - This helps in: prioritization and aids in resource allocation
        - Example: catching fraud that saves millions vs a minor optimization.  

        **3. Establish key metrics (KPIs):**  
        - *Data scientists:* accuracy, precision, recall, RMSE.  
        - *Business experts:* impact, reduced risk.  
        - *Stakeholders:* financial outcomes, cost savings.  

        **4. Ensure data quality:**  
        - ¿How well does the data serves it's intended purpose, evaluated through different dimensions:
        - *Accuracy:* does the data reflect reality?  
        - *Completeness:* are key fields missing?  
        - *Consistency:* same format across systems?  
        - *Timeliness:* is data available on time?  

        **5. Data ingestion (ETL):**  
        - Extract, Transform, Load.  
        - Validate data types, ranges, duplicates.  
        """)
# =========================
# Tab - Development
# =========================
with t_development:
    st.header("🛠️ Development Phase")

    st.markdown("""
    Once the **design phase** is complete, we move into **development**, where experimentation begins.  
    Here, data scientists and engineers work together to **transform raw data into features, design experiments, 
    and train models**.  

    The ultimate goal of this phase is to create a model that doesn’t just work in theory, but is **robust and ready 
    for production deployment**.
    """)

    # Feature Engineering
    st.subheader("🔬 Feature Engineering")
    st.write("""
    Features are the input variables that a model uses to learn patterns and make predictions.  
    For example, if we want to predict sales, useful features might include `number_of_customers`, 
    `price`, or `day_of_week`.  

    **Feature engineering** is the process of selecting, transforming, and creating these features 
    to maximize model performance.  

    **Why it matters:**  
    A well-designed feature can often improve model accuracy more than switching to a different algorithm.  

    **Common techniques:**  
    - **Feature selection:** use domain knowledge, correlation analysis, univariate selection, PCA, or RFE to 
      determine which variables matter most.  
    - **Feature creation:** combine or transform raw variables (e.g., using *average spend per customer* 
      instead of just *total spend*).  
    - **Feature stores:** centralized repositories that let teams discover, define, and reuse features across projects. 
      These are especially valuable in large organizations to ensure consistency, though smaller projects may not need them.  
    """)

    # Experimentation
    st.subheader("📊 Experimentation and Tracking")
    st.write("""
    Machine learning is **inherently experimental**. Models, hyperparameters, and datasets change constantly.  
    Without systematic tracking, results can’t be reproduced, compared, or trusted.  

    **What should be tracked:**  
    - **Dataset versions:** ensure experiments are based on consistent data.  
    - **Algorithms and hyperparameters:** document which configurations were tested.  
    - **Metrics:** record model performance in detail, not just a final score.  
    - **Code and runtime environment:** capture the exact scripts and library versions.  

    **Why tracking is crucial:**  
    - Enables reproducibility of past results.  
    - Makes it easy to compare experiments and choose the best approach.  
    - Improves collaboration between data scientists, engineers, and stakeholders.  
    - Provides clear reporting to non-technical decision makers.  

    **How to track experiments:**  
    - Basic: spreadsheets (not scalable).  
    - Proprietary platforms: often built into ML services.  
    - Open-source tools: **MLflow**, **ClearML**, or **Weights & Biases (W&B)** are the industry standard.  
    """)

    # Model Training & Evaluation
    st.subheader("🤖 Model Training and Evaluation")
    st.write("""
    After preparing features and designing experiments, models are trained on the available data.  

    **Key aspects:**  
    - Train multiple algorithms and compare results using the same evaluation metrics.  
    - Use cross-validation to estimate performance on unseen data.  
    - Evaluate results not only on accuracy, but also on **business impact** (e.g., cost savings, revenue generated).  

    At the end of this step, the best candidate model is selected and registered for deployment.  
    """)

    # Runtime Environments
    st.subheader("⚙️ Runtime Environments")
    st.write("""
    A **runtime environment** defines the context in which code is executed (libraries, OS, configurations).  
    Often, models are trained in one environment and deployed in another.  
    If library versions differ, the model may fail in production.  

    **Solution:**  
    - Use **containers** such as Docker to package models together with their dependencies.  

    **Benefits of containerization:**  
    - **Portability:** run anywhere, from a laptop to the cloud.  
    - **Consistency:** identical environments for development, testing, and production.  
    - **Efficiency:** fast startup and easier maintenance.  
    """)

# =========================
# Tab - Deployment & Monitoring
# =========================
with t_deploy:
    st.header("🚀 Deployment and Monitoring")
    st.markdown("""
        Deployment means taking the trained model and **integrating it into the business**.  

        - Expose predictions through **APIs**.  
        - Containerize models for portability.  
        - Automate deployment with **CI/CD pipelines**.  
        - Set up monitoring **from day one**.  
        """)
    st.subheader("Deployment Architectures")
    st.write("""
    - **Monolith:** one large application (simple, but hard to scale).  
    - **Microservices:** separate, independent services (scalable, fault-tolerant).  

    **Inference:** sending new input → model → prediction.  
    """)

    st.subheader("CI/CD Strategies")
    st.write("""
    - **CI (Continuous Integration):** plan, build, test code continuously.  
    - **CD (Continuous Deployment/Delivery):** automate release of code/models.  

    **Model deployment strategies:**  
    - **Basic:** replace old model with new one (fast, risky).  
    - **Shadow:** run old + new in parallel (safer, resource-heavy).  
    - **Canary:** test new model with a small portion of traffic (low risk).  
    """)

    st.subheader("📡 Monitoring Models")
    st.write("""
    Monitoring ensures models remain useful after deployment.  

    **Types:**  
    - **Statistical monitoring:** model metrics (accuracy, drift).  
    - **Computational monitoring:** latency, requests per second, resource usage.  
    - **Feedback loop:** use ground truth to correct model mistakes.  
    """)

    st.subheader("♻️ Retraining")
    st.write("""
    Models degrade over time. Reasons:  
    - **Data drift:** input data distribution changes.  
    - **Concept drift:** input-output relationship changes.  
    - **Model degradation:** accuracy naturally drops.  

    **Retraining methods:**  
    - Train only on new data.  
    - Train on new + old data.  
    - Automated retraining (e.g. monthly).  

    **Frequency depends on:**  
    - Business volatility.  
    - Cost of retraining.  
    - Required accuracy.  
    """)

# =========================
# Tab - Roles & Maturity
# =========================
with t_roles:
    st.header("👥 Roles in MLOps")

    st.subheader("Business Roles")
    st.write("""
    - **Business Stakeholder:** defines vision, budget, and assesses results.  
    - **Subject Matter Expert (SME):** provides domain expertise, validates data.  
    """)

    st.subheader("Technical Roles")
    st.write("""
    - **Data Scientist:** builds and evaluates models.  
    - **Data Engineer:** manages pipelines, ensures data quality.  
    - **ML Engineer:** bridges DS and operations, handles full ML lifecycle.  
    """)

    st.subheader("MLOps Maturity Levels")
    st.write("""
    Maturity shows how advanced an organization’s MLOps practices are.  

    **Level 1:**  
    - Manual processes, isolated teams, no monitoring.  

    **Level 2:**  
    - Automated CI, manual deployment.  
    - Some experiment tracking, limited monitoring.  

    **Level 3:**  
    - Full CI/CD automation.  
    - Strong collaboration.  
    - Continuous monitoring.  
    """)

# =========================
# Tab - Tools
# =========================
with t_tools:
    st.header("🧰 MLOps Tools")

    col1, col2, col3 = st.columns(3)

    # Feature Stores
    with col1:
        st.subheader("📦 Feature Stores")
        st.markdown("""
        - [**Feast**](https://feast.dev): open-source feature store for machine learning.  
        - [**Hopsworks**](https://www.hopsworks.ai/): feature store as part of a larger ML platform.  
        """)

    # Experiment Tracking
    with col2:
        st.subheader("🧪 Experiment Tracking")
        st.markdown("""
        - [**MLflow**](https://mlflow.org/): widely used open-source ML lifecycle platform.  
        - [**ClearML**](https://clear.ml/): experiment management and orchestration platform.  
        - [**Weights & Biases (W&B)**](https://wandb.ai/): experiment tracking and visualization.  
        """)

    # Infrastructure & CI/CD
    with col3:
        st.subheader("🐳 Infrastructure & CI/CD")
        st.markdown("""
        - [**Docker**](https://www.docker.com/): containerization for portable environments.  
        - [**Kubernetes**](https://kubernetes.io/): orchestration of containerized workloads.  
        - [**Jenkins**](https://www.jenkins.io/) / [**GitLab CI/CD**](https://docs.gitlab.com/ee/ci/): automation pipelines for integration and deployment.  
        """)

    # Monitoring & Platforms
    st.subheader("🔍 Monitoring & Platforms")
    st.markdown("""
    - [**Fiddler**](https://www.fiddler.ai/), [**Great Expectations**](https://greatexpectations.io/): tools for monitoring and validating data/models.  

    **Cloud ML Platforms:**  
    - [Amazon SageMaker](https://aws.amazon.com/sagemaker/).  
    - [Azure Machine Learning](https://azure.microsoft.com/en-us/products/machine-learning/).  
    - [Google Cloud AI Platform](https://cloud.google.com/ai-platform).  
    """)

    st.subheader("✅ Minimum Viable Production Checklist")

    st.markdown("""
    - 📦 **Versioning**  
    Track datasets, code, and models with names, dates, or unique hashes.  

    - 🧭 **Traceability**  
    Log metrics, random seeds, data splits, and hyperparameters.  

    - 📁 **Artifacts**  
    Save the full **Pipeline** (preprocessing + model) using `joblib` or `pickle`.  

    - 📝 **Schemas**  
    Validate inputs/outputs with tools like **pydantic** and enforce data types.  

    - ⚡ **Inference**  
    Provide a pure, deterministic `predict(df)` function for easy reuse.  

    - 📊 **Monitoring**  
    Watch latency, errors, data drift (e.g., PSI), and overall performance.  

    - 🔒 **Reproducibility**  
    Fix `random_state`, document dependencies, and lock `requirements.txt`.  
    """)

st.link_button(
    "📖 Learn more: MLOps Concepts (DataCamp)",
    "https://app.datacamp.com/learn/courses/mlops-concepts"
)
