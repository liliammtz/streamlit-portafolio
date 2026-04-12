import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Responsible AI Toolkit",
    layout="wide"
)

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
    
st.title("Responsible AI Toolkit")

st.markdown("""
Responsible AI ensures that artificial intelligence systems are **ethical, transparent,
and compliant with regulations**.

This toolkit helps evaluate datasets and ML projects using core principles of
Responsible AI: **Lawfulness, Fairness, Transparency, Diversity, Privacy and Security**.
""")
    
# ----------------------------------------------------
# Tabs Navigation
# ----------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Responsible AI Overview",
    "⚖️ Lawfulness",
    "⚖️ Fairness",
    "🔍 Transparency",
    "🔐 Privacy & Security",
    "📊 Data Governance"
])

# ======================================================
# PRINCIPLES
# ======================================================

with tab1:
    st.subheader("Responsible AI Principles")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
### ⚖️ Lawfulness

AI systems must comply with **laws and regulations** governing data usage.

Examples:

- GDPR
- CCPA
- HIPAA
- Basel III
- Sarbanes–Oxley
""")

        st.markdown("""
### ⚖️ Fairness

AI models should **avoid discriminatory outcomes**.

Key ideas:

- Protected attributes
- Demographic parity
- Equal outcomes
""")

        st.markdown("""
### 🔐 Privacy & Security

Sensitive data should be protected through:

- encryption
- anonymization
- access control
""")

    with col2:

        st.markdown("""
### 🔍 Transparency & Accountability

Organizations should explain:

- how models are trained
- how data is collected
- how decisions are made

Tools:

- LIME
- SHAP
""")
        
        st.markdown("""
### 🌍 Diversity & Inclusion

Datasets should represent **diverse populations** to avoid biased models.
""")
    

        st.markdown("""
### 📊 Data Governance

Responsible AI requires strong **data governance**:

- data ownership
- lifecycle management
- documentation
""")

    st.divider()

    # --------------------------------------------------
    # AI LIFECYCLE
    # --------------------------------------------------

    st.subheader("AI Project Lifecycle")

    st.markdown("""
Responsible AI must be applied **throughout the entire AI lifecycle**.
""")

    lifecycle_cols = st.columns(4)

    lifecycle_cols[0].markdown("""
**1️⃣ Data Acquisition**

- collect representative datasets
- ensure legal data use
""")

    lifecycle_cols[1].markdown("""
**2️⃣ Modeling**

- train models responsibly
- evaluate fairness and bias
""")

    lifecycle_cols[2].markdown("""
**3️⃣ Deployment**

- monitor real-world performance
- ensure explainability
""")

    lifecycle_cols[3].markdown("""
**4️⃣ Monitoring**

- detect model drift
- audit predictions
""")

    st.divider()

    # --------------------------------------------------
    # RESPONSIBLE AI CHECKLIST
    # --------------------------------------------------

    st.subheader("Responsible AI Self-Assessment")

    checks = {
        "Protected characteristics identified": False,
        "Dataset diversity evaluated": False,
        "Bias metrics calculated": False,
        "Explainability method implemented": False,
        "Data licensing verified": False,
        "Compliance with regulations verified": False,
        "Model monitoring plan defined": False
    }

    score = 0

    for key in checks:

        checks[key] = st.checkbox(key, key=f"Checklist_Selftassessment_{key}")

        if checks[key]:
            score += 1

    percent = score / len(checks) * 100

    st.progress(percent / 100)

    st.metric("Responsible AI Maturity", f"{percent:.0f}%")

    if percent < 40:
        st.error("High Responsible AI risk")

    elif percent < 70:
        st.warning("Moderate Responsible AI maturity")

    else:
        st.success("Strong Responsible AI compliance")
        
# ======================================================
# LAWFULNESS
# ======================================================
with tab2:

    st.header("⚖️ Lawfulness in Responsible AI")

    st.markdown("""
Lawfulness ensures that AI systems comply with **laws, regulations, and ethical standards**
governing how data is collected, processed, and used.

Responsible AI projects must consider regulatory frameworks across **data protection,
AI governance, and industry-specific regulations**.
""")

    st.divider()

    # -----------------------------
    # GLOBAL GOVERNANCE
    # -----------------------------

    st.subheader("🌍 Global AI Governance")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**OECD AI Principles**

Guidelines promoting trustworthy AI across member countries.

🔗 https://oecd.ai/en/ai-principles
""")

        st.markdown("""
**UNESCO AI Ethics Framework**

Global recommendations for ethical AI development.

🔗 https://www.unesco.org/en/artificial-intelligence/recommendation-ethics
""")

    with col2:

        st.markdown("""
**NIST AI Risk Management Framework**

Risk management framework widely used in AI governance.

🔗 https://www.nist.gov/itl/ai-risk-management-framework
""")

        st.markdown("""
**ISO AI Standards**

International standards for AI governance and risk management.

🔗 https://www.iso.org/artificial-intelligence.html
""")

    st.divider()

    # -----------------------------
    # DATA PROTECTION LAWS
    # -----------------------------

    st.subheader("🔐 Data Protection Laws")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**GDPR – General Data Protection Regulation (EU)**

One of the most influential data privacy laws.

Key concepts:

- Right to be forgotten
- Data minimization
- Consent requirements

🔗 https://gdpr-info.eu/
""")

        st.markdown("""
**CCPA – California Consumer Privacy Act**

Provides consumer data protection rights in California.

🔗 https://oag.ca.gov/privacy/ccpa
""")

    with col2:

        st.markdown("""
**LGPD – Brazil**

Brazil's data protection regulation similar to GDPR.

🔗 https://www.gov.br/anpd
""")

        st.markdown("""
**Mexico – Ley Federal de Protección de Datos Personales**

Regulates personal data processing in Mexico.

🔗 https://home.inai.org.mx/
""")

    st.divider()

    # -----------------------------
    # FINANCIAL REGULATIONS
    # -----------------------------

    st.subheader("💰 Financial Industry Regulations")

    st.markdown("""
AI systems used in **financial services, fintech, AML, and remittances**
must comply with additional regulatory frameworks.
""")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Basel III Accord**

International banking regulation focused on financial risk.

🔗 https://www.bis.org/basel3.htm
""")

        st.markdown("""
**Sarbanes–Oxley Act (SOX)**

U.S. regulation for financial transparency and corporate governance.

🔗 https://www.sec.gov/spotlight/sarbanes-oxley.htm
""")

        st.markdown("""
**Anti-Money Laundering (AML) Regulations**

Financial institutions must detect and prevent illicit financial activity.

Examples:

- FATF Recommendations
- Know Your Customer (KYC)

🔗 https://www.fatf-gafi.org
""")

    with col2:

        st.markdown("""
**Payment Services Directive (PSD2)**

EU regulation for payment services and fintech.

🔗 https://finance.ec.europa.eu/financial-services/payments-and-retail-payments/payment-services/payment-services-directive-psd2_en
""")

        st.markdown("""
**Dodd-Frank Act**

U.S. financial regulation aimed at reducing systemic risk.

🔗 https://www.cftc.gov/LawRegulation/DoddFrankAct/index.htm
""")

        st.markdown("""
**Fintech and Digital Payments Regulations**

Regulatory frameworks evolving for AI-driven financial systems.

Examples:

- Open banking regulations
- Digital identity verification
""")

    st.divider()

    # -----------------------------
    # AI REGULATION
    # -----------------------------

    st.subheader("🤖 AI-Specific Regulations")

    st.markdown("""
**EU AI Act**

First major regulatory framework for artificial intelligence.

Key features:

- Risk classification for AI systems
- Requirements for high-risk AI
- Transparency obligations

🔗 https://artificialintelligenceact.eu/
""")

    st.markdown("""
**U.S. AI Executive Order**

Policy framework for responsible AI development.

🔗 https://www.whitehouse.gov/ai/
""")

    st.divider()

    # -----------------------------
    # COMPLIANCE CHECKLIST
    # -----------------------------

    st.subheader("✔ Lawfulness Compliance Checklist")

    checks = [
        "Data protection regulations identified",
        "Industry regulations identified",
        "Data licensing verified",
        "Consent mechanisms implemented",
        "AI explainability requirements met",
        "Audit and compliance documentation created"
    ]

    for item in checks:
        st.checkbox(item, key=f"Lawfulness_{item}")
        

# ======================================================
# FAIRNESS
# ======================================================
with tab3:

    st.header("⚖️ Fairness in Responsible AI")

    st.markdown("""
Fairness in AI ensures that machine learning systems **do not produce discriminatory
or unequal outcomes for different groups of people**.

Bias can originate from:

- historical patterns in data
- sampling issues
- modeling decisions
- deployment context
""")

    st.divider()

    # --------------------------------------------------
    # Protected Characteristics
    # --------------------------------------------------

    st.subheader("Protected Characteristics")

    st.markdown("""
Protected characteristics refer to attributes that are legally or ethically
protected from discrimination.

Examples include:

- race
- gender
- age
- disability
- religion
- nationality
""")

    st.info("Responsible AI projects should evaluate model performance across these groups.")

    st.divider()

    # --------------------------------------------------
    # Types of Bias
    # --------------------------------------------------

    st.subheader("Common Types of Bias")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Historical Bias**

Bias that exists due to historical inequalities reflected in the data.
""")

        st.markdown("""
**Selection Bias**

Occurs when the dataset does not represent the target population properly.
""")

    with col2:

        st.markdown("""
**Sampling Bias**

Happens when the sampling method systematically favors certain groups.
""")

        st.markdown("""
**Measurement Bias**

Occurs when data collection methods introduce systematic errors.
""")

    st.divider()

    # --------------------------------------------------
    # Fairness Metrics
    # --------------------------------------------------

    st.subheader("Fairness Metrics")

    st.markdown("""
Several statistical metrics can be used to measure fairness in machine learning systems.
""")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Demographic Parity**

Ensures that prediction outcomes are independent of protected attributes.

Example:
Loan approval rates should be similar across demographic groups.
""")

        st.markdown("""
**Equal Opportunity**

Requires equal true positive rates across groups.
""")

    with col2:

        st.markdown("""
**Disparate Impact**

Measures whether a protected group receives outcomes at significantly
different rates than others.
""")

        st.markdown("""
**False Positive / False Negative Gap**

Checks if error rates differ significantly between groups.
""")

    st.divider()

    # --------------------------------------------------
    # Bias Mitigation
    # --------------------------------------------------

    st.subheader("Bias Mitigation Strategies")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Pre-processing**

- resampling
- reweighting
- relabeling
- removing sensitive attributes
""")

    with col2:

        st.markdown("""
**Model-level mitigation**

- fairness constraints
- adversarial debiasing
- algorithm selection
""")

    st.divider()

    # --------------------------------------------------
    # Interactive Fairness Check
    # --------------------------------------------------

    st.subheader("Interactive Fairness Check")

    uploaded = st.file_uploader("Upload dataset", type=["csv"], key="fairness")

    if uploaded:

        df = pd.read_csv(uploaded)

        st.dataframe(df.head())

        categorical_cols = df.select_dtypes(include="object").columns

        protected = st.selectbox(
            "Protected attribute",
            categorical_cols
        )

        target = st.selectbox(
            "Prediction / outcome variable",
            df.columns
        )

        st.markdown("### Outcome distribution by group")

        fig = px.histogram(
            df,
            x=protected,
            color=protected
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Average outcome by group")

        group_stats = df.groupby(protected)[target].mean().reset_index()

        fig2 = px.bar(
            group_stats,
            x=protected,
            y=target
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.warning("""
Large differences between groups may indicate **potential bias**
and should be investigated further.
""")
        
# ======================================================
# TRANSPARENCY
# ======================================================
with tab4:

    st.header("🔍 Transparency & Accountability in Responsible AI")

    st.markdown("""
Transparency and accountability ensure that AI systems are **understandable,
traceable, and auditable**.

Organizations must be able to explain:

- how data was collected
- how models were trained
- how predictions are generated
- who is responsible for AI decisions
""")

    st.divider()

    # -------------------------------------------
    # Transparency
    # -------------------------------------------

    st.subheader("Transparency")

    st.markdown("""
Transparency means that stakeholders can understand **how an AI system works
and how decisions are made**.

Key aspects include:

- clear documentation of datasets
- explainable model predictions
- transparency about model limitations
- disclosure of automated decision systems
""")

    st.divider()

    # -------------------------------------------
    # Explainable AI
    # -------------------------------------------

    st.subheader("Explainable AI (XAI)")

    st.markdown("""
Explainable AI techniques help interpret model predictions and identify
the factors influencing decisions.

Common methods include:
""")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**LIME (Local Interpretable Model-Agnostic Explanations)**

Explains individual predictions by approximating the model locally.
""")

        st.markdown("""
**SHAP (SHapley Additive exPlanations)**

Uses game theory to measure feature contributions to predictions.
""")

    with col2:

        st.markdown("""
**Feature Importance**

Measures how much each feature contributes to model predictions.
""")

        st.markdown("""
**Partial Dependence Plots**

Visualize relationships between features and predictions.
""")

    st.divider()

    # -------------------------------------------
    # Model Documentation
    # -------------------------------------------

    st.subheader("Model Documentation")

    st.markdown("""
Responsible AI requires documenting how models are developed and used.

Common documentation frameworks:

- Model Cards
- Data Sheets for Datasets
- AI Risk Assessments
""")

    st.info("""
Good documentation helps regulators, auditors, and stakeholders understand
AI systems and assess risks.
""")

    st.divider()

    # -------------------------------------------
    # Auditability
    # -------------------------------------------

    st.subheader("Auditability & Traceability")

    st.markdown("""
AI systems should support **audit trails** that allow organizations
to trace how predictions were produced.

Best practices:

- logging model predictions
- versioning datasets and models
- tracking feature transformations
- recording model retraining events
""")

    st.divider()

    # -------------------------------------------
    # Accountability
    # -------------------------------------------

    st.subheader("Accountability in AI Systems")

    st.markdown("""
Accountability ensures that organizations remain responsible for the
decisions made by AI systems.

Important elements:

- clearly defined ownership of models
- human oversight for high-risk systems
- governance processes for model updates
- incident response procedures
""")

    st.divider()

    # -------------------------------------------
    # Transparency Demo
    # -------------------------------------------

    st.subheader("Interactive Transparency Demo")

    uploaded = st.file_uploader("Upload dataset", type=["csv"], key="transparency")

    if uploaded:

        df = pd.read_csv(uploaded)

        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            feature = st.selectbox("Feature to analyze", numeric_cols)

            st.markdown("### Feature distribution")

            fig = px.histogram(df, x=feature)

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
Understanding the distribution of model inputs is an important step
for ensuring **transparent model behavior**.
""")

# ======================================================
# Privacy and security
# ======================================================
with tab5:

    st.header("🔐 Privacy & Security in Responsible AI")

    st.markdown("""
Privacy and security ensure that **personal and sensitive data is protected**
throughout the entire AI lifecycle.

Responsible AI systems must safeguard:

- personal information
- confidential data
- model integrity
- system access
""")

    st.divider()

    # --------------------------------------------------
    # Privacy Principles
    # --------------------------------------------------

    st.subheader("Privacy Principles")

    st.markdown("""
Responsible AI systems should follow core privacy principles:

- **Data minimization** – collect only necessary data  
- **Purpose limitation** – use data only for defined purposes  
- **User consent** – obtain informed consent from individuals  
- **Transparency** – inform users about how data is used  
- **Right to deletion** – allow users to request data removal  
""")

    st.divider()

    # --------------------------------------------------
    # Data Protection Techniques
    # --------------------------------------------------

    st.subheader("Data Protection Techniques")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Encryption**

Protects data during storage and transmission.

Examples:

- AES encryption
- TLS secure communication
""")

        st.markdown("""
**Access Control**

Restricts who can access sensitive data.

Methods include:

- role-based access control (RBAC)
- identity management
""")

    with col2:

        st.markdown("""
**Secure Storage**

Sensitive data should be stored in secure environments:

- encrypted databases
- protected cloud storage
""")

        st.markdown("""
**Monitoring and Logging**

Track system access and suspicious behavior.
""")

    st.divider()

    # --------------------------------------------------
    # Data Anonymization
    # --------------------------------------------------

    st.subheader("Data Anonymization")

    st.markdown("""
Data anonymization protects individual identities by removing
or transforming identifiable information.

Common methods:

- data masking
- pseudonymization
- aggregation
- tokenization
""")

    st.info("Anonymization reduces privacy risks but may impact model accuracy.")

    st.divider()

    # --------------------------------------------------
    # Privacy Risks in AI
    # --------------------------------------------------

    st.subheader("Privacy Risks in AI Systems")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Re-identification Risk**

Even anonymized data can sometimes be linked back to individuals.
""")

        st.markdown("""
**Model Inference Attacks**

Attackers may infer sensitive data from model outputs.
""")

    with col2:

        st.markdown("""
**Data Leakage**

Sensitive information accidentally exposed during training.
""")

        st.markdown("""
**Unauthorized Access**

Improper permissions allowing access to protected data.
""")

    st.divider()

    # --------------------------------------------------
    # Secure AI Lifecycle
    # --------------------------------------------------

    st.subheader("Security Across the AI Lifecycle")

    lifecycle_security = [
        "Secure data collection",
        "Encrypted data storage",
        "Secure model training environment",
        "Access-controlled model deployment",
        "Monitoring for security threats"
    ]

    for step in lifecycle_security:
        st.write("•", step)

    st.divider()

    # --------------------------------------------------
    # Privacy Compliance Checklist
    # --------------------------------------------------

    st.subheader("Privacy & Security Checklist")

    checklist = [
        "Sensitive data identified",
        "Encryption implemented",
        "Access controls defined",
        "Data anonymization applied",
        "Security monitoring enabled",
        "Privacy regulations identified"
    ]

    score = 0

    for item in checklist:

        if st.checkbox(item, key=f"Privacy_{item}"):
            score += 1

    percent = score / len(checklist) * 100

    st.progress(percent / 100)

    st.metric("Privacy Compliance", f"{percent:.0f}%")

# ======================================================
# Data Governance
# ======================================================
with tab6:

    st.header("📊 Data Governance")

    st.markdown("""
Data governance defines the **policies, standards, and responsibilities**
for managing data within an organization.

It ensures that data is:

- accurate
- secure
- compliant
- properly documented
- responsibly used throughout the AI lifecycle
""")

    st.divider()

    # -------------------------------------------
    # Governance Framework
    # -------------------------------------------

    st.subheader("Data Governance Framework")

    st.markdown("""
A data governance framework establishes **rules and processes**
for how data is managed across an organization.

Core components include:

- data policies
- access control
- data quality standards
- compliance procedures
- audit processes
""")

    st.divider()

    # -------------------------------------------
    # Data Management Plan
    # -------------------------------------------

    st.subheader("Data Management Plan (DMP)")

    st.markdown("""
A Data Management Plan defines **how data will be handled
throughout a project lifecycle**.

A typical DMP answers four key questions:
""")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**What data is used?**

- datasets
- sources
- formats
""")

        st.markdown("""
**Who is responsible?**

- data owners
- data stewards
- project teams
""")

    with col2:

        st.markdown("""
**When is data reviewed?**

- audit frequency
- validation steps
""")

        st.markdown("""
**How is data managed?**

- storage
- security
- retention policies
""")

    st.divider()

    # -------------------------------------------
    # Roles
    # -------------------------------------------

    st.subheader("Roles and Responsibilities")

    st.markdown("""
Effective data governance requires clear ownership of data assets.
""")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
**Data Owner**

Responsible for legal compliance and data usage policies.
""")

        st.markdown("""
**Data Steward**

Ensures data quality and proper data management.
""")

    with col2:

        st.markdown("""
**Data Engineer / Analyst**

Manages data pipelines and transformations.
""")

        st.markdown("""
**Compliance Officer**

Ensures regulatory alignment and audit readiness.
""")

    st.divider()

    # -------------------------------------------
    # Data Lifecycle
    # -------------------------------------------

    st.subheader("Data Lifecycle")

    st.markdown("""
Responsible data governance covers the **entire lifecycle of data**.
""")

    lifecycle = [
        "Data Acquisition",
        "Data Storage",
        "Data Processing",
        "Model Training",
        "Deployment",
        "Monitoring",
        "Archival / Disposal"
    ]

    st.write("Typical lifecycle stages:")

    for step in lifecycle:
        st.write("•", step)

    st.divider()

    # -------------------------------------------
    # Data Audits
    # -------------------------------------------

    st.subheader("Data Audits")

    st.markdown("""
Data audits ensure that datasets remain **accurate, compliant,
and aligned with governance policies**.

Audits typically verify:

- data quality
- regulatory compliance
- security controls
- fairness metrics
- documentation completeness
""")

    st.divider()

    # -------------------------------------------
    # Governance Checklist
    # -------------------------------------------

    st.subheader("Data Governance Checklist")

    checklist = [
        "Data ownership clearly defined",
        "Data management plan documented",
        "Access controls implemented",
        "Data audit schedule defined",
        "Compliance regulations identified",
        "Data lifecycle documented"
    ]

    score = 0

    for item in checklist:

        if st.checkbox(item, key=f"datagovernance_{item}"):
            score += 1

    percent = score / len(checklist) * 100

    st.progress(percent / 100)

    st.metric("Governance Maturity", f"{percent:.0f}%")          
