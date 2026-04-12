import streamlit as st

st.set_page_config(layout="wide")

# ---------- COLOR SYSTEM ----------
PRIMARY = "#F4B740"        # amarillo (accent)
PRIMARY_HOVER = "#D89B20"

PURPLE_DARK = "#3F3D56"   # base elegante
PURPLE_SOFT = "#6C63FF"   # accent secundario

BG = "#F7F7FB"            # 👈 CAMBIO CLAVE (ya no blanco puro)
CARD_BG = "#FFFFFF"

TEXT = "#1F2933"
TEXT_SOFT = "#6B7280"
BORDER = "#E5E7EB"

# ---------- STYLES ----------
st.markdown(f"""
<style>

/* Layout */
.block-container {{
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 2rem;
    background-color: {BG};
}}

/* Hide Streamlit UI */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* HERO */
.hero-title {{
    font-size: 3rem;
    font-weight: 700;
    color: {PURPLE_DARK};
    letter-spacing: -0.5px;
}}

.hero-sub {{
    font-size: 1.2rem;
    color: {TEXT_SOFT};
    margin-bottom: 1.5rem;
}}

/* SECTION SPACING */
section {{
    margin-bottom: 2.5rem;
}}

/* CARD (ELEVATED) */
.card {{
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid {BORDER};
    background-color: {CARD_BG};
    margin-bottom: 1rem;

    /* 👇 profundidad */
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);

    /* 👇 acento más visible */
    border-top: 3px solid {PURPLE_SOFT};

    transition: all 0.2s ease;
}}

.card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}}

.card h3, .card h4 {{
    margin-bottom: .4rem;
    color: {TEXT};
}}

.card p {{
    margin: .4rem 0 .8rem 0;
    color: {TEXT_SOFT};
}}

/* BUTTONS */
.btn {{
    display: inline-block;
    padding: 0.55rem 1.1rem;
    border-radius: 10px;
    background-color: {PRIMARY};
    color: black !important;
    text-decoration: none;
    font-weight: 600;
}}

.btn:hover {{
    background-color: {PRIMARY_HOVER};
}}

.btn.secondary {{
    background-color: {PURPLE_DARK};
    color: white !important;
}}

.btn.secondary:hover {{
    background-color: #2f2d45;
}}

/* HEADERS */
h1, h2, h3 {{
    color: {TEXT};
}}

/* SECTION TITLE CENTER */
.section-title {{
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}}

.section-sub {{
    text-align: center;
    color: {TEXT_SOFT};
    max-width: 600px;
    margin: auto;
}}

</style>
""", unsafe_allow_html=True)
# ---------- HERO ----------
col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="hero-title">Liliam Martínez</div>', unsafe_allow_html=True)
    st.markdown("### Senior Data Scientist")

    st.markdown(f"""
<div class="hero-sub">
I build analytics tools to monitor transaction performance, 
analyze trends, and support business decision-making through forecasting and data insights.
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<a class="btn" href="https://www.linkedin.com/in/liliammtz/">Work with me</a>
<a class="btn secondary" href="https://github.com/liliammtz" target="_blank">GitHub</a>
""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card" style="text-align:center;">
        <h3>Open to Consulting</h3>
        <p>Dashboards · Forecasting · Data Analytics</p>
        <p style="font-size:0.9rem;color:{TEXT_SOFT};">
            CDMX · Remote
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------- PROBLEM ----------
# ---------- PROBLEM ----------
st.markdown("""
<div style="text-align:center; margin-top:3rem; margin-bottom:2rem;">
    <h2 style="margin-bottom:0.5rem;">Why data teams struggle</h2>
    <p style="color:#6B7280; font-size:1.05rem; max-width:600px; margin:auto;">
        Even with data available, many teams lack visibility, clarity, and actionable insights.
    </p>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h4>Limited visibility</h4>
        <p>
        Data exists across systems, but there’s no clear way to monitor what’s happening in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4>Unreliable forecasts</h4>
        <p>
        Forecasts are often static, hard to interpret, and not aligned with actual business behavior.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h4>Reactive decisions</h4>
        <p>
        Issues are identified too late, making it difficult to respond proactively or prevent impact.
        </p>
    </div>
    """, unsafe_allow_html=True)
# ---------- SOLUTION ----------
st.divider()
# ---------- SOLUTION ----------
st.markdown("""
<div style="text-align:center; margin-top:3rem; margin-bottom:2rem;">
    <h2 style="margin-bottom:0.5rem;">How I help teams make better decisions</h2>
    <p style="color:#6B7280; font-size:1.05rem; max-width:600px; margin:auto;">
        I design analytics solutions that bring clarity to data and support faster, more confident decision-making.
    </p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <h4>Transaction monitoring</h4>
        <p>
        Build dashboards to track transaction performance, identify anomalies, 
        and understand behavior across segments.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4>Forecasting & planning</h4>
        <p>
        Develop forecasting models to anticipate trends, compare expected vs actuals, 
        and support planning decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="card" style="margin-top:1rem;">
    <h4>End-to-end analytics workflows</h4>
    <p>
    From data extraction and transformation to building production-ready dashboards, 
    I create solutions that teams can actually use in their day-to-day operations.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- TOOLKITS ----------
st.divider()
st.markdown("""
<div style="text-align:center; margin-top:2.5rem; margin-bottom:1.5rem;">
    <h2 style="margin-bottom:0.4rem;">Analytics toolkits</h2>
    <p style="color:#6B7280; font-size:1rem; max-width:600px; margin:auto;">
        Modular tools designed to accelerate analysis, standardize workflows, 
        and build scalable analytics solutions.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.page_link("pages/01_EDA_Toolkit.py", label="Data Exploration & Validation", icon="🔎")
c2.page_link("pages/02_Forecasting.py", label="Forecasting & Time Series", icon="📈")
c3.page_link("pages/02_ML_Toolkit.py", label="Machine Learning", icon="🧠")
c1.page_link("pages/04_LLM_Toolkit.py", label="LLM Applications", icon="🤖")
c2.page_link("pages/03_MLOps_Toolkit.py", label="MLOps & Deployment", icon="⚙️")
c3.page_link("pages/05_Responsable_AI.py", label="AI Safety and Ethics", icon="🛡️")
c2.page_link("pages/07_APIs.py", label="APIs and Data Engineering", icon="🌐")

#!st.markdown("""Designed to be reusable, structured, and adaptable to different business cases.""")

# ---------- CREDENTIALS ----------
st.divider()
st.header("Credentials")

certs = {
    "Neo4j – Certified Professional":
        "https://graphacademy.neo4j.com/c/bf6ffa2d-eb5d-46ad-a4b5-426ba5fcd022/",
    "Snowflake – BUILD 2025 - 2026: Data Engineering Bootcamp":
        "https://developerbadges.snowflake.com/9a786c79-c83b-41b7-ac02-5a556a300454#acc.fREOCm0c",
    "Snowflake – BUILD 2025 - 2026: Gen AI Bootcamp":
        "https://developerbadges.snowflake.com/1e9c2833-ff7b-43eb-9756-1714dba7e59a#acc.zJZHQEjM",
    "Snowflake – BUILD 2023 Builder Badge":
        "https://developerbadges.snowflake.com/0521f0c1-3709-4390-be0b-71dcb6503af7#acc.l91ICgm2",
    "Snowflake – BUILD 2023 LLM Bootcamp Badge":
        "https://developerbadges.snowflake.com/392e3d08-50b0-4206-9d42-d18dd960be8e#acc.3VYXCIjR",
    "Snowflake – Hands-On Essentials: Data Warehousing Workshop":
        "https://achieve.snowflake.com/0205e6b8-9552-49d6-bc73-3a5b34165697#acc.XcPpNO5g",
    "Google – IT Automation with Python":
        "https://www.coursera.org/account/accomplishments/professional-cert/V4Y6FJ4WVZZT",
    "University of Michigan – Data Science Ethics":
        "https://www.coursera.org/account/accomplishments/verify/R4RWDFN436AU",
    "University of Michigan - Applied Data Science with Python":
        "https://www.coursera.org/account/accomplishments/specialization/FS84KRYH2TRQ",
    "UCI - Effective Problem-Solving and Decision-Making":
        "https://www.coursera.org/account/accomplishments/verify/ELE4HJZ5DCQH",
    "Duke - Excel to MySQL: Analytic Techniques for Business":
        "https://www.coursera.org/account/accomplishments/specialization/3K4P5JZC78US"

}

for name, url in certs.items():
    st.markdown(f"- [{name}]({url})")

# ---------- FAQ ----------
st.divider()

st.markdown("""
<div style="text-align:center; margin-top:2rem; margin-bottom:1.5rem;">
    <h2 style="margin-bottom:0.4rem;">Frequently asked questions</h2>
    <p style="color:#6B7280; font-size:1rem;">
        A bit more about how I work and what I bring to projects
    </p>
</div>
""", unsafe_allow_html=True)


with st.expander("What kind of problems do you help solve?"):
    st.write("""
I work on analytics challenges related to transaction monitoring, forecasting, 
and performance analysis. This includes identifying anomalies, understanding trends, 
and helping teams make better data-driven decisions.
""")


with st.expander("What tools and technologies do you use?"):
    st.write("""
I primarily work with Python and SQL, using platforms like Snowflake and SQL Server 
to manage and process data. 

For building applications and dashboards, I use Streamlit, Plotly, Tableau, and Power BI, 
and I’m comfortable working with Git-based workflows for development and deployment.
""")


with st.expander("Do you work with machine learning or forecasting models?"):
    st.write("""
Yes — I develop predictive models using tools like scikit-learn, StatsForecast, and Prophet. 
These are typically used for forecasting trends, detecting anomalies, and supporting 
business planning decisions.
""")


with st.expander("Do you work with large-scale or complex data systems?"):
    st.write("""
Yes — I have experience working with structured and semi-structured data using tools 
like Snowflake, MongoDB, and Neo4j, as well as monitoring systems like ELK Stack and Splunk.
""")


with st.expander("What areas of data do you specialize in?"):
    st.write("""
My work focuses on data science, analytics, and business intelligence — particularly in 
building solutions that connect data with real decision-making processes.
""")


with st.expander("Do you offer consulting or freelance work?"):
    st.write("""
Yes — I’m open to consulting and freelance opportunities. 
I can help design dashboards, build forecasting models, or develop end-to-end analytics solutions.
""")


with st.expander("How can we work together?"):
    st.write("""
You can reach out via email or LinkedIn. I’m happy to discuss your use case and explore how I can help.
""")
    
# ---------- CTA ----------
st.divider()
st.markdown('<a name="contact"></a>', unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("## Let’s work together")
    st.write("I’m open to consulting, freelance, or full-time opportunities.")

with col2:
    st.markdown("""
<a class="btn" href="mailto:liliammtzfdz@gmail.com">Contact me</a>
""", unsafe_allow_html=True)