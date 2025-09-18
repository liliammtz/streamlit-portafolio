from datetime import date
import streamlit as st

# ---------- BASIC PAGE CONFIG ----------
st.set_page_config(
    page_title="Liliam Martínez · Data Scientist Portfolio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- STYLES ----------
PRIMARY = "#d46a92"   # acento rosa elegante
BG_SOFT = "#ffffff"   # fondo limpio blanco
CARD_BG = "#f8f0f4"   # fondo de tarjetas en rosa muy claro
TEXT_SOFT = "#5a5a66" # gris suave para texto secundario


st.markdown(
    f"""
    <style>
    /* Hide Streamlit default menu/footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Page padding + max width */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* Hero */
    .hero h1 {{
        font-size: 3rem;
        line-height: 1.1;
        margin-bottom: .2rem;
    }}
    .hero p.sub {{
        color: {TEXT_SOFT};
        font-size: 1.1rem;
        margin-top: .2rem;
    }}

    /* Soft badge */
    .badge {{
        display:inline-block; padding:.25rem .6rem; border-radius:999px; 
        background:linear-gradient(90deg, {PRIMARY}22, {PRIMARY}44);
        border:1px solid {PRIMARY}55; color:#2c2c34; font-size:.85rem; margin-right:.35rem;
        font-weight:500;
    }}

    /* Card */
    .card {{
        background: {CARD_BG}; border:1px solid #e5c6d5; border-radius: 18px;
        padding: 1.2rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 6px 18px #d46a9220;
    }}
    .card h3 {{margin: 0 0 .2rem 0; color:#2c2c34;}}
    .card p {{margin: .3rem 0 .8rem 0; color:{TEXT_SOFT};}}

    /* Button links */
    .linkrow a {{
        text-decoration:none; border:1px solid {PRIMARY}; background:{PRIMARY}; color:white; 
        padding:.55rem .8rem; border-radius: 12px; margin-right: .5rem; display:inline-block;
        font-weight:500;
    }}
    .linkrow a:hover {{background:#c1577e;}}

    /* Skills chips */
    .chips span {{
        display:inline-block; padding:.35rem .6rem; border-radius:999px; border:1px solid #e5c6d5; 
        margin:.2rem; color:#2c2c34; font-size:.9rem; background:#fff0f6;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    /* --- Botones dentro de la card --- */
    .card .actions {{
        margin-top: .6rem;
        display: flex;
        gap: .5rem;
        align-items: center;
        justify-content: flex-start; /* izquierda */
        flex-wrap: wrap;
    }}
    .btn {{
        display: inline-block;
        padding: .55rem .9rem;
        border-radius: 12px;
        border: 1px solid {PRIMARY};
        background: {PRIMARY};
        color: white !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: .9rem;
        box-shadow: 0 2px 6px #d46a9240;
        cursor: pointer;
    }}
    .btn:hover {{ background: #c1577e; border-color: #c1577e; }}
    .btn.ghost {{
        background: transparent;
        color: {PRIMARY} !important;
        border-color: {PRIMARY};
    }}
    .btn.ghost:hover {{
        background: {PRIMARY}15;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- SIDEBAR (always-visible contact + nav) ----------
with st.sidebar:
    st.markdown("### 👋 About me")
    st.write(
        "Data Scientist with a strong background in **forecasting**, **business intelligence**, and **ML-powered analytics**. "
        "I specialize in building **end-to-end data products** — from data pipelines and predictive models in **Snowflake/SQL** "
        "to polished **Streamlit apps** used daily by business teams. "
        "Passionate about turning raw data into clear, actionable insights that support **strategic decision-making**."
    )

    st.page_link("Main.py", label="Home", icon="🏠")
    st.divider()
    st.markdown("**Tools**")
    st.page_link("pages/01_EDA_Toolkit.py", label="EDA Toolkit", icon="📊")
    #!st.page_link("pages/02_Forecasting_Studio.py", label="Forecasting Toolkit", icon="📈")
    #!st.page_link("pages/03_DataOps_Toolkit.py", label="DataOps Toolkit", icon="🧰")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")
    #!st.page_link("pages/04_Save_The_Bees.py", label="save", icon="🧰")

    st.divider()
    st.markdown("**Contact**")
    st.markdown("- GitHub: [@liliam-mtz](https://github.com/liliammtz)")
    st.markdown("- LinkedIn: [Liliam Martínez](https://www.linkedin.com/in/liliammtz/)")
    st.markdown("- Email: [liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")

# ---------- HERO ----------
col1, col2 = st.columns([1.4, 1])
with col1:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.markdown("### Hi, I’m Liliam ✨")
    st.markdown("# Data Scientist")
    #!st.markdown(
    #!    """
    #!    I design **clean, fast** data apps and **explainable** ML.  
    #!    Currently building forecasting and anomaly detection tools in **Snowflake + Streamlit**.
    #!    """)
    st.markdown(
        """
        <div class="linkrow">
          <a href="https://github.com/liliammtz" target="_blank">GitHub</a>
          <a href="https://www.linkedin.com/in/liliammtz/" target="_blank">LinkedIn</a>
          <a href="mailto:liliammtzfdz@gmail.com" target="_blank">Email</a>
          <a href="https://www.datacamp.com/portfolio/liliammtzfdz" target="_blank">DataCamp</a>
          <a href="https://graphacademy.neo4j.com/u/3f4f094d-5ac2-44e0-b98a-7b9d8c2c6079/" target="_blank">Neo4j</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.write("")
    st.markdown(
        f"""
        <div class="card" style="
            text-align:center;
            background:linear-gradient(180deg,#ffffff,{CARD_BG});
        ">
            <div class="badge">Available for freelance/consulting</div>
            <h3 style="color:#2c2c34;">Open to Collaborations</h3>
            <p style="color:#2c2c34;">Dashboards · Forecasting · E2E data apps</p>
            <p style="font-size:0.9rem;color:{TEXT_SOFT};">
                Based in CDMX · Remote friendly
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- SKILLS ----------
# ---------- SKILLS ----------
st.subheader("Skills snapshot")

# Definir categorías
skills = {
    "Programming & Scripting": ["Python", "Bash", "Matlab"],
    "Databases": ["SQL Server", "Snowflake", "MongoDB", "Neo4j"],
    "Tools & Platforms": ["Git", "Office 365", "Google Suite"],
    "Visualization & BI": ["Streamlit", "Plotly", "Tableau", "Power BI"],
    "Monitoring & Logs": ["ELK Stack", "Splunk"],
    "Modeling & Forecasting": ["StatsForecast", "Prophet", "scikit-learn"],
    "Focus Areas": ["Data Science", "AI", "Big Data", "Business Intelligence", "Statistical Analysis"],
}

# Renderizar como tabla
for category, items in skills.items():
    left, right = st.columns([1, 3])
    with left:
        st.markdown(f"**{category}**")
    with right:
        st.markdown(
            "<div class='chips'>" + "".join([f"<span>{item}</span>" for item in items]) + "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------- FEATURED PROJECTS ----------
st.subheader("Featured projects")

def project_card(title: str, desc: str, tags: list[str], repo_url: str | None = None, page_path: str | None = None):
    chips = " ".join([f"<span>{t}</span>" for t in tags])

    # Construir botones como "botones" estilizados, no hipervínculos azules
    buttons_html = []
    if page_path:
        # Nota: para multipage de Streamlit, si 'page_path' es un archivo, considera pasar la URL final de la app al desplegar.
        buttons_html.append(f"<a class='btn' href='{page_path}'>Open app</a>")
    if repo_url:
        buttons_html.append(f"<a class='btn ghost' href='{repo_url}' target='_blank' rel='noopener'>GitHub repo</a>")

    st.markdown(
        f"""
        <div class='card'>
          <h3>{title}</h3>
          <p>{desc}</p>
          <div class='chips'>{chips}</div>
          <div class='actions'>
            {' '.join(buttons_html)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



c1, c2 = st.columns(2)
with c1:
    project_card(
        title="Save the bees 🐝",
        desc="Interactive dashboard ",
        tags=["Streamlit", "Data Visualization", "Plotly"],
        repo_url="https://github.com/liliammtz",
        page_path="Save_The_Bees",
    )

st.divider()

# ---------- CERTIFICATIONS ----------
st.subheader("Certifications")

certs = {
    "Neo4j – Graph Data Science Fundamentals":
        "https://graphacademy.neo4j.com/c/db3edc28-e119-4bc7-aee1-c6621d97ff98/",
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

st.divider()


# ---------- CONTACT ----------
st.subheader("Get in touch")
col_a, col_b = st.columns([1,1])
with col_a:
    st.markdown("**Email**")
    st.markdown("[liliammtzfdz@gmail.com](mailto:liliammtzfdz@gmail.com)")
with col_b:
    st.markdown("**LinkedIn**")
    st.markdown("[https://www.linkedin.com/in/liliammtz/](https://www.linkedin.com/)")

st.caption("© " + str(date.today().year) + " · Built with Streamlit")