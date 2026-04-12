import streamlit as st
import requests
import json
from urllib.request import urlopen

st.set_page_config(page_title="APIs & Data Engineering", page_icon="🌐", layout="wide")

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

st.title("🌐 APIs & Data Engineering Toolkit")

# =========================
# Tabs
# =========================
t1, t2, t3, t4, t5 = st.tabs([
    "📥 API Basics",
    "🔗 Working with APIs (Python)",
    "🧠 API Concepts",
    "🔐 Authentication",
    "🚨 Error Handling"
])

# =========================
# 📥 API Basics
# =========================
with t1:
    st.subheader("What is an API?")
    st.markdown("""
    - **API (Application Programming Interface)**  
    - Set of communication rules and capabilities  
    - Enables interactions between software applications  
    """)

    st.subheader("Web APIs")
    st.markdown("""
    - Communicate over the internet using **HTTP**  
    - Client sends a **request message** to the server  
    - Server returns a **response message**  
    """)

    st.subheader("API Architectures")
    st.markdown("""
    - **SOAP**: strict and formal API design, used in enterprise applications  
    - **REST**: simple and scalable, most common architecture  
    - **GraphQL**: flexible and optimized for performance  
    """)

# =========================
# 🔗 Working with APIs
# =========================
with t2:
    st.subheader("Working with APIs in Python")

    st.markdown("""
    Main libraries:
    - `urllib`
    - `requests`
    """)

    st.markdown("### urllib Example")
    st.code("""
from urllib.request import urlopen

api = "http://api.music.com"

with urlopen(api) as response:
    data = response.read()
    string = data.decode()
    print(string)
""", language="python")

    st.markdown("### requests Example")
    st.code("""
import requests

response = requests.get(api)
print(response.text)
""", language="python")

# =========================
# 🧠 API Concepts
# =========================
with t3:
    st.subheader("Basic Anatomy of an API Request")

    st.markdown("""
    **URL (Uniform Resource Locator)**  
    Structured address to access an API resource  

    **Structure:**
    - Protocol → `http://`
    - Domain → `www.fb.com`
    - Port → `:80`
    - Path → `/unit/2024`
    - Query → `?floor=77`
    """)

    st.markdown("### Query Parameters")
    st.code("""
query_params = {'floor':77, 'elevator':True}

response = requests.get(url, params=query_params)
print(response.url)
""", language="python")

    st.subheader("HTTP Verbs")
    st.markdown("""
    - **GET** → read  
    - **POST** → create  
    - **PUT** → update  
    - **DELETE** → delete  
    """)

    st.subheader("Headers & Status Codes")
    st.markdown("""
    **Status Codes:**
    - 1XX → informational  
    - 2XX → success  
    - 3XX → redirection  
    - 4XX → client error  
    - 5XX → server error  

    **Common:**
    - 200 → OK  
    - 401 → Unauthorized  
    - 404 → Not Found  
    - 500 → Internal Server Error  
    """)

    st.markdown("### Headers")
    st.code("""
headers = {"Accept": "application/json"}

response = requests.get(url, headers=headers)
""", language="python")

# =========================
# 🔐 Authentication
# =========================
with t4:
    st.subheader("API Authentication")

    st.markdown("""
    **Basic Authentication**
    - Uses username and password  
    """)

    st.code("""
auth = ('username','password')
requests.get(url, auth=auth)
""", language="python")

    st.markdown("""
    **API Key / Token Authentication**
    - Uses a token  
    - Can be passed as query param or header  
    """)

    st.code("""
headers = {"Authorization": "Bearer YOUR_TOKEN"}
requests.get(url, headers=headers)
""", language="python")

    st.markdown("""
    **JWT (JSON Web Token)**
    - Token-based  
    - Limited lifespan  

    **OAuth 2.0**
    - Auth framework without sharing credentials  
    """)

# =========================
# 🚨 Error Handling
# =========================
with t5:
    st.subheader("Error Handling")

    st.markdown("""
    **Types of errors:**
    - **4XX (Client errors)** → bad request, auth failure  
    - **5XX (Server errors)** → server issues  

    **Common codes:**
    - 401 → unauthorized  
    - 404 → not found  
    - 429 → too many requests  
    - 500 → internal server error  
    - 502 → bad gateway  
    - 504 → gateway timeout  
    """)

    st.markdown("""
    **Types of failures:**
    - API errors → 4xx / 5xx  
    - Connection errors → no response at all  
    """)

    st.code("""
import requests

response = requests.get(url)

try:
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print("Error:", e)
""", language="python")

# =========================
# JSON Section
# =========================
st.divider()
st.subheader("📦 Working with Structured Data")

st.markdown("""
- APIs return structured data (e.g., music albums)  
- Common formats:
    - JSON (most common)
    - XML
    - CSV
    - YAML  
""")

st.code("""
import json

# Python -> JSON
json.dumps(data)

# JSON -> Python
json.loads(string)
""", language="python")