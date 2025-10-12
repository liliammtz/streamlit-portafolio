import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Forecasting Toolkit", layout="wide")

st.title("⚙️ Forecasting Toolkit")
st.caption("An interactive guide to learn from scratch what forecasting is and how to apply it using python.")

tab_intro, tab_statistical, tab_deepl = st.tabs(['Introduction','Statistical', 'Deep Learning'])

with tab_intro:
    st.title("📈 Forecasting Toolkit: Introduction")

    st.markdown("""
    ### ⏳ What is a Time Series?
    A **time series** is a set of data points ordered in time, recorded at equal intervals 
    (e.g., every hour, day, month, or quarter).  
    Common examples include:
    - 📊 The closing value of a stock  
    - 💡 A household’s electricity consumption  
    - 🌡️ The outside temperature
    """)

    st.markdown("---")

    st.subheader("🔍 Components of a Time Series")
    st.markdown("""
    **Decomposition** is the process of breaking down a time series into its components:
    - **Trend** → Slow, long-term movement in the series (increasing or decreasing)  
    - **Seasonality** → Repeating patterns over fixed periods (e.g., yearly, weekly)  
    - **Residuals (Noise)** → Random fluctuations not explained by trend or seasonality  

    This helps us identify hidden structures in the data that aren’t obvious at first glance.
    """)

    st.markdown("---")

    st.subheader("📊 Time Series Forecasting")
    st.markdown("""
    Forecasting means predicting the future using **historical data** and any 
    **known future events** that may influence outcomes.

    ✅ **Steps before building a forecast**:
    1. Define your **goal**  
    2. Decide **what** to forecast  
    3. Set the **forecast horizon** (how far ahead to predict)  
    4. Gather and prepare data  
    5. Develop a model  
    6. Deploy the model into production  
    7. Monitor new incoming data  
    8. Update and retrain the model when necessary  
    """)

    st.markdown("---")

    st.subheader("⚖️ Key Differences: Forecasting vs. Regression")
    st.markdown("""
    - Time series **have order** (past values influence the future).  
    - Time series sometimes **lack independent features**, relying mostly on the sequence itself.  
    """)

    st.markdown("---")

    st.subheader("📌 Baseline Models")
    st.markdown("""
    A **baseline model** is a simple heuristic solution to a forecasting problem.  
    It is:
    - Easy to implement  
    - Doesn’t require fitting complex models  
    - Useful as a reference point  

    👉 **Challenge**: Can we improve our baseline by:  
    - Using the **mean of more recent periods** instead of all history?  
    - Exploiting **seasonal patterns** (e.g., using last year’s values as predictions)?  
    """)

    st.info("💡 Naive seasonal forecasting uses repeating cycles in the data as predictions.")

    # ------------------------
    # RANDOM WALK
    # ------------------------
    st.subheader("🎲 Random Walk")

    st.markdown("""
    A **random walk** is a process where each step is random, with equal probability of going up or down.  

    🔑 **Key properties**:
    - Its **first difference** is stationary and uncorrelated.  
    - Often found in **financial and economic data** (e.g., stock prices).  
    - Shows long upward or downward drifts, but also sudden changes in direction.  

    Mathematically:

    $$ y_t = C + y_{t-1} + \varepsilon_t $$

    where:
    - $y_t$ → value at time *t*  
    - $C$ → constant  
    - $y_{t-1}$ → value at previous timestep  
    - $\\varepsilon_t$ → random noise  
    """)

    st.info("💡 Example: The daily closing price of **GOOGL** can often be approximated as a random walk.")

    st.warning("⚠️ Random walks should ideally be forecast **only in the short term (next timestep)**. "
            "Otherwise, random errors accumulate and forecasts degrade quickly.")

    st.markdown("---")

    # ------------------------
    # STATIONARITY
    # ------------------------
    st.subheader("📏 Stationarity")

    st.markdown("""
    A **stationary time series** has statistical properties that do not change over time.  
    That means:
    - Constant **mean**  
    - Constant **variance**  
    - Constant **autocorrelation**  

    ✅ Many forecasting models (MA, AR, ARMA, etc.) assume stationarity.  
    ❌ If the series is non-stationary, forecasts will be unreliable.  
    """)

    st.warning("Most real-world series are **not stationary** initially because they have trends or seasonality.")

    st.markdown("""
    ### 🔧 Transformations to achieve stationarity:
    - **Differencing** → removes trend by subtracting previous value:  
    $ y'_t = y_t - y_{t-1} $  
    - **Log transform** → stabilizes variance  
    - Higher-order differencing → applied more than once if needed  

    When we apply a transformation, we must **invert it** later to return forecasts to their original scale (**inverse transform**).
    """)

    st.markdown("---")

    # ------------------------
    # ADF TEST
    # ------------------------
    st.subheader("🧪 Augmented Dickey-Fuller (ADF) Test")

    st.markdown("""
    The **ADF test** checks if a series is stationary by testing for a **unit root**.

    - **Null hypothesis (H₀):** The series has a unit root → not stationary  
    - **Alternative (H₁):** No unit root → stationary  

    Results:
    - **ADF statistic** → more negative → stronger rejection of H₀  
    - **p-value** → if $p < 0.05$, reject H₀ → series is stationary  
    """)

    st.info("💡 A stationary series has a **flat mean** over time, while a non-stationary one has a varying mean.")

    st.markdown("---")

    # ------------------------
    # AUTOCORRELATION
    # ------------------------
    st.subheader("🔗 Autocorrelation Function (ACF)")

    st.markdown("""
    The **autocorrelation function (ACF)** measures how correlated a series is with its past values.

    - **Lag** → number of timesteps between two values  
    - **ACF plot** → shows correlation at different lags  

    👉 Once the series is stationary, plotting the **ACF** helps determine if the process is a random walk or another type of model.
    """)

    st.markdown("---")

    # ------------------------
    # METRICS
    # ------------------------
    st.subheader("📊 Forecasting Metrics")

    st.markdown("""
    - **MAPE (Mean Absolute Percentage Error):**  
    An intuitive metric for forecast accuracy, measuring how far predictions deviate (in %) from actual values.
    """)



with tab_statistical:
    st.subheader("📚 Summary")

    st.markdown("""
    By the end of this section, you will have a **robust framework** for modeling time series using a variety of statistical approaches:

    ### 🔢 Core Statistical Models
    - **MA(q)** → Moving Average models  
    - **AR(p)** → Autoregressive models  
    - **ARMA(p,q)** → Combination of AR and MA for stationary series  
    - **ARIMA(p,d,q)** → Handles **non-stationary series** with differencing  
    - **SARIMA(p,d,q)(P,D,Q)m** → Adds **seasonality**  
    - **SARIMAX** → Incorporates **external variables (exogenous factors)**  

    ### 🔗 Multivariate Models
    - **VAR(p)** → Vector Autoregression, for forecasting **multiple time series** simultaneously  

    ### 📉 Other Approaches
    - **Exponential Smoothing (ETS)** → Weighted average of past values, giving more weight to recent data  
        - Can be extended to include **trend** and **seasonality**  
    - **BATS & TBATS** → Handle complex seasonality (multiple seasonal periods)  

    ✨ With these models, you’ll be equipped to model, compare, and forecast a wide range of time series problems.
    """)
    
    # ------------------------
    # MOVING AVERAGE MODEL (MA)
    # ------------------------
    st.subheader("📉 Moving Average (MA) Model")

    st.markdown("""
    A **moving average process (MA)** models a stationary time series where the **current value** depends on:
    - The mean of the series ($\\mu$)  
    - The **current error term** ($\\varepsilon_t$)  
    - Past error terms ($\\varepsilon_{t-1}, \\varepsilon_{t-2}, ...$)  

    The error terms are assumed to be **independent, normally distributed white noise**.
    """)

    st.markdown("""
    ### 🧮 General MA(q) Formula
    For order $q$, the process is written as:

    $$
    y_t = \\mu + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + \\theta_2 \\varepsilon_{t-2} + \\dots + \\theta_q \\varepsilon_{t-q}
    $$

    - $q$ → order of the MA process  
    - $\\theta_q$ → coefficient quantifying the effect of the $q$-th lagged error  
    """)

    st.info("💡 The larger **q**, the more past error terms influence the current value.")

    st.markdown("---")

    # ------------------------
    # IDENTIFICATION
    # ------------------------
    st.subheader("🔎 Identifying MA(q) Processes")

    st.markdown("""
    To identify if a series follows an MA(q) process:
    - Inspect the **ACF (Autocorrelation Function) plot**.  
    - In an MA(q), autocorrelation coefficients become **non-significant abruptly after lag q**.  

    👉 Example: If the differenced volume of sales shows significance only up to lag 2, the process is MA(2).
    """)

    st.markdown("---")

    # ------------------------
    # FORECASTING WITH MA(q)
    # ------------------------
    st.subheader("📊 Forecasting with MA(q)")

    st.markdown("""
    - You can only forecast **up to q steps ahead**.  
    - Beyond q steps, there are no past error terms available → the model defaults to predicting the **mean**.  
    - This makes long-term forecasts with MA models equivalent to a **baseline mean forecast**.  

    ✅ To extend forecasts: use **rolling forecasts** (predict repeatedly step by step).
    """)

    st.warning("⚠️ Forecasting beyond q steps with MA(q) will simply return the mean of the series.")

    st.markdown("---")

    # ------------------------
    # KEY POINTS
    # ------------------------
    st.subheader("📌 Key Takeaways")

    st.markdown("""
    - 📉 An MA(q) process: current value depends on mean, current error, and past errors.  
    - 🧩 **ACF** helps identify q (significant until lag q only).  
    - ⏳ Forecast horizon limited to **q steps ahead**.  
    - 🔁 Use **rolling forecasts** to avoid flat predictions at the mean.  
    - 🧮 If transformations are applied, remember to **inverse-transform** results back to the original scale.  
    - 📊 Assumes the data is **stationary** → must transform/difference the data first if needed.  
    """)



    # ------------------------
    # AUTOREGRESSIVE MODEL (AR)
    # ------------------------
    st.subheader("🔄 Autoregressive (AR) Model")

    st.markdown("""
    An **autoregressive process (AR)** models a variable as a **regression against its own past values**.  

    - The process is denoted as **AR(p)**, where *p* is the order.  
    - The present value depends on a constant, past values, and a white noise error term.  
    """)

    st.markdown("""
    ### 🧮 General AR(p) Formula
    $$
    y_t = C + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t
    $$

    For a **first-order AR process (AR(1))**:
    $$
    y_t = C + \phi_1 y_{t-1} + \varepsilon_t
    $$
    """)

    st.info("💡 The order *p* determines how many past values influence the present value.")

    st.markdown("---")

    # ------------------------
    # IDENTIFYING AR(p)
    # ------------------------
    st.subheader("🔎 Identifying AR(p) Processes")

    st.markdown("""
    - If the **ACF plot** of a stationary process shows **slow exponential decay**, it suggests an AR process.  
    - To determine the exact order **p**, use the **PACF (Partial Autocorrelation Function)**:
    - In an AR(p), PACF coefficients are significant **up to lag p only**.  
    - After lag p, PACF coefficients become abruptly non-significant.  
    """)

    st.markdown("""
    ### 📊 Partial Autocorrelation
    Partial autocorrelation measures the correlation between lagged values **after removing the effect of intermediate lags**.  
    This allows us to isolate the *true order* of the AR process.
    """)

    st.warning("⚠️ If both ACF and PACF show slow decay or sinusoidal patterns, the process may not be pure AR or MA → it could be an **ARMA(p,q)**.")

    st.markdown("---")

    # ------------------------
    # SUMMARY
    # ------------------------
    st.subheader("📌 Key Takeaways (AR Model)")

    st.markdown("""
    - 🔄 An autoregressive process states the present value depends on its **past values + error term**.  
    - 📉 If ACF shows **slow decay**, it is likely AR.  
    - 🔗 PACF helps identify the order *p* (significant until lag *p* only).  
    - ⚖️ If neither ACF nor PACF provides clear information, the process may be a combination → **ARMA(p,q)**.  
    """)


    # ------------------------
    # AUTOREGRESSIVE MOVING AVERAGE MODEL (ARMA)
    # ------------------------
    st.header("🔄 ARMA Model")

    st.markdown("""
    The **Autoregressive Moving Average (ARMA)** model combines the strengths of both:
    - **AR(p):** dependence on past values  
    - **MA(q):** dependence on past error terms  

    It is denoted as **ARMA(p,q)**, where:
    - *p* → order of the autoregressive part  
    - *q* → order of the moving average part  
    """)

    st.markdown("""
    ### 🧮 General ARMA(p,q) Formula
    $$
    y_t = C + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + 
        \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \dots + \theta_q \varepsilon_{t-q}
    $$

    - If $p=0$, ARMA reduces to **MA(q)**.  
    - If $q=0$, ARMA reduces to **AR(p)**.  
    """)

    st.info("💡 The orders *p* and *q* define how many past values and past errors influence the present value.")

    st.markdown("---")

    # ------------------------
    # IDENTIFICATION
    # ------------------------
    st.subheader("🔎 Identifying ARMA(p,q)")

    st.markdown("""
    - A stationary ARMA process typically shows a **decaying or sinusoidal pattern** on **both ACF and PACF plots**.  
    - Because of this, **ACF and PACF cannot be used alone** to determine p and q.  
    - Instead, we fit several ARMA(p,q) models and use **model selection criteria** and **residual analysis**.  
    """)

    st.markdown("---")

    # ------------------------
    # MODEL SELECTION: AIC
    # ------------------------
    st.subheader("📊 Model Selection with AIC")

    st.markdown("""
    The **Akaike Information Criterion (AIC)** helps select the best ARMA model:  

    - **Lower AIC → better model** (relative comparison).  
    - AIC balances:
    - **Goodness of fit** (how well the model explains the data).  
    - **Model complexity** (number of parameters $k$ increases with *p* and *q*).  

    ✅ This avoids **overfitting** (too many parameters) and **underfitting** (too few parameters).  
    """)

    st.latex(r"AIC = 2k - 2 \ln(\hat{L})")

    st.info("💡 The model with the **lowest AIC** among candidates is typically selected.")

    st.markdown("---")

    # ------------------------
    # RESIDUAL ANALYSIS
    # ------------------------
    st.subheader("🧪 Residual Analysis")

    st.markdown("""
    After selecting a model, we must check if **residuals behave like white noise**:  
    - **Uncorrelated**  
    - **Independent**  
    - **Normally distributed**  

    ### 📉 Diagnostic Tools
    - **Q-Q Plot:** compares residual distribution vs theoretical normal.  
    - Straight line ≈ normal → good fit.  
    - Curved line ≈ not normal → poor fit.  
    - **Ljung-Box Test:** checks if residuals are autocorrelated.  
    - **H₀:** residuals are independent (like white noise).  
    - If *p-value > 0.05* → fail to reject H₀ → residuals are white noise → model is valid.  
    - If *p-value < 0.05* → reject H₀ → residuals are autocorrelated → model is invalid.  
    """)

    st.warning("⚠️ A valid ARMA model must leave residuals that look like **white noise**. Otherwise, try different (p,q).")

    st.markdown("---")

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways (ARMA Model)")

    st.markdown("""
    - 🔄 **ARMA(p,q) = AR(p) + MA(q)**.  
    - 📉 ACF & PACF both show **decay/sinusoidal patterns** → ARMA suspected.  
    - 📊 Use **AIC** for model selection (lowest = best relative model).  
    - 🧪 Always confirm with **residual diagnostics** (Q-Q plot + Ljung-Box test).  
    - ✅ A valid model = residuals ≈ **white noise** (uncorrelated, normal, independent).  
    """)

    # ------------------------
    # AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA)
    # ------------------------
    st.header("🔮 ARIMA Model")

    st.markdown("""
    The **Autoregressive Integrated Moving Average (ARIMA)** model extends the ARMA(p,q) framework to handle **non-stationary series**.  

    - **AR(p):** dependence on past values  
    - **I(d):** differencing to remove non-stationarity (integration)  
    - **MA(q):** dependence on past errors  

    👉 Unlike ARMA, which requires the data to be stationary, ARIMA includes **integration (d)** so that the model can be fit directly on non-stationary series.
    """)

    # ------------------------
    # FORMULA
    # ------------------------
    st.subheader("🧮 General ARIMA(p,d,q) Formula")

    st.markdown("""
    The ARIMA model works on the **differenced series** ($y'_t$), not the raw series ($y_t$).  

    $$
    y'_t = C + \phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p} + \theta_1 \varepsilon'_{t-1} + \dots + \theta_q \varepsilon'_{t-q} + \varepsilon_t
    $$

    Where:
    - $p$ = order of the autoregressive part  
    - $d$ = number of times the series was differenced (integration order)  
    - $q$ = order of the moving average part  
    """)

    st.info("💡 $y'_t$ is the differenced series. If the data had to be differenced twice to achieve stationarity, then $d=2$.")

    # ------------------------
    # INTEGRATION (d)
    # ------------------------
    st.subheader("🔧 The Role of Integration (d)")

    st.markdown("""
    - A **time series that becomes stationary after differencing** is called an **integrated series**.  
    - The parameter **d** = minimum number of times differencing is required to make the series stationary.  

    Examples:
    - If stationary after one differencing → $d=1$  
    - If stationary after two differencings → $d=2$  
    - Rarely, $d > 2$ is needed in practice.  

    ⚠️ Unlike p and q, d is not a range to optimize. It has a **fixed definition**.
    """)

    st.warning("⚠️ Varying d changes the likelihood function, so comparing ARIMA models with different d using AIC is not valid.")

    # ------------------------
    # WHY ARIMA
    # ------------------------
    st.subheader("📊 Why ARIMA instead of ARMA?")

    st.markdown("""
    - **ARMA(p,q)** requires the input series to be stationary → you must difference first and then reverse-transform the forecasts.  
    - **ARIMA(p,d,q)** integrates differencing directly → you can work with **non-stationary data** and forecasts remain on the **original scale**.  

    👉 In simple terms, ARIMA = ARMA applied to the differenced series.
    """)

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways (ARIMA Model)")

    st.markdown("""
    - 🔄 **ARIMA(p,d,q) = AR(p) + I(d) + MA(q)**  
    - 🧩 Handles **non-stationary series** by applying differencing.  
    - 📉 $d$ = number of times the series must be differenced to become stationary.  
    - 🧮 Works directly on the original series (no manual inverse transforms needed).  
    - ⚠️ $d$ is **fixed by definition**, not tuned via model selection like p and q.  
    """)

    # ------------------------
    # SEASONAL AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (SARIMA)
    # ------------------------
    st.header("❄️ SARIMA Model")

    st.markdown("""
    The **Seasonal Autoregressive Integrated Moving Average (SARIMA)** model extends ARIMA(p,d,q) to capture **seasonal patterns** in time series.  

    It is denoted as **SARIMA(p,d,q)(P,D,Q)m**, where:
    - **p, d, q** → non-seasonal ARIMA parameters  
    - **P, D, Q** → seasonal counterparts (AR, differencing, MA)  
    - **m** → frequency = number of observations per seasonal cycle  

    👉 In other words, SARIMA = ARIMA + Seasonality.
    """)

    # ------------------------
    # FORMULA & PARAMETERS
    # ------------------------
    st.subheader("🧮 Parameters of SARIMA")

    st.markdown("""
    - **p** → non-seasonal autoregressive order  
    - **d** → non-seasonal differencing (integration)  
    - **q** → non-seasonal moving average order  
    - **P** → seasonal autoregressive order  
    - **D** → seasonal differencing (integration)  
    - **Q** → seasonal moving average order  
    - **m** → frequency (observations per seasonal cycle)  

    Examples:
    - If data is **monthly** → $m=12$ (1 cycle = 12 months)  
    - If data is **quarterly** → $m=4$ (1 cycle = 4 quarters)  
    - If data is **weekly** → $m=7$ (1 cycle = 7 days)  
    """)

    st.info("💡 Example: SARIMA(1,1,1)(1,1,1)12 → includes both monthly AR, differencing, and MA components with yearly seasonality.")

    st.markdown("---")

    # ------------------------
    # SEASONAL DIFFERENCING
    # ------------------------
    st.subheader("🔧 Seasonal Differencing")

    st.markdown("""
    To make a series with seasonal patterns stationary, we can apply **seasonal differencing**:

    For monthly data with $m=12$:
    $$
    y'_t = y_t - y_{t-12}
    $$

    - **D=1** → one seasonal differencing applied  
    - **D=2** → two seasonal differencings applied (rare in practice)  
    """)

    st.warning("⚠️ If a series has both trend and seasonality, you may need **non-seasonal differencing (d)** and **seasonal differencing (D)** together.")

    st.markdown("---")

    # ------------------------
    # IDENTIFYING SEASONALITY
    # ------------------------
    st.subheader("🔎 Identifying Seasonality")

    st.markdown("""
    We can identify seasonality with **time series decomposition**, which splits data into:
    - **Trend** → long-term direction  
    - **Seasonal component** → repeating fluctuations (e.g., yearly cycles)  
    - **Residuals (Noise)** → random variation  

    👉 If the seasonal component is **flat (≈0)**, there is no seasonality in the series.  
    """)

    st.info("💡 SARIMA is only meaningful if the data shows clear seasonal patterns.")

    st.markdown("---")

    # ------------------------
    # MODEL FITTING & SELECTION
    # ------------------------
    st.subheader("⚖️ Model Selection")

    st.markdown("""
    - Fit multiple **SARIMA(p,d,q)(P,D,Q)m** models over ranges of parameters.  
    - Use **AIC** (Akaike Information Criterion) to select the best model (lowest AIC).  
    - Validate using **residual analysis**:
    - **Q-Q Plot** → residuals follow normal distribution?  
    - **Ljung-Box Test** → residuals uncorrelated (white noise)?  
    """)

    st.success("✅ If residuals ≈ white noise and AIC is minimized → model is valid for forecasting.")

    st.markdown("---")

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways (SARIMA Model)")

    st.markdown("""
    - ❄️ **SARIMA(p,d,q)(P,D,Q)m** = ARIMA + Seasonality  
    - 📉 Seasonal parameters (P,D,Q) capture dependencies across cycles of length *m*  
    - 🧮 Seasonal differencing removes repeating patterns (e.g., yearly, weekly)  
    - 📊 Time series decomposition helps visualize trend, seasonality, and noise  
    - ⚖️ Model selection done with **AIC** + **residual diagnostics**  
    """)

    # ------------------------
    # SARIMAX MODEL
    # ------------------------
    st.header("🌐 SARIMAX Model")

    st.markdown("""
    The **Seasonal Autoregressive Integrated Moving Average with Exogenous Variables (SARIMAX)** model extends **SARIMA(p,d,q)(P,D,Q)m** by adding **external predictors** (exogenous variables).  

    - **Endogenous variable (target):** the time series we want to forecast.  
    - **Exogenous variables (predictors):** external features that may influence the target (e.g., interest rates, inflation, population, consumption).  
    """)

    # ------------------------
    # FORMULA
    # ------------------------
    st.subheader("🧮 General SARIMAX Expression")

    st.markdown("""
    The SARIMAX model can be written as:

    $$
    y_t = SARIMA(p,d,q)(P,D,Q)_m + \beta_1 X_{1,t} + \beta_2 X_{2,t} + \dots + \beta_k X_{k,t}
    $$

    Where:
    - $y_t$ → target series at time *t*  
    - $X_{i,t}$ → exogenous variable $i$ at time *t*  
    - $\beta_i$ → coefficient for predictor $i$  
    """)

    st.info("💡 SARIMAX remains a **linear model**: it’s a linear combination of past values, past errors, and external predictors.")

    st.markdown("---")

    # ------------------------
    # RELATION TO OTHER MODELS
    # ------------------------
    st.subheader("🔗 Relation to ARIMA / SARIMA")

    st.markdown("""
    - If no seasonal component → **ARIMAX**  
    - If no exogenous variables → **SARIMA**  
    - If no seasonality & no exogenous variables → **ARIMA**  
    """)

    # ------------------------
    # EXOGENOUS VARIABLES
    # ------------------------
    st.subheader("🌍 Exogenous Variables (X)")

    st.markdown("""
    Examples of exogenous variables:
    - Personal & federal consumption expenditures  
    - Interest rates  
    - Inflation rate  
    - Population  
    - Categorical features (must be encoded numerically before use)  

    ⚠️ Transformations (differencing, log, etc.) are applied **only on the target variable**, not on exogenous variables.
    """)

    st.warning("⚠️ Always encode categorical variables (e.g., one-hot, binary flags) before including them in SARIMAX.")

    st.markdown("---")

    # ------------------------
    # MODEL SELECTION & P-VALUES
    # ------------------------
    st.subheader("⚖️ Model Selection & P-Values")

    st.markdown("""
    - **AIC (Akaike Information Criterion)** remains the main tool for model selection.  
    - Statsmodels includes regression outputs (coefficients & p-values), but:  
    - The **p-value** only tests if a coefficient is significantly different from 0.  
    - ❌ It does **not** tell whether a predictor is useful for forecasting.  

    ✅ Trust **AIC** for feature/model selection, not p-values.  
    👉 Reference: [Rob Hyndman – Statistical tests for variable selection](https://robjhyndman.com/hyndsight/tests2/)  
    """)

    st.markdown("---")

    # ------------------------
    # FORECASTING WITH SARIMAX
    # ------------------------
    st.subheader("📊 Forecasting Strategy")

    st.markdown("""
    - SARIMAX forecasts **one timestep ahead reliably**.  
    - To forecast multiple timesteps:
    - You must also **forecast the exogenous variables**.  
    - This may **amplify prediction errors**.  

    👉 Practical approaches:
    - If exogenous variables are predictable (e.g., holiday calendar, known policy rate) → safe to forecast multiple steps.  
    - If exogenous variables are uncertain (e.g., economic shocks, market prices) → safer to forecast **only one step at a time** and update as new data arrives.  
    """)

    st.warning("⚠️ Forecast accuracy degrades quickly if you must also forecast exogenous variables with uncertainty.")

    st.markdown("---")

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways (SARIMAX Model)")

    st.markdown("""
    - 🌐 **SARIMAX = SARIMA + Exogenous Variables (X)**  
    - 🔎 Allows external predictors to influence the forecast  
    - ⚖️ Select model with **AIC**, not with p-values of coefficients  
    - 📉 Forecasting multiple steps requires forecasting the exogenous variables → risk of error accumulation  
    - ⏳ Safest approach: **forecast one timestep ahead**, unless exogenous variables are reliably predictable  
    """)

    # ------------------------
    # VECTOR AUTOREGRESSION (VAR)
    # ------------------------
    st.header("🔗 VAR Model (Vector Autoregression)")

    st.markdown("""
    With **SARIMAX**, the relationship between variables is **unidirectional**: exogenous variables affect the target, but the target does not affect the exogenous variables.  

    However, sometimes time series influence **each other bidirectionally**.  
    👉 Example: GDP may influence interest rates, and interest rates may influence GDP.  

    This motivates the use of the **Vector Autoregression (VAR)** model, which captures interactions between multiple time series and produces **multivariate forecasts**.
    """)

    # ------------------------
    # FORMULA
    # ------------------------
    st.subheader("🧮 General VAR(p) Formula")

    st.markdown("""
    VAR is a generalization of the AR(p) model to multiple time series.  

    For a single series AR(p):
    $$
    y_t = C + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t
    $$

    For two time series $y_{1,t}$ and $y_{2,t}$ in VAR(p):
    $$
    y_{1,t} = c_1 + \phi_{11,1} y_{1,t-1} + \dots + \phi_{11,p} y_{1,t-p} + 
            \phi_{12,1} y_{2,t-1} + \dots + \phi_{12,p} y_{2,t-p} + \varepsilon_{1,t}
    $$

    $$
    y_{2,t} = c_2 + \phi_{21,1} y_{1,t-1} + \dots + \phi_{21,p} y_{1,t-p} + 
            \phi_{22,1} y_{2,t-1} + \dots + \phi_{22,p} y_{2,t-p} + \varepsilon_{2,t}
    $$

    - Each equation includes **lagged values of itself and the other series**.  
    - Requires the series to be **stationary** (may need differencing or transformations).  
    """)

    st.info("💡 VAR(p) is essentially multiple AR(p) equations estimated together, allowing cross-dependencies.")

    st.markdown("---")

    # ------------------------
    # GRANGER CAUSALITY TEST
    # ------------------------
    st.subheader("🔎 Granger Causality Test")

    st.markdown("""
    Before using VAR, we must check if series **predict each other** with the **Granger causality test**.  

    - **H₀ (null):** Series $y_{2,t}$ does **not** Granger-cause $y_{1,t}$.  
    - **H₁ (alt):** Series $y_{2,t}$ Granger-causes $y_{1,t}$.  

    Decision rule:
    - If *p-value < 0.05* → Reject H₀ → $y_{2,t}$ helps predict $y_{1,t}$.  
    - If *p-value ≥ 0.05* → Fail to reject H₀ → no predictive relationship.  

    👉 The test is performed **after selecting the order p** of the VAR model (via AIC).  
    """)

    st.warning("⚠️ If Granger causality fails, a VAR model is invalid → revert to SARIMAX instead.")

    st.markdown("---")

    # ------------------------
    # MODEL SELECTION
    # ------------------------
    st.subheader("⚖️ Model Selection")

    st.markdown("""
    - Fit multiple **VAR(p)** models with different lag orders.  
    - Select the model with the **lowest AIC**.  
    - Perform **residual diagnostics**:
    - Residuals ≈ white noise → model is valid.  
    - If not, adjust lag order or reconsider model.  
    """)

    st.markdown("---")

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways (VAR)")

    st.markdown("""
    - 🔗 VAR(p) captures **bidirectional relationships** between time series.  
    - 📉 Requires **stationarity** (differencing may be needed).  
    - 🧪 Must pass the **Granger causality test** → validates predictive relationships.  
    - ⚖️ Model selection via **AIC**, followed by residual analysis.  
    """)

    # ------------------------
    # VARMAX MENTION
    # ------------------------
    st.subheader("🌐 VARMAX: Extension of VAR")

    st.markdown("""
    The **Vector Autoregression Moving Average with Exogenous Variables (VARMAX)** model extends VAR by:  
    - Adding a **moving average component** (like ARMA).  
    - Allowing inclusion of **exogenous predictors (X)**.  

    👉 VARMAX = VAR + MA terms + exogenous regressors.  
    This is the most general multivariate linear framework for time series forecasting.  
    """)

with tab_deepl:
    st.header("🌐 Deep Learning for Time Series Forecasting")

    st.markdown("""
    Statistical models like ARIMA, SARIMA, SARIMAX, or VAR are powerful tools for time series forecasting.  
    However, **they have important limitations**:
    - They assume *linearity* in relationships between variables.  
    - They degrade in performance when the dataset is **very large** or has **nonlinear dependencies**.  
    - They struggle when there are **multiple seasonalities** (e.g., daily + yearly).  
    - Model selection (AIC, residual analysis) becomes computationally slow when we must fit many variations.  

    👉 In these contexts, **Deep Learning** becomes the natural solution.  

    Deep learning is a subset of machine learning based on **neural networks**.  
    Unlike traditional statistical models, it does not assume linearity, stationarity, or Gaussian distributions. Instead, it **learns patterns directly from the data**, and it tends to perform better the more data we have.  
    """)

    # ------------------------
    # WHEN TO USE DL
    # ------------------------
    st.subheader("📌 When to Use Deep Learning")

    st.markdown("""
    Deep learning is especially recommended when:

    - **Large datasets**: More than **10,000 data points** (although this is not a strict rule — with ~8,000 points it may already be worth testing).  
    - **Complex seasonal patterns**: For example, hourly temperatures often have **daily seasonality** (day vs night) *and* **yearly seasonality** (winter vs summer).  
    - SARIMA can only handle one seasonality well, but DL can capture multiple.  
    - **Nonlinear dependencies**: If features interact in nonlinear ways, residuals of SARIMA/SARIMAX may remain correlated → DL captures such nonlinearities.  
    - **Performance bottlenecks**: Fitting SARIMAX on massive datasets is slow and impractical for model selection. Neural networks scale better.  
    - **Residual diagnostics fail**: When statistical models produce residuals that are *not white noise*, DL can often fix this by learning hidden structures.  
    """)

    st.info("💡 Deep learning models are **data-hungry**, but once enough data is available, they often outperform statistical methods in both accuracy and speed.")

    # ------------------------
    # TYPES OF DL MODELS
    # ------------------------
    st.subheader("🔎 Types of Deep Learning Models for Forecasting")

    st.markdown("""
    There are **three main families of neural models** for time series forecasting:

    ### 1️⃣ Single-Step Models  
    - Predict **one timestep ahead** for **one target variable**.  
    - Output is a scalar $\\hat{y}_{t+1}$.  
    - Example: Predicting tomorrow’s closing stock price from historical prices.  

    Mathematically:
    $$
    \\hat{y}_{t+1} = f(y_t, y_{t-1}, \\dots, y_{t-p})
    $$

    ---

    ### 2️⃣ Multi-Step Models  
    - Predict **multiple timesteps ahead** for **one target variable**.  
    - Output is a sequence $[\\hat{y}_{t+1}, \\hat{y}_{t+2}, \\dots, \\hat{y}_{t+k}]$.  
    - Example: Forecasting the next **24 hours** of electricity demand.  

    Mathematically:
    $$
    [\\hat{y}_{t+1}, \\hat{y}_{t+2}, ..., \\hat{y}_{t+k}] = f(y_t, y_{t-1}, \\dots, y_{t-p})
    $$

    ---

    ### 3️⃣ Multi-Output Models  
    - Predict **multiple variables simultaneously** (possibly for multiple steps).  
    - Output is a matrix with shape `(steps × features)`.  
    - Example: Forecasting **temperature and humidity** for the next 24 hours.  

    Mathematically:
    $$
    [\\hat{y}^1_{t+1}, \\hat{y}^2_{t+1}, ..., \\hat{y}^m_{t+k}] = f(y_t, y_{t-1}, ..., y_{t-p})
    $$
    where $m$ = number of target variables.  

    ---

    Each of these can be implemented using different **neural architectures**:
    - **RNN-based**: LSTMs and GRUs capture long-term dependencies.  
    - **CNN-based**: Temporal CNNs (TCN) capture local sequential patterns.  
    - **Hybrid autoregressive neural networks**: Combine AR structure with DL flexibility.  
    """)

    # ------------------------
    # DL VS STATISTICAL MODELS
    # ------------------------
    st.subheader("⚖️ Deep Learning vs Statistical Models")

    st.markdown("""
    | Aspect | Statistical Models (ARIMA, SARIMA, VAR...) | Deep Learning (LSTM, CNN, DNN...) |
    |--------|-------------------------------------------|-----------------------------------|
    | **Assumptions** | Require stationarity, linearity, Gaussian errors | No strict assumptions, learn directly from data |
    | **Data size** | Efficient for small datasets (< 10k points) | Perform better with large datasets (> 10k points) |
    | **Seasonality** | Handle one seasonal period well (SARIMA) | Can capture multiple seasonalities simultaneously |
    | **Nonlinearity** | Limited (linear structures only) | Naturally capture nonlinear dependencies |
    | **Model selection** | AIC, PACF/ACF, manual tuning | Hyperparameter optimization, deep architectures |
    | **Interpretability** | Easier to interpret (coefficients, lags) | Harder to interpret (black-box models) |
    | **Scalability** | Slows down as dataset grows | Scales better with large high-dimensional data |
    """)

    # ------------------------
    # KEY TAKEAWAYS
    # ------------------------
    st.subheader("📌 Key Takeaways")

    st.markdown("""
    - Statistical models are excellent for **small, linear, stationary datasets**.  
    - Deep learning is better for:
    - Large datasets (>10,000 points).  
    - Multiple seasonalities.  
    - Nonlinear relationships.  
    - Cases where residuals of statistical models remain correlated.  

    - **Types of DL forecasting models**:
    - Single-step → one target, one horizon.  
    - Multi-step → one target, multiple horizons.  
    - Multi-output → multiple targets, multiple horizons.  

    - Deep learning models are **fast to train with GPUs** and scale well with big data.  
    - However, they require more expertise for tuning and are often less interpretable than statistical models.  
    """)
