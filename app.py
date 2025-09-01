import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="ARIMA Forecasting", layout="centered")

# Custom CSS for minimal design
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("ARIMA Forecasting")
st.caption("Stock market time series analysis")

# Sidebar navigation
steps = ["Upload", "Visualize", "Test", "Difference", "Parameters", "Model", "Forecast"]
step = st.sidebar.selectbox("Step", steps)

# Session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'df_diff' not in st.session_state:
    st.session_state.df_diff = None
if 'model' not in st.session_state:
    st.session_state.model = None

def adf_test(data):
    result = adfuller(data.dropna())
    p_value = result[1]
    is_stationary = p_value <= 0.05
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("P-Value", f"{p_value:.4f}")
    with col2:
        status = "Stationary" if is_stationary else "Non-stationary"
        st.metric("Status", status)
    
    return is_stationary

# Step 1: Upload
if step == "Upload":
    st.subheader("Data Upload")
    
    file = st.file_uploader("CSV file", type=['csv'])
    
    if file:
        df = pd.read_csv(file)
        st.write("Data preview:")
        st.dataframe(df.head(), use_container_width=True)
        
        date_col = st.selectbox("Date column", df.columns)
        target_col = st.selectbox("Target column", [c for c in df.columns if c != date_col])
        
        if st.button("Process"):
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            st.session_state.data = df
            st.session_state.target_col = target_col
            st.success("Data processed")

# Step 2: Visualize
elif step == "Visualize":
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        st.subheader("Time Series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target], mode='lines', name=target))
        fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{df[target].mean():.2f}")
        with col2:
            st.metric("Std", f"{df[target].std():.2f}")
        with col3:
            st.metric("Points", len(df))
    else:
        st.warning("Upload data first")

# Step 3: Test
elif step == "Test":
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        st.subheader("Stationarity Test")
        adf_test(df[target])
    else:
        st.warning("Upload data first")

# Step 4: Difference
elif step == "Difference":
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        st.subheader("Differencing")
        order = st.selectbox("Order", [1, 2])
        
        if st.button("Apply"):
            df_diff = df[target].diff(order).dropna()
            st.session_state.df_diff = df_diff
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_diff.index, y=df_diff, mode='lines', name='Differenced'))
            fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Test again
            adf_test(df_diff)
    else:
        st.warning("Upload data first")

# Step 5: Parameters
elif step == "Parameters":
    if st.session_state.df_diff is not None:
        st.subheader("ACF/PACF Analysis")
        
        df_diff = st.session_state.df_diff
        lags = st.slider("Lags", 10, 50, 20)
        
        # Plot ACF/PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        sm.graphics.tsa.plot_acf(df_diff, lags=lags, ax=ax1, title="ACF")
        sm.graphics.tsa.plot_pacf(df_diff, lags=lags, ax=ax2, title="PACF")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Parameter selection
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR)", 0, 5, 1)
        with col2:
            d = st.number_input("d (I)", 0, 2, 1)
        with col3:
            q = st.number_input("q (MA)", 0, 5, 1)
        
        st.session_state.params = (p, d, q)
    else:
        st.warning("Complete differencing first")

# Step 6: Model
elif step == "Model":
    if st.session_state.data is not None and hasattr(st.session_state, 'params'):
        df = st.session_state.data
        target = st.session_state.target_col
        p, d, q = st.session_state.params
        
        st.subheader(f"ARIMA({p},{d},{q})")
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                model = ARIMA(df[target], order=(p, d, q))
                fitted = model.fit()
                st.session_state.model = fitted
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AIC", f"{fitted.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{fitted.bic:.2f}")
                
                st.success("Model trained")
    else:
        st.warning("Complete previous steps")

# Step 7: Forecast
elif step == "Forecast":
    if st.session_state.model is not None and st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        model = st.session_state.model
        
        st.subheader("Forecast")
        
        steps = st.number_input("Forecast steps", 1, 100, 30)
        
        if st.button("Generate"):
            # Forecast
            forecast = model.forecast(steps=steps)
            
            # Plot
            fig = go.Figure()
            
            # Historical (last 100 points)
            recent = df[target].tail(100)
            fig.add_trace(go.Scatter(
                x=recent.index, y=recent, 
                mode='lines', name='Historical', 
                line=dict(color='blue')
            ))
            
            # Forecast
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq='D')
            fig.add_trace(go.Scatter(
                x=future_dates, y=forecast, 
                mode='lines', name='Forecast', 
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast.round(2)
            })
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.warning("Train model first")