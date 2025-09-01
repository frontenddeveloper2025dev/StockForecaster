import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="ARIMA Forecasting", layout="centered")

# Title
st.title("ARIMA Forecasting")
st.write("Análisis de series temporales para mercado bursátil")

# Sidebar
step = st.sidebar.selectbox("Paso", ["Upload", "Visualize", "Test", "Difference", "Model", "Forecast"])

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
    """Simple ADF test"""
    try:
        result = adfuller(data.dropna())
        p_value = result[1]
        is_stationary = p_value <= 0.05
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P-Value", f"{p_value:.4f}")
        with col2:
            status = "Estacionario" if is_stationary else "No estacionario"
            st.metric("Estado", status)
        
        return is_stationary
    except Exception as e:
        st.error(f"Error en test: {e}")
        return False

def auto_arima_tuning(data, max_p=3, max_d=2, max_q=3):
    """Automatic ARIMA hyperparameter tuning using grid search"""
    best_aic = float('inf')
    best_params = None
    best_model = None
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_combination = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                current_combination += 1
                progress = current_combination / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing ARIMA({p},{d},{q})... ({current_combination}/{total_combinations})")
                
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    aic = fitted.aic
                    bic = fitted.bic
                    
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'AIC': aic, 'BIC': bic
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted
                        
                except Exception:
                    # Skip invalid parameter combinations
                    continue
    
    progress_bar.empty()
    status_text.empty()
    
    return best_model, best_params, results

# Step 1: Upload
if step == "Upload":
    st.header("1. Cargar Datos")
    
    file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
    
    if file:
        try:
            df = pd.read_csv(file)
            st.success("Archivo cargado correctamente")
            st.dataframe(df.head())
            
            date_col = st.selectbox("Columna de fecha", df.columns)
            target_col = st.selectbox("Columna objetivo", [c for c in df.columns if c != date_col])
            
            if st.button("Procesar datos"):
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                st.session_state.data = df
                st.session_state.target_col = target_col
                st.success("Datos procesados correctamente")
        except Exception as e:
            st.error(f"Error: {e}")

# Step 2: Visualize
elif step == "Visualize":
    st.header("2. Visualizar")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        st.subheader("Serie de tiempo")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target], mode='lines', name=target))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media", f"{df[target].mean():.2f}")
        with col2:
            st.metric("Desv. Std", f"{df[target].std():.2f}")
        with col3:
            st.metric("Puntos", len(df))
    else:
        st.warning("Carga los datos primero")

# Step 3: Test
elif step == "Test":
    st.header("3. Test de Estacionariedad")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        adf_test(df[target])
    else:
        st.warning("Carga los datos primero")

# Step 4: Difference
elif step == "Difference":
    st.header("4. Diferenciación")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        order = st.selectbox("Orden de diferenciación", [1, 2])
        
        if st.button("Aplicar diferenciación"):
            try:
                df_diff = df[target].diff(order).dropna()
                st.session_state.df_diff = df_diff
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_diff.index, y=df_diff, mode='lines'))
                fig.update_layout(height=300, title="Serie diferenciada")
                st.plotly_chart(fig, use_container_width=True)
                
                adf_test(df_diff)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Carga los datos primero")

# Step 5: Model
elif step == "Model":
    st.header("5. Modelo ARIMA")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        # Parameter selection method
        param_method = st.radio("Selección de parámetros:", ["Manual", "Automática"])
        
        if param_method == "Manual":
            st.subheader("Parámetros manuales")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR)", 0, 5, 1)
            with col2:
                d = st.number_input("d (I)", 0, 2, 1)
            with col3:
                q = st.number_input("q (MA)", 0, 5, 1)
            
            if st.button("Entrenar modelo manual"):
                try:
                    with st.spinner("Entrenando..."):
                        model = ARIMA(df[target], order=(p, d, q))
                        fitted = model.fit()
                        st.session_state.model = fitted
                        st.session_state.params = (p, d, q)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("AIC", f"{fitted.aic:.2f}")
                        with col2:
                            st.metric("BIC", f"{fitted.bic:.2f}")
                        
                        st.success(f"Modelo ARIMA({p},{d},{q}) entrenado")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:  # Automatic
            st.subheader("Búsqueda automática de parámetros")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                max_p = st.number_input("Máximo p", 1, 5, 3)
            with col2:
                max_d = st.number_input("Máximo d", 1, 2, 2)
            with col3:
                max_q = st.number_input("Máximo q", 1, 5, 3)
            
            if st.button("Búsqueda automática"):
                try:
                    st.info("Iniciando búsqueda automática de parámetros...")
                    best_model, best_params, results = auto_arima_tuning(df[target], max_p, max_d, max_q)
                    
                    if best_model is not None:
                        st.session_state.model = best_model
                        st.session_state.params = best_params
                        
                        # Show best results
                        st.success(f"Mejor modelo encontrado: ARIMA{best_params}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("AIC", f"{best_model.aic:.2f}")
                        with col2:
                            st.metric("BIC", f"{best_model.bic:.2f}")
                        
                        # Show top 5 results
                        if results:
                            st.subheader("Top 5 mejores modelos")
                            results_df = pd.DataFrame(results)
                            results_df = results_df.sort_values('AIC').head(5)
                            st.dataframe(results_df, use_container_width=True)
                    else:
                        st.error("No se pudo encontrar un modelo válido")
                        
                except Exception as e:
                    st.error(f"Error en búsqueda automática: {e}")
    else:
        st.warning("Carga los datos primero")

# Step 6: Forecast
elif step == "Forecast":
    st.header("6. Pronóstico")
    
    if st.session_state.model is not None and st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        model = st.session_state.model
        
        steps = st.number_input("Pasos a pronosticar", 1, 100, 30)
        
        if st.button("Generar pronóstico"):
            try:
                forecast = model.forecast(steps=steps)
                
                fig = go.Figure()
                
                # Historical
                recent = df[target].tail(100)
                fig.add_trace(go.Scatter(
                    x=recent.index, y=recent, 
                    mode='lines', name='Histórico', 
                    line=dict(color='blue')
                ))
                
                # Forecast
                last_date = df.index[-1]
                future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq='D')
                fig.add_trace(go.Scatter(
                    x=future_dates, y=forecast, 
                    mode='lines', name='Pronóstico', 
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(height=400, title="Pronóstico ARIMA")
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                forecast_df = pd.DataFrame({
                    'Fecha': future_dates,
                    'Pronóstico': forecast.round(2)
                })
                st.dataframe(forecast_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Entrena el modelo primero")

# Show current step info
st.sidebar.markdown("---")
st.sidebar.write(f"**Paso actual:** {step}")
if st.session_state.data is not None:
    st.sidebar.success("✓ Datos cargados")
if st.session_state.model is not None:
    st.sidebar.success("✓ Modelo entrenado")