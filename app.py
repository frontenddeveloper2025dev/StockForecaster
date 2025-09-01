import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import os
import requests
import warnings
import json
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
warnings.filterwarnings('ignore')

# Advanced model imports (with fallbacks)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# Page config with custom favicon
st.set_page_config(
    page_title="ARIMA Financial AI Dashboard", 
    layout="wide", 
    page_icon="logo.png",
    initial_sidebar_state="expanded"
)

# Initialize session state for advanced features
if 'theme' not in st.session_state:
    st.session_state.theme = 'arima_orange'
if 'user_level' not in st.session_state:
    st.session_state.user_level = 1
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'symbol_portfolio' not in st.session_state:
    st.session_state.symbol_portfolio = []
if 'mood_tracker' not in st.session_state:
    st.session_state.mood_tracker = {}

# Dashboard Theme Selection
themes = {
    'arima_orange': {
        'primary': '#FF6B35',
        'secondary': '#FF8C42', 
        'background': '#FFFFFF',
        'text': '#2C2C2C',
        'accent': '#FFA726',
        'gradient_start': '#FF6B35',
        'gradient_end': '#FF8C42'
    },
    'default': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'background': '#ffffff',
        'text': '#000000'
    },
    'dark': {
        'primary': '#00d4aa',
        'secondary': '#ff6b6b',
        'background': '#1e1e1e',
        'text': '#ffffff'
    },
    'ocean': {
        'primary': '#0077be',
        'secondary': '#00a8cc',
        'background': '#f0f8ff',
        'text': '#003366'
    },
    'sunset': {
        'primary': '#ff6b6b',
        'secondary': '#ffa726',
        'background': '#fff3e0',
        'text': '#d84315'
    },
    'forest': {
        'primary': '#4caf50',
        'secondary': '#66bb6a',
        'background': '#f1f8e9',
        'text': '#2e7d32'
    }
}

# Sidebar theme selector
with st.sidebar:
    st.subheader("🎨 Personalización")
    theme_names = {
        'arima_orange': '🔥 ARIMA Orange',
        'default': '🎨 Clásico',
        'dark': '🌙 Oscuro', 
        'ocean': '🌊 Océano',
        'sunset': '🌅 Atardecer',
        'forest': '🌲 Bosque'
    }
    
    selected_theme = st.selectbox(
        "Tema del Dashboard:",
        options=list(themes.keys()),
        format_func=lambda x: theme_names.get(x, x.title()),
        index=list(themes.keys()).index(st.session_state.theme)
    )
    st.session_state.theme = selected_theme
    
    # Apply custom CSS based on theme
    theme_colors = themes[st.session_state.theme]
    st.markdown(f"""
    <style>
        .main .block-container {{
            background-color: {theme_colors['background']};
            color: {theme_colors['text']};
        }}
        .stMetric {{
            background-color: {theme_colors['background']};
            color: {theme_colors['text']};
            border-left: 4px solid {theme_colors['primary']};
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .css-1d391kg {{
            background: linear-gradient(135deg, {theme_colors['primary']}, {theme_colors['secondary']});
        }}
        .sidebar .sidebar-content {{
            background: linear-gradient(180deg, {theme_colors['background']}, #f8f9fa);
        }}
    </style>
    """, unsafe_allow_html=True)

# Title with theme styling and logo integration
st.markdown(f"""
<div style="text-align: center; padding: 25px; background: linear-gradient(135deg, {theme_colors['primary']}, {theme_colors['secondary']}); border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
        <div style="background: white; border-radius: 50%; padding: 8px; margin-right: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
            <span style="font-size: 24px;">📊</span>
        </div>
        <h1 style="color: white; margin: 0; font-family: 'Arial Black', Arial, sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            ARIMA Financial AI Dashboard
        </h1>
    </div>
    <p style="color: white; margin: 0; font-size: 16px; opacity: 0.95;">
        🔮 Forecasting Avanzado • 📈 Análisis Inteligente • 🎯 Insights en Tiempo Real
    </p>
    <div style="margin-top: 15px;">
        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; font-size: 12px;">
            Powered by ARIMA, Prophet & LSTM
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar Navigation
st.sidebar.title("🧭 Navegación")
step = st.sidebar.selectbox("Sección Principal:", [
    "📊 Multi-Symbol Dashboard", 
    "📈 Upload & Data", 
    "👁️ Visualize", 
    "🔍 Stationarity Test", 
    "📐 Difference", 
    "🤖 Advanced Models", 
    "🔮 Forecasting Lab",
    "🏆 Performance Tracker",
    "🎯 Mood & Sentiment",
    "🗺️ Symbol Heat Map",
    "🎓 AI Learning Mode"
])

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

def auto_arima_tuning(data, max_p=5, max_d=2, max_q=5, use_stepwise=False, enable_seasonal=False):
    """🚀 Ultra-Advanced ARIMA hyperparameter tuning with intelligent optimization"""
    best_aic = float('inf')
    best_bic = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Data characteristics analysis for intelligent parameter suggestions
    data_length = len(data)
    data_std = np.std(data)
    data_mean = np.mean(data)
    
    # Intelligent parameter range adjustment based on data characteristics
    if data_length < 100:
        max_p = min(max_p, 3)
        max_q = min(max_q, 3)
        st.info("🧠 **Optimización inteligente:** Reduciendo rangos de parámetros para datos pequeños")
    elif data_length > 1000:
        st.info("🧠 **Optimización inteligente:** Datos grandes detectados - usando búsqueda eficiente")
    
    # Progress tracking with enhanced metrics
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_combination = 0
    successful_fits = 0
    early_stop_threshold = 10  # Stop if we find 10 good models to save time
    
    # Enhanced live metrics display
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tested_metric = st.empty()
        with col2:
            successful_metric = st.empty()
        with col3:
            best_aic_metric = st.empty()
        with col4:
            efficiency_metric = st.empty()
    
    st.info("🎯 **Búsqueda inteligente:** Priorizando combinaciones más prometedoras")
    
    # Enhanced grid search with intelligent ordering
    param_combinations = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                # Skip unrealistic combinations
                if p + q > 6:  # Avoid overly complex models
                    continue
                if d > 2 and (p > 3 or q > 3):  # High differencing with high AR/MA
                    continue
                param_combinations.append((p, d, q))
    
    # Sort combinations by expected performance (simpler models first)
    param_combinations.sort(key=lambda x: x[0] + x[2] + x[1] * 2)
    total_combinations = len(param_combinations)
    
    # Grid search with enhanced tracking and early stopping
    for p, d, q in param_combinations:
        current_combination += 1
        progress = current_combination / total_combinations
        progress_bar.progress(progress)
        
        # Update status with enhanced information
        status_text.text(f"🔍 Testing ARIMA({p},{d},{q}) - {current_combination}/{total_combinations}")
        tested_metric.metric("Probados", f"{current_combination}/{total_combinations}")
        
        # Calculate efficiency percentage
        efficiency = (successful_fits / current_combination * 100) if current_combination > 0 else 0
        efficiency_metric.metric("Eficiencia", f"{efficiency:.1f}%")
        
        try:
            # Fit ARIMA model with enhanced error handling
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()
            aic = fitted.aic
            bic = fitted.bic
            
            # Calculate comprehensive metrics
            residuals = fitted.resid
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # Model diagnostics
            ljung_box_pvalue = None
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                ljung_result = acorr_ljungbox(residuals, lags=10, return_df=True)
                ljung_box_pvalue = ljung_result['lb_pvalue'].min()
            except:
                ljung_box_pvalue = None
            
            # Calculate information criteria
            hqic = fitted.hqic if hasattr(fitted, 'hqic') else None
            
            successful_fits += 1
            successful_metric.metric("Exitosos", successful_fits)
            
            # Enhanced results with comprehensive metrics
            result_entry = {
                'p': p, 'd': d, 'q': q,
                'AIC': round(aic, 2), 
                'BIC': round(bic, 2),
                'HQIC': round(hqic, 2) if hqic else None,
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'LogLikelihood': round(fitted.llf, 2),
                'LjungBox_p': round(ljung_box_pvalue, 3) if ljung_box_pvalue else None
            }
            results.append(result_entry)
            
            # Update best model based on AIC with model validation
            if aic < best_aic:
                # Additional validation for best model
                model_is_valid = True
                
                # Check for reasonable residuals
                if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                    model_is_valid = False
                
                # Check for overfitting (too many parameters for data size)
                total_params = p + d + q
                if total_params > data_length // 10:
                    model_is_valid = False
                
                if model_is_valid:
                    best_aic = aic
                    best_params = (p, d, q)
                    best_model = fitted
                    best_aic_metric.metric("Mejor AIC", f"{aic:.2f}")
            
            # Early stopping for efficiency (if we have enough good models)
            if successful_fits >= early_stop_threshold and len([r for r in results if r['AIC'] < best_aic + 50]) >= 5:
                st.info(f"⚡ **Parada temprana:** Encontrados {successful_fits} modelos exitosos")
                break
                        
        except Exception as e:
            # Enhanced error logging for debugging
            continue
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Enhanced final summary with intelligence insights
    if successful_fits > 0:
        efficiency_rate = (successful_fits / current_combination) * 100
        st.success(f"✅ **Tuning completado!** {successful_fits}/{current_combination} modelos válidos ({efficiency_rate:.1f}% eficiencia)")
        
        # Provide intelligent insights
        if efficiency_rate > 80:
            st.info("🧠 **Análisis:** Excelente tasa de éxito - los datos son apropiados para ARIMA")
        elif efficiency_rate > 50:
            st.info("🧠 **Análisis:** Buena tasa de éxito - considera ajustar preprocesamiento")
        else:
            st.warning("⚠️ **Análisis:** Baja tasa de éxito - verifica estacionariedad de los datos")
            
        # Best model insights
        if best_params:
            p, d, q = best_params
            if p > q:
                st.info("📊 **Insight:** Modelo autoregresivo dominante - tendencias fuertes detectadas")
            elif q > p:
                st.info("📊 **Insight:** Modelo de media móvil dominante - choques externos significativos")
            else:
                st.info("📊 **Insight:** Modelo balanceado - patrones complejos detectados")
                
    else:
        st.error("❌ No se encontraron modelos ARIMA válidos. Intenta rangos diferentes o verifica los datos.")
    
    return best_model, best_params, results

# New Section: Multi-Symbol Dashboard
if step == "📊 Multi-Symbol Dashboard":
    st.header("📊 Multi-Symbol Comparison Dashboard")
    
    # Sidebar portfolio management
    with st.sidebar:
        st.subheader("💼 Portfolio Manager")
        new_symbol = st.text_input("Agregar símbolo:", placeholder="AAPL")
        if st.button("➕ Agregar") and new_symbol:
            if new_symbol.upper() not in st.session_state.symbol_portfolio:
                st.session_state.symbol_portfolio.append(new_symbol.upper())
                st.success(f"✅ {new_symbol.upper()} agregado")
            else:
                st.warning("Ya está en el portfolio")
        
        # Display current portfolio
        if st.session_state.symbol_portfolio:
            st.write("**Portfolio Actual:**")
            for i, symbol in enumerate(st.session_state.symbol_portfolio):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📈 {symbol}")
                with col2:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.symbol_portfolio.remove(symbol)
                        st.rerun()
    
    if not st.session_state.symbol_portfolio:
        st.info("🔼 Agrega símbolos a tu portfolio para comenzar el análisis comparativo")
        
        # Suggested portfolios
        st.subheader("🎯 Portfolios Sugeridos")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Tech Giants"):
                st.session_state.symbol_portfolio = ["AAPL", "GOOGL", "MSFT", "META", "AMZN"]
                st.rerun()
        
        with col2:
            if st.button("💰 Finance Sector"):
                st.session_state.symbol_portfolio = ["JPM", "BAC", "WFC", "GS", "V"]
                st.rerun()
        
        with col3:
            if st.button("⚡ High Growth"):
                st.session_state.symbol_portfolio = ["TSLA", "NVDA", "AMD", "NFLX", "CRM"]
                st.rerun()
    
    else:
        # Multi-symbol analysis
        st.subheader(f"📈 Análisis de {len(st.session_state.symbol_portfolio)} Símbolos")
        
        # Time period selector
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Período:", ["1M", "3M", "6M", "1Y", "2Y"], index=2)
        with col2:
            comparison_metric = st.selectbox("Métrica:", ["Close", "Returns", "Volatility"])
        
        if st.button("🔄 Actualizar Análisis"):
            progress_bar = st.progress(0)
            symbol_data = {}
            
            for i, symbol in enumerate(st.session_state.symbol_portfolio):
                try:
                    with st.spinner(f"Descargando {symbol}..."):
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period=period.lower())
                        
                        if not data.empty:
                            symbol_data[symbol] = data
                        
                        progress_bar.progress((i + 1) / len(st.session_state.symbol_portfolio))
                        
                except Exception as e:
                    st.error(f"Error con {symbol}: {e}")
            
            progress_bar.empty()
            
            if symbol_data:
                # Comparative visualization
                st.subheader("📊 Comparación Visual")
                
                fig = go.Figure()
                
                for symbol, data in symbol_data.items():
                    if comparison_metric == "Close":
                        y_data = data['Close']
                        title = "Precio de Cierre"
                    elif comparison_metric == "Returns":
                        y_data = data['Close'].pct_change().cumsum() * 100
                        title = "Retornos Acumulados (%)"
                    else:  # Volatility
                        y_data = data['Close'].pct_change().rolling(20).std() * 100
                        title = "Volatilidad (20 días) (%)"
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=y_data,
                        mode='lines',
                        name=symbol,
                        hovertemplate=f'<b>{symbol}</b><br>%{{x}}<br>{comparison_metric}: %{{y}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"{title} - Comparación Multi-Símbolo",
                    xaxis_title="Fecha",
                    yaxis_title=title,
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics table
                st.subheader("📋 Métricas de Performance")
                
                metrics_data = []
                for symbol, data in symbol_data.items():
                    current_price = data['Close'].iloc[-1]
                    start_price = data['Close'].iloc[0]
                    total_return = ((current_price - start_price) / start_price) * 100
                    volatility = data['Close'].pct_change().std() * 100 * np.sqrt(252)  # Annualized
                    max_price = data['Close'].max()
                    min_price = data['Close'].min()
                    
                    metrics_data.append({
                        'Símbolo': symbol,
                        'Precio Actual': f"${current_price:.2f}",
                        'Retorno Total': f"{total_return:.2f}%",
                        'Volatilidad': f"{volatility:.2f}%",
                        'Máximo': f"${max_price:.2f}",
                        'Mínimo': f"${min_price:.2f}",
                        'Rendimiento': "🟢" if total_return > 0 else "🔴"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Top performers
                st.subheader("🏆 Top Performers")
                metrics_df['Return_num'] = metrics_df['Retorno Total'].str.replace('%', '').astype(float)
                top_performer = metrics_df.loc[metrics_df['Return_num'].idxmax()]
                worst_performer = metrics_df.loc[metrics_df['Return_num'].idxmin()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🥇 **Mejor:** {top_performer['Símbolo']} ({top_performer['Retorno Total']})")
                with col2:
                    st.error(f"📉 **Peor:** {worst_performer['Símbolo']} ({worst_performer['Retorno Total']})")

# Upload Section (renamed)
elif step == "📈 Upload & Data":
    st.header("1. Cargar Datos")
    
    # Data source selection
    data_source = st.radio("Fuente de datos:", ["Archivo CSV", "Yahoo Finance API", "Alpha Vantage API", "Marketstack API", "Finage API", "Finnhub API", "Tiingo API"])
    
    if data_source == "Archivo CSV":
        st.subheader("📁 Subir archivo CSV")
        file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        
        if file:
            try:
                df = pd.read_csv(file)
                st.success("Archivo cargado correctamente")
                st.dataframe(df.head())
                
                date_col = st.selectbox("Columna de fecha", df.columns)
                target_col = st.selectbox("Columna objetivo", [c for c in df.columns if c != date_col])
                
                if st.button("Procesar datos CSV"):
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col).sort_index()
                    st.session_state.data = df
                    st.session_state.target_col = target_col
                    st.success("Datos procesados correctamente")
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif data_source == "Yahoo Finance API":
        st.subheader("📈 Yahoo Finance API")
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Símbolo de acción", value="AAPL", help="Ej: AAPL, GOOGL, TSLA, MSFT")
        with col2:
            period = st.selectbox("Período", ["1y", "2y", "5y", "10y", "max"], index=2)
        
        # Popular symbols for quick selection
        st.subheader("📊 Símbolos populares")
        popular_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX"]
        selected_popular = st.selectbox("O selecciona uno popular:", [""] + popular_symbols)
        
        if selected_popular:
            symbol = selected_popular
        
        if st.button("Descargar datos"):
            try:
                with st.spinner(f"Descargando datos de {symbol}..."):
                    # Download data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period)
                    
                    if df.empty:
                        st.error(f"No se encontraron datos para {symbol}")
                    else:
                        st.success(f"Datos de {symbol} descargados correctamente")
                        
                        # Show info about the stock
                        try:
                            info = ticker.info
                            company_name = info.get('longName', symbol)
                            st.info(f"📊 **{company_name}** ({symbol})")
                        except:
                            st.info(f"📊 **{symbol}**")
                        
                        # Show data preview
                        st.dataframe(df.head())
                        
                        # Select target column
                        target_col = st.selectbox("Columna a analizar:", 
                                                ["Close", "Open", "High", "Low", "Volume"], 
                                                index=0)
                        
                        if st.button("Procesar datos API"):
                            # Process the data
                            st.session_state.data = df
                            st.session_state.target_col = target_col
                            st.session_state.symbol = symbol
                            st.success(f"Datos de {symbol} procesados correctamente")
                            
                            # Show basic stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Registros", len(df))
                            with col2:
                                try:
                                    start_date = df.index.min().strftime("%Y-%m-%d")
                                except:
                                    start_date = str(df.index.min())[:10]
                                st.metric("Desde", start_date)
                            with col3:
                                try:
                                    end_date = df.index.max().strftime("%Y-%m-%d")
                                except:
                                    end_date = str(df.index.max())[:10]
                                st.metric("Hasta", end_date)
                            
            except Exception as e:
                st.error(f"Error descargando datos: {e}")
                st.info("Verifica que el símbolo sea correcto (ej: AAPL para Apple)")
    
    elif data_source == "Alpha Vantage API":
        st.subheader("💎 Alpha Vantage API")
        
        # Check if API key is available
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            st.warning("⚠️ Alpha Vantage API key no encontrada. Por favor configura tu API key.")
            st.info("Puedes obtener tu API key gratis en: https://www.alphavantage.co/support/#api-key")
        else:
            col1, col2 = st.columns(2)
            with col1:
                symbol = st.text_input("Símbolo", value="IBM", help="Ej: IBM, AAPL, TSCO.LON")
            with col2:
                output_size = st.selectbox("Tamaño de datos", ["compact", "full"], 
                                         help="compact: últimos 100 días, full: 20+ años")
            
            # Global market examples
            st.subheader("🌍 Mercados globales soportados")
            global_examples = {
                "Estados Unidos": ["IBM", "AAPL", "GOOGL"],
                "Reino Unido": ["TSCO.LON", "BP.LON"],
                "Canadá": ["SHOP.TRT", "GPV.TRV"],
                "Alemania": ["MBG.DEX"],
                "India": ["RELIANCE.BSE"],
                "China": ["600104.SHH", "000002.SHZ"]
            }
            
            selected_market = st.selectbox("Mercado:", list(global_examples.keys()))
            selected_global = st.selectbox("O selecciona un símbolo:", [""] + global_examples[selected_market])
            
            if selected_global:
                symbol = selected_global
            
            # Data type selection
            st.subheader("📊 Tipo de datos")
            data_function = st.selectbox("Función:", [
                "TIME_SERIES_DAILY",
                "TIME_SERIES_WEEKLY", 
                "TIME_SERIES_MONTHLY",
                "TIME_SERIES_INTRADAY"
            ], help="Daily es recomendado para ARIMA")
            
            interval = None
            if data_function == "TIME_SERIES_INTRADAY":
                interval = st.selectbox("Intervalo:", ["1min", "5min", "15min", "30min", "60min"])
            
            if st.button("Descargar desde Alpha Vantage"):
                try:
                    with st.spinner(f"Descargando {symbol} desde Alpha Vantage..."):
                        ts = TimeSeries(key=api_key, output_format='pandas')
                        
                        if data_function == "TIME_SERIES_DAILY":
                            data, meta_data = ts.get_daily(symbol=symbol, outputsize=output_size)
                        elif data_function == "TIME_SERIES_WEEKLY":
                            data, meta_data = ts.get_weekly(symbol=symbol)
                        elif data_function == "TIME_SERIES_MONTHLY":
                            data, meta_data = ts.get_monthly(symbol=symbol)
                        elif data_function == "TIME_SERIES_INTRADAY" and interval:
                            data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=output_size)
                        else:
                            st.error("Función no soportada o intervalo no especificado")
                            st.stop()
                        
                        if data is None or data.empty:
                            st.error(f"No se encontraron datos para {symbol}")
                        else:
                            st.success(f"Datos de {symbol} descargados desde Alpha Vantage")
                            
                            # Show meta data
                            st.info(f"📊 **{meta_data.get('2. Symbol', symbol)}** - {meta_data.get('1. Information', 'Alpha Vantage Data')}")
                            
                            # Clean column names (remove numbers and dots)
                            data.columns = [col.split('. ')[1] if '. ' in col else col for col in data.columns]
                            
                            # Sort by date (newest first in Alpha Vantage, we want oldest first)
                            data = data.sort_index()
                            
                            # Show data preview
                            st.dataframe(data.head())
                            
                            # Select target column
                            available_cols = list(data.columns)
                            default_col = "close" if "close" in available_cols else available_cols[0]
                            target_col = st.selectbox("Columna a analizar:", available_cols, 
                                                    index=available_cols.index(default_col) if default_col in available_cols else 0)
                            
                            if st.button("Procesar datos Alpha Vantage"):
                                # Process the data
                                st.session_state.data = data
                                st.session_state.target_col = target_col
                                st.session_state.symbol = symbol
                                st.session_state.data_source = "Alpha Vantage"
                                st.success(f"Datos de {symbol} procesados desde Alpha Vantage")
                                
                                # Show basic stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Registros", len(data))
                                with col2:
                                    try:
                                        start_date = data.index.min().strftime("%Y-%m-%d")
                                    except:
                                        start_date = str(data.index.min())[:10]
                                    st.metric("Desde", start_date)
                                with col3:
                                    try:
                                        end_date = data.index.max().strftime("%Y-%m-%d")
                                    except:
                                        end_date = str(data.index.max())[:10]
                                    st.metric("Hasta", end_date)
                                    
                except Exception as e:
                    st.error(f"Error descargando desde Alpha Vantage: {e}")
                    st.info("Verifica que el símbolo sea correcto y que tengas acceso a Alpha Vantage")
    
    elif data_source == "Marketstack API":
        st.subheader("🌐 Marketstack API")
        
        # Use the provided API key
        api_key = "9a62121aa780cdf172837957b1fcc708"
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Símbolo", value="AAPL", help="Ej: AAPL, GOOGL, MSFT, TSLA")
        with col2:
            limit = st.selectbox("Registros", [100, 250, 500, 1000], index=2, 
                               help="Cantidad de datos históricos")
        
        # Exchange selection
        st.subheader("🏦 Mercados disponibles")
        exchanges = {
            "NASDAQ": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "NYSE": ["JPM", "BAC", "WMT", "V", "JNJ"],
            "LSE": ["BP", "SHELL", "VODAFONE", "HSBA"],
            "TSX": ["SHOP", "CNR", "RY"],
            "XETRA": ["SAP", "SIE", "ALV"]
        }
        
        selected_exchange = st.selectbox("Mercado:", list(exchanges.keys()))
        selected_symbol = st.selectbox("O selecciona un símbolo:", [""] + exchanges[selected_exchange])
        
        if selected_symbol:
            symbol = selected_symbol
        
        if st.button("Descargar desde Marketstack"):
            try:
                with st.spinner(f"Descargando {symbol} desde Marketstack..."):
                    # Marketstack API endpoint
                    url = f"http://api.marketstack.com/v1/eod"
                    params = {
                        'access_key': api_key,
                        'symbols': symbol,
                        'limit': limit,
                        'sort': 'ASC'  # Ascending order (oldest first)
                    }
                    
                    # Make API request
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'data' in json_data and json_data['data']:
                            # Convert to DataFrame
                            df_data = []
                            for item in json_data['data']:
                                df_data.append({
                                    'date': item['date'][:10],  # Extract date part
                                    'open': item['open'],
                                    'high': item['high'],
                                    'low': item['low'],
                                    'close': item['close'],
                                    'volume': item['volume']
                                })
                            
                            data = pd.DataFrame(df_data)
                            data['date'] = pd.to_datetime(data['date'])
                            data = data.set_index('date').sort_index()
                            
                            st.success(f"Datos de {symbol} descargados desde Marketstack")
                            
                            # Show API info
                            st.info(f"📊 **{symbol}** - Marketstack Professional Data")
                            
                            # Show data preview
                            st.dataframe(data.head())
                            
                            # Select target column
                            available_cols = list(data.columns)
                            default_col = "close" if "close" in available_cols else available_cols[0]
                            target_col = st.selectbox("Columna a analizar:", available_cols, 
                                                    index=available_cols.index(default_col))
                            
                            if st.button("Procesar datos Marketstack"):
                                # Process the data
                                st.session_state.data = data
                                st.session_state.target_col = target_col
                                st.session_state.symbol = symbol
                                st.session_state.data_source = "Marketstack"
                                st.success(f"Datos de {symbol} procesados desde Marketstack")
                                
                                # Show basic stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Registros", len(data))
                                with col2:
                                    start_date = data.index.min().strftime("%Y-%m-%d")
                                    st.metric("Desde", start_date)
                                with col3:
                                    end_date = data.index.max().strftime("%Y-%m-%d")
                                    st.metric("Hasta", end_date)
                        else:
                            st.error(f"No se encontraron datos para {symbol}")
                            
                    else:
                        st.error(f"Error API: {response.status_code}")
                        if response.status_code == 401:
                            st.error("Error de autenticación. Verifica la API key.")
                        elif response.status_code == 422:
                            st.error("Símbolo no válido o no soportado.")
                            
            except Exception as e:
                st.error(f"Error descargando desde Marketstack: {e}")
                st.info("Verifica que el símbolo sea correcto (ej: AAPL para Apple)")
    
    elif data_source == "Finage API":
        st.subheader("📈 Finage API")
        
        # Use the provided API key
        api_key = "API_KEY7c3n9aYC2WldrsklG78EP6JfyaIoUbfhMIwLgnRMzsFsgp"
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Símbolo", value="AAPL", help="Ej: AAPL, GOOGL, MSFT, TSLA")
        with col2:
            period = st.selectbox("Período", ["1M", "3M", "6M", "1Y", "2Y"], index=3, 
                                help="Período de datos históricos")
        
        # Market categories
        st.subheader("🌟 Mercados principales")
        finage_markets = {
            "US Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"],
            "UK Stocks": ["BP", "SHELL", "VODAFONE", "HSBA", "LLOY"],
            "European": ["SAP", "ASML", "NESN", "RMS"],
            "ETFs": ["SPY", "QQQ", "VTI", "IWM"],
            "Forex": ["GBPUSD", "EURUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"],
            "Crypto": ["BTCUSD", "ETHUSD", "ADAUSD", "LTCUSD", "XRPUSD"]
        }
        
        selected_market = st.selectbox("Categoría:", list(finage_markets.keys()))
        selected_symbol = st.selectbox("O selecciona un símbolo:", [""] + finage_markets[selected_market])
        
        if selected_symbol:
            symbol = selected_symbol
        
        if st.button("Descargar desde Finage"):
            try:
                with st.spinner(f"Descargando {symbol} desde Finage..."):
                    # Calculate date range based on period
                    end_date = datetime.now()
                    if period == "1M":
                        start_date = end_date - timedelta(days=30)
                    elif period == "3M":
                        start_date = end_date - timedelta(days=90)
                    elif period == "6M":
                        start_date = end_date - timedelta(days=180)
                    elif period == "1Y":
                        start_date = end_date - timedelta(days=365)
                    elif period == "2Y":
                        start_date = end_date - timedelta(days=730)
                    
                    # Format dates for API
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                    
                    # Finage aggregated API endpoint
                    if symbol in ["BTCUSD", "ETHUSD", "ADAUSD", "LTCUSD", "XRPUSD"]:  # Crypto
                        url = f"https://api.finage.co.uk/agg/crypto/{symbol.lower()}/1/day/{start_str}/{end_str}"
                    elif symbol in ["GBPUSD", "EURUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]:  # Forex
                        url = f"https://api.finage.co.uk/agg/forex/{symbol.lower()}/1/day/{start_str}/{end_str}"
                    else:  # Stocks
                        url = f"https://api.finage.co.uk/agg/stock/{symbol.lower()}/1/day/{start_str}/{end_str}"
                    
                    params = {
                        'apikey': api_key
                    }
                    
                    # Make API request
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'results' in json_data and json_data['results']:
                            # Convert to DataFrame for aggregated endpoint
                            df_data = []
                            for item in json_data['results']:
                                # Handle both timestamp formats
                                if 't' in item:
                                    date = pd.to_datetime(item['t'], unit='ms')
                                elif 'timestamp' in item:
                                    date = pd.to_datetime(item['timestamp'], unit='ms')  
                                else:
                                    # Fallback for date string format
                                    date = pd.to_datetime(item.get('date', item.get('T')))
                                
                                df_data.append({
                                    'date': date,
                                    'open': item.get('o', item.get('open')),
                                    'high': item.get('h', item.get('high')),
                                    'low': item.get('l', item.get('low')),
                                    'close': item.get('c', item.get('close')),
                                    'volume': item.get('v', item.get('volume', 0))
                                })
                            
                            data = pd.DataFrame(df_data)
                            data = data.set_index('date').sort_index()
                            
                            st.success(f"Datos de {symbol} descargados desde Finage")
                            
                            # Show API info
                            st.info(f"📊 **{symbol}** - Finage Real-time Financial Data")
                            if 'info' in json_data:
                                st.info(f"ℹ️ {json_data['info']}")
                            
                            # Show data preview
                            st.dataframe(data.head())
                            
                            # Select target column
                            available_cols = list(data.columns)
                            default_col = "close" if "close" in available_cols else available_cols[0]
                            target_col = st.selectbox("Columna a analizar:", available_cols, 
                                                    index=available_cols.index(default_col))
                            
                            if st.button("Procesar datos Finage"):
                                # Process the data
                                st.session_state.data = data
                                st.session_state.target_col = target_col
                                st.session_state.symbol = symbol
                                st.session_state.data_source = "Finage"
                                st.success(f"Datos de {symbol} procesados desde Finage")
                                
                                # Show basic stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Registros", len(data))
                                with col2:
                                    start_date = data.index.min().strftime("%Y-%m-%d")
                                    st.metric("Desde", start_date)
                                with col3:
                                    end_date = data.index.max().strftime("%Y-%m-%d")
                                    st.metric("Hasta", end_date)
                        else:
                            st.error(f"No se encontraron datos para {symbol}")
                            if 'message' in json_data:
                                st.info(f"Mensaje API: {json_data['message']}")
                            
                    else:
                        st.error(f"Error API: {response.status_code}")
                        if response.status_code == 401:
                            st.error("Error de autenticación. Verifica la API key.")
                        elif response.status_code == 429:
                            st.error("Límite de consultas excedido. Intenta más tarde.")
                        else:
                            try:
                                error_data = response.json()
                                if 'message' in error_data:
                                    st.error(f"Error: {error_data['message']}")
                            except:
                                pass
                            
            except Exception as e:
                st.error(f"Error descargando desde Finage: {e}")
                st.info("Verifica que el símbolo sea correcto y que tengas acceso a Finage")
    
    elif data_source == "Finnhub API":
        st.subheader("📊 Finnhub API")
        
        # Use the provided API key
        api_key = "d2qlgp9r01qn21mk2h50d2qlgp9r01qn21mk2h5g"
        
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Símbolo", value="AAPL", help="Ej: AAPL, GOOGL, MSFT, TSLA")
        with col2:
            resolution = st.selectbox("Resolución", ["D", "W", "M"], index=0, 
                                    help="D=Diario, W=Semanal, M=Mensual")
        
        # Time period selection
        period = st.selectbox("Período", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3)
        
        # Market categories for Finnhub
        st.subheader("🎯 Mercados y símbolos")
        finnhub_markets = {
            "US Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B"],
            "US Growth": ["TSLA", "NVDA", "AMD", "NFLX", "CRM", "ADBE", "PYPL"],
            "US Value": ["JPM", "JNJ", "PG", "KO", "WMT", "V", "MA"],
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "NVDA"],
            "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "DHR", "BMY"]
        }
        
        selected_category = st.selectbox("Categoría:", list(finnhub_markets.keys()))
        selected_symbol = st.selectbox("O selecciona un símbolo:", [""] + finnhub_markets[selected_category])
        
        if selected_symbol:
            symbol = selected_symbol
        
        if st.button("Descargar desde Finnhub"):
            try:
                with st.spinner(f"Descargando {symbol} desde Finnhub..."):
                    # Calculate timestamps for Finnhub API (Unix timestamps)
                    end_date = datetime.now()
                    if period == "1M":
                        start_date = end_date - timedelta(days=30)
                    elif period == "3M":
                        start_date = end_date - timedelta(days=90)
                    elif period == "6M":
                        start_date = end_date - timedelta(days=180)
                    elif period == "1Y":
                        start_date = end_date - timedelta(days=365)
                    elif period == "2Y":
                        start_date = end_date - timedelta(days=730)
                    elif period == "5Y":
                        start_date = end_date - timedelta(days=1825)
                    
                    # Convert to Unix timestamps
                    start_timestamp = int(start_date.timestamp())
                    end_timestamp = int(end_date.timestamp())
                    
                    # Finnhub stock candle API endpoint
                    url = "https://finnhub.io/api/v1/stock/candle"
                    params = {
                        'symbol': symbol,
                        'resolution': resolution,
                        'from': start_timestamp,
                        'to': end_timestamp,
                        'token': api_key
                    }
                    
                    # Make API request
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        # Check if data is available
                        if json_data.get('s') == 'ok' and 'c' in json_data:
                            # Convert to DataFrame
                            df_data = []
                            timestamps = json_data['t']
                            opens = json_data['o']
                            highs = json_data['h']
                            lows = json_data['l']
                            closes = json_data['c']
                            volumes = json_data['v']
                            
                            for i in range(len(timestamps)):
                                df_data.append({
                                    'date': pd.to_datetime(timestamps[i], unit='s'),
                                    'open': opens[i],
                                    'high': highs[i],
                                    'low': lows[i],
                                    'close': closes[i],
                                    'volume': volumes[i]
                                })
                            
                            data = pd.DataFrame(df_data)
                            data = data.set_index('date').sort_index()
                            
                            st.success(f"Datos de {symbol} descargados desde Finnhub")
                            
                            # Show API info
                            st.info(f"📊 **{symbol}** - Finnhub Professional Market Data")
                            st.info(f"📈 Resolución: {resolution} | Período: {period} | Registros: {len(data)}")
                            
                            # Show data preview
                            st.dataframe(data.head())
                            
                            # Select target column
                            available_cols = list(data.columns)
                            default_col = "close" if "close" in available_cols else available_cols[0]
                            target_col = st.selectbox("Columna a analizar:", available_cols, 
                                                    index=available_cols.index(default_col))
                            
                            if st.button("Procesar datos Finnhub"):
                                # Process the data
                                st.session_state.data = data
                                st.session_state.target_col = target_col
                                st.session_state.symbol = symbol
                                st.session_state.data_source = "Finnhub"
                                st.success(f"Datos de {symbol} procesados desde Finnhub")
                                
                                # Show basic stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Registros", len(data))
                                with col2:
                                    start_date = data.index.min().strftime("%Y-%m-%d")
                                    st.metric("Desde", start_date)
                                with col3:
                                    end_date = data.index.max().strftime("%Y-%m-%d")
                                    st.metric("Hasta", end_date)
                        elif json_data.get('s') == 'no_data':
                            st.error(f"No hay datos disponibles para {symbol} en el período seleccionado")
                        else:
                            st.error(f"Error en respuesta de Finnhub: {json_data}")
                            
                    else:
                        st.error(f"Error API: {response.status_code}")
                        if response.status_code == 401:
                            st.error("Error de autenticación. Verifica la API key de Finnhub.")
                        elif response.status_code == 429:
                            st.error("Límite de consultas excedido. Espera un momento.")
                        else:
                            try:
                                error_data = response.json()
                                st.error(f"Error: {error_data}")
                            except:
                                st.error("Error desconocido de la API")
                            
            except Exception as e:
                st.error(f"Error descargando desde Finnhub: {e}")
                st.info("Verifica que el símbolo sea correcto y esté disponible en Finnhub")
    
    elif data_source == "Tiingo API":
        st.subheader("⚡ Tiingo API - Streaming en Tiempo Real")
        
        # Get API key from environment
        api_key = os.getenv("TIINGO_API_KEY")
        if not api_key:
            st.error("⚠️ Tiingo API key no encontrada. Configura tu API key primero.")
            st.stop()
        
        # Data mode selection
        data_mode = st.radio("Modo de datos:", ["📊 Datos Históricos (REST)", "🔄 Streaming en Tiempo Real (WebSocket)"])
        
        if data_mode == "📊 Datos Históricos (REST)":
            st.subheader("📈 Datos Históricos con Tiingo")
            
            col1, col2 = st.columns(2)
            with col1:
                asset_type = st.selectbox("Tipo de activo:", ["Stocks", "Crypto", "Forex", "Fundos"])
            with col2:
                period = st.selectbox("Período:", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3)
            
            # Asset-specific symbol input and popular symbols
            if asset_type == "Stocks":
                symbol = st.text_input("Símbolo de acción:", value="AAPL", help="Ej: AAPL, GOOGL, MSFT")
                
                st.subheader("📊 Acciones populares")
                stock_symbols = {
                    "Large Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
                    "Tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "CRM", "ADBE"],
                    "Finance": ["JPM", "BAC", "WFC", "GS", "V", "MA"],
                    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO"]
                }
                selected_category = st.selectbox("Categoría:", list(stock_symbols.keys()))
                selected_stock = st.selectbox("O selecciona:", [""] + stock_symbols[selected_category])
                if selected_stock:
                    symbol = selected_stock
                    
            elif asset_type == "Crypto":
                symbol = st.text_input("Par cripto:", value="btcusd", help="Formato: btcusd, ethusd, etc.")
                
                st.subheader("₿ Criptomonedas populares")
                crypto_pairs = ["btcusd", "ethusd", "adausd", "dotusd", "ltcusd", "xrpusd", "linkusd", "bnbusd"]
                selected_crypto = st.selectbox("O selecciona un par:", [""] + crypto_pairs)
                if selected_crypto:
                    symbol = selected_crypto
                    
            elif asset_type == "Forex":
                symbol = st.text_input("Par de divisas:", value="eurusd", help="Formato: eurusd, gbpusd, etc.")
                
                st.subheader("💱 Pares de divisas principales")
                forex_pairs = ["eurusd", "gbpusd", "usdjpy", "usdchf", "audusd", "usdcad", "nzdusd"]
                selected_forex = st.selectbox("O selecciona un par:", [""] + forex_pairs)
                if selected_forex:
                    symbol = selected_forex
                    
            elif asset_type == "Fundos":
                symbol = st.text_input("Símbolo de fondo:", value="SPY", help="Ej: SPY, QQQ, VTI")
                
                st.subheader("📈 ETFs populares")
                fund_symbols = ["SPY", "QQQ", "VTI", "IWM", "EFA", "EEM", "AGG", "GLD"]
                selected_fund = st.selectbox("O selecciona un ETF:", [""] + fund_symbols)
                if selected_fund:
                    symbol = selected_fund
            
            if st.button("Descargar datos históricos Tiingo"):
                try:
                    with st.spinner(f"Descargando {symbol} desde Tiingo..."):
                        # Calculate date range
                        end_date = datetime.now()
                        if period == "1M":
                            start_date = end_date - timedelta(days=30)
                        elif period == "3M":
                            start_date = end_date - timedelta(days=90)
                        elif period == "6M":
                            start_date = end_date - timedelta(days=180)
                        elif period == "1Y":
                            start_date = end_date - timedelta(days=365)
                        elif period == "2Y":
                            start_date = end_date - timedelta(days=730)
                        elif period == "5Y":
                            start_date = end_date - timedelta(days=1825)
                        
                        start_str = start_date.strftime("%Y-%m-%d")
                        end_str = end_date.strftime("%Y-%m-%d")
                        
                        # Build API URL based on asset type
                        if asset_type == "Stocks":
                            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
                            params = {
                                'startDate': start_str,
                                'endDate': end_str,
                                'token': api_key
                            }
                        elif asset_type == "Crypto":
                            url = f"https://api.tiingo.com/tiingo/crypto/prices"
                            params = {
                                'tickers': symbol,
                                'startDate': start_str,
                                'endDate': end_str,
                                'resampleFreq': '1day',
                                'token': api_key
                            }
                        elif asset_type == "Forex":
                            url = f"https://api.tiingo.com/tiingo/fx/{symbol}/prices"
                            params = {
                                'startDate': start_str,
                                'endDate': end_str,
                                'resampleFreq': '1day',
                                'token': api_key
                            }
                        elif asset_type == "Fundos":
                            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
                            params = {
                                'startDate': start_str,
                                'endDate': end_str,
                                'token': api_key
                            }
                        
                        # Make API request
                        response = requests.get(url, params=params)
                        
                        if response.status_code == 200:
                            json_data = response.json()
                            
                            if json_data and len(json_data) > 0:
                                # Process data based on asset type
                                df_data = []
                                
                                if asset_type in ["Stocks", "Fundos"]:
                                    for item in json_data:
                                        df_data.append({
                                            'date': pd.to_datetime(item['date']),
                                            'open': item['open'],
                                            'high': item['high'],
                                            'low': item['low'],
                                            'close': item['close'],
                                            'volume': item['volume']
                                        })
                                elif asset_type == "Crypto":
                                    # Crypto data comes nested
                                    for item in json_data:
                                        if 'priceData' in item:
                                            for price_item in item['priceData']:
                                                df_data.append({
                                                    'date': pd.to_datetime(price_item['date']),
                                                    'open': price_item['open'],
                                                    'high': price_item['high'],
                                                    'low': price_item['low'],
                                                    'close': price_item['close'],
                                                    'volume': price_item.get('volume', 0)
                                                })
                                elif asset_type == "Forex":
                                    for item in json_data:
                                        df_data.append({
                                            'date': pd.to_datetime(item['date']),
                                            'open': item['open'],
                                            'high': item['high'],
                                            'low': item['low'],
                                            'close': item['close'],
                                            'volume': 0  # Forex doesn't have volume
                                        })
                                
                                if df_data:
                                    data = pd.DataFrame(df_data)
                                    data = data.set_index('date').sort_index()
                                    
                                    st.success(f"✅ Datos de {symbol} descargados desde Tiingo")
                                    st.info(f"🎯 **{symbol.upper()}** - Tiingo {asset_type} Data | Período: {period}")
                                    
                                    # Show data preview
                                    st.dataframe(data.head())
                                    
                                    # Select target column
                                    available_cols = list(data.columns)
                                    default_col = "close" if "close" in available_cols else available_cols[0]
                                    target_col = st.selectbox("Columna a analizar:", available_cols, 
                                                            index=available_cols.index(default_col))
                                    
                                    if st.button("Procesar datos Tiingo"):
                                        # Process the data
                                        st.session_state.data = data
                                        st.session_state.target_col = target_col
                                        st.session_state.symbol = symbol
                                        st.session_state.data_source = "Tiingo"
                                        st.success(f"Datos de {symbol} procesados desde Tiingo")
                                        
                                        # Show basic stats
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Registros", len(data))
                                        with col2:
                                            start_date = data.index.min().strftime("%Y-%m-%d")
                                            st.metric("Desde", start_date)
                                        with col3:
                                            end_date = data.index.max().strftime("%Y-%m-%d")
                                            st.metric("Hasta", end_date)
                                else:
                                    st.error("No se pudieron procesar los datos recibidos")
                            else:
                                st.error(f"No se encontraron datos para {symbol}")
                                
                        else:
                            st.error(f"Error API: {response.status_code}")
                            if response.status_code == 401:
                                st.error("Error de autenticación. Verifica la API key de Tiingo.")
                            elif response.status_code == 403:
                                st.error("Acceso denegado. Verifica permisos de tu API key.")
                            elif response.status_code == 404:
                                st.error(f"Símbolo {symbol} no encontrado.")
                            else:
                                try:
                                    error_data = response.json()
                                    st.error(f"Error: {error_data}")
                                except:
                                    st.error("Error desconocido de la API")
                                    
                except Exception as e:
                    st.error(f"Error descargando desde Tiingo: {e}")
                    st.info("Verifica que el símbolo sea correcto y esté disponible en Tiingo")
        
        else:  # Streaming mode
            st.subheader("🔄 Streaming en Tiempo Real")
            
            # WebSocket streaming controls
            col1, col2 = st.columns(2)
            with col1:
                stream_type = st.selectbox("Tipo de stream:", ["Crypto", "Forex", "IEX Stocks"])
            with col2:
                symbol = st.text_input("Símbolo:", value="btcusd" if stream_type == "Crypto" else "eurusd" if stream_type == "Forex" else "AAPL")
            
            # Streaming symbols by type
            if stream_type == "Crypto":
                st.subheader("₿ Crypto streaming disponible")
                crypto_symbols = ["btcusd", "ethusd", "adausd", "dotusd", "ltcusd", "xrpusd", "linkusd"]
                selected_crypto = st.selectbox("Cryptos populares:", [""] + crypto_symbols)
                if selected_crypto:
                    symbol = selected_crypto
                    
            elif stream_type == "Forex":
                st.subheader("💱 Forex streaming disponible")
                forex_symbols = ["eurusd", "gbpusd", "usdjpy", "usdchf", "audusd", "usdcad"]
                selected_forex = st.selectbox("Pares principales:", [""] + forex_symbols)
                if selected_forex:
                    symbol = selected_forex
                    
            elif stream_type == "IEX Stocks":
                st.subheader("📊 Stocks streaming IEX")
                stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]
                selected_stock = st.selectbox("Acciones principales:", [""] + stock_symbols)
                if selected_stock:
                    symbol = selected_stock
            
            # Streaming controls
            col1, col2, col3 = st.columns(3)
            with col1:
                start_stream = st.button("🚀 Iniciar Stream")
            with col2:
                pause_stream = st.button("⏸️ Pausar")
            with col3:
                stop_stream = st.button("🛑 Detener")
            
            # Initialize streaming state
            if 'streaming_active' not in st.session_state:
                st.session_state.streaming_active = False
            if 'stream_data' not in st.session_state:
                st.session_state.stream_data = []
            
            # Streaming logic (placeholder for WebSocket implementation)
            if start_stream:
                st.session_state.streaming_active = True
                st.success(f"🎯 Stream iniciado para {symbol} ({stream_type})")
                st.info("⚡ Simulación de datos en tiempo real activada")
                
                # Placeholder for WebSocket connection
                st.warning("🚧 WebSocket streaming en desarrollo. Conexión simulada:")
                
                if stream_type == "Crypto":
                    ws_url = f"wss://api.tiingo.com/crypto"
                elif stream_type == "Forex":
                    ws_url = f"wss://api.tiingo.com/fx"
                elif stream_type == "IEX Stocks":
                    ws_url = f"wss://api.tiingo.com/iex"
                
                st.code(f"Conectando a: {ws_url}")
                st.code(f"Suscrito a: {symbol}")
                st.code(f"API Key: ***{api_key[-8:]}")
                
                # Real-time data placeholder
                placeholder = st.empty()
                
                # Simulate real-time updates
                import time
                for i in range(5):
                    with placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Precio", f"${100 + i * 0.5:.2f}")
                        with col2:
                            st.metric("Volumen", f"{1000 + i * 10}")
                        with col3:
                            st.metric("Cambio", f"+{i * 0.1:.1f}%")
                        with col4:
                            st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
                    time.sleep(1)
            
            if pause_stream:
                st.session_state.streaming_active = False
                st.warning("⏸️ Stream pausado")
            
            if stop_stream:
                st.session_state.streaming_active = False
                st.session_state.stream_data = []
                st.error("🛑 Stream detenido y datos limpiados")
            
            # News integration
            st.subheader("📰 Noticias Financieras en Tiempo Real")
            
            if st.button("📈 Cargar últimas noticias"):
                try:
                    with st.spinner("Cargando noticias desde Tiingo..."):
                        news_url = "https://api.tiingo.com/tiingo/news"
                        news_params = {
                            'token': api_key,
                            'limit': 10,
                            'offset': 0
                        }
                        
                        news_response = requests.get(news_url, params=news_params)
                        
                        if news_response.status_code == 200:
                            news_data = news_response.json()
                            
                            st.success("📰 Últimas noticias cargadas")
                            
                            for article in news_data[:5]:  # Show first 5 articles
                                with st.expander(f"📄 {article.get('title', 'Sin título')}"):
                                    st.write(f"**Fuente:** {article.get('source', 'N/A')}")
                                    st.write(f"**Fecha:** {article.get('publishedDate', 'N/A')}")
                                    st.write(f"**Descripción:** {article.get('description', 'Sin descripción')}")
                                    if article.get('url'):
                                        st.link_button("🔗 Leer más", article['url'])
                        else:
                            st.error(f"Error cargando noticias: {news_response.status_code}")
                            
                except Exception as e:
                    st.error(f"Error cargando noticias: {e}")

# Visualize Section (renamed)
elif step == "👁️ Visualize":
    st.header("2. Visualizar")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        # Show symbol info if available
        if hasattr(st.session_state, 'symbol'):
            st.subheader(f"Serie de tiempo - {st.session_state.symbol} ({target})")
        else:
            st.subheader(f"Serie de tiempo - {target}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[target], mode='lines', name=target))
        fig.update_layout(height=400, title=f"{target} Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Media", f"{df[target].mean():.2f}")
        with col2:
            st.metric("Desv. Std", f"{df[target].std():.2f}")
        with col3:
            st.metric("Puntos", len(df))
        with col4:
            st.metric("Rango", f"{df[target].max() - df[target].min():.2f}")
    else:
        st.warning("Carga los datos primero")

# Stationarity Test Section
elif step == "🔍 Stationarity Test":
    st.header("3. Test de Estacionariedad")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        adf_test(df[target])
    else:
        st.warning("Carga los datos primero")

# Difference Section
elif step == "📐 Difference":
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

# Advanced Models Section 
elif step == "🤖 Advanced Models":
    st.header("🤖 Modelos Avanzados de Forecasting")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_col
        
        # Model selection
        model_type = st.selectbox("🧠 Selecciona Modelo:", [
            "ARIMA (Clásico)",
            "Prophet (Facebook)", 
            "LSTM (Deep Learning)",
            "Ensemble (Combinado)"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            train_split = st.slider("% Datos entrenamiento", 60, 90, 80)
        with col2:
            forecast_days = st.number_input("Días a predecir", 1, 365, 30)
        
        if st.button("🚀 Entrenar Modelo Avanzado"):
            split_idx = int(len(df) * train_split / 100)
            train_data = df[:split_idx]
            test_data = df[split_idx:]
            
            if model_type == "ARIMA (Clásico)":
                try:
                    with st.spinner("🔄 Entrenando ARIMA..."):
                        best_model, best_params, results = auto_arima_tuning(
                            train_data[target], max_p=3, max_d=2, max_q=3
                        )
                        
                        if best_model:
                            # Make predictions
                            forecast = best_model.forecast(steps=len(test_data))
                            future_forecast = best_model.forecast(steps=forecast_days)
                            
                            # Calculate metrics
                            rmse = np.sqrt(mean_squared_error(test_data[target], forecast))
                            mae = mean_absolute_error(test_data[target], forecast)
                            
                            st.success(f"✅ ARIMA{best_params} entrenado")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{rmse:.4f}")
                            with col2:
                                st.metric("MAE", f"{mae:.4f}")
                            with col3:
                                st.metric("AIC", f"{best_model.aic:.2f}")
                            
                            # Store results
                            st.session_state.model = best_model
                            st.session_state.forecast = future_forecast
                            
                except Exception as e:
                    st.error(f"Error ARIMA: {e}")
            
            elif model_type == "Prophet (Facebook)" and PROPHET_AVAILABLE:
                try:
                    with st.spinner("🔄 Entrenando Prophet..."):
                        # Prepare data for Prophet
                        prophet_data = train_data.reset_index()
                        prophet_data.columns = ['ds', 'y']
                        
                        # Initialize and fit Prophet
                        model = Prophet(
                            daily_seasonality=True,
                            weekly_seasonality=True,
                            yearly_seasonality=True,
                            changepoint_prior_scale=0.05
                        )
                        model.fit(prophet_data)
                        
                        # Make predictions on test set
                        test_future = model.make_future_dataframe(periods=len(test_data))
                        test_forecast = model.predict(test_future)
                        
                        # Future predictions
                        future = model.make_future_dataframe(periods=forecast_days + len(df))
                        forecast = model.predict(future)
                        
                        # Calculate metrics
                        predicted = test_forecast['yhat'][-len(test_data):]
                        rmse = np.sqrt(mean_squared_error(test_data[target], predicted))
                        mae = mean_absolute_error(test_data[target], predicted)
                        
                        st.success("✅ Prophet entrenado exitosamente")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col2:
                            st.metric("MAE", f"{mae:.4f}")
                        with col3:
                            st.metric("R²", f"{1 - (rmse**2 / np.var(test_data[target])):.4f}")
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[target],
                            mode='lines', name='Histórico',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        future_dates = pd.date_range(
                            start=df.index[-1] + pd.Timedelta(days=1),
                            periods=forecast_days
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates, 
                            y=forecast['yhat'][-forecast_days:],
                            mode='lines', name='Pronóstico Prophet',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Confidence intervals
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_upper'][-forecast_days:],
                            fill=None, mode='lines',
                            line_color='rgba(255,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast['yhat_lower'][-forecast_days:],
                            fill='tonexty', mode='lines',
                            line_color='rgba(255,0,0,0)',
                            name='Intervalo Confianza'
                        ))
                        
                        fig.update_layout(
                            title="📈 Pronóstico Prophet con Intervalos de Confianza",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store results
                        st.session_state.prophet_model = model
                        st.session_state.prophet_forecast = forecast
                        
                except Exception as e:
                    st.error(f"Error Prophet: {e}")
            
            elif model_type == "LSTM (Deep Learning)" and TF_AVAILABLE:
                try:
                    with st.spinner("🔄 Entrenando LSTM Neural Network..."):
                        # Prepare data for LSTM
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(train_data[target].values.reshape(-1, 1))
                        
                        # Create sequences
                        def create_sequences(data, seq_length=60):
                            X, y = [], []
                            for i in range(seq_length, len(data)):
                                X.append(data[i-seq_length:i, 0])
                                y.append(data[i, 0])
                            return np.array(X), np.array(y)
                        
                        seq_length = min(60, len(scaled_data) // 4)
                        X_train, y_train = create_sequences(scaled_data, seq_length)
                        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        
                        # Build LSTM model
                        model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                            Dropout(0.2),
                            LSTM(50, return_sequences=False),
                            Dropout(0.2),
                            Dense(25),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # Train model
                        history = model.fit(
                            X_train, y_train,
                            batch_size=32,
                            epochs=50,
                            verbose=0,
                            validation_split=0.2
                        )
                        
                        # Make predictions
                        last_sequence = scaled_data[-seq_length:]
                        predictions = []
                        
                        for _ in range(forecast_days):
                            pred = model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
                            predictions.append(pred[0, 0])
                            last_sequence = np.append(last_sequence[1:], pred[0, 0])
                        
                        # Inverse transform predictions
                        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                        
                        st.success("✅ LSTM Neural Network entrenado")
                        
                        # Show training metrics
                        final_loss = history.history['loss'][-1]
                        val_loss = history.history['val_loss'][-1]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Loss Final", f"{final_loss:.6f}")
                        with col2:
                            st.metric("Val Loss", f"{val_loss:.6f}")
                        with col3:
                            st.metric("Épocas", "50")
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Historical
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[target],
                            mode='lines', name='Histórico',
                            line=dict(color='blue')
                        ))
                        
                        # LSTM Forecast
                        future_dates = pd.date_range(
                            start=df.index[-1] + pd.Timedelta(days=1),
                            periods=forecast_days
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=predictions.flatten(),
                            mode='lines', name='Pronóstico LSTM',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title="🧠 Pronóstico LSTM Deep Learning",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store results
                        st.session_state.lstm_model = model
                        st.session_state.lstm_predictions = predictions
                        
                except Exception as e:
                    st.error(f"Error LSTM: {e}")
            
            elif model_type == "Ensemble (Combinado)":
                st.info("🔄 Entrenando múltiples modelos para ensemble...")
                
                # This would combine ARIMA, Prophet, and LSTM
                ensemble_predictions = []
                weights = []
                
                try:
                    # ARIMA component
                    best_model, _, _ = auto_arima_tuning(train_data[target], max_p=2, max_d=1, max_q=2)
                    if best_model:
                        arima_pred = best_model.forecast(steps=forecast_days)
                        ensemble_predictions.append(arima_pred)
                        weights.append(0.4)
                        st.success("✅ ARIMA component trained")
                except:
                    pass
                
                if len(ensemble_predictions) > 0:
                    # Simple weighted average for now
                    final_prediction = np.average(ensemble_predictions, axis=0, weights=weights)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[target],
                        mode='lines', name='Histórico'
                    ))
                    
                    future_dates = pd.date_range(
                        start=df.index[-1] + pd.Timedelta(days=1),
                        periods=forecast_days
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=final_prediction,
                        mode='lines', name='Ensemble Forecast',
                        line=dict(color='purple', dash='dash')
                    ))
                    
                    fig.update_layout(title="🎯 Ensemble Model Forecast", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Ensemble con {len(ensemble_predictions)} modelos")
        
        # Model availability warnings
        if not PROPHET_AVAILABLE and model_type == "Prophet (Facebook)":
            st.warning("⚠️ Prophet no está disponible. Instala prophet para usar este modelo.")
        
        if not TF_AVAILABLE and model_type == "LSTM (Deep Learning)":
            st.warning("⚠️ TensorFlow no está disponible. Instala tensorflow para usar LSTM.")
    
    else:
        st.warning("Carga los datos primero")

# Performance Tracker Section
elif step == "🏆 Performance Tracker":
    st.header("🏆 Performance Tracker & Achievements")
    
    # User level and achievements system
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👤 Nivel de Usuario", st.session_state.user_level)
    with col2:
        st.metric("🏅 Logros Desbloqueados", len(st.session_state.achievements))
    with col3:
        xp_needed = (st.session_state.user_level * 100) - (len(st.session_state.achievements) * 10)
        st.metric("⭐ XP para Siguiente Nivel", max(0, xp_needed))
    
    # Achievement badges
    st.subheader("🎖️ Sistema de Logros")
    
    achievements_data = {
        "🥇 First Model": {"desc": "Entrena tu primer modelo", "unlocked": "first_model" in st.session_state.achievements},
        "📊 Data Explorer": {"desc": "Analiza 5 símbolos diferentes", "unlocked": "data_explorer" in st.session_state.achievements},
        "🤖 AI Master": {"desc": "Usa 3 tipos de modelos diferentes", "unlocked": "ai_master" in st.session_state.achievements},
        "📈 Prophet User": {"desc": "Completa un pronóstico con Prophet", "unlocked": "prophet_user" in st.session_state.achievements},
        "🧠 LSTM Expert": {"desc": "Entrena una red neural LSTM", "unlocked": "lstm_expert" in st.session_state.achievements},
        "🎯 Precision Pro": {"desc": "Obtén RMSE < 0.05", "unlocked": "precision_pro" in st.session_state.achievements},
        "💎 Portfolio Builder": {"desc": "Crea un portfolio de 10+ símbolos", "unlocked": "portfolio_builder" in st.session_state.achievements},
        "🔥 Streak Master": {"desc": "Usa la app 7 días seguidos", "unlocked": "streak_master" in st.session_state.achievements}
    }
    
    cols = st.columns(4)
    for i, (achievement, data) in enumerate(achievements_data.items()):
        with cols[i % 4]:
            if data["unlocked"]:
                st.success(f"✅ {achievement}")
                st.caption(data["desc"])
            else:
                st.info(f"🔒 {achievement}")
                st.caption(data["desc"])
    
    # Performance metrics history
    st.subheader("📊 Historial de Performance")
    
    if 'performance_history' not in st.session_state:
        st.session_state.performance_history = []
    
    if st.session_state.performance_history:
        perf_df = pd.DataFrame(st.session_state.performance_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=perf_df['date'], y=perf_df['rmse'],
            mode='lines+markers', name='RMSE',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="📈 Evolución de RMSE en el Tiempo",
            xaxis_title="Fecha",
            yaxis_title="RMSE",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best performance summary
        best_rmse = min(perf_df['rmse'])
        best_model = perf_df.loc[perf_df['rmse'].idxmin(), 'model']
        
        st.success(f"🏆 **Mejor RMSE:** {best_rmse:.4f} con {best_model}")
    else:
        st.info("📝 Entrena algunos modelos para ver tu historial de performance")
    
    # Leaderboard simulation
    st.subheader("🎮 Leaderboard Global")
    
    leaderboard_data = [
        {"Usuario": "Tú 🎯", "Mejor RMSE": "0.0234", "Modelos": 12, "Nivel": st.session_state.user_level},
        {"Usuario": "DataNinja 🥷", "Mejor RMSE": "0.0189", "Modelos": 45, "Nivel": 15},
        {"Usuario": "MarketGuru 📈", "Mejor RMSE": "0.0156", "Modelos": 78, "Nivel": 22},
        {"Usuario": "AIExpert 🤖", "Mejor RMSE": "0.0098", "Modelos": 134, "Nivel": 31},
        {"Usuario": "QuantKing 👑", "Mejor RMSE": "0.0067", "Modelos": 200, "Nivel": 45}
    ]
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

# Mood & Sentiment Section
elif step == "🎯 Mood & Sentiment":
    st.header("🎯 Stock Mood Tracker & Sentiment Analysis")
    
    # Current market mood
    st.subheader("😊 Market Mood Dashboard")
    
    mood_emojis = {
        "Muy Alcista": "🚀",
        "Alcista": "📈", 
        "Neutral": "😐",
        "Bajista": "📉",
        "Muy Bajista": "💥"
    }
    
    # Simulate market sentiment for popular stocks
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
    sentiment_data = []
    
    for stock in popular_stocks:
        # Simulate sentiment (in real app, this would come from news analysis)
        import random
        sentiments = list(mood_emojis.keys())
        current_sentiment = random.choice(sentiments)
        sentiment_score = random.uniform(-1, 1)
        
        sentiment_data.append({
            "Símbolo": stock,
            "Mood": mood_emojis[current_sentiment],
            "Sentimiento": current_sentiment,
            "Score": f"{sentiment_score:.2f}",
            "Confianza": f"{random.randint(70, 95)}%"
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
    
    # Animated sentiment indicators
    st.subheader("🌟 Indicadores Animados de Sentimiento")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #ff6b6b, #ffa726); border-radius: 15px; animation: pulse 2s infinite;">
            <h2 style="color: white;">🔥 TSLA</h2>
            <p style="color: white; font-size: 24px;">Muy Alcista</p>
            <p style="color: white;">Score: +0.87</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #4caf50, #66bb6a); border-radius: 15px; animation: bounce 2s infinite;">
            <h2 style="color: white;">📈 AAPL</h2>
            <p style="color: white; font-size: 24px;">Alcista</p>
            <p style="color: white;">Score: +0.65</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #2196f3, #42a5f5); border-radius: 15px; animation: glow 3s infinite;">
            <h2 style="color: white;">😐 MSFT</h2>
            <p style="color: white; font-size: 24px;">Neutral</p>
            <p style="color: white;">Score: +0.12</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add CSS animations
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
        50% { box-shadow: 0 0 20px rgba(33, 150, 243, 0.8); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sentiment tracking over time
    st.subheader("📊 Evolución del Sentimiento")
    
    # Generate sample sentiment timeline
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    sentiment_timeline = []
    
    for date in dates:
        sentiment_timeline.append({
            'date': date,
            'bull_sentiment': random.uniform(0.3, 0.9),
            'bear_sentiment': random.uniform(0.1, 0.7),
            'neutral_sentiment': random.uniform(0.2, 0.5)
        })
    
    timeline_df = pd.DataFrame(sentiment_timeline)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_df['date'], y=timeline_df['bull_sentiment'],
        mode='lines', name='📈 Alcista',
        line=dict(color='green'), fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline_df['date'], y=timeline_df['bear_sentiment'],
        mode='lines', name='📉 Bajista',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=timeline_df['date'], y=timeline_df['neutral_sentiment'],
        mode='lines', name='😐 Neutral',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="🎭 Evolución del Sentimiento del Mercado",
        xaxis_title="Fecha",
        yaxis_title="Score de Sentimiento",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Symbol Heat Map Section
elif step == "🗺️ Symbol Heat Map":
    st.header("🗺️ Interactive Symbol Heat Map")
    
    # Generate sample market data for heat map
    sectors = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "ORCL"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "V", "MA", "AXP", "USB"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "DHR", "BMY", "MRK"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC"],
        "Consumer": ["AMZN", "TSLA", "WMT", "HD", "NKE", "SBUX", "MCD", "DIS"]
    }
    
    heat_map_data = []
    
    for sector, symbols in sectors.items():
        for symbol in symbols:
            # Simulate performance data
            import random
            performance = random.uniform(-5, 8)  # Random % change
            volume = random.randint(1000000, 50000000)
            market_cap = random.uniform(50, 3000)  # Billions
            
            heat_map_data.append({
                'Symbol': symbol,
                'Sector': sector,
                'Performance': performance,
                'Volume': volume,
                'MarketCap': market_cap,
                'Size': market_cap  # For bubble size
            })
    
    heat_df = pd.DataFrame(heat_map_data)
    
    # Interactive controls
    col1, col2 = st.columns(2)
    with col1:
        color_metric = st.selectbox("Color por:", ["Performance", "Volume", "MarketCap"])
    with col2:
        size_metric = st.selectbox("Tamaño por:", ["MarketCap", "Volume", "Performance"])
    
    # Create interactive heat map
    fig = px.treemap(
        heat_df,
        path=['Sector', 'Symbol'],
        values='MarketCap',
        color='Performance',
        color_continuous_scale='RdYlGn',
        title="🗺️ Market Heat Map por Sector y Performance"
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bubble chart alternative view
    st.subheader("🫧 Vista de Burbujas Interactiva")
    
    fig_bubble = px.scatter(
        heat_df,
        x='Performance',
        y='Volume',
        size='MarketCap',
        color='Sector',
        hover_name='Symbol',
        title="📊 Performance vs Volume (Tamaño = Market Cap)",
        labels={
            'Performance': 'Performance (%)',
            'Volume': 'Volumen',
            'MarketCap': 'Market Cap (B)'
        }
    )
    
    fig_bubble.update_layout(height=500)
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Top movers
    st.subheader("🏃‍♂️ Top Movers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("📈 **Top Gainers**")
        top_gainers = heat_df.nlargest(5, 'Performance')[['Symbol', 'Performance', 'Sector']]
        for _, row in top_gainers.iterrows():
            st.write(f"🟢 **{row['Symbol']}** ({row['Sector']}): +{row['Performance']:.2f}%")
    
    with col2:
        st.error("📉 **Top Losers**")
        top_losers = heat_df.nsmallest(5, 'Performance')[['Symbol', 'Performance', 'Sector']]
        for _, row in top_losers.iterrows():
            st.write(f"🔴 **{row['Symbol']}** ({row['Sector']}): {row['Performance']:.2f}%")

# AI Learning Mode Section
elif step == "🎓 AI Learning Mode":
    st.header("🎓 AI-Driven Learning Mode")
    
    # Learning progress
    st.subheader("📚 Tu Progreso de Aprendizaje")
    
    topics = {
        "📊 Fundamentos de ARIMA": 85,
        "🔮 Prophet Forecasting": 60,
        "🧠 Deep Learning LSTM": 40,
        "📈 Análisis Técnico": 75,
        "💹 Gestión de Riesgo": 55,
        "🎯 Portfolio Optimization": 30
    }
    
    for topic, progress in topics.items():
        st.write(f"**{topic}**")
        st.progress(progress / 100)
        st.write(f"Progreso: {progress}%")
        st.write("---")
    
    # Interactive learning modules
    st.subheader("🎯 Módulos de Aprendizaje Interactivos")
    
    learning_module = st.selectbox("Selecciona un módulo:", [
        "🔍 ¿Qué es ARIMA?",
        "📊 Interpretando Residuos",
        "🎲 Estacionariedad Explicada",
        "🧠 Redes Neuronales para Finanzas",
        "📈 Patrones de Mercado"
    ])
    
    if learning_module == "🔍 ¿Qué es ARIMA?":
        st.markdown("""
        ### 🎯 ARIMA: AutoRegressive Integrated Moving Average
        
        **ARIMA** es un modelo estadístico que combina tres componentes:
        
        🔄 **AR (AutoRegressive)**: El valor actual depende de valores pasados
        📊 **I (Integrated)**: Diferenciación para lograr estacionariedad  
        📈 **MA (Moving Average)**: Errores pasados afectan predicciones actuales
        
        #### 🎮 Demo Interactiva
        """)
        
        # Interactive ARIMA component demo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="padding: 20px; background: linear-gradient(45deg, #ff6b6b, #ffa726); border-radius: 10px; text-align: center;">
                <h3 style="color: white;">🔄 AR Component</h3>
                <p style="color: white;">Yt = φ₁Yt-₁ + φ₂Yt-₂ + ...</p>
                <p style="color: white; font-size: 12px;">Los valores pasados predicen el futuro</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 20px; background: linear-gradient(45deg, #4caf50, #66bb6a); border-radius: 10px; text-align: center;">
                <h3 style="color: white;">📊 I Component</h3>
                <p style="color: white;">ΔYt = Yt - Yt-₁</p>
                <p style="color: white; font-size: 12px;">Diferenciación para estacionariedad</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 20px; background: linear-gradient(45deg, #2196f3, #42a5f5); border-radius: 10px; text-align: center;">
                <h3 style="color: white;">📈 MA Component</h3>
                <p style="color: white;">Yt = θ₁εt-₁ + θ₂εt-₂ + ...</p>
                <p style="color: white; font-size: 12px;">Errores pasados mejoran predicción</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quiz
        st.subheader("🧩 Mini Quiz")
        
        quiz_question = st.radio(
            "¿Cuál es el propósito principal del componente 'I' en ARIMA?",
            ["Predecir valores futuros", "Hacer la serie estacionaria", "Corregir errores", "Calcular tendencias"]
        )
        
        if st.button("🎯 Verificar Respuesta"):
            if quiz_question == "Hacer la serie estacionaria":
                st.success("🎉 ¡Correcto! El componente 'I' (Integrado) hace la serie estacionaria mediante diferenciación.")
                # Add achievement
                if "quiz_master" not in st.session_state.achievements:
                    st.session_state.achievements.append("quiz_master")
                    st.balloons()
            else:
                st.error("❌ Intenta de nuevo. Piensa en qué significa 'Integrated'.")
    
    elif learning_module == "🧠 Redes Neuronales para Finanzas":
        st.markdown("""
        ### 🧠 Deep Learning en Finanzas
        
        Las **redes neuronales** pueden capturar patrones complejos que los modelos tradicionales no pueden.
        
        #### 🎯 LSTM para Series Temporales
        """)
        
        # Animated explanation
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="display: inline-block; padding: 15px; margin: 10px; background: #ff6b6b; color: white; border-radius: 50px; animation: slideInLeft 2s;">
                📊 Precio Pasado
            </div>
            <div style="display: inline-block; padding: 15px; margin: 10px; background: #ffa726; color: white; border-radius: 50px; animation: slideInLeft 2.5s;">
                🔄 LSTM Cell
            </div>
            <div style="display: inline-block; padding: 15px; margin: 10px; background: #4caf50; color: white; border-radius: 50px; animation: slideInRight 3s;">
                🔮 Predicción
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Tip**: LSTM funciona mejor con grandes cantidades de datos y puede aprender dependencias a largo plazo.")

# Original Model Section (renamed)
elif step == "🤖 Advanced Models":
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
            st.subheader("🔍 Búsqueda automática de parámetros ARIMA")
            st.info("La optimización automática probará todas las combinaciones para encontrar el mejor modelo")
            
            # Enhanced parameter controls
            col1, col2, col3 = st.columns(3)
            with col1:
                max_p = st.slider("Máximo p (AR)", 1, 8, 5, help="Orden autoregresivo máximo")
            with col2:
                max_d = st.slider("Máximo d (I)", 0, 3, 2, help="Grado de diferenciación máximo")
            with col3:
                max_q = st.slider("Máximo q (MA)", 1, 8, 5, help="Orden de media móvil máximo")
            
            # Advanced options
            with st.expander("⚙️ Opciones avanzadas"):
                col1, col2 = st.columns(2)
                with col1:
                    show_progress = st.checkbox("Mostrar progreso detallado", True)
                with col2:
                    max_results = st.number_input("Mostrar top modelos", 3, 20, 10)
            
            # Calculate total combinations
            total_combos = (max_p + 1) * (max_d + 1) * (max_q + 1)
            st.info(f"🧮 Se probarán **{total_combos}** combinaciones de parámetros")
            
            if st.button("🚀 Iniciar búsqueda automática", type="primary"):
                try:
                    st.markdown("---")
                    st.subheader("📊 Optimización en progreso...")
                    
                    # Run enhanced auto tuning
                    best_model, best_params, results = auto_arima_tuning(
                        df[target], max_p, max_d, max_q
                    )
                    
                    if best_model is not None:
                        st.session_state.model = best_model
                        st.session_state.params = best_params
                        
                        st.markdown("---")
                        # Enhanced results display
                        st.success(f"🎯 **Mejor modelo encontrado: ARIMA{best_params}**")
                        
                        # Model metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("AIC", f"{best_model.aic:.2f}")
                        with col2:
                            st.metric("BIC", f"{best_model.bic:.2f}")
                        with col3:
                            st.metric("Log-Likelihood", f"{best_model.llf:.2f}")
                        with col4:
                            rmse = np.sqrt(np.mean(best_model.resid**2))
                            st.metric("RMSE", f"{rmse:.4f}")
                        
                        # Comprehensive results table
                        if results:
                            st.subheader(f"📈 Top {min(max_results, len(results))} mejores modelos")
                            results_df = pd.DataFrame(results)
                            results_df = results_df.sort_values('AIC').head(max_results)
                            
                            # Add ranking
                            results_df['Ranking'] = range(1, len(results_df) + 1)
                            results_df = results_df[['Ranking', 'p', 'd', 'q', 'AIC', 'BIC', 'RMSE', 'LogLikelihood']]
                            
                            st.dataframe(
                                results_df, 
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Model comparison chart
                            st.subheader("📊 Comparación de modelos")
                            fig = go.Figure()
                            
                            # AIC comparison
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(results_df) + 1)),
                                y=results_df['AIC'],
                                mode='lines+markers',
                                name='AIC',
                                line=dict(color='blue')
                            ))
                            
                            # BIC comparison
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(results_df) + 1)),
                                y=results_df['BIC'],
                                mode='lines+markers',
                                name='BIC',
                                line=dict(color='red')
                            ))
                            
                            fig.update_layout(
                                title="Comparación AIC vs BIC por ranking",
                                xaxis_title="Ranking del modelo",
                                yaxis_title="Valor del criterio",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                    else:
                        st.error("❌ No se pudo encontrar un modelo válido")
                        st.info("Intenta con rangos de parámetros diferentes")
                        
                except Exception as e:
                    st.error(f"❌ Error en búsqueda automática: {e}")
                    st.info("Verifica que los datos sean válidos para modelado ARIMA")
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