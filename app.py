import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import os
import requests
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

def auto_arima_tuning(data, max_p=5, max_d=2, max_q=5, use_stepwise=False):
    """Enhanced ARIMA hyperparameter tuning with advanced grid search"""
    best_aic = float('inf')
    best_bic = float('inf')
    best_params = None
    best_model = None
    results = []
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1)
    current_combination = 0
    successful_fits = 0
    
    # Live metrics display
    with metrics_container:
        col1, col2, col3 = st.columns(3)
        with col1:
            tested_metric = st.empty()
        with col2:
            successful_metric = st.empty()
        with col3:
            best_aic_metric = st.empty()
    
    # Grid search with enhanced tracking
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                current_combination += 1
                progress = current_combination / total_combinations
                progress_bar.progress(progress)
                
                # Update status
                status_text.text(f"🔍 Testing ARIMA({p},{d},{q}) - {current_combination}/{total_combinations}")
                tested_metric.metric("Tested", f"{current_combination}/{total_combinations}")
                
                try:
                    # Fit ARIMA model
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    aic = fitted.aic
                    bic = fitted.bic
                    
                    # Calculate additional metrics
                    residuals = fitted.resid
                    mse = np.mean(residuals**2)
                    rmse = np.sqrt(mse)
                    
                    successful_fits += 1
                    successful_metric.metric("Successful", successful_fits)
                    
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'AIC': round(aic, 2), 
                        'BIC': round(bic, 2),
                        'RMSE': round(rmse, 4),
                        'LogLikelihood': round(fitted.llf, 2)
                    })
                    
                    # Update best model based on AIC
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted
                        best_aic_metric.metric("Best AIC", f"{aic:.2f}")
                        
                except Exception as e:
                    # Log failed attempts for debugging
                    continue
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Final summary
    if successful_fits > 0:
        st.success(f"✅ Tuning completed! Tested {successful_fits}/{total_combinations} valid combinations")
    else:
        st.error("❌ No valid ARIMA models found. Try different parameter ranges.")
    
    return best_model, best_params, results

# Step 1: Upload
if step == "Upload":
    st.header("1. Cargar Datos")
    
    # Data source selection
    data_source = st.radio("Fuente de datos:", ["Archivo CSV", "Yahoo Finance API", "Alpha Vantage API", "Marketstack API", "Finage API"])
    
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
            "Crypto": ["BTCUSD", "ETHUSD", "ADAUSD"]
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
                    
                    # Finage historical API endpoint
                    if symbol.endswith("USD"):  # Crypto
                        url = f"https://api.finage.co.uk/history/crypto/{symbol}"
                    else:  # Stocks
                        url = f"https://api.finage.co.uk/history/stock/{symbol}"
                    
                    params = {
                        'apikey': api_key,
                        'period': '1d',  # Daily data
                        'from': start_str,
                        'to': end_str
                    }
                    
                    # Make API request
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 200:
                        json_data = response.json()
                        
                        if 'results' in json_data and json_data['results']:
                            # Convert to DataFrame
                            df_data = []
                            for item in json_data['results']:
                                df_data.append({
                                    'date': pd.to_datetime(item['t'], unit='ms'),
                                    'open': item['o'],
                                    'high': item['h'],
                                    'low': item['l'],
                                    'close': item['c'],
                                    'volume': item.get('v', 0)  # Volume might not be available for all symbols
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

# Step 2: Visualize
elif step == "Visualize":
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