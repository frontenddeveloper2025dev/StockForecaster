import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from scipy import stats

st.set_page_config(
    page_title="Time Series Forecasting - Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Forecasting using ARIMA Models")
st.markdown("### Stock Market Trends Analysis and Prediction")

st.sidebar.title("Navigation")
analysis_steps = [
    "📊 Data Upload & Exploration",
    "📈 Data Visualization", 
    "🔍 Stationarity Testing",
    "⚖️ Making Data Stationary",
    "📊 ACF/PACF Analysis",
    "🤖 ARIMA Model Building",
    "🎯 Predictions & Results"
]

selected_step = st.sidebar.selectbox("Select Analysis Step:", analysis_steps)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'df_diff' not in st.session_state:
    st.session_state.df_diff = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None

def perform_adf_test(data, title="ADF Test Results"):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    """
    try:
        dftest = adfuller(data.dropna(), autolag='AIC')
        
        st.subheader(title)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ADF Statistic", f"{dftest[0]:.6f}")
            st.metric("P-Value", f"{dftest[1]:.6f}")
            
        with col2:
            st.write("**Critical Values:**")
            try:
                # ADF test returns: (adf_stat, p_value, usedlag, nobs, critical_values, icbest)
                critical_values = dftest[4] if len(dftest) > 4 else None
                if critical_values and isinstance(critical_values, dict):
                    for key, val in critical_values.items():
                        st.write(f"• {key}: {val:.6f}")
                else:
                    st.write("• Critical values not available")
            except Exception:
                st.write("• Critical values not available")
        
        # Interpretation
        if dftest[1] <= 0.05:
            st.success("✅ Data is stationary (p-value ≤ 0.05)")
            return True
        else:
            st.warning("⚠️ Data is non-stationary (p-value > 0.05)")
            return False
            
    except Exception as e:
        st.error(f"Error performing ADF test: {str(e)}")
        return False

def plot_acf_pacf(data, lags=40):
    """
    Plot Autocorrelation and Partial Autocorrelation functions
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF plot
        sm.graphics.tsa.plot_acf(data.dropna(), lags=lags, ax=ax1, title="Autocorrelation Function (ACF)")
        ax1.grid(True)
        
        # PACF plot
        sm.graphics.tsa.plot_pacf(data.dropna(), lags=lags, ax=ax2, title="Partial Autocorrelation Function (PACF)")
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Guidelines for parameter selection
        st.info("""
        **Parameter Selection Guidelines:**
        - **q (MA order)**: Look at ACF plot - first lag that crosses the confidence interval
        - **p (AR order)**: Look at PACF plot - first lag that crosses the confidence interval
        - **d (Differencing)**: Number of differencing operations needed for stationarity
        """)
        
    except Exception as e:
        st.error(f"Error creating ACF/PACF plots: {str(e)}")

# Step 1: Data Upload & Exploration
if selected_step == "📊 Data Upload & Exploration":
    st.header("📊 Data Upload & Exploration")
    
    uploaded_file = st.file_uploader(
        "Upload your stock market CSV file", 
        type=['csv'],
        help="Expected format: CSV with Date, Open, High, Low, Close, Volume columns"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Display basic info about the dataset
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Date Range", f"{len(df)} days")
            
            # Show column names and types
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info)
            
            # Show first few rows
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
            
            # Date column selection
            st.subheader("Data Processing Configuration")
            date_column = st.selectbox(
                "Select Date Column", 
                df.columns,
                help="Choose the column containing date information"
            )
            
            target_column = st.selectbox(
                "Select Target Column for Analysis", 
                [col for col in df.columns if col != date_column],
                help="Choose the column to analyze (e.g., Close price)"
            )
            
            if st.button("Process Data"):
                try:
                    # Process the data
                    df_processed = df.copy()
                    df_processed[date_column] = pd.to_datetime(df_processed[date_column])
                    df_processed = df_processed.set_index(date_column)
                    df_processed = df_processed.sort_index()
                    
                    # Store in session state
                    st.session_state.data = df_processed
                    st.session_state.target_column = target_column
                    
                    st.success("✅ Data processed successfully!")
                    
                    # Show processed data stats
                    st.subheader("Processed Data Statistics")
                    st.dataframe(df_processed[target_column].describe())
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        # Show expected format
        st.subheader("Expected Data Format")
        sample_data = {
            'Date': ['2000-01-03', '2000-01-04', '2000-01-05'],
            'Symbol': ['HDFC', 'HDFC', 'HDFC'],
            'Open': [293.5, 317.0, 290.0],
            'High': [293.50, 317.00, 303.90],
            'Low': [293.5, 297.0, 285.0],
            'Close': [293.50, 304.05, 292.80],
            'Volume': [22744, 255251, 269087]
        }
        st.dataframe(pd.DataFrame(sample_data))

# Step 2: Data Visualization
elif selected_step == "📈 Data Visualization":
    st.header("📈 Data Visualization")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target_col = st.session_state.target_column
        
        # Time series plot
        st.subheader(f"Time Series Plot - {target_col}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[target_col],
            mode='lines',
            name=target_col,
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=f"{target_col} Over Time",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution")
            fig_hist = px.histogram(
                df, 
                x=target_col, 
                nbins=50,
                title=f"Distribution of {target_col}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            st.subheader("Monthly Statistics")
            monthly_stats = df[target_col].resample('M').agg(['mean', 'std', 'min', 'max'])
            st.dataframe(monthly_stats.round(2))
        
        # Trend analysis
        st.subheader("Trend Analysis")
        window = st.slider("Moving Average Window", 5, 100, 30)
        
        df_ma = df[target_col].rolling(window=window).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df.index,
            y=df[target_col],
            mode='lines',
            name='Original',
            line=dict(color='lightblue', width=1)
        ))
        fig_trend.add_trace(go.Scatter(
            x=df.index,
            y=df_ma,
            mode='lines',
            name=f'{window}-Day Moving Average',
            line=dict(color='red', width=2)
        ))
        
        fig_trend.update_layout(
            title=f"{target_col} with Moving Average",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
    else:
        st.warning("⚠️ Please upload and process data first in the 'Data Upload & Exploration' step.")

# Step 3: Stationarity Testing
elif selected_step == "🔍 Stationarity Testing":
    st.header("🔍 Stationarity Testing")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target_col = st.session_state.target_column
        
        st.subheader("What is Stationarity?")
        st.info("""
        A stationary time series has:
        - **Constant mean**: The average value doesn't change over time
        - **Constant variance**: The spread of data remains the same
        - **No seasonality**: No repeating patterns over fixed periods
        
        Most time series models (including ARIMA) require stationary data.
        """)
        
        # Perform ADF test on original data
        is_stationary = perform_adf_test(df[target_col], "Original Data - ADF Test")
        
        if is_stationary:
            st.success("🎉 Great! Your data is already stationary. You can proceed to model building.")
        else:
            st.warning("📈 Your data shows trend/non-stationarity. We'll need to apply differencing.")
            
            st.subheader("Visual Stationarity Check")
            
            # Plot with trend line
            fig = go.Figure()
            
            # Original data
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[target_col],
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1)
            ))
            
            # Add trend line
            x_numeric = np.arange(len(df))
            z = np.polyfit(x_numeric, df[target_col].dropna(), 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=p(x_numeric),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Time Series with Trend Line",
                xaxis_title="Date",
                yaxis_title=target_col,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("☝️ The trend line shows the overall direction of your data. A clear upward or downward trend indicates non-stationarity.")
        
    else:
        st.warning("⚠️ Please upload and process data first in the 'Data Upload & Exploration' step.")

# Step 4: Making Data Stationary
elif selected_step == "⚖️ Making Data Stationary":
    st.header("⚖️ Making Data Stationary")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        target_col = st.session_state.target_column
        
        st.subheader("Differencing Method")
        st.info("""
        **Differencing** removes trend by subtracting the previous value from the current value:
        - First difference: Y'(t) = Y(t) - Y(t-1)
        - Second difference: Y''(t) = Y'(t) - Y'(t-1)
        """)
        
        # Differencing options
        diff_order = st.selectbox("Select Differencing Order", [1, 2], 
                                help="Start with 1st order differencing")
        
        if st.button("Apply Differencing"):
            try:
                # Apply differencing
                df_diff = df[target_col].diff(periods=diff_order).dropna()
                st.session_state.df_diff = df_diff
                st.session_state.diff_order = diff_order
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Data")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=df.index,
                        y=df[target_col],
                        mode='lines',
                        name='Original',
                        line=dict(color='blue')
                    ))
                    fig1.update_layout(
                        title="Original Time Series",
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.subheader("Differenced Data")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df_diff.index,
                        y=df_diff,
                        mode='lines',
                        name=f'{diff_order}st Order Difference',
                        line=dict(color='green')
                    ))
                    fig2.update_layout(
                        title=f"After {diff_order}st Order Differencing",
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Test stationarity of differenced data
                st.subheader("Stationarity Test - Differenced Data")
                is_stationary = perform_adf_test(df_diff, f"After {diff_order}st Order Differencing")
                
                if is_stationary:
                    st.success(f"✅ Great! Data is now stationary after {diff_order}st order differencing.")
                else:
                    st.warning("⚠️ Data might need higher order differencing or other transformations.")
                
                # Show statistics comparison
                st.subheader("Statistics Comparison")
                stats_comparison = pd.DataFrame({
                    'Original Data': df[target_col].describe(),
                    'Differenced Data': df_diff.describe()
                })
                st.dataframe(stats_comparison.round(4))
                
            except Exception as e:
                st.error(f"Error applying differencing: {str(e)}")
        
        # Show existing differenced data if available
        if st.session_state.df_diff is not None:
            st.subheader("Current Differenced Data")
            diff_order = st.session_state.diff_order
            df_diff = st.session_state.df_diff
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Differencing Order", diff_order)
                st.metric("Data Points", len(df_diff))
            with col2:
                st.metric("Mean", f"{df_diff.mean():.6f}")
                st.metric("Std Dev", f"{df_diff.std():.6f}")
        
    else:
        st.warning("⚠️ Please upload and process data first in the 'Data Upload & Exploration' step.")

# Step 5: ACF/PACF Analysis
elif selected_step == "📊 ACF/PACF Analysis":
    st.header("📊 ACF/PACF Analysis")
    
    if st.session_state.df_diff is not None:
        df_diff = st.session_state.df_diff
        
        st.subheader("Autocorrelation Analysis")
        st.info("""
        **ACF (Autocorrelation Function)**: Shows correlation between observations at different time lags
        **PACF (Partial Autocorrelation Function)**: Shows correlation after removing effects of shorter lags
        
        These plots help determine the optimal parameters (p, q) for the ARIMA model.
        """)
        
        # Parameters for plots
        col1, col2 = st.columns(2)
        with col1:
            max_lags = st.slider("Maximum Lags to Display", 10, 50, 40)
        with col2:
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
        
        if st.button("Generate ACF/PACF Plots"):
            plot_acf_pacf(df_diff, lags=max_lags)
            
            # Parameter suggestion
            st.subheader("ARIMA Parameter Suggestions")
            
            try:
                # Simple heuristic for parameter suggestion
                acf_values = sm.tsa.acf(df_diff.dropna(), nlags=max_lags)
                pacf_values = sm.tsa.pacf(df_diff.dropna(), nlags=max_lags)
                
                # Find first significant lag (simplified approach)
                significance_level = 1.96 / np.sqrt(len(df_diff))  # 95% confidence
                
                # Suggest q (from ACF)
                q_suggest = 1
                for i in range(1, len(acf_values)):
                    if abs(acf_values[i]) < significance_level:
                        q_suggest = max(1, i-1)
                        break
                
                # Suggest p (from PACF) 
                p_suggest = 1
                for i in range(1, len(pacf_values)):
                    if abs(pacf_values[i]) < significance_level:
                        p_suggest = max(1, i-1)
                        break
                
                d_suggest = st.session_state.diff_order
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Suggested p (AR order)", min(p_suggest, 5))
                with col2:
                    st.metric("Suggested d (Differencing)", d_suggest)
                with col3:
                    st.metric("Suggested q (MA order)", min(q_suggest, 5))
                
                st.session_state.suggested_params = (min(p_suggest, 5), d_suggest, min(q_suggest, 5))
                
                st.info("💡 These are automated suggestions. You can also manually select parameters in the next step.")
                
            except Exception as e:
                st.warning(f"Could not generate parameter suggestions: {str(e)}")
        
    else:
        st.warning("⚠️ Please complete the differencing step first to generate ACF/PACF plots.")

# Step 6: ARIMA Model Building
elif selected_step == "🤖 ARIMA Model Building":
    st.header("🤖 ARIMA Model Building")
    
    if st.session_state.df_diff is not None and st.session_state.data is not None:
        df_diff = st.session_state.df_diff
        original_data = st.session_state.data[st.session_state.target_column]
        
        st.subheader("ARIMA Model Parameters")
        st.info("""
        **ARIMA(p,d,q) Parameters:**
        - **p**: Number of autoregressive terms (AR order)
        - **d**: Number of differencing operations (Integration order)  
        - **q**: Number of moving average terms (MA order)
        """)
        
        # Parameter selection
        col1, col2, col3 = st.columns(3)
        
        # Use suggested parameters if available
        default_params = getattr(st.session_state, 'suggested_params', (1, 1, 1))
        
        with col1:
            p = st.number_input("p (AR order)", min_value=0, max_value=10, value=default_params[0])
        with col2:
            d = st.number_input("d (Differencing)", min_value=0, max_value=3, value=default_params[1])
        with col3:
            q = st.number_input("q (MA order)", min_value=0, max_value=10, value=default_params[2])
        
        # Model training
        if st.button("Build ARIMA Model"):
            try:
                with st.spinner("Training ARIMA model..."):
                    # Build model
                    model = ARIMA(original_data, order=(p, d, q))
                    model_fit = model.fit()
                    
                    st.session_state.arima_model = model_fit
                    st.session_state.arima_params = (p, d, q)
                
                st.success(f"✅ ARIMA({p},{d},{q}) model trained successfully!")
                
                # Model summary
                st.subheader("Model Summary")
                
                # Extract key statistics
                aic = model_fit.aic
                bic = model_fit.bic
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{aic:.2f}")
                with col2:
                    st.metric("BIC", f"{bic:.2f}")
                with col3:
                    st.metric("Log Likelihood", f"{model_fit.llf:.2f}")
                
                # Model diagnostics
                st.subheader("Model Diagnostics")
                
                # Residuals analysis
                residuals = model_fit.resid
                
                fig_residuals = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Residuals Plot', 'Residuals Distribution', 
                                  'Q-Q Plot', 'ACF of Residuals'),
                    vertical_spacing=0.1
                )
                
                # Residuals over time
                fig_residuals.add_trace(
                    go.Scatter(x=residuals.index, y=residuals, mode='lines', name='Residuals'),
                    row=1, col=1
                )
                
                # Residuals histogram
                fig_residuals.add_trace(
                    go.Histogram(x=residuals, name='Distribution', nbinsx=30),
                    row=1, col=2
                )
                
                # Q-Q plot data
                qq_data = stats.probplot(residuals.dropna(), dist="norm")
                fig_residuals.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'),
                    row=2, col=1
                )
                fig_residuals.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0]*qq_data[0][0], 
                             mode='lines', name='Normal Line'),
                    row=2, col=1
                )
                
                # ACF of residuals
                residual_acf = sm.tsa.acf(residuals.dropna(), nlags=20)
                fig_residuals.add_trace(
                    go.Bar(x=list(range(len(residual_acf))), y=residual_acf, name='ACF'),
                    row=2, col=2
                )
                
                fig_residuals.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_residuals, use_container_width=True)
                
                # Model performance on training data
                st.subheader("Model Fit Visualization")
                
                # Get fitted values
                fitted_values = model_fit.fittedvalues
                
                fig_fit = go.Figure()
                
                # Original data
                fig_fit.add_trace(go.Scatter(
                    x=original_data.index,
                    y=original_data,
                    mode='lines',
                    name='Original Data',
                    line=dict(color='blue', width=1)
                ))
                
                # Fitted values
                fig_fit.add_trace(go.Scatter(
                    x=fitted_values.index,
                    y=fitted_values,
                    mode='lines',
                    name='ARIMA Fit',
                    line=dict(color='red', width=2)
                ))
                
                fig_fit.update_layout(
                    title=f"ARIMA({p},{d},{q}) Model Fit",
                    xaxis_title="Date",
                    yaxis_title=st.session_state.target_column,
                    height=500
                )
                
                st.plotly_chart(fig_fit, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error building ARIMA model: {str(e)}")
                st.info("💡 Try different parameter values or check if your data needs further preprocessing.")
        
        # Show existing model info if available
        if st.session_state.arima_model is not None:
            st.subheader("Current Model")
            params = st.session_state.arima_params
            st.info(f"ARIMA({params[0]},{params[1]},{params[2]}) model is ready for forecasting!")
    
    else:
        st.warning("⚠️ Please complete the data processing and differencing steps first.")

# Step 7: Predictions & Results
elif selected_step == "🎯 Predictions & Results":
    st.header("🎯 Predictions & Results")
    
    if st.session_state.arima_model is not None and st.session_state.data is not None:
        model_fit = st.session_state.arima_model
        original_data = st.session_state.data[st.session_state.target_column]
        params = st.session_state.arima_params
        
        st.subheader("Forecasting Configuration")
        
        # Forecasting parameters
        col1, col2 = st.columns(2)
        with col1:
            forecast_steps = st.number_input("Forecast Steps", min_value=1, max_value=100, value=30)
        with col2:
            confidence_level = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1)
        
        if st.button("Generate Forecast"):
            try:
                with st.spinner("Generating forecasts..."):
                    # Generate forecast
                    forecast_result = model_fit.forecast(steps=forecast_steps, alpha=1-confidence_level/100)
                    forecast_values = forecast_result
                    
                    # Get prediction intervals
                    prediction_result = model_fit.get_prediction(start=len(original_data)-50, 
                                                               end=len(original_data)+forecast_steps-1)
                    prediction_ci = prediction_result.conf_int()
                    
                    # Create future dates
                    last_date = original_data.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                               periods=forecast_steps, freq='D')
                    
                    # Forecasting visualization
                    st.subheader("Forecast Visualization")
                    
                    fig_forecast = go.Figure()
                    
                    # Historical data (last 100 points for clarity)
                    recent_data = original_data.tail(100)
                    fig_forecast.add_trace(go.Scatter(
                        x=recent_data.index,
                        y=recent_data,
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Forecast
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast_values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Confidence intervals (if available)
                    forecast_ci = None
                    try:
                        forecast_ci = model_fit.get_forecast(steps=forecast_steps, alpha=1-confidence_level/100).conf_int()
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast_ci.iloc[:, 1],  # Upper bound
                            mode='lines',
                            name=f'{confidence_level}% Upper CI',
                            line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
                            showlegend=False
                        ))
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast_ci.iloc[:, 0],  # Lower bound
                            mode='lines',
                            name=f'{confidence_level}% Lower CI',
                            line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)',
                            showlegend=True
                        ))
                    except:
                        pass
                    
                    fig_forecast.update_layout(
                        title=f"ARIMA({params[0]},{params[1]},{params[2]}) Forecast",
                        xaxis_title="Date",
                        yaxis_title=st.session_state.target_column,
                        height=600
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast table
                    st.subheader("Forecast Values")
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Forecast': forecast_values
                    })
                    
                    try:
                        if 'forecast_ci' in locals() and forecast_ci is not None:
                            forecast_df['Lower_CI'] = forecast_ci.iloc[:, 0].values
                            forecast_df['Upper_CI'] = forecast_ci.iloc[:, 1].values
                    except:
                        pass
                    
                    st.dataframe(forecast_df.round(2))
                    
                    # Model performance metrics
                    st.subheader("Model Performance")
                    
                    # Calculate metrics on training data
                    fitted_values = model_fit.fittedvalues
                    residuals = original_data - fitted_values
                    
                    mae = np.mean(np.abs(residuals.dropna()))
                    mse = np.mean(residuals.dropna()**2)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs(residuals.dropna() / original_data.loc[residuals.dropna().index])) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{mae:.2f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.2f}")
                    with col3:
                        st.metric("MAPE (%)", f"{mape:.2f}")
                    with col4:
                        st.metric("AIC", f"{model_fit.aic:.2f}")
                    
                    # Export functionality
                    st.subheader("Export Results")
                    
                    # Prepare export data
                    export_data = {
                        'model_params': params,
                        'forecast_steps': forecast_steps,
                        'forecast_values': forecast_values.tolist(),
                        'forecast_dates': future_dates.strftime('%Y-%m-%d').tolist(),
                        'model_metrics': {
                            'AIC': model_fit.aic,
                            'BIC': model_fit.bic,
                            'MAE': mae,
                            'RMSE': rmse,
                            'MAPE': mape
                        }
                    }
                    
                    st.download_button(
                        label="Download Forecast Results (CSV)",
                        data=forecast_df.to_csv(index=False),
                        file_name=f"arima_forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("💡 Check your model parameters and data quality.")
    
    else:
        st.warning("⚠️ Please build an ARIMA model first in the previous step.")

# Footer
st.markdown("---")
st.markdown("""
### 📚 About This Application

This application implements a complete time series forecasting workflow using ARIMA models:

1. **Data Exploration**: Upload and explore your stock market data
2. **Stationarity Testing**: Check if data is stationary using ADF test
3. **Data Transformation**: Apply differencing to make data stationary
4. **Parameter Selection**: Use ACF/PACF plots to determine optimal parameters
5. **Model Building**: Train ARIMA model with selected parameters
6. **Forecasting**: Generate future predictions with confidence intervals

**ARIMA Model**: AutoRegressive Integrated Moving Average model is suitable for time series forecasting when data shows trends but no strong seasonality.

Built with ❤️ using Streamlit, Statsmodels, and Plotly.
""")
