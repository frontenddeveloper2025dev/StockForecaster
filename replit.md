# Overview

ARIMA Financial AI Dashboard is a comprehensive financial time series forecasting application built with Streamlit. The application provides advanced stock market analysis and prediction capabilities using multiple machine learning models including ARIMA, Prophet, and LSTM neural networks. It features a gamified user experience with interactive visualizations, sentiment analysis, and multi-symbol portfolio management designed for both novice and experienced financial analysts.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with wide layout optimization for financial data visualization
- **Navigation**: Sidebar-based workflow with sequential analysis phases from data upload through predictions
- **State Management**: Streamlit session state for maintaining user data and analysis progress across interactions
- **Custom Branding**: ARIMA Orange theme with custom logo and favicon integration
- **Responsive Design**: Multi-column layouts optimized for dashboard-style financial data presentation

## Data Processing Pipeline
- **Multi-source Data Ingestion**: 
  - File upload support (CSV/Excel) through Streamlit components
  - Real-time financial data via Yahoo Finance (yfinance) and Alpha Vantage APIs
  - Tiingo API integration for enhanced market data coverage
- **Time Series Processing**: Sequential pipeline including data exploration, stationarity testing (ADF), differencing transformations, and ACF/PACF analysis
- **Model Training Pipeline**: Automated parameter optimization for ARIMA models with diagnostic testing and validation

## Machine Learning Architecture
- **Primary Models**: ARIMA (Auto-Regressive Integrated Moving Average) using statsmodels
- **Advanced Models**: 
  - Prophet for trend and seasonality decomposition (optional dependency)
  - LSTM neural networks via TensorFlow/Keras for deep learning predictions (optional)
- **Model Selection**: Automated parameter tuning with grid search and information criteria (AIC/BIC)
- **Validation Framework**: Train/test splits with performance metrics (MSE, MAE) and residual analysis

## Visualization and Analytics
- **Primary Visualization**: Plotly Graph Objects and Express for interactive financial charts
- **Supporting Libraries**: Matplotlib and Seaborn for statistical plots and heatmaps
- **Chart Types**: Time series plots, correlation matrices, volatility analysis, and forecast confidence intervals
- **Gamification Elements**: Performance scoring, achievement badges, and progress tracking
- **Sentiment Integration**: TextBlob-based sentiment analysis with emoji representations

## Error Handling and Fallbacks
- **Graceful Degradation**: Optional imports with fallback handling for Prophet, TensorFlow, and TextBlob
- **API Resilience**: Multiple data source options with automatic failover between providers
- **Validation Pipeline**: Comprehensive data quality checks and user feedback for invalid inputs

# External Dependencies

## Core Financial Data APIs
- **Tiingo API**: Primary market data provider for real-time and historical financial data (required)
- **Alpha Vantage API**: Secondary financial data source for enhanced coverage (optional)
- **Yahoo Finance (yfinance)**: Backup data source for broad market access

## Machine Learning and Analytics
- **statsmodels**: Time series analysis, ARIMA modeling, and statistical testing
- **scikit-learn**: Model evaluation metrics, preprocessing, and validation tools
- **TensorFlow/Keras**: Deep learning framework for LSTM neural network implementations (optional)
- **Prophet**: Facebook's time series forecasting library for trend analysis (optional)
- **NumPy/Pandas**: Numerical computing and data manipulation backbone
- **SciPy**: Advanced statistical functions and optimization algorithms

## Visualization and UI
- **Streamlit**: Web application framework and user interface components
- **Plotly**: Interactive charting and financial visualization library
- **Matplotlib**: Statistical plotting and chart generation
- **Seaborn**: Enhanced statistical visualization and heatmap generation

## Natural Language Processing
- **TextBlob**: Sentiment analysis for market sentiment integration (optional)

## Deployment Infrastructure
- **Render Platform**: Cloud deployment with automated build and start scripts
- **Environment Configuration**: Custom Streamlit server settings for production deployment
- **Build Pipeline**: Bash scripts for dependency installation and application startup