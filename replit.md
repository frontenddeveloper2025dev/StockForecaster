# Overview

ARIMA Financial AI Dashboard is a comprehensive financial time series analysis and forecasting application built with Streamlit. The dashboard provides advanced machine learning capabilities for stock market analysis, including ARIMA modeling, Prophet forecasting, LSTM neural networks, and sentiment analysis. It features a gamified user experience with portfolio management, interactive visualizations, and multi-symbol analysis capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework providing rapid development and deployment for data science applications
- **Layout**: Wide layout configuration optimized for financial data visualization and interactive charts
- **Navigation**: Sidebar-based sequential workflow with 7 distinct analysis phases from data ingestion to prediction
- **State Management**: Streamlit session state for maintaining user data and analysis results across interactions
- **Theming**: Custom ARIMA Orange branding with personalized logo and color scheme

## Application Structure
- **Monolithic Design**: Single app.py file containing all application logic with conditional rendering based on user navigation
- **Workflow-driven Interface**: Step-by-step analysis pipeline guiding users through complete financial analysis process
- **Interactive Components**: File upload capabilities, parameter selection widgets, real-time chart generation, and gamification elements
- **Multi-model Support**: Graceful fallbacks when advanced ML libraries (TensorFlow, Prophet) are unavailable

## Data Processing Pipeline
- **Multi-source Data Ingestion**: Support for 7 different financial data sources including Yahoo Finance and Tiingo API
- **Time Series Analysis Pipeline**:
  - Data exploration and statistical visualization
  - Stationarity testing using Augmented Dickey-Fuller (ADF) test
  - Data transformation through differencing to achieve stationarity
  - Autocorrelation and Partial Autocorrelation Function (ACF/PACF) analysis
  - ARIMA model parameter optimization with automated selection
  - Advanced forecasting with Prophet and LSTM models (when available)
  - Model performance evaluation and comparison

## Visualization and Analytics
- **Primary Visualization**: Plotly for interactive financial charts with both Graph Objects and Express APIs
- **Chart Types**: Time series plots, correlation matrices, heat maps, residual analysis, and multi-model forecast comparisons
- **Advanced Features**: Portfolio analysis, sentiment analysis with emoji indicators, achievement tracking system
- **Performance Metrics**: Comprehensive model evaluation using MSE, MAE, and statistical diagnostics

## Deployment Architecture
- **Platform**: Optimized for Render cloud deployment with containerized approach
- **Build System**: Multi-stage build process with dependency optimization for cloud environments
- **Environment Management**: Flexible configuration supporting both full-featured and lightweight deployments
- **Error Handling**: Robust fallback mechanisms for missing dependencies and API failures

# External Dependencies

## Core Python Libraries
- **streamlit**: Web application framework and user interface rendering
- **pandas**: Financial data manipulation and time series data structures
- **numpy**: Numerical computations and array operations
- **plotly**: Interactive financial visualizations and charting
- **statsmodels**: ARIMA modeling and statistical time series analysis

## Advanced Machine Learning (Optional)
- **tensorflow**: LSTM neural network implementation for deep learning forecasting
- **prophet**: Facebook's time series forecasting library for trend analysis
- **scikit-learn**: Machine learning utilities and model evaluation metrics
- **textblob**: Natural language processing for sentiment analysis

## Financial Data APIs
- **yfinance**: Yahoo Finance API for stock market data retrieval
- **alpha-vantage**: Alpha Vantage API for additional financial data sources
- **tiingo**: Professional financial data API (primary data source)

## Supporting Libraries
- **matplotlib**: Statistical plotting and chart generation
- **seaborn**: Statistical data visualization and correlation analysis
- **requests**: HTTP client for API communications
- **scipy**: Scientific computing and statistical functions
- **websockets**: Real-time data streaming capabilities

## Deployment Dependencies
- **Cloud Platform**: Render web service for application hosting
- **Environment Variables**: API key management for financial data services
- **Build Tools**: Bash scripts for optimized cloud deployment and dependency management