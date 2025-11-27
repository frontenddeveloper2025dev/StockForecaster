# ARIMA Financial AI Dashboard

## Overview

This is a financial analytics dashboard built with Streamlit that provides ARIMA-based time series forecasting and analysis for financial data. The application focuses on stock market analysis using various statistical and machine learning models, with support for advanced forecasting techniques including Prophet and LSTM neural networks when available.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework for rapid data app development
- **UI Components**: Wide layout with expandable sidebar for enhanced user experience
- **Visualization**: Plotly for interactive charts and graphs, with Seaborn integration for statistical plots
- **Styling**: Custom favicon support with responsive design

### Data Processing & Analytics Engine
- **Core Analytics**: NumPy and Pandas for data manipulation and statistical computations
- **Time Series Analysis**: 
  - Primary: ARIMA models using statsmodels for forecasting
  - Advanced: Prophet for trend decomposition and seasonality analysis (optional)
  - ML-based: LSTM neural networks via TensorFlow/Keras for deep learning forecasting (optional)
- **Model Evaluation**: Scikit-learn metrics for performance assessment (MSE, MAE)
- **Statistical Testing**: Augmented Dickey-Fuller test for stationarity analysis

### Data Sources & Integration
- **Primary Data Provider**: Yahoo Finance via yfinance library for real-time and historical stock data
- **Secondary Source**: Alpha Vantage API for additional financial data and validation
- **Data Preprocessing**: MinMaxScaler for neural network input normalization

### Optional Features (Graceful Degradation)
- **Sentiment Analysis**: TextBlob integration for market sentiment evaluation when available
- **Advanced ML Models**: Prophet and TensorFlow models with fallback handling
- **Error Handling**: Comprehensive warning suppression and graceful feature degradation

### Application Configuration
- **Deployment**: Configured for wide-screen dashboard experience
- **Error Management**: Warnings filtered to provide clean user experience
- **Performance**: Optimized imports with conditional loading for heavy dependencies

## External Dependencies

### Core Financial Data APIs
- **Yahoo Finance (yfinance)**: Primary data source for stock prices, trading volumes, and market data
- **Alpha Vantage API**: Secondary financial data provider requiring API key configuration

### Machine Learning & Analytics Libraries
- **Statsmodels**: Statistical analysis and ARIMA modeling capabilities
- **Scikit-learn**: Model evaluation metrics and data preprocessing utilities
- **TensorFlow/Keras**: Optional deep learning framework for LSTM time series forecasting
- **Prophet**: Optional Facebook's time series forecasting tool for trend analysis

### Visualization & UI Framework
- **Streamlit**: Web application framework and deployment platform
- **Plotly**: Interactive charting and visualization library
- **Matplotlib/Seaborn**: Statistical plotting and data visualization support

### Data Processing Stack
- **Pandas**: Data manipulation and analysis framework
- **NumPy**: Numerical computing foundation
- **SciPy**: Scientific computing utilities for statistical operations

### Utility Dependencies
- **Requests**: HTTP client for API communications
- **WebSockets**: Real-time data streaming capabilities
- **TextBlob**: Optional natural language processing for sentiment analysis
