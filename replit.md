# Overview

This project is a comprehensive time series forecasting application built with Streamlit for analyzing and predicting stock market trends using ARIMA (AutoRegressive Integrated Moving Average) models. The application provides an interactive interface for users to upload financial data, perform stationarity testing, visualize trends, and generate forecasts through a step-by-step guided workflow.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application framework for rapid prototyping and data science applications
- **Layout**: Wide layout configuration optimized for data visualization and charts
- **Navigation**: Sidebar-based step-by-step workflow with 7 distinct analysis phases
- **State Management**: Streamlit session state for maintaining data persistence across user interactions

## Application Structure
- **Single-page Application**: Monolithic app.py structure with conditional rendering based on user navigation
- **Workflow-driven Design**: Sequential analysis steps from data upload through final predictions
- **Interactive Components**: File upload, parameter selection, and real-time chart generation

## Data Processing Pipeline
- **Data Ingestion**: Support for CSV/Excel file uploads through Streamlit file uploader
- **Time Series Analysis**: Multi-stage pipeline including:
  - Data exploration and visualization
  - Stationarity testing using Augmented Dickey-Fuller (ADF) test
  - Data transformation through differencing to achieve stationarity
  - Autocorrelation and Partial Autocorrelation Function (ACF/PACF) analysis
  - ARIMA model parameter optimization and fitting
  - Forecasting and prediction visualization

## Visualization Strategy
- **Primary Library**: Plotly for interactive charts with both Graph Objects and Express APIs
- **Secondary Library**: Matplotlib for statistical plots and compatibility
- **Chart Types**: Time series plots, correlation plots, residual analysis, and forecast visualizations
- **Interactivity**: Zoom, pan, hover tooltips, and dynamic updates

## Statistical Computing
- **Core Library**: Statsmodels for time series analysis and ARIMA implementation
- **Supporting Libraries**: 
  - NumPy for numerical computations
  - Pandas for data manipulation and time series handling
  - SciPy for additional statistical functions
- **Model Architecture**: ARIMA models with automated parameter selection and diagnostic testing

# External Dependencies

## Python Libraries
- **streamlit**: Web application framework and user interface
- **pandas**: Data manipulation and time series data structures
- **numpy**: Numerical computing and array operations
- **matplotlib**: Static plotting and visualization fallback
- **plotly**: Interactive visualization and charting
- **statsmodels**: Time series analysis, ARIMA modeling, and statistical tests
- **scipy**: Statistical functions and probability distributions

## Data Sources
- **File Upload System**: Local file support for CSV and Excel formats
- **Time Series Data**: Expected format with date/timestamp and numerical value columns
- **Stock Market Data**: Primary use case for financial time series analysis

## Deployment Requirements
- **Python Runtime**: Python 3.7+ environment
- **Memory**: Sufficient RAM for time series data processing and model fitting
- **Processing**: CPU resources for statistical computations and model training