#!/bin/bash
# Start script for Render deployment

echo "🎯 Starting ARIMA Financial AI Dashboard..."

# Start Streamlit app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false