#!/bin/bash
# Start script for Render deployment

echo "🎯 Starting ARIMA Financial AI Dashboard..."

# Export environment variables
export STREAMLIT_SERVER_PORT=${PORT:-10000}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start Streamlit app
streamlit run app.py \
  --server.port=$STREAMLIT_SERVER_PORT \
  --server.address=$STREAMLIT_SERVER_ADDRESS \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false