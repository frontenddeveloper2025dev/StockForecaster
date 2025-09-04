#!/bin/bash
set -e

echo "🚀 Iniciando ARIMA Financial AI Dashboard"

# Variables de entorno
export PORT=${PORT:-10000}
export STREAMLIT_SERVER_PORT=$PORT
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "🌐 Puerto configurado: $PORT"
echo "📡 Dirección: $STREAMLIT_SERVER_ADDRESS"

# Iniciar Streamlit
exec streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.fileWatcherType=none \
  --browser.gatherUsageStats=false