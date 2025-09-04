#!/bin/bash
set -e

echo "🔧 Build iniciado para ARIMA Dashboard"

# Verificar Python
python --version

# Upgrade pip sin timeout
echo "📦 Actualizando pip..."
python -m pip install --upgrade pip --no-cache-dir

# Instalar dependencias básicas primero
echo "🚀 Instalando dependencias básicas..."
pip install --no-cache-dir streamlit==1.49.1
pip install --no-cache-dir pandas==2.3.2
pip install --no-cache-dir numpy==2.3.2
pip install --no-cache-dir plotly==6.3.0

echo "📊 Instalando librerías de análisis..."
pip install --no-cache-dir statsmodels==0.14.5
pip install --no-cache-dir yfinance==0.2.65
pip install --no-cache-dir scikit-learn==1.7.1

echo "🎨 Instalando visualización..."
pip install --no-cache-dir matplotlib==3.10.6
pip install --no-cache-dir seaborn==0.13.2

echo "🌐 Instalando utilities..."
pip install --no-cache-dir requests==2.32.5
pip install --no-cache-dir alpha-vantage==3.0.0
pip install --no-cache-dir websockets==15.0.1

echo "✅ Build completado exitosamente!"