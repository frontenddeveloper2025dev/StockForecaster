# 🚀 ARIMA Financial AI Dashboard - Deployment Guide

## 📋 Pre-requisitos para Render

### Archivos principales:
- `app.py` - Aplicación principal Streamlit
- `render.txt` - Dependencias Python (renombrar a requirements.txt en Render)
- `render-build.sh` - Script de construcción
- `render-start.sh` - Script de inicio
- `render.yaml` - Configuración de Render
- `.streamlit/config.toml` - Configuración Streamlit
- `logo.png` - Logo personalizado

## 🔧 Instrucciones de Deployment en Render

### 1. Subir a GitHub/GitLab
```bash
git init
git add .
git commit -m "ARIMA Financial Dashboard"
git push origin main
```

### 2. Configurar en Render
1. Crear nuevo Web Service
2. Conectar repositorio
3. Configurar:
   - **Build Command**: `./render-build.sh`
   - **Start Command**: `./render-start.sh`
   - **Environment**: Python 3.11+

### 3. Variables de entorno
```
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=10000
STREAMLIT_SERVER_ADDRESS=0.0.0.0
TIINGO_API_KEY=tu_api_key_aqui
ALPHA_VANTAGE_API_KEY=tu_api_key_aqui (opcional)
```

### 4. Renombrar archivos en Render
- `render.txt` → `requirements.txt`

## 🎯 Características del Dashboard

✅ Multi-symbol portfolio analysis
✅ Advanced forecasting (ARIMA, Prophet, LSTM)
✅ Performance metrics & gamification
✅ Sentiment analysis with emojis
✅ Interactive heat maps
✅ AI learning modules
✅ Custom ARIMA Orange theme
✅ Real-time data streaming

## 🔑 APIs Requeridas

- **Tiingo API**: Para datos financieros y streaming
- **Alpha Vantage**: Opcional, fuente adicional de datos

## 🎨 Branding

- Tema personalizado ARIMA Orange (#FF6B35)
- Logo integrado como favicon
- Gradientes y efectos visuales personalizados

## ⚡ Performance

- Optimizado para deployment en la nube
- Cache de datos implementado
- Configuración headless para servidores