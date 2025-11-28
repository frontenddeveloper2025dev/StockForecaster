# ARIMA Financial AI Dashboard

# 📈 ARIMA Financial AI Dashboard

**ARIMA Financial AI Dashboard** es una aplicación web de analítica financiera desarrollada con **Streamlit**, que permite realizar pronósticos de series temporales del mercado bursátil utilizando modelos estadísticos y de aprendizaje automático. Ofrece herramientas interactivas para el análisis de acciones, evaluaciones de rendimiento y visualización de datos, todo desde una interfaz sencilla y responsiva.

---

## 🖼️ Vistas Previas

| Dashboard Principal | Predicción ARIMA | Modelos Alternativos |
|---------------------|------------------|----------------------|
| ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%201.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster2%20.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster3.png) |

| Comparación de Modelos | Resultados ML | Configuración ARIMA |
|------------------------|---------------|----------------------|
| ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%204.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%205.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%206.png) |

| Resultados Prophet | Métricas de Evaluación | Gráficas Interactivas |
|--------------------|------------------------|------------------------|
| ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%207.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%2013.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%2015.png) |

| Análisis Final | Exportación |
|----------------|-------------|
| ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster%2016.png) | ![](https://github.com/frontenddeveloper2025dev/StockForecaster/blob/main/StockForecaster17.png) |

---

## 🔍 ¿Qué Puedes Hacer?

- Analizar precios históricos de acciones y volúmenes de trading
- Aplicar modelos ARIMA para pronóstico de series temporales
- Probar modelos avanzados como Prophet o redes neuronales LSTM (opcional)
- Evaluar el rendimiento de los modelos con métricas como MSE y MAE
- Visualizar datos con gráficas interactivas y comparativas
- Descargar resultados y exportar datos analizados

---

## ⚙️ Arquitectura del Sistema

### 🔹 Frontend – Interfaz de Usuario
- **Framework:** Streamlit
- **Diseño:** Layout amplio y responsivo con sidebar expandible
- **Visualizaciones:** Plotly para gráficas interactivas, Seaborn para gráficos estadísticos
- **Estilo:** Soporte para favicon y diseño limpio

### 🔹 Backend – Procesamiento & Modelado
- **Manejo de Datos:** `pandas`, `numpy`
- **Análisis de Series Temporales:**
  - Principal: `ARIMA` con `statsmodels`
  - Opcional: `Prophet` (tendencias y estacionalidades)
  - Opcional: `LSTM` (forecasting profundo con `TensorFlow/Keras`)
- **Evaluación de Modelos:** `scikit-learn` (MSE, MAE)
- **Test Estadístico:** Augmented Dickey-Fuller para verificar estacionariedad
- **Normalización:** `MinMaxScaler` para redes neuronales

---

## 🌐 Fuentes de Datos

- **Yahoo Finance (`yfinance`)**: Fuente principal para precios y volúmenes históricos
- **Alpha Vantage API**: Fuente secundaria (requiere API Key)

---

## 🧠 Funcionalidades Avanzadas (Opcionales)

- **Análisis de Sentimiento**: con `TextBlob` (cuando disponible)
- **Modelos Avanzados:** `Prophet` y `LSTM`, con degradación elegante si no están activos
- **Control de Errores:** Manejo silencioso de warnings y fallbacks automáticos

---

## 📦 Dependencias Principales

### 🔹 Data y Estadística
- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `scikit-learn`

### 🔹 Visualización
- `plotly`
- `matplotlib`
- `seaborn`
- `streamlit`

### 🔹 ML y Forecasting (Opcionales)
- `tensorflow` / `keras`
- `prophet`

### 🔹 Integración y Utilidades
- `yfinance`
- `requests`
- `websockets`
- `textblob`

---

## 🚀 Cómo Ejecutar la App

1. Clona este repositorio:

```bash
git clone https://github.com/frontenddeveloper2025dev/StockForecaster.git
cd StockForecaster
Instala las dependencias:

pip install -r requirements.txt


### Ejecuta la app con Streamlit:

streamlit run app.py


### Asegúrate de tener conexión a internet para acceder a los datos de Yahoo Finance.

### 🛠️ Consideraciones

Si no tienes Prophet o TensorFlow instalados, la app sigue funcionando usando solo ARIMA.

La API de Alpha Vantage es opcional. Puedes configurar tu clave si deseas usarla como respaldo.

Algunas funciones avanzadas están desactivadas por defecto para optimizar el rendimiento.

### 👩‍💻 Autora

 Como proyecto de análisis financiero con enfoque en pronóstico estadístico y visualización interactiva.

¿Te gusta este proyecto? ¡Dale ⭐ en GitHub y compártelo!
