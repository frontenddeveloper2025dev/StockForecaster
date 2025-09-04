# ⚡ Deployment Rápido - ARIMA Dashboard

## 🎯 Pasos Simples para Render

### 1️⃣ Preparar archivos
✅ **Renombrar antes de subir a GitHub:**
- `render.txt` → `requirements.txt`

### 2️⃣ Configuración en Render
```
Build Command: bash render-build.sh
Start Command: bash render-start.sh
Environment: Python 3.11
```

### 3️⃣ Variables de entorno obligatorias
```
TIINGO_API_KEY=tu_clave_aqui
```

### 4️⃣ ¡Listo! 🚀
Tu dashboard estará disponible en la URL que te proporcione Render.

---

## 📱 Tu Dashboard incluye:
- 📊 Análisis multi-símbolo
- 🤖 IA: ARIMA, Prophet, LSTM
- 🏆 Sistema gamificado
- 🎯 Análisis de sentimiento
- 🗺️ Mapas de calor interactivos
- 🎨 Branding ARIMA Orange personalizado

## 🔑 APIs necesarias:
- **Tiingo API**: Obligatoria (datos financieros)
- **Alpha Vantage**: Opcional (fuente adicional)

## 🛠️ Soporte técnico:
Si tienes problemas con el deployment, revisa:
1. Las claves de API están configuradas
2. Los archivos tienen los permisos correctos
3. El puerto está configurado como variable de entorno