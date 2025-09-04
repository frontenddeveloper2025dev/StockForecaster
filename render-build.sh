#!/bin/bash
# Build script for Render deployment

echo "🚀 Starting ARIMA Dashboard deployment..."

# Install Python dependencies
pip install --upgrade pip
pip install -r render.txt

echo "✅ Dependencies installed successfully!"
echo "🎯 ARIMA Financial AI Dashboard ready for deployment"