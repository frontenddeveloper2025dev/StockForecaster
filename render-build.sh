#!/bin/bash
# Build script for Render deployment

echo "🚀 Starting ARIMA Dashboard deployment..."

# Update system packages
echo "📦 Updating system packages..."

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r render-requirements.txt

echo "✅ Dependencies installed successfully!"
echo "🎯 ARIMA Financial AI Dashboard ready for deployment"