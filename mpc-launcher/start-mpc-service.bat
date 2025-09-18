@echo off
echo 🎬 Starting MPC Launcher Service...
echo.
echo This service allows direct MPC launching from React frontend
echo Keep this window open while using the video retrieval system
echo.
echo Press Ctrl+C to stop the service
echo.

cd /d "%~dp0"

if not exist node_modules (
    echo 📦 Installing dependencies...
    npm install
    echo.
)

echo 🚀 Starting service on http://127.0.0.1:3001
npm start

pause