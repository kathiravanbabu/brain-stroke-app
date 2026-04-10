@echo off
echo =======================================
echo   Brain Stroke Prediction App
echo =======================================
echo.
echo Starting Streamlit... please wait.
echo Your browser will open automatically.
echo.
cd /d "%~dp0"
streamlit run app.py
pause
