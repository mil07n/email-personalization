@echo off
title NLP Executive Email Assistant
color 0A
echo.
echo  ================================================================
echo   NLP-Powered Executive Email Assistant - MailMind AI
echo  ================================================================
echo.
echo  [1] Starting backend server on http://127.0.0.1:8000
echo  [2] Open your browser to: http://127.0.0.1:8000
echo.
echo  DO NOT open http://0.0.0.0:7860 — that was the old Gradio app.
echo.
cd /d "%~dp0"
python backend\main.py
pause
