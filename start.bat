@echo off
echo.
echo ===============================================
echo       IntelCLaw - Autonomous AI Agent
echo ===============================================
echo.

:: Check if .env exists
if not exist ".env" (
    echo [!] No .env file found. Running onboarding...
    echo.
    uv run python scripts/onboarding.py
    echo.
)

:: Run the application
echo [*] Starting IntelCLaw...
echo [*] Press Ctrl+Shift+Space to summon the agent
echo.
uv run python main.py %*
