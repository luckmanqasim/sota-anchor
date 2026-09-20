: << 'CMDBLOCK'
@echo off
REM Cross-platform polyglot launcher for sota-anchor hook scripts.
REM On Windows cmd.exe runs the batch half, which locates bash and calls it.
REM On Unix the shell treats this as a script (: is a no-op) and falls through.
REM
REM Pattern adapted from the superpowers plugin (MIT), including its two
REM hard-won details: hook scripts use extensionless names because Claude Code
REM on Windows prepends "bash" to any command containing .sh, and a missing
REM bash exits 0 rather than erroring, so a seeding hook can never be the
REM reason a session fails to start.
REM
REM Usage: run-hook.cmd <script-name> [args...]

if "%~1"=="" (
    echo run-hook.cmd: missing script name >&2
    exit /b 1
)

set "HOOK_DIR=%~dp0"

if exist "C:\Program Files\Git\bin\bash.exe" (
    "C:\Program Files\Git\bin\bash.exe" "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)
if exist "C:\Program Files (x86)\Git\bin\bash.exe" (
    "C:\Program Files (x86)\Git\bin\bash.exe" "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

where bash >nul 2>nul
if %ERRORLEVEL% equ 0 (
    bash "%HOOK_DIR%%~1" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

REM No bash available: degrade silently. The plugin still works, the session
REM just starts without the injected model landscape.
exit /b 0
CMDBLOCK

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_NAME="$1"
shift
exec bash "${SCRIPT_DIR}/${SCRIPT_NAME}" "$@"
