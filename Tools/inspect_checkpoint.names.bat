chcp 65001
cd /d "%~dp0"

FOR %%i IN (%*) DO python inspect_checkpoint.py --file_name %%~dpni

pause
