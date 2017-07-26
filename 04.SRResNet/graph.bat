cd /d "%~dp0"

FOR %%i IN (62) DO python SRResNet_graph.py --postfix %%i

pause
