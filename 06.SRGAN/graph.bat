cd /d "%~dp0"

FOR %%i IN (21) DO python graph.py --postfix %%i --graph_dir ./graph_gpu.tmp --data_format NCHW
FOR %%i IN (21) DO python graph.py --postfix %%i --graph_dir ./graph_cpu.tmp --data_format NHWC

pause
