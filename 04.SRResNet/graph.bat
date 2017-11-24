cd /d "%~dp0"

FOR %%i IN (1003) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se False
FOR %%i IN (1003) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se False

pause

FOR %%i IN (199) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_gpu --data_format NCHW
FOR %%i IN (199) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_cpu --data_format NHWC

FOR %%i IN (201) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW
FOR %%i IN (201) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC
