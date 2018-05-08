cd /d "%~dp0"


pause

FOR %%i IN (220) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW --use_se 1 --g_depth 8 --channels 64 --activation swish
FOR %%i IN (220) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC --use_se 1 --g_depth 8 --channels 64 --activation swish
FOR %%i IN (223) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW --use_se 1 --g_depth 8 --channels 64 --activation swish
FOR %%i IN (223) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC --use_se 1 --g_depth 8 --channels 64 --activation swish
FOR %%i IN (224) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW --use_se 1 --g_depth 16 --channels 64 --activation swish
FOR %%i IN (224) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC --use_se 1 --g_depth 16 --channels 64 --activation swish
FOR %%i IN (225) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_gpu --data_format NCHW --use_se 1 --g_depth 8 --channels 64 --activation swish
FOR %%i IN (225) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_cpu --data_format NHWC --use_se 1 --g_depth 8 --channels 64 --activation swish

FOR %%i IN (1020) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish
FOR %%i IN (1020) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish
FOR %%i IN (1021) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish
FOR %%i IN (1021) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish

FOR %%i IN (1054) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish
FOR %%i IN (1054) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish
FOR %%i IN (1055) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish
FOR %%i IN (1055) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish

pause

FOR %%i IN (199) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_gpu --data_format NCHW
FOR %%i IN (199) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRMedium_cpu --data_format NHWC

FOR %%i IN (201) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW
FOR %%i IN (201) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC

FOR %%i IN (207) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_gpu --data_format NCHW --k_resize 5
FOR %%i IN (207) DO python graph.py --postfix %%i --graph_dir ./model%%i/SRHeavy_cpu --data_format NHWC --k_resize 5

FOR %%i IN (1014) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish
FOR %%i IN (1014) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish

FOR %%i IN (1016) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_gpu --data_format NCHW --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish
FOR %%i IN (1016) DO python graph.py --postfix %%i --graph_dir ./model%%i/Restore_cpu --data_format NHWC --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish
