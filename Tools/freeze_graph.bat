chcp 65001
cd /d "%~dp0"

FOR %%i IN (%*) DO python freeze_graph.py --input_binary False --input_graph="%%~dpni.graphdef" --input_checkpoint="%%~dpni" --output_graph="%%~dpni.pb" --output_node_names=Output

pause
exit

FOR %%i IN (%*) DO python freeze_graph.py --input_binary True --input_meta_graph="%%~dpni.meta" --input_checkpoint="%%~dpni" --output_graph="%%~dpni.pb" --output_node_names=Output
