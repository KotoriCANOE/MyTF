cd /d "%~dp0"

python SR_app.py --model_dir ./SRHeavy_gpu --data_format NCHW --scaling 2 --src_dir ./src --dst_dir ./dst --dst_postfix .SRHeavy --patch_width 1024 --patch_height 1024

pause
