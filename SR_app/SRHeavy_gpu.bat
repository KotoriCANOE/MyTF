cd /d "%~dp0"

python SR_app.py --model_dir ./SRHeavy_gpu --data_format NCHW --src_dir ./src --dst_dir ./dst --dst_postfix .SRHeavy --patch_height 360 --patch_width 360

pause
