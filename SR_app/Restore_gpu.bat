cd /d "%~dp0"

python SR_app.py --model_dir ./Restore_gpu --data_format NCHW --scaling 1 --src_dir ./src --dst_dir ./dst --dst_postfix .Restore --patch_height 2000 --patch_width 1200

pause
