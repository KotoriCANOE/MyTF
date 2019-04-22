cd /d "%~dp0"

python SR_app.py --model_dir ./Restore_gpu1 --data_format NCHW --scaling 1 --src_dir ./src --dst_dir ./dst --dst_postfix .Restore1 --patch_width 2000 --patch_height 1200

pause

python SR_app.py --model_dir ./Restore_gpu2 --data_format NCHW --scaling 1 --src_dir ./src --dst_dir ./dst --dst_postfix .Restore2 --patch_width 2000 --patch_height 1200
