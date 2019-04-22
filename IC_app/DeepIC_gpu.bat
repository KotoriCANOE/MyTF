cd /d "%~dp0"

python IC_app.py --model_dir ./DeepIC_gpu --data_format NCHW --src_dir ./src --dst_dir ./dst --dst_postfix .DeepIC --patch_height 512 --patch_width 512

pause
