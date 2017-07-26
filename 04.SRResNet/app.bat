cd /d "%~dp0"

python SRResNet_app.py --src_dir ./src --dst_dir ./dst --patch_height 360 --patch_width 360 --patch_pad 8

pause
