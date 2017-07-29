cd /d "%~dp0"

python SR_app.py --model_dir ./SRResNet_cpu --data_format NHWC --src_dir ./src --dst_dir ./dst --dst_postfix .SRResNet --patch_height 480 --patch_width 480

pause
