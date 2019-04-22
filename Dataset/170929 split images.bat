cd /d "%~dp0"

python "170929 split images.py" --src_dir "K:\Dataset.SR\Games" --dst_dir "K:\Dataset.SR\Train\Games_split"

pause

python "170929 split images.py" --src_dir "K:\Dataset.SR\DIV2K_train_HR" --dst_dir "K:\Dataset.SR\Train\DIV2K_train_split"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Flickr2K_HR" --dst_dir "K:\Dataset.SR\Train\Flickr2K_split"
