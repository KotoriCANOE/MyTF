cd /d "%~dp0"


pause

python "170929 split images.py" --src_dir "K:\Dataset.SR\DIV2K\DIV2K_train_HR" --dst_dir "I:\Dataset.SR\DIV2K_train"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Flickr2K_HR" --dst_dir "I:\Dataset.SR\Flickr2K"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Train\Games" --dst_dir "I:\Dataset.SR\Games"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Train\ACG picture" --dst_dir "I:\Dataset.SR\ACG picture"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Train\PIXIV" --dst_dir "I:\Dataset.SR\PIXIV"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Train\Konachan" --dst_dir "I:\Dataset.SR\Konachan"

python "170929 split images.py" --src_dir "K:\Dataset.SR\DIV2K\DIV2K_train_HR" --dst_dir "K:\Dataset.SR\Train\DIV2K_train"
python "170929 split images.py" --src_dir "K:\Dataset.SR\Flickr2K_HR" --dst_dir "K:\Dataset.SR\Train\Flickr2K"
python "170929 split images.py" --src_dir "K:\Dataset.SR\ACG picture" --dst_dir "K:\Dataset.SR\Train\ACG picture"
