cd /d "%~dp0"

FOR %%i IN (1) DO python MRS_train.py --train_dir train%%i.tmp --num_epochs 40

pause
