cd /d "%~dp0"

FOR %%i IN (1) DO python MRS_test.py --train_dir train%%i.tmp --test_dir test%%i.tmp

pause
