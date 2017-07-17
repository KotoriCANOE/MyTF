cd /d "%~dp0"

FOR %%i IN (41) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --num_epochs 40

pause

FOR %%i IN (4) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (5) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 5 --k_last 5 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (6) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 7 --k_last 7 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (7) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 9 --k_last 9 --res_blocks 8 --channels 64 --batch_norm 0.999

FOR %%i IN (8) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999
FOR %%i IN (9) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 10 --channels 64 --batch_norm 0.999

FOR %%i IN (10) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 32 --batch_norm 0.999
FOR %%i IN (11) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 48 --batch_norm 0.999

FOR %%i IN (12) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 4 --channels 64 --batch_norm 0.999
FOR %%i IN (8.1) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999 --max_steps 20000
FOR %%i IN (8.2) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999

FOR %%i IN (21) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0
FOR %%i IN (22) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --batch_size 32
FOR %%i IN (23) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --loss_moving_average 0
FOR %%i IN (24) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --train_moving_average 0
FOR %%i IN (25) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --stddev_activation 2.0
FOR %%i IN (26) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --stddev_factor 2.0 --stddev_activation 4.0

FOR %%i IN (27) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --channels2 32
FOR %%i IN (28) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --channels2 16
FOR %%i IN (29) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 64 --batch_norm 0 --channels2 32

FOR %%i IN (30) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 4 --init_factor 1.0 --init_activation 2.0
FOR %%i IN (31) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation prelu --initializer 4 --init_factor 1.0 --init_activation 2.0
FOR %%i IN (32) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0
FOR %%i IN (33) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --lr_min 0

FOR %%i IN (34) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0
FOR %%i IN (35) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --weight_decay 0.0001
FOR %%i IN (36) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --lr_min 0.00002 --max_steps 500000
FOR %%i IN (37) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --max_steps 200000
FOR %%i IN (38) DO python SRResNet_train.py --train_dir SRResNet_train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --max_steps 200000 --weight_decay 0.0001

pause
