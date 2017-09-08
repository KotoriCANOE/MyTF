cd /d "%~dp0"

FOR %%l IN (12) DO (
)

:: --smoothing 0.5 --noise_scale 0.03

pause
exit

FOR %%i IN (1) DO python train.py --postfix %%i --num_epochs 40
FOR %%i IN (2) DO python train.py --postfix %%i --num_epochs 2 --epoch_size 100000
FOR %%i IN (2) DO python test.py --postfix %%i >> test.log
FOR %%i IN (3) DO python train.py --postfix %%i --num_epochs 4 --epoch_size 50000
FOR %%i IN (3) DO python test.py --postfix %%i >> test.log
FOR %%i IN (4) DO python train.py --postfix %%i --num_epochs 8 --epoch_size 25000
FOR %%i IN (4) DO python test.py --postfix %%i >> test.log
FOR %%i IN (5) DO python train.py --postfix %%i --num_epochs 20 --epoch_size 10000
FOR %%i IN (5) DO python test.py --postfix %%i >> test.log
FOR %%i IN (6) DO python train.py --postfix %%i --num_epochs 200 --epoch_size 1000
FOR %%i IN (6) DO python test.py --postfix %%i >> test.log

FOR %%i IN (10) DO python train.py --postfix %%i --num_epochs 40 --epoch_size 100000

FOR %%l IN (4) DO (
    FOR %%i IN (1) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 20
    FOR %%i IN (2) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 5
)

FOR %%l IN (12) DO (
    FOR %%i IN (100) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16
    FOR %%i IN (101) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --weight_decay 1e-6
    FOR %%i IN (102) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --weight_decay 0
    FOR %%i IN (103) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --k_first 5
    FOR %%i IN (104) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --k_first 3
    FOR %%i IN (105) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 8 --initializer 1
    FOR %%i IN (106) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 8 --initializer 2
    FOR %%i IN (107) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 8 --initializer 4
    FOR %%i IN (108) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 8 --initializer 5
    FOR %%i IN (109) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --channels 64
    FOR %%i IN (110) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --channels 32
    FOR %%i IN (111) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --batch_norm 0.999
    FOR %%i IN (112) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3
    FOR %%i IN (112) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (113) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_beta1 0.5 --learning_beta2 0.9
    FOR %%i IN (113) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (114) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3 --smoothing 0.5 --noise_scale 0
    FOR %%i IN (114) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --smoothing 0.5 --noise_scale 0
    FOR %%i IN (115) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3 --smoothing 0 --noise_scale 0.03
    FOR %%i IN (115) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --smoothing 0.5 --noise_scale 0 --smoothing 0 --noise_scale 0.03
    FOR %%i IN (116) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3 --smoothing 0 --noise_scale 0
    FOR %%i IN (116) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --smoothing 0.5 --noise_scale 0 --smoothing 0 --noise_scale 0
    FOR %%i IN (117) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 160 --learning_rate 1e-3 --epoch_size 100000
    FOR %%i IN (117) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (118) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 1600 --learning_rate 1e-3 --epoch_size 10000
    FOR %%i IN (118) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (119) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16000 --learning_rate 1e-3 --epoch_size 1000
    FOR %%i IN (119) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (120) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --learning_rate 1e-3
    FOR %%i IN (120) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore True --num_epochs 16 --learning_rate 1e-3
    FOR %%i IN (120) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (121) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3
    FOR %%i IN (121) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    :: 122: mse loss
    FOR %%i IN (122) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --learning_rate 1e-3
    FOR %%i IN (122) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    
    FOR %%i IN (3) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 40
    FOR %%i IN (4) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 40
    FOR %%i IN (5) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 1e-1
    FOR %%i IN (6) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 1e-2
    FOR %%i IN (7) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 1e-3
    FOR %%i IN (8) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 1e-4
    FOR %%i IN (9) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 5e-4
    FOR %%i IN (10) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 2 --learning_rate 3e-4
    FOR %%i IN (11) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 40 --batch_size 16
    FOR %%i IN (12) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 57 --batch_size 32
    FOR %%i IN (13) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 80 --batch_size 64
    FOR %%i IN (14) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --learning_rate 1e-4
    FOR %%i IN (15) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --learning_rate 2e-4
    FOR %%i IN (16) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --learning_rate 5e-4
    FOR %%i IN (17) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --learning_rate 1e-3
    FOR %%i IN (18) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 2e-5
    FOR %%i IN (19) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 3e-5
    FOR %%i IN (20) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 4e-5
    FOR %%i IN (21) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 5e-5
    FOR %%i IN (22) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 7e-5
    FOR %%i IN (23) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 1e-4
    FOR %%i IN (24) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 7e-6
    FOR %%i IN (25) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 5e-6
    FOR %%i IN (26) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 4e-6
    FOR %%i IN (27) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 3e-6
    FOR %%i IN (28) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 2e-6
    FOR %%i IN (29) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --weight_decay 1e-6
    FOR %%i IN (30) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 80
    FOR %%i IN (31) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_size 32 --channels 64
    FOR %%i IN (32) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_size 32 --channels 56
    FOR %%i IN (33) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_size 32 --channels 48
    FOR %%i IN (34) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_size 32 --channels 40
    FOR %%i IN (35) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_size 64
    FOR %%i IN (36) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --batch_norm 0.999
    FOR %%i IN (37) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (38) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (39) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 3
    FOR %%i IN (40) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 4
    FOR %%i IN (41) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (42) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --activation prelu
    FOR %%i IN (43) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --activation relu
    FOR %%i IN (44_1) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (44_2) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (44_3) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (44_4) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (44_5) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 1
    FOR %%i IN (45_1) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (45_2) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (45_3) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (45_4) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (45_5) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 2
    FOR %%i IN (46_1) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (46_2) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (46_3) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (46_4) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (46_5) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 10 --initializer 5
    FOR %%i IN (47) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60 --activation relu
    FOR %%i IN (48) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 60
    FOR %%i IN (49) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --num_epochs 60 --res_blocks 8
    FOR %%i IN (50) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --num_epochs 60 --res_blocks 9
    FOR %%i IN (51) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --num_epochs 60 --res_blocks 10
    FOR %%i IN (52) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore True --num_epochs 120 --res_blocks 8
    FOR %%i IN (53) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --res_blocks 7
    FOR %%i IN (54) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --res_blocks 8
    FOR %%i IN (55) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --res_blocks 9
    FOR %%i IN (56) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --res_blocks 8 --batch_norm 0.999
    FOR %%i IN (57) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --res_blocks 8 --activation prelu
    FOR %%i IN (60) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --smoothing 0
    FOR %%i IN (61) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --noise_scale 0
    FOR %%i IN (62) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --smoothing 0.5 --noise_scale 0.03
    FOR %%i IN (62) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore True --num_epochs 16 --smoothing 0.5 --noise_scale 0.03
    FOR %%i IN (63) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train2" --num_labels %%l --restore False --num_epochs 16 --smoothing 0.5 --noise_scale 0.03
    FOR %%i IN (64) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --initializer 1
    rem FOR %%i IN (62) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --initializer 2
    FOR %%i IN (65) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --initializer 3
    FOR %%i IN (66) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --initializer 4
    FOR %%i IN (67) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train1" --num_labels %%l --restore False --num_epochs 8 --initializer 5
)

FOR %%l IN (16) DO (
    FOR %%i IN (2) DO python train.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Train" --num_labels %%l --num_epochs 5
)
