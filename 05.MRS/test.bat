cd /d "%~dp0"

FOR %%l IN (12) DO (
    FOR %%i IN (112) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress False --batch_size 1 >>test.log
    FOR %%i IN (122) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress False --batch_size 1 >>test.log
)

pause
exit

FOR %%l IN (4) DO (
    FOR %%i IN (2) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
)

FOR %%l IN (12) DO (
    FOR %%i IN (100) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (101) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (102) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (103) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --k_first 5
    FOR %%i IN (104) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --k_first 3
    FOR %%i IN (105) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (106) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (107) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (108) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log
    FOR %%i IN (109) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --channels 64
    FOR %%i IN (110) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --channels 32
    FOR %%i IN (111) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress True >>test.log --batch_norm 0.999
    FOR %%i IN (111) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --progress False >>test.log --batch_norm 0.999
    FOR %%i IN (112) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --progress True >>test.log
    FOR %%i IN (120) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --progress True >>test.log
    FOR %%i IN (121) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --progress True >>test.log
    
    FOR %%i IN (3) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (4) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (11) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (12) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (13) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (14) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (15) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (16) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (17) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (18) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (19) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (20) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (21) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (22) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (23) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (24) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (25) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (26) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (27) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (28) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (29) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (30) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (31) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --channels 64 >>test.log
    FOR %%i IN (32) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --channels 56 >>test.log
    FOR %%i IN (33) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --channels 48 >>test.log
    FOR %%i IN (34) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --channels 40 >>test.log
    FOR %%i IN (35) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (36) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --batch_norm 0.999 >>test.log
    FOR %%i IN (37) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (38) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (39) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (40) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (41) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (42) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --activation prelu >>test.log
    FOR %%i IN (43) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --activation relu >>test.log
    FOR %%i IN (44_1) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (44_2) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (44_3) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (44_4) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (44_5) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (45_1) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (45_2) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (45_3) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (45_4) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (45_5) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (46_1) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (46_2) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (46_3) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (46_4) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (46_5) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (47) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --activation relu >>test.log
    FOR %%i IN (48) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
    FOR %%i IN (49) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (50) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --res_blocks 9 >>test.log
    FOR %%i IN (51) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l --res_blocks 10 >>test.log
    FOR %%i IN (49) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (50) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 9 >>test.log
    FOR %%i IN (51) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 10 >>test.log
    FOR %%i IN (52) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (52) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (53) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --res_blocks 7 >>test.log
    FOR %%i IN (53) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 7 >>test.log
    FOR %%i IN (54) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (54) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 8 >>test.log
    FOR %%i IN (55) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --res_blocks 9 >>test.log
    FOR %%i IN (55) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 9 >>test.log
    FOR %%i IN (56) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --res_blocks 8 --batch_norm 0.999 >>test.log
    FOR %%i IN (56) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --res_blocks 8 --batch_norm 0.999 >>test.log
    FOR %%i IN (57) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --k_first 3 >>test.log
    FOR %%i IN (57) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --k_first 3 >>test.log
    FOR %%i IN (60) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --smoothing 0 >>test.log
    FOR %%i IN (60) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --smoothing 0 >>test.log
    FOR %%i IN (61) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --noise_scale 0 >>test.log
    FOR %%i IN (61) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --noise_scale 0 >>test.log
    FOR %%i IN (63) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l --smoothing 0.5 --noise_scale 0.03 >>test.log
    FOR %%i IN (63) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l --smoothing 0.5 --noise_scale 0.03 >>test.log
    FOR %%i IN (64) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l >>test.log
    FOR %%i IN (64) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l >>test.log
    FOR %%i IN (65) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l >>test.log
    FOR %%i IN (65) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l >>test.log
    FOR %%i IN (66) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l >>test.log
    FOR %%i IN (66) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l >>test.log
    FOR %%i IN (67) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test1" --num_labels %%l >>test.log
    FOR %%i IN (67) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test2" --num_labels %%l >>test.log
)

FOR %%l IN (16) DO (
    FOR %%i IN (2) DO python test.py --postfix %%l_%%i --dataset "../../Dataset.MRS/%%l/Test" --num_labels %%l >>test.log
)
