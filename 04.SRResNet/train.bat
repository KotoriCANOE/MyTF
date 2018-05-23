cd /d "%~dp0"

:: remove COCO from dataset
FOR %%i IN (1103) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1103) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

pause
exit

:: remove Konachan from dataset
FOR %%i IN (1102) DO python train.py --postfix %%i --num_epochs 7 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1102) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

pause

:: replace L2 regularization with weight decay (prior to apply_grad)
FOR %%i IN (1060) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-7
FOR %%i IN (1060) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1061) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1061) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1062) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-5
FOR %%i IN (1062) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: apply missing weight decay to SE units
FOR %%i IN (1063) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1063) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: remove bias before batch norm (1st weight layer in Residual Blocks)
FOR %%i IN (1064) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1064) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: remove bias before batch norm (2nd weight layer in Residual Blocks)
FOR %%i IN (1065) DO python train.py --postfix %%i --num_epochs 4 --restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1065) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1067) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1067) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: (unchanged) use bias for 2nd weight layer in the last Residual Block
FOR %%i IN (1066) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1066) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: (unchanged) Local SE Unit
FOR %%i IN (1068) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 2 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1068) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 2 --g_depth 8 --channels 32 --activation swish >>test.log
:: (unchanged) Replace ReLU with Swish in SE Unit
FOR %%i IN (1069) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1069) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: weight decay test (1e-6)
FOR %%i IN (1070) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 2e-6
FOR %%i IN (1070) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1071) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-7
FOR %%i IN (1071) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: --weight_decay 1e-6
FOR %%i IN (1072) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3
FOR %%i IN (1072) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: replace weight decay (prior to apply_grad) with weight decay (post to apply_grad)
FOR %%i IN (1073) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1073) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1074) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-7
FOR %%i IN (1074) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1075) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-5
FOR %%i IN (1075) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1076) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-7
FOR %%i IN (1076) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1077) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 2e-6
FOR %%i IN (1077) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1078) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 5e-3 --weight_decay 1e-6
FOR %%i IN (1078) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: (unchanged) replace weight decay (post to apply_grad) with L2 regularization
FOR %%i IN (1079) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1079) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: use Batch Renormalization
FOR %%i IN (1080) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0.99
FOR %%i IN (1080) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0.99 >>test.log
:: add back bias before batch norm (2nd weight layer in Residual Blocks)
FOR %%i IN (1081) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1081) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: (unchanged) replace weight decay (post to apply_grad) with L2 regularization
FOR %%i IN (1082) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1082) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: using normalized weight decay
:: weight decay test
FOR %%i IN (1083) DO python train.py --postfix %%i --num_epochs 1 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 2e-5
FOR %%i IN (1083) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1084) DO python train.py --postfix %%i --num_epochs 1 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-5
FOR %%i IN (1084) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1085) DO python train.py --postfix %%i --num_epochs 1 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-4
FOR %%i IN (1085) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1086) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-5
FOR %%i IN (1086) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1087) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-4
FOR %%i IN (1087) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1088) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 2e-5
FOR %%i IN (1088) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
::
FOR %%i IN (1089) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-5
FOR %%i IN (1089) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: (revert) normalized weight decay
:: (revert) replace L2 regularization with weight decay (post to apply_grad)
:: (revert) batch renormalization
:: (revert) remove bias before batch norm (1st weight layer in Residual Blocks)
:: (revert) apply missing weight decay to SE units
FOR %%i IN (1090) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0
FOR %%i IN (1090) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0 >>test.log
:: remove bias before batch norm (1st weight layer in Residual Blocks)
FOR %%i IN (1091) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0
FOR %%i IN (1091) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0 >>test.log
:: (unchanged) apply missing weight decay to SE units
FOR %%i IN (1092) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0
FOR %%i IN (1092) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0 >>test.log
:: (unchanged) weight decay (post to apply_grad)
FOR %%i IN (1093) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0
FOR %%i IN (1093) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0 >>test.log
:: (unchanged) batch renormalization, patch_size=160, weight decay (post to apply_grad)
FOR %%i IN (1094) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0.99
FOR %%i IN (1094) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0.99 >>test.log
:: patch_size=128
FOR %%i IN (1095) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0
FOR %%i IN (1095) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0 >>test.log
:: batch renormalization
FOR %%i IN (1096) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --batch_renorm 0.99
FOR %%i IN (1096) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --batch_renorm 0.99 >>test.log
:: apply missing weight decay to SE units
FOR %%i IN (1097) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1097) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1098) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 5e-7
FOR %%i IN (1098) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1099) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 2e-6
FOR %%i IN (1099) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: --init_activation 1.0
FOR %%i IN (1100) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6 --init_activation 1.0
FOR %%i IN (1100) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: replicate 1097
FOR %%i IN (1101) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16 --random_seed 0 --learning_rate 2e-3 --weight_decay 1e-6
FOR %%i IN (1101) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

pause

FOR %%i IN (220) DO python train.py --postfix %%i --num_epochs 24 --no-restore --g_depth 8 --channels 64 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --batch_norm 0.99 --train_moving_average 0.9999
FOR %%i IN (220) DO python test.py --postfix %%i --progress --pre_down --g_depth 8 --channels 64 --activation swish >>test.log
FOR %%i IN (221) DO python train.py --postfix %%i --num_epochs 24 --no-restore --g_depth 16 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.99 --train_moving_average 0.9999
FOR %%i IN (221) DO python test.py --postfix %%i --progress --pre_down --g_depth 16 --channels 32 --activation swish >>test.log
FOR %%i IN (222) DO python train.py --postfix %%i --num_epochs 24 --no-restore --g_depth 8 --channels 64 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.99 --train_moving_average 0.9999 --multistage_resize 1
FOR %%i IN (222) DO python test.py --postfix %%i --progress --pre_down --g_depth 8 --channels 64 --activation swish >>test.log --multistage_resize 1
FOR %%i IN (223) DO python train.py --postfix %%i --num_epochs 24 --no-restore --multistage_resize 1 --use_se 1 --g_depth 8 --channels 64 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1e-3
FOR %%i IN (223) DO python test.py --postfix %%i --progress --pre_down --multistage_resize 1 --use_se 1 --g_depth 8 --channels 64 --activation swish >>test.log
FOR %%i IN (224) DO python train.py --postfix %%i --num_epochs 24 --no-restore --multistage_resize 1 --use_se 1 --g_depth 16 --channels 64 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1e-3
FOR %%i IN (224) DO python test.py --postfix %%i --progress --pre_down --multistage_resize 1 --use_se 1 --g_depth 16 --channels 64 --activation swish >>test.log
FOR %%i IN (225) DO python train.py --postfix %%i --num_epochs 24 --no-restore --multistage_resize 0 --jpeg_coding 1.0 --use_se 1 --g_depth 8 --channels 64 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1e-3
FOR %%i IN (225) DO python test.py --postfix %%i --progress --pre_down --multistage_resize 0 --jpeg_coding 1.0 --use_se 1 --g_depth 8 --channels 64 --activation swish >>test.log

FOR %%i IN (1020) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.99 --train_moving_average 0.9999
FOR %%i IN (1020) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1021) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --batch_norm 0.99 --train_moving_average 0.9999
FOR %%i IN (1021) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log
FOR %%i IN (1022) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.999 --train_moving_average 0.9999
FOR %%i IN (1022) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1023) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.999 --train_moving_average 0.999
FOR %%i IN (1023) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1024) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --batch_norm 0.9999 --train_moving_average 0.9999
FOR %%i IN (1024) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: fixed random seed
FOR %%i IN (1025) DO python train.py --postfix %%i --num_epochs 6 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --batch_norm 0.999 --train_moving_average 0.999
FOR %%i IN (1025) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1026) DO python train.py --postfix %%i --num_epochs 6 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --batch_norm 0.9999 --train_moving_average 0.999
FOR %%i IN (1026) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1027) DO python train.py --postfix %%i --num_epochs 6 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --batch_norm 0.99 --train_moving_average 0.999
FOR %%i IN (1027) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1028) DO python train.py --postfix %%i --num_epochs 6 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --batch_norm 0.999 --train_moving_average 0.99
FOR %%i IN (1028) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: --batch_norm 0.999 --train_moving_average 0.999
:: learning rate test (--g_depth 8 --patch_height 192 --patch_width 192 --batch_size 16)
FOR %%i IN (1029) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 1e-5 --lr_decay_steps 500 --lr_decay_factor -0.05 --save_steps 500
FOR %%i IN (1029) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1030) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --random_seed 0 --learning_rate 1e-5 --lr_decay_steps 500 --lr_decay_factor -0.2 --save_steps 500
FOR %%i IN (1030) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: --learning_rate 1.25e-2
FOR %%i IN (1031) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --weight_decay 0
FOR %%i IN (1031) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1032) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --weight_decay 1e-6
FOR %%i IN (1032) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1033) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --weight_decay 1e-5
FOR %%i IN (1033) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1034) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --weight_decay 2e-6
FOR %%i IN (1034) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1035) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --weight_decay 4e-6
FOR %%i IN (1035) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: --weight_decay 1e-6
FOR %%i IN (1036) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --initializer 3
FOR %%i IN (1036) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1037) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --initializer 5
FOR %%i IN (1037) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: --initializer 3
FOR %%i IN (1038) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --init_activation 1.0
FOR %%i IN (1038) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1039) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --init_factor 2.0
FOR %%i IN (1039) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: (unchanged) --init_factor 1.0 --init_activation 2.0
FOR %%i IN (1040) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --k_first 7
FOR %%i IN (1040) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --k_first 7 >>test.log
FOR %%i IN (1041) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --k_last 7
FOR %%i IN (1041) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --k_last 7 >>test.log
FOR %%i IN (1042) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --k_first 7 --k_last 7
FOR %%i IN (1042) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --k_first 7 --k_last 7 >>test.log

:: (unchanged) --k_first 3 --k_last 3
FOR %%i IN (1043) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --k_resize 3
FOR %%i IN (1043) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --k_resize 3 >>test.log
FOR %%i IN (1044) DO python train.py --postfix %%i --num_epochs 3 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2 --random_seed 0 --k_resize 11
FOR %%i IN (1044) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --k_resize 11 >>test.log

:: (unchanged) --k_resize 7
FOR %%i IN (1045) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1.25e-2
FOR %%i IN (1045) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: learning rate test2 (--g_depth 8 --patch_height 192 --patch_width 192 --batch_size 16)
FOR %%i IN (1047) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1e-2
FOR %%i IN (1047) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1048) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 5e-3
FOR %%i IN (1048) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1049) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 2e-3
FOR %%i IN (1049) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1050) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 1e-3
FOR %%i IN (1050) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
:: --learning_rate 2e-3 (--g_depth 8 --patch_height 192 --patch_width 192 --batch_size 16)
FOR %%i IN (1054) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 192 --patch_width 192 --batch_size 16 --learning_rate 2e-3
FOR %%i IN (1054) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log

:: learning rate test (--g_depth 16 --patch_height 160 --patch_width 160 --batch_size 16)
FOR %%i IN (1046) DO python train.py --postfix %%i --num_epochs 4 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --random_seed 0 --learning_rate 1e-5 --lr_decay_steps 500 --lr_decay_factor -0.2 --save_steps 500
FOR %%i IN (1046) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log
:: --learning_rate 2e-3 (--g_depth 16 --patch_height 160 --patch_width 160 --batch_size 16)
:: learning rate test2 (--g_depth 16 --patch_height 160 --patch_width 160 --batch_size 16)
FOR %%i IN (1051) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --learning_rate 2e-3 --random_seed 0
FOR %%i IN (1051) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log
FOR %%i IN (1052) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --learning_rate 1e-3 --random_seed 0
FOR %%i IN (1052) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log
FOR %%i IN (1053) DO python train.py --postfix %%i --num_epochs 2 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --learning_rate 5e-4 --random_seed 0
FOR %%i IN (1053) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log
:: --learning_rate 1e-3 (--g_depth 16 --patch_height 160 --patch_width 160 --batch_size 16)
FOR %%i IN (1055) DO python train.py --postfix %%i --num_epochs 24 --no-restore --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16 --learning_rate 1e-3
FOR %%i IN (1055) DO python test.py --postfix %%i --progress --pre_down --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log

pause

FOR %%i IN (1000) DO python train.py --postfix %%i --num_epochs 24 --restore False --scaling 1 --multistage_resize 3 --patch_height 96 --patch_width 96 --batch_size 16
FOR %%i IN (1000) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --multistage_resize 3 >>test.log
FOR %%i IN (1001) DO python train.py --postfix %%i --num_epochs 12 --restore False --scaling 1 --multistage_resize 3 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1001) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --multistage_resize 3 >>test.log
FOR %%i IN (1002) DO python train.py --postfix %%i --num_epochs 12 --restore False --scaling 1 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1002) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 >>test.log
FOR %%i IN (1003) DO python train.py --postfix %%i --num_epochs 12 --restore False --pre_down True --scaling 1 --use_se 0 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1003) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 0 >>test.log
FOR %%i IN (1004) DO python train.py --postfix %%i --num_epochs 12 --restore False --pre_down False --scaling 1 --use_se 1 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1004) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 >>test.log
FOR %%i IN (1005) DO python train.py --postfix %%i --num_epochs 12 --restore False --pre_down False --scaling 1 --use_se 2 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1005) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 2 >>test.log
FOR %%i IN (1006) DO python train.py --postfix %%i --num_epochs 12 --restore False --pre_down False --scaling 1 --use_se 0 --patch_height 64 --patch_width 64 --batch_size 16
FOR %%i IN (1006) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 0 >>test.log
FOR %%i IN (1007) DO python train.py --postfix %%i --num_epochs 12 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 96 --patch_width 96 --batch_size 16
FOR %%i IN (1007) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log
FOR %%i IN (1008) DO python train.py --postfix %%i --num_epochs 24 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 96 --patch_width 96 --batch_size 16
FOR %%i IN (1008) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log
FOR %%i IN (1009) DO python train.py --postfix %%i --num_epochs 24 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 112 --patch_width 112 --batch_size 16
FOR %%i IN (1009) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log
FOR %%i IN (1010) DO python train.py --postfix %%i --num_epochs 24 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 112 --patch_width 112 --batch_size 16
FOR %%i IN (1010) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log
FOR %%i IN (1011) DO python train.py --postfix %%i --num_epochs 24 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 112 --patch_width 112 --batch_size 16
FOR %%i IN (1011) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log
FOR %%i IN (1012) DO python train.py --postfix %%i --num_epochs 24 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 64 --patch_height 112 --patch_width 112 --batch_size 16
FOR %%i IN (1012) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 64 >>test.log

FOR %%i IN (1013) DO python train.py --postfix %%i --num_epochs 8 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation su --patch_height 160 --patch_width 160 --batch_size 16
FOR %%i IN (1013) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation su >>test.log
FOR %%i IN (1014) DO python train.py --postfix %%i --num_epochs 16 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish --patch_height 160 --patch_width 160 --batch_size 16
FOR %%i IN (1014) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 8 --channels 32 --activation swish >>test.log
FOR %%i IN (1015) DO python train.py --postfix %%i --num_epochs 4 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 5 --channels 32 --activation su --patch_height 160 --patch_width 160 --batch_size 16
FOR %%i IN (1015) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 5 --channels 32 --activation su >>test.log
FOR %%i IN (1016) DO python train.py --postfix %%i --num_epochs 16 --restore False --pre_down False --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish --patch_height 128 --patch_width 128 --batch_size 16
FOR %%i IN (1016) DO python test.py --postfix %%i --progress True --pre_down True --scaling 1 --use_se 1 --g_depth 16 --channels 32 --activation swish >>test.log

pause
exit

FOR %%i IN (202) DO python train.py --postfix %%i --num_epochs 24 --restore False
FOR %%i IN (202) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (203) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (203) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (204) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (204) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (205) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (205) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (206) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (206) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (207) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (207) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (208) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (208) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (209) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (209) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (210) DO python train.py --postfix %%i --num_epochs 6 --restore False --k_resize 7
FOR %%i IN (210) DO python test.py --postfix %%i --progress True --pre_down True --k_resize 7 >>test.log
FOR %%i IN (211) DO python train.py --postfix %%i --num_epochs 6 --restore False --k_resize 5
FOR %%i IN (211) DO python test.py --postfix %%i --progress True --pre_down True --k_resize 5 >>test.log

pause

FOR %%i IN (180) DO python train.py --postfix %%i --num_epochs 12 --restore False --lr_decay_steps -200 --lr_decay_factor 0.50
FOR %%i IN (180) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (181) DO python train.py --postfix %%i --num_epochs 12 --restore False --lr_decay_steps -200 --lr_decay_factor 0.50
FOR %%i IN (181) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (182) DO python train.py --postfix %%i --num_epochs 12 --restore False --lr_decay_steps -200 --lr_decay_factor 0.29
FOR %%i IN (182) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (183) DO python train.py --postfix %%i --num_epochs 1 --restore False --lr_decay_steps -200 --lr_decay_factor 0.16
FOR %%i IN (183) DO python train.py --postfix %%i --num_epochs 12 --restore True --lr_decay_steps -200 --lr_decay_factor 0.16
FOR %%i IN (183) DO python train.py --postfix %%i --num_epochs 24 --restore True --lr_decay_steps -200 --lr_decay_factor 0.16
FOR %%i IN (183) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (184) DO python train.py --postfix %%i --num_epochs 12 --restore False --learning_rate 1e-1
FOR %%i IN (184) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (185) DO python train.py --postfix %%i --num_epochs 50 --restore False
FOR %%i IN (185) DO python test.py --postfix %%i --progress True --pre_down True >>test.log

FOR %%i IN (186) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (186) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (186) DO python train.py --postfix %%i --num_epochs 16 --restore True
FOR %%i IN (186) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (187) DO python train.py --postfix %%i --num_epochs 16 --restore False
FOR %%i IN (187) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (188) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (188) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (189) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (189) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (190) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (190) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (191) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (191) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (192) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (192) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (193) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (193) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (194) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (194) DO python train.py --postfix %%i --num_epochs 12 --restore True
FOR %%i IN (194) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (195) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (195) DO python test.py --postfix %%i --progress True --pre_down True >>test.log

FOR %%i IN (196) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (196) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (197) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (197) DO python train.py --postfix %%i --num_epochs 24 --restore True
FOR %%i IN (197) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (198) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (198) DO python test.py --postfix %%i --progress True --pre_down True >>test.log

FOR %%i IN (199) DO python train.py --postfix %%i --num_epochs 24 --restore False
FOR %%i IN (199) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (200) DO python train.py --postfix %%i --num_epochs 24 --restore False
FOR %%i IN (200) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (201) DO python train.py --postfix %%i --num_epochs 12 --restore False --pretrain_dir ./model199/SRMedium_gpu --multistage_resize 2 --jpeg_coding 2.0 --learning_rate 4e-4
FOR %%i IN (201) DO python test.py --postfix %%i --progress True --pre_down True --multistage_resize 2 --jpeg_coding 2.0 >>test.log

pause

FOR %%i IN (120) DO python train.py --postfix %%i --num_epochs 50 --restore False
FOR %%i IN (120) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (121) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (121) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (122) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu
FOR %%i IN (122) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (123) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu
FOR %%i IN (123) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (124) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.9
FOR %%i IN (124) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.9 >>test.log
FOR %%i IN (125) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0
FOR %%i IN (125) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0 >>test.log
FOR %%i IN (126) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --init_activation 2.0
FOR %%i IN (126) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (127) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.9
FOR %%i IN (127) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.9 >>test.log
FOR %%i IN (128) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.99
FOR %%i IN (128) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.99 >>test.log
FOR %%i IN (129) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.9968
FOR %%i IN (129) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.9968 >>test.log
FOR %%i IN (130) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.999
FOR %%i IN (130) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.999 >>test.log
FOR %%i IN (131) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --batch_norm 0.9999
FOR %%i IN (131) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.9999 >>test.log
FOR %%i IN (132) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --init_activation 2.0 --lr_min 1e-6
FOR %%i IN (132) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (133) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --batch_norm 0.99
FOR %%i IN (133) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.99 >>test.log
FOR %%i IN (134) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --batch_norm 0.968
FOR %%i IN (134) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.968 >>test.log
FOR %%i IN (135) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --batch_norm 0.9968
FOR %%i IN (135) DO python test.py --postfix %%i --progress True --pre_down True --activation relu --batch_norm 0.9968 >>test.log

FOR %%i IN (136) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --weight_decay 0
FOR %%i IN (136) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (137) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --weight_decay 1e-5
FOR %%i IN (137) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (138) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --weight_decay 1e-6
FOR %%i IN (138) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (139) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --weight_decay 5e-6
FOR %%i IN (139) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log

FOR %%i IN (140) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --initializer 3
FOR %%i IN (140) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (141) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --initializer 4
FOR %%i IN (141) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (142) DO python train.py --postfix %%i --num_epochs 12 --restore False --activation lrelu0.3
FOR %%i IN (142) DO python test.py --postfix %%i --progress True --pre_down True --activation lrelu0.3 >>test.log
FOR %%i IN (143) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation elu
FOR %%i IN (143) DO python test.py --postfix %%i --progress True --pre_down True --activation elu >>test.log
FOR %%i IN (144) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation prelu
FOR %%i IN (144) DO python test.py --postfix %%i --progress True --pre_down True --activation prelu >>test.log
FOR %%i IN (145) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --learning_rate 2e-3
FOR %%i IN (145) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (146) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --learning_rate 5e-3
FOR %%i IN (146) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (147) DO python train.py --postfix %%i --num_epochs 18 --restore False --activation relu --learning_rate 5e-4
FOR %%i IN (147) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (148) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation relu --learning_rate 2e-4
FOR %%i IN (148) DO python test.py --postfix %%i --progress True --pre_down True --activation relu >>test.log
FOR %%i IN (149) DO python train.py --postfix %%i --num_epochs 12 --restore False --activation lrelu0.05
FOR %%i IN (149) DO python test.py --postfix %%i --progress True --pre_down True --activation lrelu0.05 >>test.log
FOR %%i IN (150) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation su
FOR %%i IN (150) DO python test.py --postfix %%i --progress True --pre_down True --activation su >>test.log
FOR %%i IN (151) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation su --batch_norm 0
FOR %%i IN (151) DO python test.py --postfix %%i --progress True --pre_down True --activation su --batch_norm 0 >>test.log
FOR %%i IN (152) DO python train.py --postfix %%i --num_epochs 6 --restore False --activation su --batch_norm 0 --channels2 64
FOR %%i IN (152) DO python test.py --postfix %%i --progress True --pre_down True --activation su --batch_norm 0 --channels2 64 >>test.log
FOR %%i IN (153) DO python train.py --postfix %%i --num_epochs 2 --restore False
FOR %%i IN (153) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (154) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (154) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (155) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (155) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (156) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (156) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (157) DO python train.py --postfix %%i --num_epochs 6 --restore False
FOR %%i IN (157) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (158) DO python train.py --postfix %%i --num_epochs 6 --restore False --channels 80 --channels2 40
FOR %%i IN (158) DO python test.py --postfix %%i --progress True --pre_down True --channels 80 --channels2 40 >>test.log
FOR %%i IN (158.1) DO python train.py --postfix %%i --num_epochs 12 --restore True --channels 80 --channels2 40
FOR %%i IN (158.1) DO python test.py --postfix %%i --progress True --pre_down True --channels 80 --channels2 40 >>test.log
FOR %%i IN (158.2) DO python train.py --postfix %%i --num_epochs 12 --restore True --channels 80 --channels2 40 --lr_min 5e-5
FOR %%i IN (158.2) DO python test.py --postfix %%i --progress True --pre_down True --channels 80 --channels2 40 --lr_min 5e-5 >>test.log
FOR %%i IN (159) DO python train.py --postfix %%i --num_epochs 6 --restore False --g_depth 16
FOR %%i IN (159) DO python test.py --postfix %%i --progress True --pre_down True --g_depth 16 >>test.log
FOR %%i IN (160) DO python train.py --postfix %%i --num_epochs 6 --restore False --channels 96 --channels2 48
FOR %%i IN (160) DO python test.py --postfix %%i --progress True --pre_down True --channels 96 --channels2 48 >>test.log
FOR %%i IN (161) DO python train.py --postfix %%i --num_epochs 6 --restore False --channels 128 --channels2 64
FOR %%i IN (161) DO python test.py --postfix %%i --progress True --pre_down True --channels 128 --channels2 64 >>test.log
FOR %%i IN (162) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (162) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (163) DO python train.py --postfix %%i --num_epochs 12 --restore False --lr_decay_steps 1000
FOR %%i IN (163) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (164) DO python train.py --postfix %%i --num_epochs 12 --restore False --lr_decay_steps 250
FOR %%i IN (164) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (165) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (165) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (166) DO python train.py --postfix %%i --num_epochs 6 --restore False --epsilon 1e-3
FOR %%i IN (166) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (167) DO python train.py --postfix %%i --num_epochs 6 --restore False --epsilon 1e-1
FOR %%i IN (167) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (168) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (168) DO python test.py --postfix %%i --progress True --pre_down True >>test.log

FOR %%i IN (169) DO python train.py --postfix %%i --num_epochs 6 --restore False --noise_scale 0 --jpeg_coding False
FOR %%i IN (169) DO python test.py --postfix %%i --progress True --pre_down True --noise_scale 0 --jpeg_coding False >>test.log
FOR %%i IN (170) DO python train.py --postfix %%i --num_epochs 6 --restore False --noise_scale 0 --jpeg_coding False --random_resizer 0.05
FOR %%i IN (170) DO python test.py --postfix %%i --progress True --pre_down True --noise_scale 0 --jpeg_coding False --random_resizer 0.05 >>test.log
FOR %%i IN (171) DO python train.py --postfix %%i --num_epochs 100 --restore False --dataset I:\Dataset.SR\DIV2K_train --noise_scale 0 --jpeg_coding False --random_resizer 0.05
FOR %%i IN (171) DO python test.py --postfix %%i --progress True --pre_down True --noise_scale 0 --jpeg_coding False --random_resizer 0.05 >>test.log
FOR %%i IN (172) DO python train.py --postfix %%i --num_epochs 100 --restore False --dataset I:\Dataset.SR\DIV2K_train --noise_scale 0 --jpeg_coding False --random_resizer 0.4
FOR %%i IN (172) DO python test.py --postfix %%i --progress True --pre_down True --noise_scale 0 --jpeg_coding False --random_resizer 0.4 >>test.log

FOR %%i IN (173) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (173) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (174) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (174) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (175) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (175) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (176) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (176) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (177) DO python train.py --postfix %%i --num_epochs 12 --restore False --train_moving_average 0.999
FOR %%i IN (177) DO python test.py --postfix %%i --progress True --pre_down True >>test.log
FOR %%i IN (178) DO python train.py --postfix %%i --num_epochs 12 --restore False --train_moving_average 0
FOR %%i IN (178) DO python test.py --postfix %%i --progress True --pre_down True --train_moving_average 0 >>test.log
FOR %%i IN (179) DO python train.py --postfix %%i --num_epochs 12 --restore False
FOR %%i IN (179) DO python test.py --postfix %%i --progress True --pre_down True >>test.log

FOR %%i IN (162.2) DO python train.py --postfix %%i --num_epochs 12 --restore False --pretrain_dir "K:\MyTF\04.SRResNet\graph_gpu.tmp" --learning_rate 1e-5 --lr_decay_steps 1000
FOR %%i IN (162.2) DO python test.py --postfix %%i --progress True --pre_down True >>test.log


pause

FOR %%i IN (4) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (5) DO python train.py --train_dir train%%i.tmp --k_first 5 --k_last 5 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (6) DO python train.py --train_dir train%%i.tmp --k_first 7 --k_last 7 --res_blocks 8 --channels 64 --batch_norm 0.999
FOR %%i IN (7) DO python train.py --train_dir train%%i.tmp --k_first 9 --k_last 9 --res_blocks 8 --channels 64 --batch_norm 0.999

FOR %%i IN (8) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999
FOR %%i IN (9) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 10 --channels 64 --batch_norm 0.999

FOR %%i IN (10) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 32 --batch_norm 0.999
FOR %%i IN (11) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 48 --batch_norm 0.999

FOR %%i IN (12) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 4 --channels 64 --batch_norm 0.999
FOR %%i IN (8.1) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999 --max_steps 20000
FOR %%i IN (8.2) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0.999

FOR %%i IN (21) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0
FOR %%i IN (22) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --batch_size 32
FOR %%i IN (23) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --loss_moving_average 0
FOR %%i IN (24) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --train_moving_average 0
FOR %%i IN (25) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --stddev_activation 2.0
FOR %%i IN (26) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --stddev_factor 2.0 --stddev_activation 4.0

FOR %%i IN (27) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --channels2 32
FOR %%i IN (28) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --batch_norm 0 --channels2 16
FOR %%i IN (29) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 8 --channels 64 --batch_norm 0 --channels2 32

FOR %%i IN (30) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 4 --init_factor 1.0 --init_activation 2.0
FOR %%i IN (31) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation prelu --initializer 4 --init_factor 1.0 --init_activation 2.0
FOR %%i IN (32) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0
FOR %%i IN (33) DO python train.py --train_dir train%%i.tmp --k_first 3 --k_last 3 --res_blocks 6 --channels 64 --channels2 32 --batch_norm 0 --activation relu --initializer 5 --init_factor 1.0 --init_activation 1.0 --lr_min 0

FOR %%i IN (34) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0
FOR %%i IN (35) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0 --weight_decay 0.0001
FOR %%i IN (36) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0 --lr_min 0.00002 --max_steps 500000
FOR %%i IN (37) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0 --max_steps 200000
FOR %%i IN (38) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0 --max_steps 200000 --weight_decay 0.0001

FOR %%i IN (41) DO python train.py --train_dir train%%i.tmp --initializer 5 --init_factor 1.0 --init_activation 1.0 --num_epochs 40
FOR %%i IN (42) DO python train.py --train_dir train%%i.tmp --num_epochs 80
FOR %%i IN (43) DO python train.py --train_dir train%%i.tmp --num_epochs 80

FOR %%i IN (51) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --patch_height 96 --patch_width 96 --noise_level 0.0
FOR %%i IN (52) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --patch_height 96 --patch_width 96 --noise_level 0.005
FOR %%i IN (53) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --patch_height 96 --patch_width 96 --lr_decay_factor 0.95
FOR %%i IN (54) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --patch_height 96 --patch_width 96 --lr_decay_factor 0.95 --color_augmentation 0
FOR %%i IN (55) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --patch_height 96 --patch_width 96 --lr_decay_factor 0.90 --color_augmentation 0

FOR %%i IN (56) DO python train.py --train_dir train%%i.tmp --num_epochs 20
FOR %%i IN (57) DO python train.py --train_dir train%%i.tmp --num_epochs 20
FOR %%i IN (58) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --mixed_alpha 0.84
FOR %%i IN (59) DO python train.py --train_dir train%%i.tmp --num_epochs 20 --mixed_alpha 0.40
FOR %%i IN (60) DO python train.py --postfix %%i --num_epochs 20 --mixed_alpha 0.84
FOR %%i IN (61) DO python train.py --postfix %%i --num_epochs 20 --mixed_alpha 0.84
FOR %%i IN (62) DO python train.py --postfix %%i --num_epochs 20 --mixed_alpha 0.50

FOR %%i IN (63) DO python train.py --postfix %%i --num_epochs 20 --channels2 16
FOR %%i IN (64) DO python train.py --postfix %%i --num_epochs 1 --channels2 16 --data_format NHWC
FOR %%i IN (65) DO python train.py --postfix %%i --num_epochs 1 --channels2 16 --data_format NCHW
FOR %%i IN (66) DO python train.py --postfix %%i --num_epochs 1
FOR %%i IN (67) DO python train.py --postfix %%i --num_epochs 1
FOR %%i IN (68) DO python train.py --postfix %%i --num_epochs 20 --weight_decay 1e-5

FOR %%i IN (71) DO python train.py --postfix %%i --num_epochs 20 --noise_scale 0.01 --noise_corr 0.25
FOR %%i IN (72) DO python train.py --postfix %%i --num_epochs 20 --noise_scale 0.01 --noise_corr 0.25
FOR %%i IN (73) DO python train.py --postfix %%i --num_epochs 20 --noise_scale 0.01 --noise_corr 0.75
FOR %%i IN (74) DO python train.py --postfix %%i --num_epochs 20 --noise_scale 0.01 --noise_corr 0.75

FOR %%i IN (75) DO python train.py --postfix %%i --restore True --num_epochs 500 --pre_down True --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding False --lr_decay_factor 0.98 --lr_min 1e-8

FOR %%i IN (76) DO python train.py --postfix %%i --num_epochs 100 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding False --lr_decay_factor 0.98 --lr_min 1e-8
FOR %%i IN (76) DO python train.py --postfix %%i --num_epochs 100 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --restore True

FOR %%i IN (80) DO python train.py --postfix %%i --num_epochs 10 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8
FOR %%i IN (81) DO python train.py --postfix %%i --num_epochs 10 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --res_blocks 8
FOR %%i IN (81) DO python train.py --postfix %%i --num_epochs 100 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --res_blocks 8 --restore True
FOR %%i IN (82) DO python train.py --postfix %%i --num_epochs 10 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --channels 80 --channels2 40
FOR %%i IN (83) DO python train.py --postfix %%i --num_epochs 10 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --res_blocks 10
FOR %%i IN (83) DO python train.py --postfix %%i --num_epochs 100 --pre_down False --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True --lr_decay_factor 0.98 --lr_min 1e-8 --res_blocks 10 --restore True

FOR %%i IN (91) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 5 --weight_decay 0 --learning_rate 1e-3 --lr_decay_steps 100
FOR %%i IN (92) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 5 --weight_decay 0 --learning_rate 1e-4 --lr_decay_steps 100
FOR %%i IN (93) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 1 --weight_decay 0 --learning_rate 1e-3 --lr_decay_steps 100
FOR %%i IN (94) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 2 --weight_decay 0 --learning_rate 1e-3 --lr_decay_steps 100
FOR %%i IN (95) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 3 --weight_decay 0 --learning_rate 1e-3 --lr_decay_steps 100
FOR %%i IN (96) DO python train.py --postfix %%i --num_epochs 2 --restore False --initializer 4 --weight_decay 0 --learning_rate 1e-3 --lr_decay_steps 100
FOR %%i IN (97) DO python train.py --postfix %%i --num_epochs 2 --restore False --weight_decay 1e-5 --lr_decay_steps 100
FOR %%i IN (98) DO python train.py --postfix %%i --num_epochs 2 --restore False --weight_decay 4e-5 --lr_decay_steps 100
FOR %%i IN (99) DO python train.py --postfix %%i --num_epochs 2 --restore False --lr_decay_steps 100
FOR %%i IN (99) DO python test.py --postfix %%i --pre_down True --noise_scale 0.01 --noise_corr 0.75 --jpeg_coding True

FOR %%i IN (100) DO python train.py --postfix %%i --num_epochs 2 --restore False --weight_decay 0 --lr_decay_steps 100 --input_range 1 --output_range 2
FOR %%i IN (101) DO python train.py --postfix %%i --num_epochs 2 --restore False --weight_decay 0 --lr_decay_steps 100 --input_range 1 --output_range 2 --batch_norm 0.999
FOR %%i IN (102) DO python train.py --postfix %%i --num_epochs 2 --restore False --weight_decay 0 --lr_decay_steps 100 --input_range 2 --output_range 2
FOR %%i IN (103) DO python train.py --postfix %%i --num_epochs 2 --restore False --lr_decay_steps 100
FOR %%i IN (110) DO python train.py --postfix %%i --num_epochs 50 --restore False --lr_decay_steps 400 --input_range 1 --output_range 2

FOR %%i IN (111) DO python train.py --postfix %%i --num_epochs 50 --restore False --input_range 2 --output_range 2 --weight_decay 5e-6
FOR %%i IN (112) DO python train.py --postfix %%i --num_epochs 50 --restore False --input_range 2 --output_range 2 --weight_decay 2e-6


