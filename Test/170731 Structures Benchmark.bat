cd /d "%~dp0"

python "170731 Structures Benchmark.py" --batch_size 1 --patch_height 512 --patch_width 512

pause

python "170731 Structures Benchmark.py" --channels 32
python "170731 Structures Benchmark.py" --channels 48
python "170731 Structures Benchmark.py" --channels 64
python "170731 Structures Benchmark.py" --channels 80
python "170731 Structures Benchmark.py" --channels 96
python "170731 Structures Benchmark.py" --channels 112

python "170731 Structures Benchmark.py" --res_blocks 2
python "170731 Structures Benchmark.py" --res_blocks 4
python "170731 Structures Benchmark.py" --res_blocks 6
python "170731 Structures Benchmark.py" --res_blocks 8
python "170731 Structures Benchmark.py" --res_blocks 10
python "170731 Structures Benchmark.py" --res_blocks 12
python "170731 Structures Benchmark.py" --res_blocks 14

python "170731 Structures Benchmark.py" --batch_size 16 --patch_height 96 --patch_width 96
python "170731 Structures Benchmark.py" --batch_size 32 --patch_height 96 --patch_width 96
python "170731 Structures Benchmark.py" --batch_size 16 --patch_height 128 --patch_width 128
python "170731 Structures Benchmark.py" --batch_size 8 --patch_height 192 --patch_width 192
python "170731 Structures Benchmark.py" --batch_size 4 --patch_height 256 --patch_width 256
python "170731 Structures Benchmark.py" --batch_size 1 --patch_height 512 --patch_width 512
