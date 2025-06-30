## resnet34
# baseline # 0.252 0.232
nohup python -u baseline.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--eval_frequency 2000 --pr 0.5 --tag id-1 --seed 2021 >logs/baseline-resnet34-imagenet-50-pr0.5-id-1.txt 2>&1 &&
# CMa 0.252 0.232
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.252 --target_drp 0.232 --tag id-1 --seed 2021 >logs/CMa-resnet34-imagenet-50-drf-0.252-drp-0.232-id-1.txt 2>&1 &&
# CMa 0.313 0.303
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.313 --target_drp 0.303 --tag id-1 --seed 2021 >logs/CMa-resnet34-imagenet-50-drf-0.313-drp-0.303-id-1.txt 2>&1 &&
# CMa 0.313 0.303 rand sample
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.313 --target_drp 0.303 --tag id-1 --seed 2021 --rand_sample True >logs/CMa-resnet34-imagenet-50-randsample-drf-0.313-drp-0.303-id-1.txt 2>&1 &&
# CMa 0.335 0.235
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.335 --target_drp 0.235 --tag id-1 --seed 2021 >logs/CMa-resnet34-imagenet-50-drf-0.335-drp-0.235-id-1.txt 2>&1 &&
# D2
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--prune_level block --num_del_block 2 --tag id-1 --seed 2021 >logs/CMa-resnet34-imagenet-50-D2-id-1.txt 2>&1 &&
# D3
nohup python -u main.py --model resnet34 --num_sample 50 --dataset imagenet --data_root /your/datasets/imagenet/ \
--prune_level block --num_del_block 3--tag id-1 --seed 2021 >logs/CMa-resnet34-imagenet-50-D3-id-1.txt 2>&1 &&
## mobilenet_v2
# CMa 0.403 0.246 
nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.403 --target_drp 0.246 --tag id-1 --seed 2021 >logs/CMa-mobilenet_v2-imagenet-500-drf-0.403-drp-0.246-id-1.txt 2>&1 &&
# CMa 0.216 0.129
nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.216 --target_drp 0.129 --tag id-1 --seed 2021 >logs/CMa-mobilenet_v2-imagenet-500-drf-0.216-drp-0.129-id-1.txt 2>&1 &&
# CMa 0.133 0.077
nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.133 --target_drp 0.077 --tag id-1 --seed 2021 >logs/CMa-mobilenet_v2-imagenet-500-drf-0.133-drp-0.077-id-1.txt 2>&1 &&
## D1
nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--prune_level block --num_del_block 1 --tag id-1 --seed 2021 >logs/CMa-mobilenet_v2-imagenet-500-D1-id-1.txt 2>&1 &&
## D2
nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--prune_level block --num_del_block 2 --tag id-1 --seed 2021 >logs/CMa-mobilenet_v2-imagenet-500-D2-id-1.txt 2>&1 &&