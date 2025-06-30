nohup python -u main.py --model mobilenet_v2 --num_sample 500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-500-drf-0.505-drp-0-id-1.txt 2>&1 &&

nohup python -u main.py --model mobilenet_v2 --num_sample 1000 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-1000-drf-0.505-drp-0-id-1.txt 2>&1 &&

nohup python -u main.py --model mobilenet_v2 --num_sample 1500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-1500-drf-0.505-drp-0-id-1.txt 2>&1 &&

nohup python -u main.py --model mobilenet_v2 --num_sample 2000 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-2000-drf-0.505-drp-0-id-1.txt 2>&1 &&

nohup python -u main.py --model mobilenet_v2 --num_sample 2500 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-2500-drf-0.505-drp-0-id-1.txt 2>&1 &&

nohup python -u main.py --model mobilenet_v2 --num_sample 3000 --dataset imagenet --data_root /your/datasets/imagenet/ \
--target_drf 0.505 --target_drp 0 --tag id-1 --seed 2021 >full_logs/CMa-mobilenet_v2-imagenet-3000-drf-0.505-drp-0-id-1.txt 2>&1 &&
