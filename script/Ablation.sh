### num_sample 50
## baseline
# kd
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer kd >logs_ablation/ablation-resnet34-imagenet-50-baseline-kd.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer kd-mixup >logs_ablation/ablation-resnet34-imagenet-50-baseline-kd-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer kd-cutmix >logs_ablation/ablation-resnet34-imagenet-50-baseline-kd-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer kd-CMa >logs_ablation/ablation-resnet34-imagenet-50-baseline-kd-CMa.txt 2>&1 &&
# bp
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer bp >logs_ablation/ablation-resnet34-imagenet-50-baseline-bp.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer bp-mixup >logs_ablation/ablation-resnet34-imagenet-50-baseline-bp-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer bp-cutmix >logs_ablation/ablation-resnet34-imagenet-50-baseline-bp-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer bp-CMa >logs_ablation/ablation-resnet34-imagenet-50-baseline-bp-CMa.txt 2>&1 &&
# MiR
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer MiR >logs_ablation/ablation-resnet34-imagenet-50-baseline-MiR.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer MiR-mixup >logs_ablation/ablation-resnet34-imagenet-50-baseline-MiR-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer MiR-cutmix >logs_ablation/ablation-resnet34-imagenet-50-baseline-MiR-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer MiR-CMa >logs_ablation/ablation-resnet34-imagenet-50-baseline-MiR-CMa.txt 2>&1 &&

## CMa
# kd
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer kd >logs_ablation/ablation-resnet34-imagenet-50-CMa-kd.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer kd-mixup >logs_ablation/ablation-resnet34-imagenet-50-CMa-kd-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer kd-cutmix >logs_ablation/ablation-resnet34-imagenet-50-CMa-kd-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer kd-CMa >logs_ablation/ablation-resnet34-imagenet-50-CMa-kd-CMa.txt 2>&1 &&
# bp
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer bp >logs_ablation/ablation-resnet34-imagenet-50-CMa-bp.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer bp-mixup >logs_ablation/ablation-resnet34-imagenet-50-CMa-bp-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer bp-cutmix >logs_ablation/ablation-resnet34-imagenet-50-CMa-bp-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer bp-CMa >logs_ablation/ablation-resnet34-imagenet-50-CMa-bp-CMa.txt 2>&1 &&
# MiR
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer MiR >logs_ablation/ablation-resnet34-imagenet-50-CMa-MiR.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer MiR-mixup >logs_ablation/ablation-resnet34-imagenet-50-CMa-MiR-mixup.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer MiR-cutmix >logs_ablation/ablation-resnet34-imagenet-50-CMa-MiR-cutmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer MiR-CMa >logs_ablation/ablation-resnet34-imagenet-50-CMa-MiR-CMa.txt 2>&1 &&


##
nohup python -u ablation.py --num_sample 50 --pruner CMa --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-50-CMa-MiR-gridmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 100 --pruner CMa --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-100-CMa-MiR-gridmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 500 --pruner CMa --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-500-CMa-MiR-gridmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 50 --pruner baseline --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-50-baseline-MiR-gridmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 100 --pruner baseline --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-100-baseline-MiR-gridmix.txt 2>&1 &&
nohup python -u ablation.py --num_sample 500 --pruner baseline --trainer MiR-gridmix >logs_ablation/ablation-resnet34-imagenet-500-baseline-MiR-gridmix.txt 2>&1 &&
