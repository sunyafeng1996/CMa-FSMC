## baseline
# 0.192 0.200 50
nohup python -u baseline.py --model fcn_resnet101 --pr 0.5 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-1 >logs/baseline-fcn_resnet101-voc-50-pr0.5-id-1.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.192 --target_drp 0.200 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-1 >logs/CMa-fcn_resnet101-voc-50-0.192-0.200-id-1.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.5 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-2 >logs/baseline-fcn_resnet101-voc-50-pr0.5-id-2.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.192 --target_drp 0.200 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-2 >logs/CMa-fcn_resnet101-voc-50-0.192-0.200-id-2.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.5 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-3 >logs/baseline-fcn_resnet101-voc-50-pr0.5-id-3.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.192 --target_drp 0.200 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-3 >logs/CMa-fcn_resnet101-voc-50-0.192-0.200-id-3.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.5 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-4 >logs/baseline-fcn_resnet101-voc-50-pr0.5-id-4.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.192 --target_drp 0.200 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-4 >logs/CMa-fcn_resnet101-voc-50-0.192-0.200-id-4.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.5 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-5 >logs/baseline-fcn_resnet101-voc-50-pr0.5-id-5.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.192 --target_drp 0.200 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-5 >logs/CMa-fcn_resnet101-voc-50-0.192-0.200-id-5.txt 2>&1 &&

# 0.269 0.280 50
nohup python -u baseline.py --model fcn_resnet101 --pr 0.7 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-1 >logs/baseline-fcn_resnet101-voc-50-pr0.7-id-1.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.269 --target_drp 0.280 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-1 >logs/CMa-fcn_resnet101-voc-50-0.269-0.280-id-1.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.7 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-2 >logs/baseline-fcn_resnet101-voc-50-pr0.7-id-2.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.269 --target_drp 0.280 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-2 >logs/CMa-fcn_resnet101-voc-50-0.269-0.280-id-2.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.7 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-3 >logs/baseline-fcn_resnet101-voc-50-pr0.7-id-3.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.269 --target_drp 0.280 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-3 >logs/CMa-fcn_resnet101-voc-50-0.269-0.280-id-3.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.7 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-4 >logs/baseline-fcn_resnet101-voc-50-pr0.7-id-4.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.269 --target_drp 0.280 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-4 >logs/CMa-fcn_resnet101-voc-50-0.269-0.280-id-4.txt 2>&1 &&

nohup python -u baseline.py --model fcn_resnet101 --pr 0.7 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-5 >logs/baseline-fcn_resnet101-voc-50-pr0.7-id-5.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --target_drf 0.269 --target_drp 0.280 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-5 >logs/CMa-fcn_resnet101-voc-50-0.269-0.280-id-5.txt 2>&1 &&


## PRACTISE
# D2
nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 2 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-1 >logs/PRACTISE-fcn_resnet101-voc-50-D2-id-1.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 2 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-1 >logs/CMa-fcn_resnet101-voc-50-D2-id-1.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 2 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-2 >logs/PRACTISE-fcn_resnet101-voc-50-D2-id-2.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 2 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-2 >logs/CMa-fcn_resnet101-voc-50-D2-id-2.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 2 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-3 >logs/PRACTISE-fcn_resnet101-voc-50-D2-id-3.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 2 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-3 >logs/CMa-fcn_resnet101-voc-50-D2-id-3.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 2 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-4 >logs/PRACTISE-fcn_resnet101-voc-50-D2-id-4.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 2 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-4 >logs/CMa-fcn_resnet101-voc-50-D2-id-4.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 2 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-5 >logs/PRACTISE-fcn_resnet101-voc-50-D2-id-5.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 2 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-5 >logs/CMa-fcn_resnet101-voc-50-D2-id-5.txt 2>&1 &&

# D3
nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 3 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-1 >logs/PRACTISE-fcn_resnet101-voc-50-D3-id-1.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 3 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-1 >logs/CMa-fcn_resnet101-voc-50-D3-id-1.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 3 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-2 >logs/PRACTISE-fcn_resnet101-voc-50-D3-id-2.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 3 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-2 >logs/CMa-fcn_resnet101-voc-50-D3-id-2.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 3 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-3 >logs/PRACTISE-fcn_resnet101-voc-50-D3-id-3.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 3 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-3 >logs/CMa-fcn_resnet101-voc-50-D3-id-3.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 3 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-4 >logs/PRACTISE-fcn_resnet101-voc-50-D3-id-4.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 3 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-4 >logs/CMa-fcn_resnet101-voc-50-D3-id-4.txt 2>&1 &&

nohup python -u PRACTISE.py --model fcn_resnet101 --num_del_block 3 --num_sample 50 --dataset voc2012 --data_root /your/datasets/ \
--eval_frequency 200 --train_batch_size 8 --train_workers 0 --lr 0.001 --tag id-5 >logs/PRACTISE-fcn_resnet101-voc-50-D3-id-5.txt 2>&1 &&
nohup python -u main.py --model fcn_resnet101 --prune_level block --num_del_block 3 --num_sample 50 \
--dataset voc2012 \--data_root /your/datasets/ --eval_frequency 200 --train_batch_size 8 \
--train_workers 0 --lr 0.001 --tag id-5 >logs/CMa-fcn_resnet101-voc-50-D3-id-5.txt 2>&1 &&
