import argparse, os, time
os.environ['TORCH_HOME'] = '/your'
import pytorchcv
from models.imagenet.mobilenetv2 import InvertedResidual
from utils.compute_flops_paras import get_flops_paras
import utils.dataset as dataset
import utils.evaluators as evaluators
from utils.log import get_logger
from utils.prune_utils import l1_pruning_with_prs_for_fcn_resnet101, l1_pruning_with_prs_for_mobilenet_v2, l1_pruning_with_prs_for_resnet34
from utils.registry import get_model
from utils.registry import get_model
from utils.trainer import MiRTrainer
import models.imagenet.resnet as res

""" running parameters"""
parser = argparse.ArgumentParser(
    description="Parameters for runing L1-norm pruning and MiR training."
)
# model
parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["resnet34","mobilenet_v2","fcn_resnet101"])
parser.add_argument("--pre_val", type=bool, default=False)
parser.add_argument("--pr", type=float, default=0.8)
# few sample dataset
parser.add_argument("--num_sample", type=int, default=50)
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet","voc2012"])
parser.add_argument("--data_root", type=str, default="/your/datasets/imagenet/")
parser.add_argument("--only_target_classes", type=bool, default=False)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--train_workers", type=int, default=-1)
parser.add_argument("--test_workers", type=int, default=-1)
parser.add_argument("--seed", type=int, default=2021)
# train
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--decrease_lr", type=float, default=0.4)
parser.add_argument("--reduction_factor", type=float, default=0.1)
parser.add_argument("--eval_frequency", type=float, default=2000)
parser.add_argument("--device", type=str, default="cuda")
# log
parser.add_argument("--save_dir", type=str, default="run/")
parser.add_argument("--tag", type=str, default="")
# best num_works
NUM_WORKERS_DICT = {
    50: 0,
    100: 2,
    500: 4,
    1000: 4,
    2000: 4,
    3000: 4
}
VAL_NUM_WORKERS_DICT = {
    'imagenet': 8,
    'voc2012': 2
}

def main():
    args = parser.parse_args()
    if args.train_workers == -1:
        args.train_workers = NUM_WORKERS_DICT[args.num_sample]
    if args.test_workers == -1:
        args.test_workers = VAL_NUM_WORKERS_DICT[args.dataset]
    latency = {}

    ## log
    save_dir = args.save_dir + 'baseline-' + args.model + '-' + args.dataset + '-' + str(args.num_sample) + '-pr-' + str(args.pr)
    args.save_dir = save_dir + '-' + args.tag + '/' if args.tag != '' else save_dir + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = get_logger(args.save_dir)
    for k, v in vars(args).items():
        logger.info(str(k) + " = " + str(v))

    ## load dataset
    st = time.time()
    if args.dataset == 'imagenet':
        val_loader = dataset.__dict__['imagenet'](False, args.test_batch_size, args.test_workers, imagenet_path = args.data_root)
        evaluator = evaluators.classification_evaluator(val_loader)
        sampled_loader = dataset.__dict__['imagenet_fewshot'](args.num_sample, args.train_batch_size, args.train_workers, seed=args.seed,imagenet_path = args.data_root)
        sampled_loader.dataset.samples_to_file(os.path.join(args.save_dir, "samples.txt"))
        num_classes = 1000
        HW = 224
    elif args.dataset == 'voc2012':
        val_loader = dataset.__dict__['voc2012'](args.data_root, "val", args.test_batch_size, args.test_workers)
        evaluator = evaluators.segmentation_evaluator(val_loader)
        sampled_loader = dataset.__dict__['voc2012_fewshot'](args.num_sample, args.seed, args.save_dir, args.data_root, args.train_batch_size, args.train_workers)
        num_classes = 21
        HW = 480
    et = time.time()
    latency["dataset"] = et - st
    logger.info("==> Load dataset success. Cost {:.3}s".format(latency["dataset"]))

    ## load model
    st = time.time()
    model = get_model(args.model, num_classes, args.dataset, pretrained=True)
    et = time.time()
    latency["model"] = et - st
    logger.info("==> Load model success. Cost {:.3}s".format(latency["model"]))

    ## pre-val
    if args.pre_val:
        model = model.to(args.device)
        model.eval()
        logger.info('==> Eval model')
        st = time.time()
        eval_results = evaluator(model, device=args.device)
        et = time.time()
        latency["eval"] = et - st
        if args.dataset == 'voc2012':
            acc, miou = eval_results[0], eval_results[1]
            logger.info('    [model] Pixel Acc = {:.2%}  mIOU = {:.1%}  Cost = {:.2f}s'.format(acc, miou, latency["eval"]))
        else:
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
            logger.info('    [model] Acc@1 = {:.2%}  Acc@5 = {:.2%} Loss = {:.4f}  Cost = {:.2f}s'.format(acc1,acc5,val_loss,latency["eval"]))

    ''' pruning '''
    flops_ori, paras_ori = get_flops_paras(model, args.device, HW)
    prs = {}
    if args.model == 'resnet34':
        tabu_layers = ['layer1.0', 'layer2.0', 'layer3.0','layer4.0', 'layer1.2', 'layer2.3','layer3.5','layer4.2']
    elif args.model == 'fcn_resnet101':
        tabu_layers = ['backbone.1.unit1.body', 'backbone.2.unit1.body', 'backbone.3.unit1.body','backbone.4.unit1.body', \
                       'backbone.1.unit3.body', 'backbone.2.unit4.body','backbone.3.unit23.body','backbone.4.unit3.body']
    else:
        tabu_layers = []
    for n, m in model.named_modules():
        if (isinstance(m, res.BasicBlock) or isinstance(m, pytorchcv.models.resnet.ResBottleneck)) and n not in tabu_layers:
            prs[n] = args.pr
        elif isinstance(m, InvertedResidual) and m.use_res_connect:
            prs[n] = args.pr
    logger.info('==> Pruning the model')
    st = time.time()
    if args.model == 'resnet34':
        pm = l1_pruning_with_prs_for_resnet34(model, prs)
    elif args.model == 'mobilenet_v2':
        pm = l1_pruning_with_prs_for_mobilenet_v2(model, prs)
    elif args.model == 'fcn_resnet101':
        pm = l1_pruning_with_prs_for_fcn_resnet101(model, prs, True)
    et = time.time()
    latency['pruning'] = et - st
    flops, paras = get_flops_paras(pm, args.device, HW)
    drf = (flops_ori - flops) / flops_ori
    drp = (paras_ori - paras) / paras_ori
    logger.info('    Pruning success. Flops drop rate = {:.1%}. Paras drop rate = {:.1%}. Cost {:.3f}s'\
                .format(drf, drp, latency["pruning"]))

    ''' training '''
    st = time.time()
    trainer = MiRTrainer(
        model=model,
        pm=pm,
        sampled_loader=sampled_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        decrease_lr=args.decrease_lr,
        gamma=args.reduction_factor,
        eval_frequency=args.eval_frequency,
        device=args.device,
        logger=logger,
        save_dir=args.save_dir,
        evaluator=evaluator,
        epochs=args.epochs
    )
    trainer.train()
    et = time.time()
    latency['training'] = et - st
    logger.info('    Train success. Cost {:.3f}s'.format(latency["training"]))
    # save
    with open(args.save_dir+'cost.txt', "w", encoding="utf-8") as file:
        file.write(str(latency))


if __name__ == "__main__":
    main()
