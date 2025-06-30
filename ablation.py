import torch
import argparse, os, time
from models.imagenet.mobilenetv2 import InvertedResidual
from utils.compute_flops_paras import get_flops_paras
from utils.concept_mask_difference import get_difference_of_concept_mask_fcn_resnet101, get_difference_of_concept_mask_mobilenet_v2, get_difference_of_concept_mask_resnet34, get_featuremap_for_mobilenet_v2, get_featuremap_for_resnet, get_mask, normalize_dict
import utils.dataset as dataset
import utils.evaluators as evaluators
from utils.evolutionary_search import DE, objective_function_pruning
from utils.log import get_logger
from utils.prune_utils import l1_pruning_with_prs_for_mobilenet_v2, l1_pruning_with_prs_for_resnet34
from utils.registry import get_model
from utils.trainer import BPTrainer, KDTrainer, MiRTrainer
import models.imagenet.resnet as res

""" running parameters"""
parser = argparse.ArgumentParser(
    description="Parameters for runing Concept Mask guided Few Sample Model Compression."
)
# model
parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34","mobilenet_v2"])
# ablation
parser.add_argument("--trainer", type=str, default="MiR-gridmix", choices=["kd","bp","MiR","kd-mixup","bp-mixup","MiR-mixup",\
                                                    "kd-cutmix","bp-cutmix","MiR-cutmix","kd-CMa","bp-CMa","MiR-CMa","MiR-gridmix"])
parser.add_argument("--pruner", type=str, default="CMa", choices=["CMa","baseline"])
parser.add_argument("--baseline_pr", type=float, default=0.8)
# few sample dataset
parser.add_argument("--num_sample", type=int, default=50)
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet"])
parser.add_argument("--data_root", type=str, default="/your/datasets/imagenet/")
parser.add_argument("--only_target_classes", type=bool, default=False)
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--train_workers", type=int, default=-1)
parser.add_argument("--test_workers", type=int, default=8)
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
# CMa
parser.add_argument("--target_drf", type=float, default=0.401) # target flops drop ratio
parser.add_argument("--target_drp", type=float, default=0.370) # target parameter size drop ratio
parser.add_argument("--upper_bound", type=float, default=0.7) # upper bound of scaling
parser.add_argument("--lower_bound", type=float, default=0.4) # lower bound of scaling
## evolutionary search
parser.add_argument("--pop_size", type=int, default=50)
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--F", type=float, default=0.5)
parser.add_argument("--CR", type=float, default=0.9)
parser.add_argument("--print_interval", type=int, default=5)
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

def main():
    args = parser.parse_args()
    if args.train_workers == -1:
        args.train_workers = NUM_WORKERS_DICT[args.num_sample]
    latency = {}

    ## log
    save_dir = args.save_dir + 'Ablation-' + args.pruner + '-' + args.trainer + '-' + args.model + '-' + args.dataset + '-' + str(args.num_sample)
    if args.pruner == "CMa":
        save_dir = save_dir + '-target_drf-' + str(args.target_drf) + '-target_drp-' + str(args.target_drp)
    elif args.pruner == "baseline":
        save_dir = save_dir + '-pr-' + str(args.baseline_pr)
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
    else:
        raise('No dataset')
    et = time.time()
    latency["dataset"] = et - st
    logger.info("==> Load dataset success. Cost {:.3}s".format(latency["dataset"]))

    ## load model
    st = time.time()
    model = get_model(args.model, num_classes, args.dataset, pretrained=True)
    et = time.time()
    latency["model"] = et - st
    logger.info("==> Load model success. Cost {:.3}s".format(latency["model"]))

    ''' pruning '''
    flops_ori, paras_ori = get_flops_paras(model, args.device, HW)

    if args.pruner == 'CMa':
        prs_file = 'configs/' + args.model + '-' + args.dataset + '-target_drf-' + str(args.target_drf) + '-target_drp-' + str(args.target_drp) + '.txt'
        if os.path.exists(prs_file):
            logger.info('==> Load existed prs file'.format(k, v))
            with open(prs_file, "r", encoding="utf-8") as file:
                content = file.read()
                prs = eval(content)
            for k, v in prs.items():
                logger.info('    {}  :  {}'.format(k, v))
        else:
            ## step 1: obtain the featuremap
            logger.info('==> Obtain the featuremap')
            st = time.time()
            if 'resnet' in args.model:
                fm_in, fm_out = get_featuremap_for_resnet(model, sampled_loader, args.device)
            elif 'mobilenet' in args.model:
                fm_in, fm_out = get_featuremap_for_mobilenet_v2(model, sampled_loader, args.device)
            else:
                raise ValueError("No suitable function for obtain featuremap!")
            et = time.time()
            latency["featuremap"] = et - st
            logger.info('    Obtain featuremap finshed. Cost {:.3f}s'.format(latency["featuremap"]))
            ## step 2: obtain the concept mask
            logger.info('==> Obtain the concept mask')
            st = time.time()
            mask_fm_out, mask_fm_in = {}, {}
            for k in fm_out.keys():
                x_in, x_out = fm_in[k].to(args.device), fm_out[k].to(args.device)
                ## For segmentation tasks, overly large batches can lead to OOM
                if args.model == "fcn_resnet101":
                    x_in, x_out = x_in[0:2], x_out[0:2]
                mask_fm_in[k] = get_mask(x_in,HW).detach().cpu()
                mask_fm_out[k] = get_mask(x_out,HW).detach().cpu()
                del x_in, x_out
                torch.cuda.empty_cache()
            et = time.time()
            latency["mask"] = et - st
            logger.info('    Obtain the mask finshed. Cost {:.3f}s'.format(latency["mask"]))
            ## step 3: calculate the difference of concept mask between input and output
            logger.info('==> Calculate the difference of concept mask')
            st = time.time()
            if args.model == 'resnet34':
                doc = get_difference_of_concept_mask_resnet34(model, mask_fm_in, mask_fm_out, args.device)
            elif args.model == 'mobilenet_v2':
                doc = get_difference_of_concept_mask_mobilenet_v2(model, mask_fm_in, mask_fm_out, args.device)
            elif args.model == 'fcn_resnet101':
                doc = get_difference_of_concept_mask_fcn_resnet101(model, mask_fm_in, mask_fm_out, args.device)
            else:
                raise ValueError("No suitable function for obtain difference of concept mask!")
            et = time.time()
            latency["doc"] = et - st
            logger.info('    Calculate the difference of concept mask finshed. Cost {:.3f}s'.format(latency["doc"]))
            ## step 4: scaling doc
            logger.info('==> Scaling the difference')
            st = time.time()
            scaled_doc = normalize_dict(doc, args.upper_bound, args.lower_bound)
            for k, v in scaled_doc.items():
                logger.info('    {}  :  {:.4f}'.format(k, v))
            et = time.time()
            latency["scaled_doc"] = et - st
            logger.info('    scaling the difference of concept mask finshed. Cost {:.3f}s'.format(latency["scaled_doc"]))
            ## step 5: searching for optimal pruning rates using evolutionary algorithm
            logger.info('==> Searching for optimal pruning rates')
            st = time.time()
            additional_parameter = {'model' : model,
            'flops_ori' : flops_ori,
            'paras_ori' : paras_ori,
            'device' : args.device,
            'doc' : scaled_doc,
            'target_drf' : args.target_drf,
            'target_drp' : args.target_drp
            }
            bounds_prs = []
            for i in range(len(doc)):
                bounds_prs.append((0.1, 0.9))
            de = DE(
                obj_func=objective_function_pruning,
                additional_parameter=additional_parameter,
                bounds=bounds_prs,
                pop_size=args.pop_size,
                max_iter=args.max_iter,
                F=args.F,
                CR=args.CR,
                print_interval=args.print_interval,
                logger=logger
            )
            de.iter_search()
            prs = {}
            idx = 0
            for k, v in  doc.items():
                prs[k] = de.best_sol[idx]
                idx += 1
            with open(prs_file, "w", encoding="utf-8") as file:
                file.write(str(prs))
            for k, v in prs.items():
                logger.info('    {}  :  {:.4f}'.format(k, v))
            et = time.time()
            latency['search'] = et - st
            logger.info('    Search success. Best obj {:.3f}. violation {:.3f}. Cost {:.3f}s'.format(de.best_obj, de.best_violation, latency["search"]))

    elif args.pruner == 'baseline':
        prs = {}
        if args.model == 'resnet34':
            tabu_layers = ['layer1.0', 'layer2.0', 'layer3.0','layer4.0', 'layer1.2', 'layer2.3','layer3.5','layer4.2']
        else:
            tabu_layers = []
        for n, m in model.named_modules():
            if (isinstance(m, res.BasicBlock) or isinstance(m, res.Bottleneck)) and n not in tabu_layers:
                prs[n] = args.baseline_pr
            elif isinstance(m, InvertedResidual) and m.use_res_connect:
                prs[n] = args.baseline_pr

    ## pruning model
    logger.info('==> Pruning the model')
    st = time.time()
    if args.model == 'resnet34':
        pm = l1_pruning_with_prs_for_resnet34(model, prs)
    elif args.model == 'mobilenet_v2':
        pm = l1_pruning_with_prs_for_mobilenet_v2(model, prs)
    else:
        raise ValueError("No suitable function for pruning model!")
    flops, paras = get_flops_paras(pm, args.device, HW)
    drf = (flops_ori - flops) / flops_ori
    drp = (paras_ori - paras) / paras_ori
    et = time.time()
    latency['pruning'] = et - st
    logger.info('    Pruning success. Flops drop rate = {:.1%}. Paras drop rate = {:.1%}. Cost {:.3f}s'\
                .format(drf, drp, latency["pruning"]))
    
    ''' training '''
    st = time.time()
    TRAINER_DICT = {
        "kd": (KDTrainer, 'none'),
        "bp": (BPTrainer, 'none'),
        "MiR": (MiRTrainer, 'none'),
        "kd-mixup": (KDTrainer, 'mixup'),
        "bp-mixup": (BPTrainer, 'mixup'),
        "MiR-mixup": (MiRTrainer, 'mixup'),
        "kd-cutmix": (KDTrainer, 'cutmix'),
        "bp-cutmix": (BPTrainer, 'cutmix'),
        "MiR-cutmix": (MiRTrainer, 'cutmix'),
        "kd-CMa": (KDTrainer, 'CMa'),
        "bp-CMa": (BPTrainer, 'CMa'),
        "MiR-CMa": (MiRTrainer, 'CMa'),
        "MiR-gridmix": (MiRTrainer, 'gridmix'),
    }
    Trainer, data_enhence = TRAINER_DICT[args.trainer]
    trainer = Trainer(
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
        epochs=args.epochs,
        data_enhancer = data_enhence
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
