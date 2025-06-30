import copy
import torch
import argparse, os, time
from utils.compute_flops_paras import get_flops_paras
from utils.concept_mask_difference import get_difference_of_concept_mask_fcn_resnet101, get_difference_of_concept_mask_mobilenet_v2, get_difference_of_concept_mask_resnet34, get_featuremap_for_mobilenet_v2, get_featuremap_for_resnet, get_mask, normalize_dict
import utils.dataset as dataset
import utils.evaluators as evaluators
from utils.evolutionary_search import DE, objective_function_pruning
from utils.log import get_logger
from utils.prune_utils import _set_module, l1_pruning_with_prs_for_fcn_resnet101, l1_pruning_with_prs_for_mobilenet_v2, l1_pruning_with_prs_for_resnet34
from utils.registry import get_model
from utils.trainer import CMaTrainer

""" running parameters"""
parser = argparse.ArgumentParser(
    description="Parameters for runing Concept Mask guided Few Sample Model Compression."
)
# model
parser.add_argument("--model", type=str, default="resnet34", choices=["resnet34","mobilenet_v2","fcn_resnet101"])
parser.add_argument("--pre_val", type=bool, default=False)
# few sample dataset
parser.add_argument("--num_sample", type=int, default=50)
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet","voc2012"])
parser.add_argument("--data_root", type=str, default="/your/datasets/imagenet/")
parser.add_argument("--rand_sample", type=bool, default=False)
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
# CMa
parser.add_argument("--prune_level", type=str, default="filter", choices=["filter","block"])
parser.add_argument("--num_del_block", type=int, default=2)
parser.add_argument("--target_drf", type=float, default=0.318) # target flops drop ratio
parser.add_argument("--target_drp", type=float, default=0.331) # target parameter size drop ratio
parser.add_argument("--upper_bound", type=float, default=0.7) # upper bound of scaling
parser.add_argument("--lower_bound", type=float, default=0.4) # lower bound of scaling
parser.add_argument("--th", type=float, default=0.2)
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
VAL_NUM_WORKERS_DICT = {
    'imagenet': 8,
    'voc2012': 2
}

def main():
    args = parser.parse_args()
    if args.train_workers == -1:
        try:
            args.train_workers = NUM_WORKERS_DICT[args.num_sample]
        except:
            args.train_workers = 4
    if args.test_workers == -1:
        args.test_workers = VAL_NUM_WORKERS_DICT[args.dataset]
    latency = {}

    ## log
    if args.prune_level == 'block':
        save_dir = args.save_dir + 'CMa-' + args.model + '-' + args.dataset + '-' + str(args.num_sample) + '-num_del_block-' +str(args.num_del_block)
    else:
        save_dir = args.save_dir + 'CMa-' + args.model + '-' + args.dataset + '-' + str(args.num_sample) + '-target_drf-' + str(args.target_drf) + '-target_drp-' + str(args.target_drp)
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
    prs_file = 'configs/' + args.model + '-' + args.dataset + '-target_drf-' + str(args.target_drf) + '-target_drp-' + str(args.target_drp) + '.txt'
    flops_ori, paras_ori = get_flops_paras(model, args.device, HW)

    if os.path.exists(prs_file) and args.prune_level == "filter":

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
            mask_fm_in[k] = get_mask(x_in,HW,th=args.th).detach().cpu()
            mask_fm_out[k] = get_mask(x_out,HW,th=args.th).detach().cpu()
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

        ## block pruning
        if args.prune_level == "block":
            logger.info('==> Pruning the model')
            st = time.time()
            deleted_block_name = sorted(doc, key=lambda k: doc[k])[:args.num_del_block]
            pm = copy.deepcopy(model)
            for name in deleted_block_name:
                _set_module(pm, name, torch.nn.Sequential())
            logger.info(deleted_block_name)
            flops, paras = get_flops_paras(pm, args.device, HW)
            drf = (flops_ori - flops) / flops_ori
            drp = (paras_ori - paras) / paras_ori
            et = time.time()
            latency['pruning'] = et - st
            del fm_in, fm_out, mask_fm_in, mask_fm_out, doc
            logger.info('    Pruning success. Flops drop rate = {:.1%}. Paras drop rate = {:.1%}. Cost {:.3f}s'\
                        .format(drf, drp, latency["pruning"]))
        ## search prs for filter
        elif args.prune_level == "filter":
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

    ## pruning model
    if args.prune_level == "filter":
        logger.info('==> Pruning the model')
        st = time.time()
        if args.model == 'resnet34':
            pm = l1_pruning_with_prs_for_resnet34(model, prs)
        elif args.model == 'mobilenet_v2':
            pm = l1_pruning_with_prs_for_mobilenet_v2(model, prs)
        elif args.model == 'fcn_resnet101':
            pm = l1_pruning_with_prs_for_fcn_resnet101(model, prs, True)
        else:
            raise ValueError("No suitable function for pruning model!")
        flops, paras = get_flops_paras(pm, args.device, HW)
        drf = (flops_ori - flops) / flops_ori
        drp = (paras_ori - paras) / paras_ori
        et = time.time()
        latency['pruning'] = et - st
        try:
            del fm_in, fm_out, mask_fm_in, mask_fm_out, doc, scaled_doc
        except:
            pass
        logger.info('    Pruning success. Flops drop rate = {:.1%}. Paras drop rate = {:.1%}. Cost {:.3f}s'\
                    .format(drf, drp, latency["pruning"]))
    
    ''' training '''
    st = time.time()
    trainer = CMaTrainer(
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
        th=args.th
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
