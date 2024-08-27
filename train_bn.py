import argparse
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.office_home
import datasets.pacs
import datasets.vlcs
import datasets.domainnet

import trainers.meta_aug
import trainers.zsclip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.META = CN()
    cfg.TRAINER.META.MODE = 'D'  # number of context vectors
    cfg.TRAINER.META.N_CTX = 2  # number of context vectors
    cfg.TRAINER.META.N_PRO = 2  # number of prompt vectors
    cfg.TRAINER.META.ALPHA = 0.1
    cfg.TRAINER.META.ADAPT_LR = 0.0025
    cfg.TRAINER.META.LR_RATIO = 1.0
    cfg.TRAINER.META.LAYERS = 12  # number of prompt vectors
    cfg.TRAINER.META.PREC = "fp16"  # fp16, fp32, amp
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.ID = 0

    cfg.OPTIM_META = CN()
    cfg.OPTIM_META.NAME = "adam"
    cfg.OPTIM_META.LR = 0.0003
    cfg.OPTIM_META.WEIGHT_DECAY = 5e-4
    cfg.OPTIM_META.MOMENTUM = 0.9
    cfg.OPTIM_META.SGD_DAMPNING = 0
    cfg.OPTIM_META.SGD_NESTEROV = False
    cfg.OPTIM_META.RMSPROP_ALPHA = 0.99
    # The following also apply to other
    # adaptive optimizers like adamw
    cfg.OPTIM_META.ADAM_BETA1 = 0.9
    cfg.OPTIM_META.ADAM_BETA2 = 0.999
    # STAGED_LR allows different layers to have
    # different lr, e.g. pre-trained base layers
    # can be assigned a smaller lr than the new
    # classification layer
    cfg.OPTIM_META.STAGED_LR = False
    cfg.OPTIM_META.NEW_LAYERS = ()
    cfg.OPTIM_META.BASE_LR_MULT = 0.1
    # Learning rate scheduler
    cfg.OPTIM_META.LR_SCHEDULER = "single_step"
    # -1 or 0 means the stepsize is equal to max_epoch
    cfg.OPTIM_META.STEPSIZE = (-1, )
    cfg.OPTIM_META.GAMMA = 0.1
    cfg.OPTIM_META.MAX_EPOCH = 10
    # Set WARMUP_EPOCH larger than 0 to activate warmup training
    cfg.OPTIM_META.WARMUP_EPOCH = -1
    # Either linear or constant
    cfg.OPTIM_META.WARMUP_TYPE = "linear"
    # Constant learning rate when type=constant
    cfg.OPTIM_META.WARMUP_CONS_LR = 1e-5
    # Minimum learning rate when type=linear
    cfg.OPTIM_META.WARMUP_MIN_LR = 1e-5
    # Recount epoch for the next scheduler (last_epoch=-1)
    # Otherwise last_epoch=warmup_epoch
    cfg.OPTIM_META.WARMUP_RECOUNT = True
    
def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../DATA", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default=".", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/Meta_B2N_AUG/vit_b16_aug.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/b2n/eurosat.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="Meta_B2N_AUG", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true",default=False, help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
