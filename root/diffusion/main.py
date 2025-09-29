import argparse
import copy
import os
import sys
import time

# Add current working directory to sys.path for local imports
sys.path.append(os.getcwd())

# === Third-party libraries ===
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from tensorboardX import SummaryWriter
from rdkit import Chem
from rdkit.Chem import AllChem

# === Local utility modules ===
from config import Config, update_config
from utils import create_logger, seed_set
from utils.script import *

from utils.demo_visualize import demo_visualize
from utils.evaluation import compute_stats
from utils.mol_gen import mol_generator
from utils.training import Trainer

# === Models ===
from models.transformer import EMA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('--cfg',
                        #default='h36m', help='h36m or humaneva')
    parser.add_argument('--mode', default='train', help='train / pred ')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=10)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ckpt.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    args = parser.parse_args()

    """setup"""

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    dataset, dataset_multi_test = dataset_split(cfg)

    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg,args)

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
            #multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
            trainer = Trainer(
                model=model,
                diffusion=diffusion,
                dataset=dataset,
                cfg=cfg,
                logger=logger,
                tb_logger=tb_logger,
            )
            trainer.loop()
    else:

        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)




