import os
import numpy as np
from utils.mol_gen import mol_generator



def demo_visualize(mode, cfg, model, diffusion, dataset):

    """
    script for drawing gifs in different modes
    """


    mol_list = dataset['test'].prepare_iter_action(cfg.dataset)
    print(mol_list)
    for i in range(0, len(mol_list)):
        mol_gen = mol_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=mol_list[i], nrow=cfg.vis_row)
        suffix = mol_list[i]

            

