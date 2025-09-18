import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset):

    """
    script for drawing gifs in different modes
    """


    action_list = dataset['test'].prepare_iter_action(cfg.dataset)
    print(action_list)
    for i in range(0, len(action_list)):
        pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
        suffix = action_list[i]
        print(pose_gen)
            

