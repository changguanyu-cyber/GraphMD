from torch import tensor
from utils import *
from utils.script import sample_preprocessing

def mol_generator(data_set, model_select, diffusion, cfg, mode=None,
                   action=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    all_generated_trajectories = []
    uni_trajectories = []

    poses = {}
    draw_order_indicator = -1
    for k in range(0, nrow):


        data = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result_.npy')
        data = data[:1000]
        data = data[None, ...]
        print(data.shape)

        gt = data[0].copy()
        #gt[:, :1, :] = 0
        #data[:, :, :1, :] = 0
            #print(gt.shape)
        uni_trajectories.append(gt)

        gt = np.expand_dims(gt, axis=0)
        first_frame = gt[..., 0, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])

        traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])


        traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)

        mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode=mode)
        sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_mod,
                                                   mode_dict,)


        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = post_process(traj_est, cfg, first_frame)

            #print(traj_est.shape)

        all_generated_trajectories.append(traj_est)
    uni_trajectories=uni_trajectories[0]
    all_generated_trajectories=all_generated_trajectories[0]
    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/uni_traj.npy' , uni_trajectories)


    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy', all_generated_trajectories)


