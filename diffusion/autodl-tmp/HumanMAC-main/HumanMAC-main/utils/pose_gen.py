from torch import tensor
from utils import *
from utils.script import sample_preprocessing
from models.features import get_features

def pose_generator(data_set, model_select, diffusion, cfg, mode=None,
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
        data2 = data[0]
        atomic_number_to_symbol = {
            1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
            15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
            # 如有更多元素，请扩展
        }

        # 假设你有如下原子序号：
        molecule = np.load('/root/rmd17/npz_data/rmd17_aspirin.npz')
        nuclear_charges = molecule['nuclear_charges']

        # 转换为元素符号
        species = [atomic_number_to_symbol[z] for z in nuclear_charges]
        coord = data2.tolist()
        bond_connectivity_list = [[2, 5, 14], [3, 6, 15], [0, 3, 16], [1, 2, 17], [11, 18, 19, 20], [0, 6, 10], [1, 5, 12], [10], [11], [10, 13], [5, 7, 9], [4, 8, 12], [6, 11], [9], [0], [1], [2], [3], [4], [4], [4]]

        features = get_features(coord, species, bond_connectivity_list)
        num_atoms = len(nuclear_charges)
        λ = 0.01
        λ = λ/num_atoms
        sigma = features['nonbond'].sum()*λ

        # 加和 bond + angle + diheral
        mu = features['bond'].sum()*λ + features['angle'].sum()*λ + features['diheral'].sum()*λ
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
                                                   mode_dict,
                                                   sigma,
                                                   mu,
                                                   features)


        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        traj_est = traj_est.cpu().numpy()
        traj_est = post_process(traj_est, cfg, first_frame)

            #print(traj_est.shape)

        all_generated_trajectories.append(traj_est)
    uni_trajectories=uni_trajectories[0]
    all_generated_trajectories=all_generated_trajectories[0]
    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/uni_traj.npy' , uni_trajectories)


    np.save('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy', all_generated_trajectories)
    print('yi bao cun')


