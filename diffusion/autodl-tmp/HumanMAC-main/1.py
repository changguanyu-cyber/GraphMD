import numpy as np

def classical_mds(D, d=3):
    D2 = D ** 2
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    L = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
    V = eigvecs[:, :d]
    X = V @ L
    return X

def restore_batch_trajectory_from_distances(D_all):
    B, T, N, _ = D_all.shape
    traj_reconstructed = np.zeros((B, T, N, 3))
    for b in range(B):
        for t in range(T):
            D = D_all[b, t]
            coords_rec = classical_mds(D)
            traj_reconstructed[b, t] = coords_rec
    return traj_reconstructed

def compute_distance_matrices_batch(traj):
    diff = traj[..., :, np.newaxis, :] - traj[..., np.newaxis, :, :]  # (B, T, N, N, 3)
    dist_mats = np.linalg.norm(diff, axis=-1)  # (B, T, N, N)
    return dist_mats

def restore_traj_125_9_3(traj):
    """
    应用于 (125, 9, 3) 的轨迹
    """
    traj = traj[np.newaxis, ...]  # (1, 125, 9, 3)
    dist_mats = compute_distance_matrices_batch(traj)  # (1, 125, 9, 9)
    traj_rec = restore_batch_trajectory_from_distances(dist_mats)  # (1, 125, 9, 3)
    return traj_rec[0]  # 去掉 batch 维度 → (125, 9, 3)
traj = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/uni_traj.npy')
traj_reconstructed = restore_traj_125_9_3(traj)
np.save()

