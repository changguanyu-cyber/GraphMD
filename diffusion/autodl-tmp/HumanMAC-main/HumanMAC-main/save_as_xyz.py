import numpy as np

positions = np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/result.npy')
positions =positions[0]
nuclear_charge = np.load('/root/rmd17/npz_data/rmd17_ethanol.npz')['nuclear_charges']

# 原子序数 → 元素符号映射表
periodic_table = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'
}

T, N, _ = positions.shape

with open('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/uff.xyz', 'w') as f:
    for t in range(T):
        f.write(f"{N}\n")
        f.write(f"Frame {t}\n")
        for i in range(N):
            symbol = periodic_table.get(nuclear_charge[i], 'X')  # fallback: 'X'
            x, y, z = positions[t, i]
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

