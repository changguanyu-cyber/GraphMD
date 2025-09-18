import numpy as np
h=np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/energy_model/energy_pred.npy')
k=np.load('/root/autodl-tmp/HumanMAC-main/HumanMAC-main/inference/energy_model/energy_uni.npy')
h=h[0]
k=k[0]
print(h)
print(k)