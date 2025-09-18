#查看版本
import torch
print(torch.__version__)
print(torch.version.cuda)
1.8.0
11.1

#安装
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric


#运行
source activate pytorch1.8

cd GNN

python -u main.py --cuda --cutoff 5