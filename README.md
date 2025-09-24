# GRAPHMD:A TWO-MODULE DIFFUSION FRAMEWORK FOR SMOOTH AND CONSISTENT MOLECULAR DYNAMICS
Source code for our 

**Task Example**: A two-module molecular dynamics simulation approach: a molecular graph interaction module enhanced with classical potential functions, and a diffusion module that leverages the Discrete Cosine Transform (DCT) to better capture smooth molecular motions.

# Approach

# Download and prepare the datasets
1. Download the datasets:

   - [ActivityNet Captions C3D feature](https://example.com/ActivityNet_C3D.zip)
   - [TACoS C3D feature](https://example.com/TACoS_C3D.zip)

2. After downloading, unzip the files and place them under the `data/` directory:

```
data/
├── ActivityNet/
│ └── C3D_features/
└── TACoS/
└── C3D_features/
```

3. Extracting the `.npy` files.  Place all `.npy` files into the `data/` directory as follows:

```
data/
├── ActivityNet/
│ └── C3D_features/
└── TACoS/
└── C3D_features/
```
# Dependencies
we recommend installing the following packages:

```
Python >= 3.8
PyTorch >= 1.10
torchvision
numpy
tqdm
```

# Training
You can start training our two modules by using the following two commands, respectively:

```
data/
python train 1
```

```
data/
python train 2
```
