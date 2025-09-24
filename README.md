# GRAPHMD:A TWO-MODULE DIFFUSION FRAMEWORK FOR SMOOTH AND CONSISTENT MOLECULAR DYNAMICS
Source code for our 

**Task Example**: A two-module molecular dynamics simulation approach: a molecular graph interaction module enhanced with classical potential functions, and a diffusion module that leverages the Discrete Cosine Transform (DCT) to better capture smooth molecular motions.

# Approach

# Download and prepare the datasets
1. Download the datasets:

   - [MD17](https://figshare.com/articles/Revised_MD17_dataset_rMD17_/12672038/3)

2. After downloading, unzip the files and place them under the `root/` directory:

```
root/
├── rmd17/
│ └── rmd17_aspirin.npz/
└── diffusion/
└── MG_interaction/
```

3. Extracting the `.npy` files.  Place all `.npy` files into the `root/` directory as follows:

```
root/
├── extracted_data/
│ └── aspirin_coord.npy/
└── rmd17/
└── diffusion/
└── MG_interaction/
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

## Molecular Graph Interaction Module
```
python train 1
```

## Diffusion Module
```
python train 2
```

# Sampling
