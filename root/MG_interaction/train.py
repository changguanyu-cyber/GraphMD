import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
forcetut = './nap_20000'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)


from schnetpack.datasets import MD17
from ase.db import connect
ethanol_data = spk.data.AtomsDataModule(
    '/root/new_dataset_nap_20000.db',
    batch_size=64,
    distance_unit='Ang',
    property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
    num_train=16000,
    num_val=2000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=1,
    pin_memory=True, # set to false, when not using a GPU
)
# 在创建数据集后检查长度是否匹配

ethanol_data.prepare_data()

# 检查数据库内容

ethanol_data.setup()
properties = ethanol_data.dataset[0]
print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])


print('Forces:\n', properties[MD17.forces])
print('Shape:\n', properties[MD17.forces].shape)



cutoff = 5.
n_atom_basis = 30

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=MD17.energy)
pred_forces = spk.atomistic.Forces(energy_key=MD17.energy, force_key=MD17.forces)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets(MD17.energy, add_mean=True, add_atomrefs=False)
    ]
)
output_energy = spk.task.ModelOutput(
    name=MD17.energy,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_forces = spk.task.ModelOutput(
    name=MD17.forces,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)
logger = pl.loggers.TensorBoardLogger(save_dir=forcetut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(forcetut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=forcetut,
    max_epochs=300, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=ethanol_data)

from ase import Atoms


# set device
device = torch.device("cuda")

# load model
model_path = os.path.join(forcetut, "best_inference_model")
best_model = torch.load(model_path, map_location=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)

# create atoms object from dataset
structure = ethanol_data.test_dataset[0]
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)

# convert atoms to SchNetPack inputs and perform prediction
inputs = converter(atoms)
results = best_model(inputs)

print(results)