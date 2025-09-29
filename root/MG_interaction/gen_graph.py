import os
import numpy as np
import schnetpack as spk
from schnetpack.md import System
from ase.io import read
from schnetpack.md import UniformInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack import properties
from schnetpack.md import Simulator
import torch
from schnetpack.md.simulation_hooks import LangevinThermostat
from schnetpack.md.simulation_hooks import callback_hooks
from schnetpack.md.data import HDF5Loader
from ase.io import write
import shutil
def remove_trailing_blank_lines(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 从末尾开始移除空白行
    while lines and lines[-1].strip() == '':
        lines.pop()

    # 写回文件
    with open(file_path, 'w') as file:
        file.writelines(lines)
md_workdir = 'mdtut'
def create_empty_xyz_files(n, output_dir="/root/autodl-tmp/output_xyz_files"):
    num_files = 2 ** n
    print(f"Creating {num_files} empty .xyz files in folder '{output_dir}'...")

    # 创建目标文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 创建空白文件
    for i in range(num_files):
        file_path = os.path.join(output_dir, f"frame_{i:04d}.xyz")
        with open(file_path, 'w') as f:
            pass  # 不写任何内容，保持空白

# 示例：创建 2^3 = 8 个空白文件
# Gnerate a directory of not present
if not os.path.exists(md_workdir):
    os.mkdir(md_workdir)
# Get the parent directory of SchNetPack
spk_path = os.path.abspath(os.path.join(os.path.dirname(spk.__file__), '../..'))
# Get the path to the test data
test_path = '/root/tests/testdata'
# Load model and structure
model_path = '/root/asp_20000/best_inference_model'
molecule_path = '/root/aspirin.xyz'
# Load atoms with ASE
molecule = read(molecule_path)
# Number of molecular replicas
n_replicas = 1
# Create system instance and load molecule
md_system = System()
md_system.load_molecules(
    molecule,
    n_replicas,
    position_unit_input="Angstrom"
)
system_temperature = 500 # Kelvin

# Set up the initializer
md_initializer = UniformInit(
    system_temperature,
    remove_center_of_mass=True,
    remove_translation=True,
    remove_rotation=True,
)
# Initialize the system momenta
md_initializer.initialize_system(md_system)
print(md_system.velocities.shape)
print(md_system.velocities)
time_step = 0.5 # fs
# Set up the integrator
md_integrator = VelocityVerlet(time_step)
# set cutoff and buffer region
cutoff = 5.0  # Angstrom (units used in model)
cutoff_shell = 2.0  # Angstrom

# initialize neighbor list for MD using the ASENeighborlist as basis
md_neighborlist = NeighborListMD(
    cutoff,
    cutoff_shell,
    ASENeighborList,
)
md_calculator = SchNetPackCalculator(
    model_path,  # path to stored model
    "forces",  # force key
    "kcal/mol",  # energy units
    "Angstrom",  # length units
    md_neighborlist,  # neighbor list
    energy_key="energy",  # name of potential energies
    required_properties=[],  # additional properties extracted from the model
)
md_simulator = Simulator(
    md_system,
    md_integrator,
    md_calculator
)
# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    md_device = "cuda"
else:
    md_device = "cpu"
# use single precision
md_precision = torch.float32
# set precision
# Set temperature and thermostat constant
bath_temperature = 500  # K
time_constant = 100  # fs

# Initialize the thermostat
langevin = LangevinThermostat(bath_temperature, time_constant)

simulation_hooks = [
    langevin
]
# Path to database
log_file = os.path.join(md_workdir, "simulation.hdf5")

# Size of the buffer
buffer_size = 100

# Set up data streams to store positions, momenta and the energy
data_streams = [
    callback_hooks.MoleculeStream(store_velocities=True),
    callback_hooks.PropertyStream(target_properties=[properties.energy]),
]

# Create the file logger
file_logger = callback_hooks.FileLogger(
    log_file,
    buffer_size,
    data_streams=data_streams,
    every_n_steps=1,  # logging frequency
    precision=32,  # floating point precision used in hdf5 database
)

# Update the simulation hooks
simulation_hooks.append(file_logger)
#Set the path to the checkpoint file
chk_file = os.path.join(md_workdir, 'simulation.chk')

# Create the checkpoint logger
checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

# Update the simulation hooks
simulation_hooks.append(checkpoint)
# directory where tensorboard log will be stored to
tensorboard_dir = os.path.join(md_workdir, 'logs')

tensorboard_logger = callback_hooks.TensorBoardLogger(
    tensorboard_dir,
    ["energy", "temperature"], # properties to log
)

# update simulation hooks
simulation_hooks.append(tensorboard_logger)
md_simulator = Simulator(md_system, md_integrator, md_calculator, simulator_hooks=simulation_hooks)

md_simulator = md_simulator.to(md_precision)
md_simulator = md_simulator.to(md_device)
temp = 1000
md_simulator.simulate(temp)
data = HDF5Loader(log_file)
for prop in data.properties:
    print(prop)
# extract structure information from HDF5 data

velocities = data.properties['velocities']
velocities = velocities.reshape(-1,velocities.shape[-2],3)
np.save('/root/autodl-tmp/velocities.npy',velocities)
import numpy as np
import matplotlib.pyplot as plt
from schnetpack import units as spk_units


# Get the energy logged via PropertiesStream
import numpy as np
import matplotlib.pyplot as plt

# 获取能量
energies_calculator = data.get_property(properties.energy, atomistic=False)
energies_system = data.get_potential_energy()

# 时间轴 (fs)
time_axis = np.arange(data.entries) * data.time_step / spk_units.fs

# 单位转换 (kJ/mol → kcal/mol)
energies_system *= spk_units.convert_units("kJ/mol", "kcal/mol")

# 计算差值 ΔE
delta_energy = energies_system - energies_calculator
delta_energy = delta_energy-0.2
# 画图
plt.figure()
plt.plot(time_axis, delta_energy, label="ΔE = System - Logger", color="blue")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.ylabel("ΔE [kcal/mol]")
plt.xlabel("t [fs]")
plt.title("Difference between System and Logger Energies")
plt.legend()
plt.tight_layout()
plt.savefig("/root/autodl-tmp/potential_energy.png", dpi=300)
plt.show()
def plot_temperature(data):

    # Read the temperature
    temperature = data.get_temperature()

    # Compute the cumulative mean
    temperature_mean = np.cumsum(temperature) / (np.arange(data.entries)+1)

    # Get the time axis
    time_axis = np.arange(data.entries) * data.time_step / spk_units.fs  # in fs

    plt.figure(figsize=(8,4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("/root/autodl-tmp/temperature.png", dpi=300)
    plt.show()

#plot_temperature(data)
md_atoms = data.convert_to_atoms()

# write list of Atoms to XYZ file
write(
    "/root/autodl-tmp/traj.xyz",
    md_atoms,
    format="xyz"
)
shutil.rmtree('/root/mdtut')