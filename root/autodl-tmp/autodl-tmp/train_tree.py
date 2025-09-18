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
import os
import shutil
def extract_last_molecule_xyz(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 获取原子数
    num_atoms = int(lines[0].strip())
    lines_per_molecule = num_atoms + 2

    # 计算总帧数
    total_frames = len(lines) // lines_per_molecule

    # 获取最后一个分子的起始行索引
    start_line = (total_frames - 1) * lines_per_molecule
    end_line = start_line + lines_per_molecule

    # 提取并写入
    last_molecule = lines[start_line:end_line]
    with open(output_file, 'w') as f:
        f.writelines(last_molecule)
def expand_xyz_files(folder, i):
        start = 0
        end = 2**i - 1
        target_start = 2**i

        for j in range(start, end + 1):
            src_file = os.path.join(folder, f"frame_{j:04d}.xyz")
            dst_file = os.path.join(folder, f"frame_{target_start + j:04d}.xyz")
            shutil.copyfile(src_file, dst_file)


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
# Gnerate a directory of not present
md_workdir = 'mdtut'
if not os.path.exists(md_workdir):
    os.mkdir(md_workdir)
# Get the parent directory of SchNetPack
spk_path = os.path.abspath(os.path.join(os.path.dirname(spk.__file__), '../..'))
# Get the path to the test data
test_path = '/root/tests/testdata'
# Load model and structure
model_path = '/root/forcetut/best_inference_model'
molecule_path = '/root/n_p_y/aspirin.xyz'
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
system_temperature = 300 # Kelvin

# Set up the initializer
md_initializer = UniformInit(
    system_temperature,
    remove_center_of_mass=True,
    remove_translation=True,
    remove_rotation=True,
)
# Initialize the system momenta
md_initializer.initialize_system(md_system)
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
bath_temperature = 300  # K
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
n_steps = 10
create_empty_xyz_files(n_steps)
temp = 1
md_simulator.simulate(temp)
data = HDF5Loader(log_file)
# extract structure information from HDF5 data
md_atoms = data.convert_to_atoms()

# write list of Atoms to XYZ file
write(
    "/root/autodl-tmp/output_xyz_files/frame_0000.xyz",
    md_atoms,
    format="xyz"
)
shutil.rmtree('/root/mdtut')
for i in range(n_steps):
    expand_xyz_files("/root/autodl-tmp/output_xyz_files", i)
    for j in range(2**(i+1)):
        md_workdir = 'mdtut'
        if not os.path.exists(md_workdir):
            os.mkdir(md_workdir)
        extract_last_molecule_xyz(f'/root/autodl-tmp/output_xyz_files/frame_{j:04d}.xyz', '/root/autodl-tmp/test.xyz')
        model_path = '/root/forcetut/best_inference_model'
        molecule_path = '/root/autodl-tmp/test.xyz'
        molecule = read(molecule_path)
        n_replicas = 1
        md_system = System()
        md_system.load_molecules(
            molecule,
            n_replicas,
            position_unit_input="Angstrom"
        )
        system_temperature = 300  # Kelvin

        # Set up the initializer
        md_initializer = UniformInit(
            system_temperature,
            remove_center_of_mass=True,
            remove_translation=True,
            remove_rotation=True,
        )
        # Initialize the system momenta
        md_initializer.initialize_system(md_system)
        time_step = 0.5  # fs
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
        md_precision = torch.float32
        bath_temperature = 300  # K
        time_constant = 100  # fs
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
        # Set the path to the checkpoint file
        chk_file = os.path.join(md_workdir, 'simulation.chk')

        # Create the checkpoint logger
        checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

        # Update the simulation hooks
        simulation_hooks.append(checkpoint)
        # directory where tensorboard log will be stored to
        tensorboard_dir = os.path.join(md_workdir, 'logs')

        tensorboard_logger = callback_hooks.TensorBoardLogger(
            tensorboard_dir,
            ["energy", "temperature"],  # properties to log
        )

        # update simulation hooks
        simulation_hooks.append(tensorboard_logger)
        md_simulator = Simulator(md_system, md_integrator, md_calculator, simulator_hooks=simulation_hooks)

        md_simulator = md_simulator.to(md_precision)
        md_simulator = md_simulator.to(md_device)
        temp = 1
        md_simulator.simulate(temp)
        data = HDF5Loader(log_file)
        # extract structure information from HDF5 data
        md_atoms = data.convert_to_atoms()
        with open('/root/autodl-tmp/test.xyz', 'w') as f:
            pass  # 打开并立即关闭，将文件内容清空
        write(
            "/root/autodl-tmp/test.xyz",
            md_atoms,
            format="xyz"
        )
        # 源文件（要复制的文件）
        source_file = '/root/autodl-tmp/test.xyz'

        # 目标文件（要追加到的文件）
        target_file = f'/root/autodl-tmp/output_xyz_files/frame_{j:04d}.xyz'

        # 打开源文件读取内容
        with open(source_file, 'r') as src:
            data = src.read()
        with open(target_file, 'a') as tgt:
            tgt.write(data)

        with open('/root/autodl-tmp/test.xyz', 'w') as f:
            pass  # 打开并立即关闭，将文件内容清空
        shutil.rmtree('/root/mdtut')


