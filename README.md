# Curiosity Sampling
![image](https://user-images.githubusercontent.com/128594297/226975861-8a8556cb-a8b8-40a5-b9ae-9f76f23e0e03.png)


Curiosity Sampling is an algorithm that enables efficient sampling of protein conformational landscapes using a Reinforcement Learning-based algorithm that specifically samples configurations that maximize the "novelty" or curiosity reward. This method can overcome the limitations of traditional molecular dynamics (MD) simulations and efficiently explore the high-dimensional configurational space of proteins.

# Installation
Expected time of tinstallation is about 30 minutes, so prepare your something tasty to drink. 

## Required libraries

To install Curiosity Sampling you will need Python 3.9 or newer and Conda package manager. You can find instructions to install them here.

The additional libraries needed to work with the package are:

* OpenMM
* SciPy
* NumPy
* OpenMMTools
* tqdm
* MDTraj
* Parmed
* Sci-Kit-learn
* DiskCache
* ray-all
* yaml
* TensorFlow
* TensorFlow_probability
* TensorFlow_addons
* TensorBoard (tensorflow-rocm for AMD)
* CUDNN (for Nvidia GPU)
* CUDA toolkit (for Nvidia GPU)
* ROCm libraries and ROCm Tensorflow/OpenMM if you use AMD graphic cards


For Python and Conda, you can download and install them from the official websites:

* [Python](https://www.python.org/downloads/)
* [Conda](https://docs.conda.io/en/latest/miniconda.html)

Where Python is installed automatically through conda

## Library installation through conda 

To switch the solver from the basic [Conda solver to the Mumba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community), run the following commands:

```python
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Then, depending on your platform, run one of the following commands:

### For Nvidia Graphic Cards and CPU platforms (OpenMM requires CUDA through Conda):

```bash
conda create -n env_curiosity_cuda -c conda-forge openmm scipy cudnn numpy openmmtools tqdm mdtraj parmed scikit-learn conda-forge::cudatoolkit diskcache ray-all yaml tqdm -y
conda activate env_curiosity_cuda
yes | pip install tensorflow tensorflow_probability tensorflow_addons tensorboard
```

### For AMD cards and other supporting ROCm HIP platforms:

Installation of ROCm on AMD platforms is [here](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.1/page/How_to_Install_ROCm.html)

```bash
conda create -n env_curiosity_hip -c conda-forge openmm scipy numpy openmmtools tqdm mdtraj parmed scikit-learn diskcache ray-all yaml tqdm -y 
conda install -c streamhpc -c conda-forge --strict-channel-priority openmm-hip -y
conda activate env_curiosity_hip
yes | pip install tensorflow-rocm tensorflow_probability tensorflow_addons tensorboard
```

## Curiosity Sampling installation when all libraries are installed

After setting up the Conda environment, clone the Curiosity Sampling repository:

```bash
git clone https://github.com/CuriositySampling/CuriositySampling.git
cd CuriositySampling
```
Then, install Curiosity Sampling by running:

```bash
pip install .
```

# Usage
## Jupyter Notebook
In the doc/examples there is a Jupyter Notebook with Alanine Dipeptide, in detail explanations and excepted outputs. 

## General version
To start using Curiosity Sampling, you first need to prepare an [OpenMM system](http://docs.openmm.org/latest/userguide/application/02_running_sims.html).
The running time for the example is 2h for CPU only platform, though preparing the input should not exceed 10 minutes.

Here is an example:

```python
from openmm.app import *
from openmm import *
from openmm.unit import *

# Load PDB
# You can use Chignolin as an example
# wget https://files.rcsb.org/view/1UAO.pdb input.pdb; mv 1UAO.pdb input.pdb 
# other proteins will need a preparation with PDBfixer or CHARMM-GUI
pdb = PDBFile('input.pdb')

# Define forcefield 
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
# Start building system
modeller = Modeller(pdb.topology, pdb.positions)
print('Adding hydrogens...')
modeller.addHydrogens(forcefield)
print('Adding solvent...')
modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer)
print('Minimizing...')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

# Define integrator and other
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
print('Saving...')
# provide asNumpy=True to work with Curiosity Sampling
positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
topology = modeller.topology
```

Then we need to initialize the Curiosity Sampling algorithm. First we need to initialize `Ray` parallel library, through:
```python
import ray
# num of cpus on your computer, if you use cuda=True, you have to specify num_of_gpus=1 also
ray.init(num_cpus=6)
```
Then we need to initialize two modules of Curiosity Sampling:
* OpenMMManager - class which manages MD simulations inside Curiosity Sampling
* CuriousSampling - class that manages Curiosity Sampling simulation in general

The easiest way is to start from the default parameters and modify them (see Jupyter Notebook in the examples). Here we show without modification (go to the examples directory for **Alanine Dipeptide Jupyter Notebook example, this one should take about 20 minutes to run and provides excepted output in the bottom of the notebook**).

```python
from curiositysampling.utils import DEFAULT_OMM_SETTINGS
from curiositysampling.utils import DEFAULT_RND_SETTINGS
```

Setup only the most important parameters, such as the number of integration steps per episode, temperature, number of slow dynamics processes, and lagtime.

```python
from copy import deepcopy

# copy default settings
my_omm_settings = copy.deepcopy(DEFAULT_OMM_SETTINGS)
my_rnd_settings = copy.deepcopy(DEFAULT_RND_SETTINGS)

my_omm_settings['temperature'] = 300 # 300K
my_omm_settings['steps'] = 50000 # 50000 integration steps per episode per agent, 200 ps simulation with timestep of 4 fs


my_rnd_settings['model']['target']['dense_out'] = 3 # number k of slow processes target network
my_rnd_settings['model']['predictor']['dense_out'] = 3 # number k of slow processes predictor network
my_rnd_settings['autoencoder_lagtime'] = 10 # lag time tau in units of frames
```

Then we have to pass OpenMM's topology, system, integrator, initial positions:

```python
# set initial positions for all agents. Have to be numpy array (units nm) or Quanity object from OpenMM
my_omm_settings['positions'] = positions
# system object from OpenMM
my_omm_settings['system'] = system
# topology object from OpenMM
my_omm_settings['topology'] = topology
# integrator object from OpenMM
my_omm_settings['integrator'] = integrator

```



Define objects based on the OpenMMManager and Curious Sampling and pass settings.

```python
from curiositysampling.ray import OpenMMManager
from curiositysampling.core import CuriousSampling

omm = OpenMMManager.remote(**my_omm_settings)
my_rnd_settings = copy.deepcopy(DEFAULT_RND_SETTINGS)
```

Set the number of agents in parallel, the number of episodes to calculate, and the working directory and we are ready to go!


```python
# we define three parallel actors
number_of_agents = 3
# we will calculate 10 episodes
number_of_episodes = 10
# we will use current directory to save simulation
working_directory = os.getcwd()
# omm object encapsulated
config_env = {'openmmmanager': omm}

# Create CuriositySampling object - it may take a while, as it performs one initial episode
csm = CuriousSampling(rnd_config=my_rnd_settings, env_config=config_env,
                      number_of_agents=number_of_agents, working_directory=working_directory)
                      
# Run the Curiosity Sampling for N episodes
csm.run(number_of_episodes)

```

Remember to remove `tmp_data` directory before a new run, that is generated by the Curiosity Sampling algorithm.


# Licence
This work is licensed under the CC BY-NC-SA 4.0 (see LICENSE.txt for more)
