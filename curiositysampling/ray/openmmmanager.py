from curiositysampling.ray import SingleBiasedSim
from curiositysampling.utils.trajreporter import TrajReporter
from curiositysampling.utils.checkpointutils import save_json_object, save_pickle_object
import time
from collections import deque
import logging
import numpy as np
import os
import openmm as omm
import ray
from uuid import uuid4
from itertools import combinations
from itertools import chain


@ray.remote
class OpenMMManager:
    """OpenMMManager controls MD simulations , so that it allows for running,
    parallerization (using Ray framework) for several actors (or walkers).
    The OMM prepares files (pdb, force field), creates  directories, and instantiates new ray actors.
    Each actor represnets one agent, that takes it's own series of actions and states.
    Also it allows to control several parameters resposible for MD simulations and
    their featurization.
    Arguments:
        positions: initial positions for MD simulation.
        system: OpenMM compatibile system configuration for a given molecular system.
        topology: OpenMM compatibile topology for a given molecular system.
        integrator: OpenMM compatibile integrator.
        reporter_stride: How many integration steps have to be passed in order to
                         save an microstate as a observation. The observation is further
                         used as a training example.
        temperature: Temperature for given molecular system
        steps: Number of integration steps, before an observation is returned.
        file_per_cycle: If true, separate trajectory files are saved every cycle.
        saving_frequency: After every integration steps save trajectory in every cycle.
        warmup_cycles: number of cycles before trajectory is saved to file/files.
                       Generally, it's should at least of buffer's size.
        warmup_steps: number of integration steps while performing warmup cycles.
        warmup_reporter_stride: The same as reporter_stride, but for warmup period.
        regular_md: perform regular md simulation, without setting new positions and
                    resetting thermostat.
        cuda: If to use CUDA gpus or CPUs.

        use_dihedral: If to use dihedral as features for a given selection, if false, distances are used.
        add_chi_angles: If to add chi1, chi2, chi3 to the features of the dihedral angles. Requires to `use_dihedral` to be True.
        selection: Selection used for feature calculations. Default is "protein". The selection is of MDTraj/VMD style.
    """

    def __init__(
        self,
        positions=None,
        system=None,
        integrator=None,
        topology=None,
        saving_frequency=100,
        steps=1000,
        temperature=300,
        reporter_stride=100,
        regular_md=False,
        cuda=False,
        hip=False,
        gpus_per_agent=None,
        cpus_per_agent=None,
        file_per_cycle=True,
        warmup_cycles=0,
        warmup_steps=0,
        warmup_reporter_stride=10,
        use_dihedrals=False,
        use_chi1=False,
        use_chi2=False,
        use_chi3=False,
        use_chi4=False,
        use_positions=False,
        use_distances=False,
        distance_cutoff=None,
        selection="protein",
        use_distances_scheme="closest-heavy",
        boosting=False,
        boosting_temp=200,
        boosting_steps=5000,
        boosting_amd=False,
        boosting_amd_coef=0.2,
        boosting_amd_coef_tors=0.05,
        boosting_amd_Ecoef=1.0,
        boosting_amd_Ecoef_tors=1.0,
        position_list=None,
        box_list=None,
    ):
        self.cuda = cuda
        self.hip = hip
        self.gpus_per_agent = gpus_per_agent
        self.cpus_per_agent = cpus_per_agent
        self.total_cpus = 0
        self.total_gpus = 0
        self.total_agents = 0
        self.first_time_step = True
        self.logger = logging
        self.logger.basicConfig(
            format="Actor::{0} - %(asctime)s :: %(levelname)s - %(message)s :: (%(filename)s:%(lineno)d)".format(
                "OpenMMManager"
            ),
            level=logging.INFO,
        )

        if self.cpus_per_agent != os.environ["OMP_NUM_THREADS"]:
            self.log(
                "OMP_NUM_THREADS is set to {0} while cpus per agent to {1}".format(
                    os.environ["OMP_NUM_THREADS"], self.cpus_per_agent
                )
            )
            if self.cpus_per_agent is not None:
                to_show = str(self.cpus_per_agent)
            else:
                to_show = "<number of choice>"
            self.log(
                "Set export OMP_NUM_THREADS={0} before running the python code".format(
                    to_show
                ),
            )
        self.file_per_cycle = file_per_cycle
        self.warmup_cycles = warmup_cycles
        self.warmup_steps = warmup_steps
        self.warmup_reporter_stride = warmup_reporter_stride
        # MD simulation prep
        self.positions = positions
        self.system = system
        self.integrator = integrator
        self.topology = topology
        if position_list is not None and box_list is not None:
            self.set_init_actionsboxes(position_list, box_list)
            self.initial_actionsboxes = True
        else:
            self.initial_actionsboxes = False

        # MD simulation parameters
        self.steps = steps
        self.reporter_stride = reporter_stride
        self.regular_md = regular_md
        self.temperature = temperature
        self.saving_frequency = saving_frequency

        # Technical
        self.instances = {}
        self.max_conf_buffer_size = -5
        # self.one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.float32)
        # self.one_hot_encoder.fit(np.arange(start=0, stop=self.num_of_disjoint_cvs).reshape(-1, 1))
        self.distance_indices_global = None
        self.use_dihedrals = use_dihedrals
        self.use_distances = use_distances
        self.contact_cache = True
        self.distance_cutoff = distance_cutoff
        self.use_positions = use_positions
        self.use_chi1 = use_chi1
        self.use_chi2 = use_chi2
        self.use_chi3 = use_chi3
        self.use_chi4 = use_chi4
        self.selection = selection
        self.use_distances_scheme = use_distances_scheme
        self.boosting = boosting
        self.boosting_temp = boosting_temp
        self.boosting_steps = boosting_steps
        self.boosting_amd = boosting_amd
        self.boosting_amd_coef = boosting_amd_coef
        self.boosting_amd_coef_tors = boosting_amd_coef_tors
        self.boosting_amd_Ecoef = boosting_amd_Ecoef
        self.boosting_amd_Ecoef_tors = boosting_amd_Ecoef_tors

    def set_working_directory(self, working_directory=None):
        self.working_directory = working_directory

    def get_numbers_of_atoms(self, topology=None, selection=["N", "CA", "O", "CB"]):
        """Returns atom ids in the order from smallest to biggest, based on the
        passed selection.
        Arguments:
            toplogy: OpenMM topology object
            selection: List of atoms names, those should be considered.
        Returns:
            Sorted list of atom ids for a given selection
        """
        list_of_atom_id = []
        for atom in topology.atoms():
            if atom.name in selection:
                list_of_atom_id.append(int(atom.id))
        return list_of_atom_id

    def get_initial_positions(self):
        return self.positions

    def get_instance(self, sim_id=None):
        return self.instances[sim_id]["instance"]

    def create_new_instance(self):
        """Creates new ray actor that runs openmm biased simulation.
        All necessary files (force field, pdb, etc) and directories
        are prepared during the step.
        The method assigns new id to every ray actor, so that it is
        possible to run several instances at once through the class.
        Args:
            None
        Returns:
            instance id as a string
        """
        new_id = str(uuid4())
        self.total_agents += 1
        if self.cpus_per_agent is not None:
            if self.cuda or self.hip:
                if self.gpus_per_agent is not None:
                    RemoteClass = ray.remote(
                        num_gpus=self.gpus_per_agent, num_cpus=self.cpus_per_agent
                    )(SingleBiasedSim)
                    self.total_gpus += self.gpus_per_agent
                else:
                    RemoteClass = ray.remote(num_gpus=1, num_cpus=self.cpus_per_agent)(
                        SingleBiasedSim
                    )
                    self.total_gpus += 1
                self.total_cpus += self.cpus_per_agent
            else:
                RemoteClass = ray.remote(num_cpus=self.cpus_per_agent)(SingleBiasedSim)
                self.total_cpus += self.cpus_per_agent
                self.total_gpus = None
        else:
            if self.cuda or self.hip:
                self.total_cpus = None
                if self.gpus_per_agent is not None:
                    RemoteClass = ray.remote(num_gpus=self.gpus_per_agent)(
                        SingleBiasedSim
                    )
                    self.total_gpus += self.gpus_per_agent
                else:
                    RemoteClass = ray.remote(num_gpus=1)(SingleBiasedSim)
                    self.total_gpus += 1
            else:
                RemoteClass = ray.remote(SingleBiasedSim)
                self.total_gpus = None
                self.total_cpus = None

        new_instance = RemoteClass.remote(
            working_directory=self.working_directory,
            sim_id=new_id,
            positions=self.positions,
            system=self.system,
            topology=self.topology,
            integrator=self.integrator,
            reporter_stride=self.reporter_stride,
            temperature=self.temperature,
            steps=self.steps,
            cuda=self.cuda,
            hip=self.hip,
            num_gpus=self.gpus_per_agent,
            num_cpus=self.cpus_per_agent,
            file_per_cycle=self.file_per_cycle,
            warmup_cycles=self.warmup_cycles,
            warmup_steps=self.warmup_steps,
            warmup_reporter_stride=self.warmup_reporter_stride,
            use_contact_cache=self.contact_cache,
            regular_md=self.regular_md,
            saving_frequency=self.saving_frequency,
            selection=self.selection,
            boosting=self.boosting,
            boosting_temp=self.boosting_temp,
            boosting_steps=self.boosting_steps,
            boosting_amd=self.boosting_amd,
            boosting_amd_coef=self.boosting_amd_coef,
            boosting_amd_coef_tors=self.boosting_amd_coef_tors,
            boosting_amd_Ecoef=self.boosting_amd_Ecoef,
            boosting_amd_Ecoef_tors=self.boosting_amd_Ecoef_tors,
            max_conf_buffer_size=self.max_conf_buffer_size,
        )
        cur_dir = ray.get(new_instance.get_cur_dir.remote())
        self.instances[new_id] = {
            "instance": new_instance,
            "locked": False,
            "running_id": None,
            "cur_dir": cur_dir,
        }
        del cur_dir
        return new_id

    def step(self, sim_id=None, action=None, random_seed=None, box=None, energy=None):
        """Performs specified in `__init__` number of steps of MD simulation.
        The method is non-blocking and results are obtained by calling the
        method several times with the same sim_id.
        Arguments:
            sim_id: simulation id, that distingushes this instance from other.
            action: An Quantity object in units of nm, that contains
                    molecular system positions (including water, ions etc.).
            random_seed: random seed used to draw velocities
        """
        if self.first_time_step:
            self.log("Total number of CPUs used: {}".format(self.total_cpus))
            self.log("Total number of GPUs used: {}".format(self.total_gpus))
            self.log(
                "If CPUs are None, they weren't set explicitly and default to the Ray's default settings"
            )
            self.log("If GPUs are None, they aren't used at all")
            self.log(
                "Total number of Agents started by OPMM: {}".format(self.total_agents)
            )
            self.first_time_step = False
        instance = self.instances[sim_id]["instance"]
        if self.instances[sim_id]["locked"] == False:

            self.instances[sim_id]["locked"] = True
            running_id = instance.run.remote(
                action=action,
                box=box,
                energy=energy,
                use_distances_scheme=self.use_distances_scheme,
                use_dihedrals=self.use_dihedrals,
                use_distances=self.use_distances,
                distance_cutoff=self.distance_cutoff,
                use_positions=self.use_positions,
                use_chi1=self.use_chi1,
                use_chi2=self.use_chi2,
                use_chi3=self.use_chi3,
                use_chi4=self.use_chi4,
                random_seed=random_seed,
            )
            self.instances[sim_id]["running_id"] = running_id
        else:
            running_id = self.instances[sim_id]["running_id"]

        # We save running id for later, to be checked by `results` method
        ready, not_ready = ray.wait([running_id], timeout=0)

        if len(ready) == 0:
            return None
        else:
            results = ray.get(ready[0])
            # remove reference
            (
                box_sizes,
                dssp_calc,
                energy,
                trajectory,
                trajectory_obs,
                md_time,
            ) = results

            # deref
            del results, running_id, ready, not_ready

            self.instances[sim_id]["locked"] = False
            return box_sizes, dssp_calc, energy, trajectory, trajectory_obs, md_time

    def get_initbox(self):
        return self.system.getDefaultPeriodicBoxVectors()

    def get_perframe_unit(self):
        reporter_stride = self.reporter_stride
        stepsize = self.integrator.getStepSize()
        onestep = reporter_stride * stepsize

        return onestep

    def get_if_positions(self):

        return self.use_positions

    def save_checkpoint_data(self, data_path=None):

        openmm_dict = {}
        for i, key in enumerate(self.instances.keys()):
            openmm_dict["instance_" + str(i)] = {}
            openmm_dict["instance_" + str(i)]["directory"] = str(
                self.instances[key]["cur_dir"]
            )
            openmm_dict["instance_" + str(i)]["counter"] = int(
                ray.get(self.instances[key]["instance"].get_reporter_counter.remote())
            )

        dist_fname = save_pickle_object(
            self.distance_indices_global, path=data_path, fname="openmm_dist_ind"
        )
        full_fname = save_json_object(openmm_dict, path=data_path, fname="openmm_data")

        return full_fname, dist_fname

    def set_init_actionsboxes(self, position_list=None, box_list=None):

        self.position_list_deque = deque(position_list)
        self.box_list_deque = deque(box_list)

    def get_init_oneactionbox(self):
        if self.initial_actionsboxes:
            try:
                return (self.position_list_deque.pop(), self.box_list_deque.pop())
            except IndexError:
                raise ValueError(
                    "Most probably you didnt provide the same number of boxes/positions as number of agents/walkers"
                )
        else:
            return (None, None)

    def get_if_dihedrals(self):
        return self.use_dihedrals

    def get_simlength(self):
        stepsize = self.integrator.getStepSize()
        return (self.steps * stepsize).value_in_unit(omm.unit.nanoseconds)

    def get_if_distances(self):
        return self.use_distances

    def set_max_buffer(self, max_conf_buffer=-5):
        self.max_conf_buffer_size = max_conf_buffer

    def log(self, msg):
        self.logger.info(msg)
