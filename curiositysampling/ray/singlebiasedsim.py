import copy
import logging
import os
import random
from collections import deque
from itertools import combinations
from time import time

import mdtraj as md
import numpy as np
from curiositysampling.utils import (
    DCDReporterMultiFile,
    EnergySelect,
    TrajReporter,
    atom_sequence,
    compute_contacts,
)
from openmm import *
from openmm.app import *
from openmm.app import DCDFile
from openmm.unit import *

import ray


class SingleBiasedSim:
    """Class that implements Molecular Dynamics simulation, that can be restarted from cetain configuration,
    that is provided externally. Furthermore, the class allows for parallerization for several walkers using
    Ray framework.
    Arguments:
        sim_id: simulation id, that distingushes this instance from other. Usually an uuid.
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
        saving_frequency: after how many integration steps, energies, positions are saved
        warmup_cycles: number of cycles before trajectory is saved to file/files.
                       Generally, it's should at least of buffer's size.
        warmup_steps: number of integration steps while performing warmup cycles.
        warmup_reporter_stride: The same as reporter_Stride, but during warmup
        regular_md: perform regular md simulation, without setting new positions and
                    resetting thermostat.
        cuda: If set to true, use CUDA and GPUs, if set to False, you CPU platform

        frequency: how often (in number of integration steps) deposit a gaussian.
        selection: Selection used for feature calculations. The selection is of MDtraj/VMD.
    """

    def __init__(
        self,
        working_directory=None,
        sim_id=None,
        positions=None,
        system=None,
        topology=None,
        integrator=None,
        reporter_stride=100,
        temperature=None,
        frequency=1000,
        steps=1000,
        cuda=False,
        hip=False,
        num_gpus=None,
        num_cpus=None,
        saving_frequency=100,
        file_per_cycle=False,
        warmup_cycles=0,
        warmup_steps=1000,
        warmup_reporter_stride=10,
        regular_md=False,
        selection="protein",
        use_distances_scheme=None,
        use_contact_cache=None,
        boosting=False,
        boosting_temp=200,
        boosting_steps=5000,
        boosting_amd=False,
        boosting_amd_coef=None,
        boosting_amd_coef_tors=None,
        boosting_amd_Ecoef=None,
        boosting_amd_Ecoef_tors=None,
        max_conf_buffer_size=-5,
    ):
        self.working_directory = working_directory
        self.max_conf_buffer_size = max_conf_buffer_size
        self.sim_id = sim_id
        self.logger = logging
        self.logger.basicConfig(
            format="Actor::{0} - %(asctime)s :: %(levelname)s - %(message)s :: (%(filename)s:%(lineno)d)".format(
                self.sim_id
            ),
            level=logging.INFO,
        )

        self.system = system
        # group forces for energy calc
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force.setForceGroup(i)

        self.topology = topology
        self.topology_mdtraj = md.Topology.from_openmm(self.topology)
        self.integrator = integrator
        # initial positions
        if isinstance(positions, Quantity):
            if isinstance(positions[0]._value, vec3.Vec3):
                positions = positions._value * positions.unit
        elif isinstance(positions, list):
            if isinstance(positions[0], vec3.Vec3):
                positions = positions._value
            elif positions[0].shape[1] > 1:
                positions = positions[0]
            else:
                raise ValueError("Input positions are of wrong format")
            if isinstance(positions, Quantity):
                pass
            else:
                positions = Quantity(positions, nanometer)
        else:
            # they were converted from nanometers
            positions = Quantity(positions, nanometer)
        self.positions = positions
        # base it on initial positions
        pospros = (
            np.array([v for v in self.positions.value_in_unit(nanometer)])
            .reshape(1, -1, 3)
            .astype(np.float32)
        )
        if pospros.shape[1] > 1:
            self.superpose_traj = md.Trajectory(
                pospros,
                self.topology_mdtraj,
            )
            # set periodic vectors
            if self.system.usesPeriodicBoundaryConditions():
                vec = [
                    v.value_in_unit(nanometer)
                    for v in system.getDefaultPeriodicBoxVectors()
                ]
                self.superpose_traj.unitcell_vectors = np.array(vec).reshape(1, 3, 3)
            self.superpose_traj.center_coordinates()
        if pospros.shape[1] < 1:
            raise ValueError("Wrong input positions")
        self.temperature = temperature
        self.file_per_cycle = file_per_cycle
        self.warmup_cycles = warmup_cycles
        self.warmup_steps = warmup_steps
        self.warmup_cycles_to_go = self.warmup_cycles
        self.warmup_reporter_stride = warmup_reporter_stride
        self.reporter_stride = reporter_stride
        self.regular_md = regular_md
        self.selection = selection
        self.distance_ind = None
        self.contact_cache = use_contact_cache

        self.steps = steps
        self.cache_distance_ind = None
        self.torsion_id = None
        self.max_energy = None
        self.energy = None
        self.max_torsenergy = None
        self.boosting = boosting
        self.boosting_temp = boosting_temp
        self.boosting_steps = boosting_steps
        self.boosting_amd = boosting_amd
        self.boosting_amd_coef = boosting_amd_coef
        self.boosting_amd_coef_tors = boosting_amd_coef_tors
        self.boosting_amd_Ecoef = boosting_amd_Ecoef
        self.boosting_amd_Ecoef_tors = boosting_amd_Ecoef_tors
        if self.boosting:
            if not self.boosting_amd:
                self.log(
                    "Using boosting to boost barrier crossing, temperature +{0} K, {1}k steps".format(
                        self.boosting_temp, self.boosting_steps // 1000
                    )
                )
            else:
                self.log(
                    "Using boosting to boost barrier crossing, alpha coef {0}, accelerated md with {1} of max E, {2}k steps".format(
                        self.boosting_amd_coef,
                        self.boosting_amd_Ecoef,
                        self.boosting_steps // 1000,
                    )
                )
                self.log(
                    "Using boosting to boost barrier crossing, alpha coef torsional {0}, accelerated md with {1} of max E torsional, {2}k steps".format(
                        self.boosting_amd_coef_tors,
                        self.boosting_amd_Ecoef_tors,
                        self.boosting_steps // 1000,
                    )
                )

        # Metadynamics parameters
        if self.boosting:
            self.system_boost = copy.deepcopy(self.system)
            # remove barostat, so the boosting is in NVT
            for i in range(self.system_boost.getNumForces()):
                forname = self.system_boost.getForce(i).__class__.__name__
                if forname.count("Barostat"):
                    self.system_boost.removeForce(i)
                    break
            # check if there are torsional forces
            for i in range(self.system_boost.getNumForces()):
                forname = self.system_boost.getForce(i).__class__.__name__
                # check if there are torsional forces
                if forname.count("Torsion"):
                    self.torsion_id = i
                    break
                else:
                    self.torsion_id = None
            if not self.boosting_amd:
                self.integrator_boost = copy.deepcopy(self.integrator)
                # Halve the timestep for stability
                self.integrator_boost.setStepSize(self.integrator.getStepSize() / 2)
                try:
                    self.temperature_boost = self.temperature + self.boosting_temp
                except:
                    self.temperature_boost = (
                        self.temperature + self.boosting_temp * kelvin
                    )
                self.integrator_boost.setTemperature(self.temperature_boost)
            else:
                if self.torsion_id is None:
                    self.integrator_boost = AMDIntegrator(
                        self.integrator.getStepSize() / 2,
                        1e16 * kilojoule / mole,
                        -1e16 * kilojoule / mole,
                    )
                else:
                    self.log("Torsional angles detected, dual aMD is used")
                    self.integrator_boost = DualAMDIntegrator(
                        self.integrator.getStepSize() / 2,
                        self.torsion_id,
                        1e16 * kilojoule / mole,
                        -1e16 * kilojoule / mole,
                        1e16 * kilojoule / mole,
                        -1e16 * kilojoule / mole,
                    )

        # Simulation
        if cuda or hip:
            if cuda:
                platform = Platform.getPlatformByName("CUDA")
            else:
                platform = Platform.getPlatformByName("HIP")
            if num_gpus is not None:
                properties = {
                    "Precision": "mixed",
                    "DeviceIndex": ",".join([str(i) for i in range(num_gpus)]),
                }
            else:
                properties = {"Precision": "mixed"}
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, platform, properties
            )
            if self.boosting:
                self.simulation_boost = Simulation(
                    self.topology,
                    self.system_boost,
                    self.integrator_boost,
                    platform,
                    properties,
                )

        else:
            platform = Platform.getPlatformByName("CPU")
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, platform
            )
            if self.boosting:
                self.simulation_boost = Simulation(
                    self.topology, self.system_boost, self.integrator_boost, platform
                )

        self.simulation.context.setPositions(self.positions)
        # max size of frames
        self.max_traj_size = self.steps // self.reporter_stride
        # temp. conf buffer
        self.conf_buffer = deque(maxlen=self.max_conf_buffer_size)
        self.log("Configuration Buffer size is {}".format(self.max_conf_buffer_size))
        # Create reporters
        # Create folder for results:
        self.cur_dir = self.working_directory + "/" + self.sim_id
        if not os.path.exists(self.cur_dir):
            os.makedirs(self.cur_dir)
        # reporter
        if self.warmup_cycles > 0:
            self.trajectory_reporter = TrajReporter(
                report_interval=self.warmup_reporter_stride,
                all_frames_dims=(
                    self.max_traj_size,
                    self.positions.shape[0],
                    self.positions.shape[1],
                ),
            )
        else:
            self.trajectory_reporter = TrajReporter(
                report_interval=self.reporter_stride,
                all_frames_dims=(
                    self.max_traj_size,
                    self.positions.shape[0],
                    self.positions.shape[1],
                ),
            )

        self.simulation.reporters.append(self.trajectory_reporter)
        if self.file_per_cycle:
            self.dcdreporter = DCDReporterMultiFile(
                self.cur_dir + "/traj" + ".dcd", saving_frequency
            )
        else:
            self.dcdreporter = DCDReporter(
                self.cur_dir + "/traj" + ".dcd", saving_frequency
            )
        self.simulation.reporters.append(
            StateDataReporter(
                self.cur_dir + "/scalars" + ".csv",
                saving_frequency,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                density=True,
                volume=True,
                temperature=True,
                speed=True,
                remainingTime=True,
                totalSteps=True,
            )
        )
        self.out_dcd_file = open(self.cur_dir + "/actionpostions" + ".dcd", "wb")
        self._dcd = DCDFile(
            self.out_dcd_file,
            self.topology,
            self.integrator.getStepSize(),
            self.simulation.currentStep,
            1,
            False,
        )
        # other
        self.run_once = True

        self.log(
            "First actions are usually shown as coordinates, later on they are index of the trajectory. The first index is the index chosen either from configuration buffer (then the number is conf. buffer index + max(MD traj length)) or trajectory. The second index always describes index chosen directly from trajectory"
        )

    def get_initial_positions(self):
        """Return initial positions for MD simulation"""
        return self.positions

    def run(
        self,
        action,
        box,  # TODO: Remove it from the turnover
        energy,
        use_distances_scheme,
        use_dihedrals=False,
        use_distances=False,
        distance_cutoff=None,
        use_positions=False,
        use_chi1=False,
        use_chi2=False,
        use_chi3=False,
        use_chi4=False,
        save_action_to_file=True,
        random_seed=None,
    ):
        """Performs one MD simulation step.
        Arguments:
            action: An Quantity object in units of nm, that contains
                    molecular system positions (including water, ions etc.).
            box: box size from action
            energy: energy tuple obtained from action
            dihedral: If use cos(dihedral_angle) of protein's backbone dihedral angles instead
                      of distance matrixces as a feature vector.
            save_action_to_file: If save every action to a file, in order to investigate what
                                 molecular configurations were chosen by the algorithm.
            add_chi_angles=: If to add chi angles to features. `dihedral` must be True then.
            positions_only: Use positions as features, overrides all other options.
            random_seed: random seed used to draw velocities

        Returns:
            Returns a tuple of three variables, free energy at the end of simulation,
            trajectory of all structures saved every `reporter_stride` and trajectory_obs
            with distance_matrices or dihedrals, saved every `reporter_stride`.
        """
        self.log("Action entering agent {0} is {1}".format(self.sim_id, action))

        # we assume than it started with indices, not files
        if isinstance(action[0], np.int64):
            pass
        elif isinstance(action[0][0], np.int64):
            action = action[0]

        # propagate actions from indices
        if isinstance(action[0], np.int64):
            # add to the configuration buffer
            # only top confs from trajectory are added here, and in curioussampling.py
            if action[1] >= 0:
                self.conf_buffer.append(
                    (
                        copy.deepcopy(
                            self.trajectory_reporter.get_trajectory()["xyz"][action[1]]
                        ),
                        copy.deepcopy(
                            self.trajectory_reporter.get_trajectory()[
                                "unitcell_vectors"
                            ][action[1]]
                        ),
                    )
                )

            # it means, configuration comes from configuration buffer
            if action[0] >= self.max_traj_size:
                # -1 because len(buffer) is added, which is between 0 ... N
                try:
                    action, box = self.conf_buffer[action[0] - self.max_traj_size]
                except Exception as e:
                    self.log(e)
                    self.log(self.conf_buffer)
                    self.log("Len of buffer: {}".format(len(self.conf_buffer)))
                    raise ValueError("The action index was {}".format(action[0]))
            elif action[0] >= 0:
                # action pased to MD
                action = self.trajectory_reporter.get_trajectory()["xyz"][action[0]]
            else:
                raise ValueError("Action number have to be positive number")

        start_time = time()
        if not isinstance(action, (np.ndarray, np.generic)):
            if isinstance(action, Quantity):
                action = np.array(action._value) * action.unit
            else:
                action = np.array(action._value) * nanometer

        if action.ndim > 2:
            position = np.squeeze(action)
        elif action.ndim < 2:
            position = np.expand_dims(action, axis=0)
        else:
            position = action
        if isinstance(position, Quantity):
            pass
        else:
            # they were converted from nanometers
            position = Quantity(position, nanometer)
        if isinstance(box, Quantity):
            pass
        else:
            # they were converted from nanometers
            box = Quantity(box, nanometer)
        # make sure the shape is correct
        position = position.reshape((-1, 3))
        # save action to a file
        if save_action_to_file:
            self._dcd.writeModel(
                position,
                periodicBoxVectors=box,
            )

        # use boosting
        if self.boosting:
            position, box = self.boost_sim(position, box, energy)

        if not self.regular_md:
            self.set_position(position, random_seed, box)
        # perform one step MD
        if self.warmup_cycles_to_go > 0:
            steps = self.warmup_steps
            self.warmup_cycles_to_go -= 1
            self.log("Performing warmup cycle")
        else:
            steps = self.steps
            if self.file_per_cycle:
                self.dcdreporter.nextCycle(self.simulation)

            # start saving after warmup is set
            if self.run_once:
                self.trajectory_reporter._reportInterval = self.reporter_stride
                self.simulation.reporters.append(self.dcdreporter)
                # set a new file if file per cycle set
                self.dcdreporter.describeNextReport(self.simulation)
                self.log("Trajectory is going to be saved from now")
                self.run_once = False

        # flush at the beginning
        self.trajectory_reporter.flush_reporter()

        valid = False
        for i in range(10):
            try:
                # it will still save the broken simulation to the file
                # it can be prevented when simulation is saved as separate files instead of single one
                # TODO remove last traj file if simulation fails
                self.simulation.step(steps)
                valid = True
                break
            except OpenMMException:
                self.log(
                    "Warning, particle coord is nan, restarting from last position. Restart times {}".format(
                        i
                    )
                )
                self.set_position(
                    position, random.randint(-2147483646, 2147483646), box
                )
        if not valid:
            raise OpenMMException(
                "Tried 10 times to restart simulation, Particle is still NaN"
            )

        # get saved trajectory positions
        # TODO: Save only features, not whole trajectory
        # TODO: pass a method for self.metric to reporter, and use it to
        # on-the-fly featurization
        trajectory = self.trajectory_reporter.get_trajectory()
        # get box sizes
        # box_sizes = np.array([b for b in self.trajectory_reporter.get_box()])
        box_sizes = self.trajectory_reporter.get_box()
        # get energies
        self.calc_energystats()
        # get nparray object out of the Quantity object
        # trajectory = [positions._value for positions in trajectory]
        # TODO: Save only features, not whole trajectory
        # get positions of backbone atoms:
        trajectory_obs = self.metric(
            trajectory,
            self.selection,
            self.topology_mdtraj,
            use_distances_scheme,
            use_dihedrals=use_dihedrals,
            use_distances=use_distances,
            distance_cutoff=distance_cutoff,
            use_positions=use_positions,
            use_chi1=use_chi1,
            use_chi2=use_chi2,
            use_chi3=use_chi3,
            use_chi4=use_chi4,
        )

        md_time = time() - start_time
        sel = self.topology_mdtraj.select("protein")
        if sel.shape[0] > 0:
            traj = md.Trajectory(
                trajectory["xyz"].astype(np.float32), self.topology_mdtraj
            )
            traj.unitcell_vectors = trajectory["unitcell_vectors"].astype(np.float32)
            sub_traj = traj.atom_slice(sel)
            dssp_calc = md.compute_dssp(sub_traj, simplified=True)
        else:
            dssp_calc = None

        return (
            box_sizes,
            dssp_calc,
            self.energy,
            # instead sending whole trajectory, we send the action indices
            # first index = top trajectory
            # second index = top trajectory, without buffer
            # we distinguis with -42 to double check
            np.stack(
                [
                    np.arange(trajectory["xyz"].shape[0]),
                    np.arange(trajectory["xyz"].shape[0]) * 0 - 42,
                ]
            ).T,
            trajectory_obs,
            md_time,
        )

    def set_position(self, positions, random_seed, box):
        """Set molecular configuration, that is going to be sampled during the next step.
        Arguments:
            positions: molecular configuration of type Quantity
            random_seed: random seed used to draw velocities
            box: box vectors of the simulation, if PBC is set
        """
        self.simulation.context.reinitialize()
        if self.system.usesPeriodicBoundaryConditions():
            a, b, c = box
            a = np.squeeze(a)
            b = np.squeeze(b)
            c = np.squeeze(c)
            self.simulation.context.setPeriodicBoxVectors(a, b, c)
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocitiesToTemperature(
            self.temperature, random_seed
        )
        # set different random seed than velocites
        self.integrator.setRandomNumberSeed(random_seed + random.randint(-200, 200))

    def boost_sim(self, positions, box_vectors, energy):
        """Perform few barrier-boosted steps to leave a local minima
        Arguments:
            pass
        Returns:
            positons
        """
        self.simulation_boost.context.reinitialize()
        energy = np.squeeze(energy)
        _, e_max, e_min, e_torsion_max, e_torsion_min = energy
        if self.system.usesPeriodicBoundaryConditions():
            a, b, c = np.squeeze(box_vectors)
            a = np.squeeze(a)
            b = np.squeeze(b)
            c = np.squeeze(c)
            self.simulation_boost.context.setPeriodicBoxVectors(a, b, c)

        self.simulation_boost.context.setPositions(positions)
        if not self.boosting_amd:
            self.simulation_boost.context.setVelocitiesToTemperature(
                self.temperature_boost
            )
            self.integrator_boost.setRandomNumberSeed(
                random.randint(-2147483646, 2147483646)
            )

        else:
            if e_max is not None:
                max_energy = e_max
                alpha = (
                    np.abs(e_max - e_min) * self.boosting_amd_coef * kilojoule / mole
                )
                E = (
                    (max_energy + np.abs(max_energy) * (self.boosting_amd_Ecoef - 1))
                    * kilojoule
                    / mole
                )
                if self.torsion_id is None:
                    self.integrator_boost.setE(E)
                    self.integrator_boost.setAlpha(alpha)
                else:
                    self.integrator_boost.setETotal(E)
                    self.integrator_boost.setAlphaTotal(alpha)

                    torsion_max_energy = e_torsion_max
                    E_group = (
                        (
                            torsion_max_energy
                            + np.abs(torsion_max_energy)
                            * (self.boosting_amd_Ecoef_tors - 1)
                        )
                        * kilojoule
                        / mole
                    )
                    alpha_group = (
                        np.abs(e_torsion_max - e_torsion_min)
                        * self.boosting_amd_coef_tors
                        * kilojoule
                        / mole
                    )
                    self.integrator_boost.setEGroup(E_group)
                    self.integrator_boost.setAlphaGroup(alpha_group)

        valid = False
        for i in range(10):
            try:
                self.simulation_boost.step(self.boosting_steps)
                valid = True
                break
            except OpenMMException:
                self.log(
                    "Warning, particle coord is nan for BOOSTING, restarting from last position. Restart times {}".format(
                        i
                    )
                )
                self.simulation_boost.context.setPositions(positions)
                self.simulation_boost.context.setVelocitiesToTemperature(
                    self.temperature_boost, random.randint(-2147483646, 2147483646)
                )
                self.integrator_boost.setRandomNumberSeed(
                    random.randint(-2147483646, 2147483646)
                )
        if not valid:
            raise OpenMMException(
                "Tried 10 times to restart simulation, Particle is still NaN"
            )

        state = self.simulation_boost.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        positions = state.getPositions(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        return positions, box_vectors

    def calc_positions(self, trajectory, distance_based=False):
        """Calculates xyz positions of the trajectory"""
        # frames, atoms, 3
        if distance_based and trajectory.xyz.shape[1] > 1:
            self.log("Warning, using distance_based positions")
            ind = np.array(
                list(combinations(list(range(trajectory.xyz.shape[1])), r=2))
            )
            distances = md.compute_distances(trajectory, ind)
            traj_feat = distances
        else:
            traj_feat = trajectory.xyz
            traj_feat = traj_feat.reshape(-1, 3 * traj_feat.shape[1])
        return traj_feat

    def calc_distances(
        self,
        trajectory,
        trajectory_ref,
        use_distances_scheme="closest-heavy",
        cutoff=None,
    ):
        """Returns distance matrix, of coodinates for a  given
        selection (passed through selection argument).
        Dimension of the matrix depends on the system
        and selection.
        Arguments:
            trajectory: a list of Quantity objects from the OpenMM simulation.
        """
        if self.cache_distance_ind is None and self.contact_cache:
            distances, res, ind = compute_contacts(
                trajectory_ref, scheme=use_distances_scheme
            )
            # Check if order of distances is the same as of indices
            # otherwise masking wont work
            traj_dist = md.compute_distances(trajectory, ind[0])
            argdist = np.argmax(traj_dist[0])
            assert np.isclose(
                np.max(traj_dist[0]),  # check for trajectory, not input pos
                md.compute_distances(trajectory, ind[0][argdist].reshape(1, -1))[0],
            )
            # check for traj not input pos
            assert traj_dist.ndim == 2
            assert ind[0].ndim == 2
            if cutoff is not None:
                # calculate cutoff based on the input positions
                mask = distances[0] <= cutoff
                # take indices from first trajectory
                self.cache_distance_ind = ind[0][mask]
            else:
                # take indices from first trajectory
                self.cache_distance_ind = ind[0]
        if self.contact_cache:
            distances = md.compute_distances(trajectory, self.cache_distance_ind)
        else:
            distances, res, ind = compute_contacts(
                trajectory, scheme=use_distances_scheme
            )
        # exp transform distances
        distances = np.exp(-distances)
        return distances

    def calc_dih(self, trajectory):
        phi = md.compute_phi(trajectory)[1]
        psi = md.compute_psi(trajectory)[1]

        angles = np.concatenate([phi, psi], axis=-1)
        dihedral_matrix = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

        return dihedral_matrix

    def calc_chi(self, trajectory, chi):
        if chi == "chi1":
            chi_val = md.compute_chi1(trajectory)[1]
        elif chi == "chi2":
            chi_val = md.compute_chi2(trajectory)[1]
        elif chi == "chi3":
            chi_val = md.compute_chi3(trajectory)[1]
        elif chi == "chi4":
            chi_val = md.compute_chi4(trajectory)[1]
        chi_matrix = np.concatenate([np.sin(chi_val), np.cos(chi_val)], axis=-1)

        return chi_matrix

    def metric(
        self,
        traj_org,
        selection,
        mdtraj_top,
        use_distances_scheme,
        use_distances=False,
        distance_cutoff=None,
        use_dihedrals=False,
        use_positions=False,
        use_chi1=False,
        use_chi2=False,
        use_chi3=False,
        use_chi4=False,
    ):
        """Calculates feature matrix"""

        sel = self.topology_mdtraj.select(selection)
        traj = md.Trajectory(
            xyz=traj_org["xyz"].astype(np.float32),
            time=traj_org["time"].astype(np.float32),
            topology=mdtraj_top,
        )

        if self.system.usesPeriodicBoundaryConditions():
            traj.unitcell_vectors = traj_org["unitcell_vectors"].astype(np.float32)
            traj.make_molecules_whole(inplace=True)
        if traj_org["xyz"].shape[1] > 1:
            traj.center_coordinates()
            traj.superpose(self.superpose_traj)
        # would need very small timestep between frames
        # traj.smooth(3, inplace=True)

        # All indices from sub_traj correspond to the sub_traj!
        # Not to the full topology!
        sub_traj = traj.atom_slice(sel)
        sub_traj_ref = self.superpose_traj.atom_slice(sel)
        feat_array = []

        if use_distances:
            feat_matrix_dist = self.calc_distances(
                sub_traj,
                sub_traj_ref,
                use_distances_scheme=use_distances_scheme,
                cutoff=distance_cutoff,
            )
            feat_array.append(feat_matrix_dist)
        if use_dihedrals:
            feat_matrix_dih = self.calc_dih(sub_traj)
            feat_array.append(feat_matrix_dih)
        if use_chi1:
            feat_matrix_chione = self.calc_chi(sub_traj, chi="chi1")
            if feat_matrix_chione.shape[1] != 0:
                feat_array.append(feat_matrix_chione)
        if use_chi2:
            feat_matrix_chitwo = self.calc_chi(sub_traj, chi="chi2")
            if feat_matrix_chitwo.shape[1] != 0:
                feat_array.append(feat_matrix_chitwo)
        if use_chi3:
            feat_matrix_chithree = self.calc_chi(sub_traj, chi="chi3")
            if feat_matrix_chithree.shape[1] != 0:
                feat_array.append(feat_matrix_chithree)
        if use_chi4:
            feat_matrix_chifour = self.calc_chi(sub_traj, chi="chi4")
            if feat_matrix_chifour.shape[1] != 0:
                feat_array.append(feat_matrix_chifour)
        if use_positions:
            feat_matrix_positions = self.calc_positions(sub_traj)
            feat_array.append(feat_matrix_positions)
        if len(feat_array) == 0:
            raise ValueError("At least one featureset has to be chosen")
        feat_matrix = np.float32(np.concatenate(feat_array, axis=1))
        return feat_matrix

    def calc_energystats(self):
        """Calculates based on the MD energy from trajectroy, energy, max energy in the trajectory,
        min energy in the trajectory, max torsional energy in the trajectory and min torsional energy in
        the trajectory
        Arguments: None,
        Returns: None"""
        energy = np.array(
            [e._value for e in self.trajectory_reporter.get_energy()], dtype=np.float32
        )
        assert len(energy) > 0
        tors_energy = np.array(
            [e._value for e in self.trajectory_reporter.get_torsenergy()],
            dtype=np.float32,
        )
        self.max_energy = np.max(energy)
        self.min_energy = np.min(energy)
        if len(tors_energy) > 0:
            self.max_torsenergy = np.max(tors_energy)
            self.min_torsenergy = np.min(tors_energy)
        else:
            self.max_torsenergy = None
            self.min_torsenergy = None
        new_energy = []
        for i in range(len(energy)):
            ene_tup = [
                energy[i],
                self.max_energy,
                self.min_energy,
                self.max_torsenergy,
                self.min_torsenergy,
            ]
            new_energy.append(ene_tup)
        assert len(new_energy) > 0
        # replace energy with new energy
        self.energy = np.array(new_energy)

    def get_cur_dir(self):
        return self.cur_dir

    def get_reporter_counter(self):
        if self.file_per_cycle:
            return self.dcdreporter.get_counter()
        else:
            return -1

    def get_distance_ind(self):
        return self.cache_distance_ind

    def set_distance_ind(self, dist_ind):
        self.cache_distance_ind = dist_ind

    def log(self, msg):
        self.logger.info(msg)
