import numpy as np
import ray
import time
from copy import deepcopy


class SimpleMDEnv:
    """An Biased Molecular Dynamics Environment, that allows for interaction between biased MD simulation
    and machine learning algorithms. Since MD simulations are very slow in their nature, the env is designed to so that,
    it allows for multiple workers to operate.

    Arguments:
        config: A dictionary that, contains are necessary parameters, at the moment the only one parameter is OpenMMManager object.
    """

    def __init__(self, config):
        # MD variables
        self.openmmmanager = config["openmmmanager"]
        self.sim_id = ray.get(self.openmmmanager.create_new_instance.remote())
        # RND object
        # Observations
        # every discrete space is number of action types,
        # self.action_space = gym.spaces.Discrete(len(self.all_available_actions))
        # RL variables
        self.results_sleep = 0.001
        # Init
        time.sleep(self.results_sleep)
        self.lock_reset_vars = None

    def reset(self, random_seed, energy):
        """Resets state of the environment. Initial positions are loaded and
        one step is performed.
        Returns: Initial observation
        """
        if self.lock_reset_vars is None:
            self.cur_pos = 0
            action, box = ray.get(self.openmmmanager.get_init_oneactionbox.remote())
            if action is None or box is None:
                action = ray.get(self.openmmmanager.get_initial_positions.remote())
                box = ray.get(self.openmmmanager.get_initbox.remote())
                print("Use default set positions for every agent/walker")
            box = [b._value for b in box]
            self.lock_reset_vars = {
                "action": action,
                "random_seed": random_seed,
                "box": box,
                "energy": energy,
            }
            del action, box
        obs = self.step(**self.lock_reset_vars)
        return obs

    def step(self, action, random_seed, box, energy):
        """Performs one RL step, during which fallowing things happen:
        1. Pass action (openmm's system positions) to the md simulation.
        2. Perform md simulation
        3. Return observation

        Arguments:
            action: positions as a Quantity object from simtk framework, in the
                    nm units.
            random_seed: random seed used to draw velocities
        Returns:
            Observation dict with `trajectory` key and `dist_matrix` key. The first
            contains Quantity moleculary structures, where the second contains
            feature tensors.
        """
        state = action
        box_sizes, dssp_calc, energy, trajectory, trajectory_obs = self._md_simulation(
            state, random_seed, box, energy
        )
        if any([isinstance(trajectory, str), isinstance(trajectory_obs, str)]):
            if any([trajectory == "Wait", trajectory_obs == "Wait"]):
                return "Wait"
        # The obs is still a dictionary
        obs = self._next_observation(
            box_sizes,
            dssp_calc,
            energy,
            trajectory_obs,
            trajectory,
        )
        #
        return obs

    def _next_observation(
        self, box_sizes, dssp_calc, energy, trajectory_obs, current_trajectory
    ):
        """Calculates next observation ready to be used by `step` method.
        Arguments:
            dssp_calc: matrix with dssp decomposition of the protein structure. None
                       if it's not a protein.
            energy: potential energy per frame
            trajectory_obs: features of the observation
            current_trajectory: Quantity objetcs, from the OpenMM simulation
        """
        return {
            "box": box_sizes,
            "dssp": dssp_calc,
            "energy": energy,
            "dist_matrix": trajectory_obs,
            "trajectory": current_trajectory,
        }

    # TODO: get all walkers in one ray.get not separately
    def _md_simulation(self, action, random_seed, box, energy):
        """Perform molecular dynamics simulation in the context of
        openmm manager object. After the simulation, state of
        the system is returned as a free energy matrix, trajectory
        (positions as Quantity objects) and features.
        If `metadynamics` was set to false, free energy is
        None.
        Arguments:
            action: An simtk's Quantity object with molecular system positions
            random_seed: random seed used to draw velocities
        """
        results = None
        while True:
            results = ray.get(
                self.openmmmanager.step.remote(
                    sim_id=self.sim_id,
                    action=action,
                    random_seed=random_seed,
                    box=box,
                    energy=energy,
                )
            )
            if results is not None:
                break
            else:
                # by 10, because we check 10 times
                # per simulation time (thus it slows
                # performance max by 1/10)
                # otherwise we wait double of it
                # remove ref
                del results
                return "Wait", "Wait", "Wait", "Wait", "Wait"
        box_sizes, dssp_calc, energy, trajectory, trajectory_obs, md_time = results
        del results
        return box_sizes, dssp_calc, energy, trajectory, trajectory_obs

    def close(self):
        """The method closes openmmmanager and his
        subsequent agents.
        """
        pass
        # Status TODO
        # self.openmmmanager.close()
        # ray.kill(slf.openmmmanager)
