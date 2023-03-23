import copy
import gc
import os
import pprint
import random
import sys
from collections import deque
from datetime import datetime
from time import sleep, time

import numpy as np
import openmm as omm
import ray
from curiositysampling.utils.checkpointutils import (
    append_json_object,
    save_json_object,
    save_pickle_object,
)
from diskcache import FanoutCache

from .base_loger import logger


class CuriousSampling:
    """The class defines object, that allows running molecular curiosity sampling with openmm framework.
    Example:
        ```python
        from curiositysampling.core import OpenMMManager
        from curiositysampling.core import CuriousSampling
        import ray
        from openmm.app import *
        from openmm import *
        from unit import *
        from openmmtools.testsystems import AlanineDipeptideImplicit


        ray.init()
        testsystem = AlanineDipeptideExplicit(hydrogenMass=4*amu)
        system = testsystem.system
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 4*femtoseconds)
        topology = testsystem.topology
        positions = testsystem.positions
        omm = OpenMMManager.remote(positions=testsystem.positions, system=system, topology=testsystem.topology,
                                   integrator=integrator, steps=2000, reporter_stride=10)
        config_env = {'openmmmanager': omm}
        config_rnd = {'model': 'dense_units': [layer_size]*layers,
                               'dense_units_ae_enc': [layer_size]*layers,
                               'dense_units_ae_dec': [layer_size]*layers,
                               'dense_activ': 'fullsort',
                               'dense_layernorm': False,
                               'dense_out': 1,
                               'dense_out_activ': 'linear',
                               'curiosity_activ': 'tanh',
                               'initializer': 'glorot_uniform',
                               'spectral': True,
                               'ae_spectral_only': True},
                      'autoencoder': True,
                      'autoencoder_lagtime': 450,
                      'minibatch_size': 200,
                      'clip_by_global_norm': False,
                      'num_of_train_updates': iterations,
                      'num_of_ae_train_updates': 2,
                      'learning_rate_cur': 0.0001,
                      'learning_rate_ae': 0.0001,
                      'obs_stand': False,
                      'reward_stand': False,
                      'ae_train_buffer_size': 50000,
                      'train_buffer_size': 50000,
                      'optimizer': 'sgd',
                      'optimizer_ae': 'nsgd',
                      'target_network_update_freq': 20
                     }

         csm = CuriousSampling(rnd_config=config_rnd, env_config=config_env,
                               number_of_agents=1)
         # define arrays to report output values:
         intrinsic_reward_reporter = []
         action_reporter = []
         state_mean_var_reporter = []
         reward_mean_var_reporter = []

         # run for 20 cycles
         csm.run(20, action_reporter=action_reporter,
         max_reward_reporter=intrinsic_reward_reporter,
         state_mean_var_reporter=state_mean_var_reporter,
         reward_mean_var_reporter=reward_mean_var_reporter)
         ```

         Arguments:
             rnd_config: A dictionary which defines RND model, number of iterations,
                         minibatch size and other parameters, see RNDtrain documentation
                         for more details.
             env_config: A dictionary which defines objects (like OpenMMManager) and
                         parameters associated with molecular dynamics. At the moment
                         the only its key is `omm`, which should be set to OpenMMManager
                         object.

             number_of_agents: How many envs should be sampled in parallel

             random_actions: Pickpup actions from uniform distribution. Used for
                             testing purporse.
             latent_save_frequency: After how many train examples, latent
                                    space is saved (by default 0, not saved).
                                    The latent space can be obtained by
                                    `get_saved_latent()` method.
             latent_save_action: If to save action's latent space into array.
                                 Used for testing purpose.
    """

    # hide from RAY
    # import it inside class, needs self for reference

    def __init__(
        self,
        rnd_config=None,
        env_config=None,
        number_of_agents=1,
        diskcache=True,
        buffer_size=20,
        use_metadynamics=False,
        random_actions=False,
        latent_save_frequency=0,
        latent_space_action=False,
        working_directory=None,
        checkpoint_save_frequency=1,
        action_mode="async",
    ):
        logger.info("Node list")
        logger.info(ray.nodes())

        # checkpoint
        self.checkpoint_save_frequency = checkpoint_save_frequency
        # config
        self.rnd_config = rnd_config
        self.env_config = env_config
        # diskcache
        self.diskcache = diskcache
        # directories
        self.fname_checkpoint_filenames = "saved_filenames"
        if working_directory is not None and isinstance(working_directory, str):
            if not os.path.isdir(working_directory):
                raise ValueError(
                    "Directory {} doesn't exists".format(working_directory)
                )
            self.working_directory = working_directory
        else:
            raise ValueError(
                "Working directory has to be set as a string, now {}".format(
                    working_directory
                )
            )

        # set working directory for OpenMMManager
        env_config["openmmmanager"].set_working_directory.remote(
            working_directory=self.working_directory
        )
        env_config["openmmmanager"].set_max_buffer.remote(max_conf_buffer=buffer_size)

        # create data directory
        self.data_path = self.working_directory + "/" + "tmp_data"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        else:
            logger.warning(
                "temporary data exists as {}, using existing one".format(self.data_path)
            )

        # env control vars
        self.results_sleep = 0.001
        self.results_sleep_beta = 0.9
        self.number_of_agents = number_of_agents

        # prepare env instances
        self.env_instances = []
        self.env_encoder = {}

        # relative import
        from .simplemdenv import SimpleMDEnv

        for i in range(number_of_agents):
            self.env_instances.append((i, SimpleMDEnv(env_config)))
            self.env_encoder[self.env_instances[i][1].sim_id] = i
        initial_observations = []

        start_time_md = time()

        temp_obs = {}
        restr = []
        self.oneframe_stepsize = ray.get(
            env_config["openmmmanager"].get_perframe_unit.remote()
        )

        self.one_agent_sim_length = ray.get(
            env_config["openmmmanager"].get_simlength.remote()
        )

        self.if_use_positions = ray.get(
            env_config["openmmmanager"].get_if_positions.remote()
        )

        self.if_use_dihedrals = ray.get(
            env_config["openmmmanager"].get_if_dihedrals.remote()
        )

        self.if_use_distances = ray.get(
            env_config["openmmmanager"].get_if_distances.remote()
        )
        while True:
            # does not go further, envs starts simulation from beginning if it's called twice
            # if obs != "Wait", then env have to be removed from checking
            for i, env in self.env_instances:
                if i not in restr:
                    random_seed = random.randint(-2147483646, 2147483646)
                    box = ray.get(env_config["openmmmanager"].get_initbox.remote())
                    box = [b._value for b in box]
                    energy = (-1e16, -1e14, -1e16, -1e14, -1e16)
                    obs = env.reset(random_seed, energy)
                    # make sure it is deleted
                    sleep(0.5 * random.random())
                    if obs != "Wait":
                        temp_obs[env.sim_id] = copy.deepcopy(obs)
                        del obs
                        restr.append(i)
            sleep(self.results_sleep / 1000)
            if len(temp_obs) == len(self.env_instances):
                break
        time_md = time() - start_time_md

        self.results_sleep = self.results_sleep * self.results_sleep_beta + time_md * (
            1 - self.results_sleep_beta
        )

        self.actions = []
        self.boxes = []
        self.energy = []
        if action_mode not in ["top", "ranked", "async"]:
            raise ValueError('Action mode can be either "top", "ranked" or "async"')
        self.action_mode = action_mode
        self.action_mode_perm = self.action_mode
        # Here load previous action if checkpoint
        # code ...
        #
        for i, env in self.env_instances:
            init_obs = temp_obs[env.sim_id]
            ind = random.randint(0, len(init_obs["trajectory"]) - 1)
            action = init_obs["trajectory"][ind]
            energy = init_obs["energy"][ind]
            box = init_obs["box"][ind]
            self.actions.append(action)
            self.boxes.append(box)
            self.energy.append(energy)
        # import RndTrain here to avoid Ray init see Tensorflow
        from .rndtrain import RndTrain

        self.rnd_train_instance = RndTrain(
            diskcache=self.diskcache,
            data_path=self.data_path,
            fname_checkpoint_filenames=self.fname_checkpoint_filenames,
            checkpoint_save_frequency=self.checkpoint_save_frequency,
            config=rnd_config,
            oneframe_stepsize=self.oneframe_stepsize,
            if_use_positions=self.if_use_positions,
            if_use_dihedrals=self.if_use_dihedrals,
            if_use_distances=self.if_use_distances,
        )
        self.rnd_train_instance.initialise(init_obs)
        # add init obs to buffer
        for i, env in self.env_instances:
            init_obs = temp_obs[env.sim_id]
            self._add_to_train(init_obs, shuffle=False)

        del init_obs, temp_obs, box, action, energy, ind
        # buffer
        if buffer_size < 0 or not isinstance(buffer_size, int):
            raise ValueError("buffer_size has to be integer and greater than 0")
        self.buffer_size = buffer_size
        # Buffer loading from file for checkpoit
        # code
        # ...
        if self.diskcache:
            from curiositysampling.utils.mltools import Dequemax

            self.buffer = Dequemax(
                maxlen=self.buffer_size,
                directory=self.data_path + "/diskcachedeque_obsbuff",
                pickle_protocol=5,
            )
        else:
            # at the moment make checkpoint for this to raise "nonimplemented"
            self.buffer = deque(maxlen=self.buffer_size)

        # It does not reduce memory usage
        # TODO: Move to unified memory model
        # if self.diskcache:
        #    self.current_cycle_obs = FanoutCache(
        #        directory=self.data_path + "/diskcache_internode_obs",
        #        pickle_protocol=5,
        #        size_limit=1e14,
        #        eviction_policy="none",
        #    )
        # else:
        self.current_cycle_obs = {}

        # other

        # hide imports from RAY
        import tensorflow as tf

        self.tb_writer = tf.summary.create_file_writer(
            self.data_path
            + "/tensorboard/curiosty_stats/"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + "/curiosity"
        )

        self.random_actions = random_actions
        self.autoencoder = rnd_config["autoencoder"]
        self.checkpoint_save_frequency = checkpoint_save_frequency
        # init json object for filenames
        save_json_object({}, path=self.data_path, fname=self.fname_checkpoint_filenames)
        # latent space save
        self.latent_save_frequency = latent_save_frequency
        self.latent_space_counter = 0
        self.latent_space_array = []
        self.latent_space_action = latent_space_action
        self.latent_space_action_array = []

    def get_sim_ids(self):
        """Returns all simulations ids, added to this point in time."""
        sim_ids = []
        for i, env_instance in self.env_instances:
            sim_id = env_instance.sim_id
            sim_ids.append(sim_id)
        return sim_ids

    def run(
        self,
        cycles,
        action_reporter=None,
        max_reward_reporter=None,
        state_mean_var_reporter=None,
        reward_mean_var_reporter=None,
    ):
        """Runs curious sampling for parameters set in the curious sampling
        objects.
        Arguments:
            cycles: Number of cycles, where one cycle is one MD simulation
                    of number of steps length, and training the RND network
                    with samples drawn from the MD sampler.
            action_reporter: Structures chosen with the highest reward. Be careful while storing
                             them, they are Nx3 matrices, where N is number of all atoms.
                             Therefore, they can easily lead to Out of memory situation.
            max_reward_reporter: Maximum reward associated with each env (may change to
                                 the maximum reward at all in the future).
            state_mean_var_reporter: Mean and Variance used to standarize observation,
                                     reported in each cycle.
            reward_mean_var_reporter: Mean and Variance used to standarize reward,
                                      in each cycle.

        """
        # hide it from RAY
        import tensorflow as tf

        cycl_list = list(range(cycles))
        for c in cycl_list:
            logger.info("Cycle {0} out of {1} cycles".format(c + 1, cycles))
            logger.info("Current buffer size {}".format(len(self.buffer)))
            start_time_total = time()

            total_num_of_obs = 0

            # get observations from MD
            time_md = self.make_one_md_step()
            # release actions
            del self.actions, self.boxes, self.energy
            # TODO: It forces to read from the disk, so is super expensive
            # Move it to a method and call the method from inside make_one_md_step
            # TODO: checkpoint load - you have to load a list of keys, they have to be the same
            # as in the previous database
            for w, env in self.env_instances:
                instance = env.sim_id
                obs = self.current_cycle_obs[instance]
                total_num_of_obs += obs["dist_matrix"].shape[0]
                # add observations to train buffer
                self._add_to_train(obs, shuffle=False)
                # plot distribution of energy
                with self.tb_writer.as_default(c):
                    tf.summary.histogram(
                        "mdstats/Potential energy for the simulation {0}".format(w),
                        tf.convert_to_tensor(obs["energy"][:, 0].astype("float32")),
                    )

                for j, e in enumerate(tf.convert_to_tensor(obs["energy"][:, 0].astype("float32"))):
                    with self.tb_writer.as_default(c * int(obs["energy"].shape[0]) + j):
                        tf.summary.scalar(
                            "mdstats/Potential energy over frames for the simulation {0}".format(
                                w
                            ),
                            e,
                        )
            # del references at the end
            del obs
            del instance
            # one train step through all the examples in the rnd model
            start_time_train = time()
            predict_network_loss_list, ae_loss_list, vamp2_score = self._train(
                total_num_of_obs
            )

            time_train = time() - start_time_train
            if len(predict_network_loss_list) > 0:
                logger.info(
                    "Predictor Network loss is {}".format(
                        np.mean(np.hstack(predict_network_loss_list))
                    )
                )
            if self.autoencoder and len(ae_loss_list) > 0:
                logger.info(
                    "Autoencoder loss is {}".format(np.mean(np.hstack(ae_loss_list)))
                )
                if len(vamp2_score) > 0:
                    logger.info(
                        "VAMP2 score is {}".format(np.mean(np.hstack(vamp2_score)))
                    )

            # here we start predicting actions based on the reward
            rewards = self.distribute_actions(mode=self.action_mode)

            logger.info("Top rewards for each agent: {}".format(rewards))

            # allow for real time printing
            sys.stdout.flush()
            # only top rewards for chosen actions for every env instance
            rewards = rewards[: len(self.env_instances)]
            if action_reporter is not None:
                action_reporter.append(self.actions)
            if max_reward_reporter is not None:
                max_reward_reporter.append(rewards)
            if state_mean_var_reporter is not None:
                state_mean_var_reporter.append(
                    self.rnd_train_instance.get_state_mean_variance()
                )
            if reward_mean_var_reporter is not None:
                reward_mean_var_reporter.append(
                    self.rnd_train_instance.get_reward_mean_variance()
                )

            # del ref for rewards
            del rewards

            time_total = time() - start_time_total
            # print times
            logger.info(
                (
                    "Total time: {0:.2f} s"
                    + "   MD time: {1:.2f} s"
                    + "   ML train time {2:.2f} s"
                ).format(time_total, time_md, time_train)
            )
            one_day = 24 * 3600
            ns_per_day = (
                (self.one_agent_sim_length * self.number_of_agents)
                / time_total
                * one_day
            )

            md_eff = time_md / time_total * 100
            logger.info("Simulation performance: {0:.2f} ns/day".format(ns_per_day))
            logger.info("Simulation efficiency vs pure MD: {0:.2f} %".format(md_eff))

            with self.tb_writer.as_default(c):
                tf.summary.scalar("performance/Total episode time", time_total)
                tf.summary.scalar(
                    "performance/Simulation performance in ns/day", ns_per_day
                )
                tf.summary.scalar(
                    "performance/MD efficiency with respect to pure MD", md_eff
                )
                tf.summary.scalar(
                    "performance/Number of observations per second",
                    total_num_of_obs / time_total,
                )
                tf.summary.scalar(
                    "performance/Number of observations per second per process",
                    total_num_of_obs / (time_total * self.number_of_agents),
                )

                tf.summary.scalar("performance/MD simulation time", time_md)
                tf.summary.scalar("performance/ML RND time", time_train)

            # release obs that are very heavy
            self.current_cycle_obs.clear()

            # free up memory after training
            gc.collect()
            sleep(1)

            # save checkpoint
            if c % self.checkpoint_save_frequency == 0:
                obs_data = {
                    "energy": self.energy,
                    "boxes": self.boxes,
                    "actions": self.actions,
                }
                fname_obs = "obs_data"
                full_fname = save_pickle_object(
                    obs_data, path=self.data_path, fname=fname_obs
                )
                append_json_object(
                    {"cur_filename_obs": full_fname},
                    path=self.data_path,
                    fname=self.fname_checkpoint_filenames,
                )

                conf_data = {
                    "rnd_config": self.rnd_config,
                    "diskcache": self.diskcache,
                    "number_of_agents": self.number_of_agents,
                    "buffer_size": self.buffer_size,
                    "random_actions": self.random_actions,
                    "working_directory": self.working_directory,
                }
                full_fname = save_pickle_object(
                    conf_data, path=self.data_path, fname="conf_data"
                )
                append_json_object(
                    {"cur_filename_conf": full_fname},
                    path=self.data_path,
                    fname=self.fname_checkpoint_filenames,
                )

                intcur_data = {
                    "cycles": c,
                }
                full_fname = save_json_object(
                    intcur_data, path=self.data_path, fname="intcur_data"
                )
                append_json_object(
                    {"cur_filename_intcur": full_fname},
                    path=self.data_path,
                    fname=self.fname_checkpoint_filenames,
                )

                openmm_full_fname, openmm_dist_ind_fname = ray.get(
                    self.env_config["openmmmanager"].save_checkpoint_data.remote(
                        data_path=self.data_path
                    )
                )
                append_json_object(
                    {"filename_openmmm": openmm_full_fname},
                    path=self.data_path,
                    fname=self.fname_checkpoint_filenames,
                )

                append_json_object(
                    {"filename_openmmm_dist_ind": openmm_dist_ind_fname},
                    path=self.data_path,
                    fname=self.fname_checkpoint_filenames,
                )
                del intcur_data, conf_data, obs_data
                logger.info("Saving checkpoint ...")

            logger.debug("Cluster resources")
            logger.debug(ray.cluster_resources())

            logger.debug("Cluster available resources")
            logger.debug(ray.available_resources())

            gc.collect()

    def _concatenate_obs(self, obs_list):
        """Concatenates observations from a list
        into a single observation - dictionary
        with two ndarrays.
        Arguments:
            obs_list: list of dictionaries with trajectory and feature matrix (called dist_matrix).
        Returns:
            dictionary with trajectory (key: trajectory) and feature matrix (key: dist_matrix).
        """
        # TODO: concatenate and copy increase memory consumption few times
        # Use original data
        obs_base = copy.deepcopy(obs_list[0])
        if len(obs_list) > 1:
            for obs in obs_list[1:]:
                obs_base["trajectory"] = np.concatenate(
                    [obs_base["trajectory"], obs["trajectory"]]
                )
                obs_base["dist_matrix"] = np.concatenate(
                    [obs_base["dist_matrix"], obs["dist_matrix"]]
                )
                obs_base["energy"] = np.concatenate([obs_base["energy"], obs["energy"]])
                obs_base["box"] = np.concatenate(
                    [np.squeeze(obs_base["box"]), obs["box"]]
                )
                if obs_base["dssp"] is not None:
                    obs_base["dssp"] = np.concatenate([obs_base["dssp"], obs["dssp"]])
            if obs_base["dssp"] is not None:
                dssp_check = obs_base["dssp"].shape[0] < 2
            else:
                dssp_check = False
            if (
                obs_base["trajectory"].shape[0] < 2
                or obs_base["dist_matrix"].shape[0] < 2
                or obs_base["energy"].shape[0] < 2
                or dssp_check
            ):
                raise Exception("The shape should be at least 2 after contatenate")

        return obs_base

    def _train(self, total_num_of_obs):
        """Trains RND network for all stored examples."""
        return self.rnd_train_instance.train(total_num_of_obs)

    def _train_last_n_batch(self, n=None):
        """Trains last n example from training buffer.
        Arguments:
            n: should be negative and indicate position in the
            buffer, e.g. -1, -2 ... -N.
        """
        self.rnd_train_instance.train_last_n_batch(n)

    def _add_to_train(self, obs, shuffle=False):
        """Adds observations to the buffer from MD simulations.
        Arguments:
            obs: dictionary with observations
            shuffle: shuffles input against first dimension
        """
        self.rnd_train_instance.add_to_buffer(obs, shuffle=shuffle)

    def _get_new_actions(self, obs, n, update_stats=False, instance=None):
        """Calculates `n` observations with the highest rewards.
        Number of the input observations have to be higher than n.
        Arguments:
            obs: dictionary with observations.
            n: number of returned observations with highest rewards.
        Returns:
            A tuple of four, that contains sorted `n` actions from highest to lowest,
            `n` top observations sorted from highest to lowest, `n` rewards sorted
            from highest to lowest and unsorted buffer's rewards (used for debug).
        """
        (
            actions,
            dist_matrixs,
            rewards,
            reward_unsorted,
            energy_matrix,
            dssp_matrix,
            box_list,
        ) = self.rnd_train_instance.predict_action(
            obs,
            n,
            random_actions=self.random_actions,
            update_stats=update_stats,
            instance=instance,
        )
        if dssp_matrix is not None:
            top_obss = [
                {
                    "box": box_el,
                    "dssp": dssp_el,
                    "energy": energy,
                    "trajectory": action,
                    "dist_matrix": dist_matrix,
                }
                for box_el, dssp_el, energy, action, dist_matrix in zip(
                    box_list, dssp_matrix, energy_matrix, actions, dist_matrixs
                )
            ]
        else:
            top_obss = [
                {
                    "box": box_el,
                    "energy": energy,
                    "trajectory": action,
                    "dist_matrix": dist_matrix,
                }
                for box_el, energy, action, dist_matrix in zip(
                    box_list, energy_matrix, actions, dist_matrixs
                )
            ]

        # ensure that input shapes are the same as output shapes
        if not reward_unsorted.shape[0] == obs["dist_matrix"].shape[0]:
            msg = "objects shapes are not equal, {0} != {1}".format(
                reward_unsorted.shape[0], obs["dist_matrix"].shape[0]
            )
            raise ValueError(msg)
        # always the indices with numbers higher than shape of obs are buffer's
        buffer_reward_unsorted = reward_unsorted[
            reward_unsorted.shape[0] - len(self.buffer) :
        ]  # len of buffer, since it's size changes at the beginning
        if n > 1 and len(self.buffer) > 0:
            if not buffer_reward_unsorted.shape[0] == len(self.buffer):
                msg = "objects shapes are not equal, {0} != {1}".format(
                    buffer_reward_unsorted.shape[0], len(self.buffer)
                )
                raise ValueError(msg)
        return actions, top_obss, rewards, buffer_reward_unsorted

    def make_one_md_step(self):
        start_time_md = time()
        restr = []
        while True:
            for i_env, action, box, energy in zip(
                self.env_instances, self.actions, self.boxes, self.energy
            ):
                i, env = i_env
                # remove first dimension from action
                # at the moment step is blocking
                # it should be changed in future to do MD in parallel
                if i not in restr:
                    sleep(0.1)
                    random_seed = random.randint(-2147483646, 2147483646)
                    obs = env.step(action, random_seed, box, energy)
                    # let is safely transfer everything
                    sleep(0.2)
                    if obs != "Wait":
                        # TODO: pass directly object (dictionary) to the step, then
                        # openmmmanager, then walker. So that it can be saved directly,
                        # not by transfering through network.
                        # It can even be transfered during the simulation as a shared container
                        # by creating one, that works like dictionary but is shared among all walkers
                        self.current_cycle_obs[env.sim_id] = copy.deepcopy(obs)
                        del obs
                        restr.append(i)
                        # let is safely save everything
                        sleep(0.5)
            sleep(self.results_sleep / 1000)

            if len(self.current_cycle_obs) == len(self.env_instances):
                break

        time_md = time() - start_time_md
        self.results_sleep = self.results_sleep * self.results_sleep_beta + time_md * (
            1 - self.results_sleep_beta
        )

        return time_md

    def add_obsbuffer(self, instance=None, shared=True):
        """Adds obs to the buffer
        Arguments:
            instance: index for the walker instance
            shared: if to share the buff obs among the walkers or not

        Returns:
            obs concatenated with the buffer's obs

        """
        # only if buffer is 0
        if len(self.buffer) == 0:
            if not shared:
                con_obs = self.current_cycle_obs[instance]
            else:
                con_obs = self._concatenate_obs(list(self.current_cycle_obs.values()))
        # buffer not 0, but not shared
        elif not shared:
            # add buffered struct for a given instance
            buff_temp = []
            for buff_episode in self.buffer:
                for buff_el in buff_episode:
                    # encode instance with predefined id
                    env_ind = self.env_encoder[instance]
                    # take element of buffer with id of the instance
                    if env_ind == buff_el[0]:
                        buff_temp.append(buff_el[1])
            con_obs = self._concatenate_obs(
                [self.current_cycle_obs[instance]] + buff_temp
            )
        # if buffer not 0, but shared
        elif shared:
            if instance is not None:
                raise ValueError("instances has to be None when shared=True")
            # add buffered struct for a given instance
            buff_temp = []
            for buff_el in self.buffer:
                for one_buff in buff_el:
                    buff_temp.append(one_buff)
            con_obs = self._concatenate_obs(
                list(self.current_cycle_obs.values()) + buff_temp
            )
        else:
            raise ValueError("shared has to be True or False")

        return con_obs

    def distribute_actions(self, mode=None):
        """Distributes actions among walkers, based on one of the strategies listed below.
        There are three strategies: async, ranked and top.
        In async mode, every walkers works independently, but use the same
        neural network to score.
        In ranked, every walker is ranked from 0 to M, where M is number of walkers.
        It is used to reduce bias in choosing an action.
        In top, every walker picks up the top one from all the obs.

        Arguments:
            mode: mode of distribution, either async, ranked or top

        """
        if not isinstance(mode, str) and not mode in ["async", "ranked", "top"]:
            raise ValueError("The mode should be one of async, ranked or top")
        assert self.current_cycle_obs is not None

        copy_actions = []
        copy_boxes = []
        copy_energies = []
        buff_temp_top = []
        rewards_array = []
        num_of_instances = len(self.env_instances)

        if not mode == "async":
            raise NotImplementedError(
                "Due to changes with configurations turnover, other modes than async, don't work!"
            )
            # TODO: Adapt to the new configurations turnover, e.g. add big number to every buffer
            for i, env in self.env_instances:
                instance = env.sim_id
                # it will consume more CPU, but memory can be released earlier
                con_obs = self.add_obsbuffer(
                    instance=None,
                    shared=True,
                )
                # get top structures, including those from buffer
                (
                    self.actions,
                    top_obss,
                    rewards,
                    buffer_reward_unsorted,
                ) = self._get_new_actions(
                    con_obs, num_of_instances, update_stats=True, instance=instance
                )
                # release con_obs
                del con_obs
                # populate buffer only with non-buffered obs - top_obss_nonbuff
                # TODO: it double-allocate the same buffer list
                # TODO: It takes tons of memory, tons!
                (
                    _,
                    top_obss_nonbuff,
                    _,
                    _,
                ) = self._get_new_actions(
                    self._concatenate_obs(list(self.current_cycle_obs.values())),
                    num_of_instances,
                    update_stats=False,
                )

                if mode == "ranked":
                    env_ind = self.env_encoder[instance]
                    action = self.actions[env_ind]
                    copy_actions.append(action)
                    copy_boxes.append(top_obss[env_ind]["box"][0])
                    copy_energies.append(top_obss[env_ind]["energy"])
                    rewards_array.append(rewards[env_ind])

                    buff_temp_top.append(top_obss_nonbuff[env_ind])

                elif mode == "top":
                    # top action
                    action = self.actions[0]
                    copy_actions.append(action)
                    copy_actions.append(action)
                    copy_boxes.append(top_obss[0]["box"][0])
                    copy_energies.append(top_obss[0]["energy"])
                    rewards_array.append(rewards[0])

                    buff_temp_top.append(top_obss_nonbuff[0])

                # train on the examples from the walker where action was chosen
                # self._train_last_n_batch(-num_of_instances + env_ind)

            self.buffer.append(buff_temp_top)
            self.actions = copy_actions
            self.boxes = copy_boxes
            self.energy = copy_energies

        elif mode == "async":
            for i, env in self.env_instances:
                instance = env.sim_id
                # here it goes per instance, so it's to be recalculated
                con_obs = self.add_obsbuffer(
                    instance=instance,
                    shared=False,
                )

                # assuming the same order as in env, which should be true
                action, top_obss, rewards, _ = self._get_new_actions(
                    con_obs, 1, update_stats=True, instance=instance
                )

                # first index [0] - top action, second index [1] - top action without buffer, staight from traj
                # top action
                action = action[0]
                box = top_obss[0]["box"][0]
                copy_boxes.append(box)
                energy = top_obss[0]["energy"]
                rewards_array.append(rewards[0])
                copy_energies.append(energy)
                env_ind = self.env_encoder[instance]

                # populate buffer only with non-buffered obs - top_obss_nonbuff
                # TODO: it double-allocate the same buffer list
                top_action_nonbuff, top_obss_nonbuff, _, _ = self._get_new_actions(
                    self.current_cycle_obs[instance], 1, update_stats=False
                )

                # add at the end with modifed second indiex, to indicate which was top outside buffer
                # number -42 from simplebiasedsim.py, number -777 indicates that it was from buffer
                assert (action[0][1] == -42) or (action[0][1] == -777)

                action[0][1] = int(top_obss_nonbuff[0]["trajectory"][0][0])
                copy_actions.append(action)

                # first index [0] - top action, second index [1] - top action without buffer, staight from traj
                # to move buff obs outside indexes of normal sym
                max_size_obs = self.current_cycle_obs[instance]["trajectory"].shape[0]
                top_obss_nonbuff = copy.deepcopy(top_obss_nonbuff[0])
                # should be between 0 .. N
                # if it's e.g. 5, then the sixth one will be added
                # so the index should be 5, and if 5 is max, then
                # one is removed, so forth one
                add_int = (
                    len(self.buffer)
                    if len(self.buffer) < self.buffer_size
                    else len(self.buffer) - 1
                )
                # We change numbering outside normal range, so that's easy to detect if it's from buffer
                # The order of buffer is the same in every walker
                top_obss_nonbuff["trajectory"][0][0] = max_size_obs + add_int
                top_obss_nonbuff["trajectory"][0][1] = -777
                buff_temp_top.append((env_ind, top_obss_nonbuff))
                # train on the examples from the walker where action w
                # self._train_last_n_batch(-num_of_instances + self.env_encoder[instance])
            self.actions = copy_actions
            self.boxes = copy_boxes
            self.energy = copy_energies
            if len(self.buffer) >= self.buffer_size:
                trigger_renumber = True
            else:
                trigger_renumber = False
            self.buffer.append(buff_temp_top)
            # mod all buffers by -1 to all if they reach limit
            # to make them numbered from 0 ... N
            if trigger_renumber:
                # last number has correct, because deque is FIFO
                for i, one_buff_el in enumerate(self.buffer):
                    one_buff_el_copy = copy.deepcopy(one_buff_el)
                    # don't renumerate last element
                    if i == len(self.buffer) - 1:
                        continue
                    for buff_agent in one_buff_el_copy:
                        buff_agent[1]["trajectory"][0][0] -= 1
                    self.buffer[i] = one_buff_el_copy
                # check in separate loop if it's saved
                sum_index = np.zeros(self.number_of_agents, dtype=np.int64)
                sum_index += self.buffer_size * (self.buffer_size - 1) // 2
                for i, one_buff_el in enumerate(self.buffer):
                    for j, buf_walk in enumerate(one_buff_el):
                        index = buf_walk[1]["trajectory"][0][0] - max_size_obs
                        assert index < self.buffer_size
                        sum_index[j] -= index
                # assert sum_index <= self.buffer_size * (self.buffer_size - 1) // 2

        return rewards_array

    def get_free_energy(self):
        """Returns free energy surface, if metadynamics is turned on.
        Otherwise it returns None
        """
        return self.env_instances[0][1].current_free_energy

    def get_saved_latent(self):
        """Returns saved latent space array"""
        raise NotImplementedError()

    def get_saved_latent_action(self):
        """Returns saved latent space array"""
        raise NotImplementedError()
