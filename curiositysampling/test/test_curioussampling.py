from curiositysampling.core import OpenMMManager
from curiositysampling.core import SimpleMDEnv
from curiositysampling.core import RndTrain
from curiositysampling.core import CuriousSampling
import ray
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmtools.testsystems import AlanineDipeptideVacuum
import numpy as np
import random
import pytest
import time
import shutil
from pprint import pprint
from itertools import combinations

pytest.globalvariable = 0


def env_creator(env_config):
    return SimpleMDEnv(env_config)


def pytest_namespace():
    return {"sim_id": 0}


@pytest.fixture(scope="module")
def prepare_env_config():
    testsystem = AlanineDipeptideVacuum()
    system = testsystem.system
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
    topology = testsystem.topology
    positions = testsystem.positions
    ray.init()
    phi = openmm.CustomTorsionForce("theta")
    phi.addTorsion(4, 6, 8, 14)
    phi_bias = app.BiasVariable(phi, -np.pi, np.pi, 0.35, True, 360)
    psi = openmm.CustomTorsionForce("theta")
    psi.addTorsion(6, 8, 14, 16)
    psi_bias = app.BiasVariable(psi, -np.pi, np.pi, 0.35, True, 360)
    omm = OpenMMManager.remote(
        positions=testsystem.positions,
        system=system,
        topology=testsystem.topology,
        integrator=integrator,
        steps=500,
        frequency=np.inf,
        variables=[psi_bias, phi_bias],
    )

    config = {
        "num_of_nodes": 9,
        "num_of_node_types": 2,
        "num_of_edges": 9,
        "openmmmanager": omm,
    }

    return config


@pytest.fixture(scope="module")
def prepare_rndtrain_config():
    config = {
        "model": {
            "dense_units": [9, 6, 3],
            "dense_activ": "relu",
            "dense_batchnorm": False,
            "dense_batchnorm_renorm": False,
            "dense_out": 2,
            "dense_out_activ": "linear",
        },
        "minibatch_size": 64,
        "clip_by_global_norm": True,
    }

    return config


def test_single_env_sampling(prepare_env_config, prepare_rndtrain_config):
    num_of_cycles = 20
    csm = CuriousSampling(
        rnd_config=prepare_rndtrain_config,
        env_config=prepare_env_config,
        number_of_parralel_envs=1,
    )
    sim_id = csm.get_sim_ids()[0]
    pytest.sim_id = sim_id
    intrinsic_reward_reporter = []
    action_reporter = []
    state_mean_var_reporter = []
    reward_var_reporter = []
    csm.run(
        num_of_cycles,
        action_reporter=action_reporter,
        max_reward_reporter=intrinsic_reward_reporter,
        state_mean_var_reporter=state_mean_var_reporter,
        reward_var_reporter=reward_var_reporter,
    )
    for env_actions in action_reporter:
        for action in env_actions:
            assert isinstance(action, quantity.Quantity)
    # check if reward is a float number
    for env_rewards in intrinsic_reward_reporter:
        assert isinstance(env_rewards, np.ndarray)
        for reward in env_rewards:
            assert isinstance(reward, np.floating)
            assert not np.any(reward < 0)
    # check if actions change
    for action_1, action_2 in combinations(action_reporter, r=2):
        assert (
            np.linalg.norm(action_1[0] - action_2[0]) > 0
        ), "The actions should differ"
    first_action = action_reporter[0][0]
    # check if state's mean and var change
    for state_mean_var_1, state_mean_var_2 in combinations(
        state_mean_var_reporter, r=2
    ):
        # check mean
        assert np.linalg.norm(state_mean_var_1[0] - state_mean_var_2[0]) > 0
        # check variance
        assert np.linalg.norm(state_mean_var_1[1] - state_mean_var_2[1]) > 0

    # report few numbers
    for env_actions, env_rewards in zip(action_reporter, intrinsic_reward_reporter):
        for action, reward in zip(env_actions, env_rewards):
            print(
                "Action's Forbenius norm with respect to first action {0} and its associated reward {1}".format(
                    np.linalg.norm(action - first_action), reward
                )
            )
    avg_sum_reward = np.mean(intrinsic_reward_reporter)
    print(
        "\nCumulative reward for {0} cycles: {1:6.2}".format(
            num_of_cycles, avg_sum_reward
        )
    )
    assert isinstance(avg_sum_reward, np.floating)
    assert avg_sum_reward is not np.NaN
    time.sleep(0.5)


def teardown_module(module):
    shutil.rmtree(pytest.sim_id)
    time.sleep(2.5)
