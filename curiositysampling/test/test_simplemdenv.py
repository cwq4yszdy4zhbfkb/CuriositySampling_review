from curiositysampling.core import OpenMMManager
from curiositysampling.core import SimpleMDEnv
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

pytest.globalvariable = 0


def env_creator(env_config):
    return SimpleMDEnv(env_config)


def pytest_namespace():
    return {"sim_id": 0}


@pytest.fixture(scope="module")
def prepare_env():
    testsystem = AlanineDipeptideVacuum()
    system = testsystem.system
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
    topology = testsystem.topology
    positions = testsystem.positions
    ray.init(num_cpus=4, ignore_reinit_error=True)
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
        steps=5000,
        variables=[psi_bias, phi_bias],
    )

    config = {
        "num_of_nodes": 9,
        "num_of_node_types": 2,
        "num_of_edges": 9,
        "openmmmanager": omm,
    }
    env = env_creator(config)
    # Give time for initialization
    time.sleep(0.5)
    return env


def test_random_sample(prepare_env):
    num_of_samples = 10
    sim_id = prepare_env.sim_id
    pytest.sim_id = sim_id
    reward_arr = []
    obs = prepare_env.reset()
    assert isinstance(obs["trajectory"], list)
    action = random.choice(obs["trajectory"])
    for i in range(num_of_samples):
        pprint("Random action taken with shape {}".format(action.shape))
        obs, reward = prepare_env.step(action)
        reward_arr.append(reward)
        print("Step number {0}".format(i))
        print(
            "Observation shapes: {0} and {2}".format(
                obs["distance_matrix"].shape, np.array(obs["trajectory"]).shape
            )
        )
        print(
            "Step avg reward: {0:6.2}, shape: {1}".format(np.mean(reward), reward.shape)
        )
        action = random.choice(obs["trajectory"])
    avg_sum_reward = np.mean(reward_arr)
    assert isinstance(avg_sum_reward, float)
    assert avg_sum_reward is not np.NaN
    print(
        "\nCumulative reward for {0} steps: {1:6.2}".format(
            num_of_samples, avg_sum_reward
        )
    )
    time.sleep(0.5)


def teardown_module(module):
    time.sleep(0.5)
    shutil.rmtree(pytest.sim_id)
