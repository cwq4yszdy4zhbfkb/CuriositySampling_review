import pytest
import ray
from openmm.app import *
from openmm.app.metadynamics import *
from openmm import *
from openmm.unit import *
from openmmtools.testsystems import AlanineDipeptideVacuum
import numpy as np
from curiositysampling.core import SingleBiasedSim
from curiositysampling.utils import SharedBiases
import shutil
import time


def setup_module(module):
    ray.init(num_cpus=4, ignore_reinit_error=True)
    time.sleep(0.5)


@pytest.fixture(scope="module")
def prepare():
    class NewInstance:
        def get(self):
            testsystem = AlanineDipeptideVacuum()
            system = testsystem.system
            integrator = LangevinIntegrator(
                300 * kelvin, 1 / picosecond, 2 * femtoseconds
            )
            topology = testsystem.topology
            positions = testsystem.positions
            phi = openmm.CustomTorsionForce("theta")
            phi.addTorsion(4, 6, 8, 14)
            phi_bias = app.BiasVariable(phi, -np.pi, np.pi, 0.35, True, 360)

            psi = openmm.CustomTorsionForce("theta")
            psi.addTorsion(6, 8, 14, 16)
            psi_bias = app.BiasVariable(psi, -np.pi, np.pi, 0.35, True, 180)
            bias_share_object = SharedBiases(variables=[psi_bias, phi_bias])
            actor = SingleBiasedSim.remote(
                sim_id="1",
                positions=positions,
                system=system,
                topology=topology,
                integrator=integrator,
                bias_share_object=bias_share_object,
                variables=[psi_bias, phi_bias],
                reporter_stride=10,
                temperature=300,
                frequency=1000,
                steps=2000,
            )
            return actor

    return NewInstance()


@pytest.fixture(scope="module")
def action(prepare):
    instance = prepare.get()
    action = ray.get(instance.get_initial_positions.remote())
    raylet = instance.run.remote(action=action)
    free_energy, trajectory, trajectory_obs = ray.get(raylet)
    new_actions = list(trajectory)
    action = new_actions[np.random.randint(0, len(new_actions))]
    return action


@pytest.mark.parametrize("step", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_one_step(prepare, action, step):
    print("step number {}".format(step))
    instance = prepare.get()
    print("testing for action with shape {}".format(action.shape))
    try:
        raylet = instance.run.remote(action=action)
    except:
        print("Failed to run `run` method")
    assert isinstance(
        raylet, ray._raylet.ObjectRef
    ), "The output of SingleBiasedSim run method is not ray object"
    getdata = ray.get(raylet)
    assert isinstance(
        getdata[0], quantity.Quantity
    ), "After calling `ray.get`, the outputs are not instanced of Quantity"
    assert isinstance(
        getdata[1], quantity.Quantity
    ), "After calling `ray.get`, the outputs are not instanced of Quantity"
    assert (
        getdata[0].ndim == 2
    ), "Dimension of the output's first index should be bigger than 1"
    assert (
        getdata[1].ndim == 3
    ), "Dimension of the output's second index is different than 2"
    assert isinstance(
        getdata[2], tuple
    ), "The type of the output's third index, should be tuple"
    assert len(getdata[2]) == 2, "The third output should be tuple of len 2"
    # assert getdata[2].ndim == 3, 'The tuple should contain two openmm arrays'
    ray.kill(instance)
    return None


@pytest.mark.parametrize("step", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_multiple_instanaces(prepare, action, step):
    print("step number {}".format(step))
    n_instances = 10
    instance_list = []
    time.sleep(2)
    print("testing for action with shape {}".format(action.shape))
    for i in range(n_instances):
        instance = prepare.get()
        instance_list.append(instance)
        print("Creating instance number {}".format(i))
    for i in range(n_instances):
        try:
            raylet = instance_list[i].run.remote(action=action)
        except:
            print("Failed to run `run` method")
            raise RuntimeError(
                "Failed to run one simulation." + " The instance is {}".format(i)
            )

    for i in range(n_instances):
        assert isinstance(
            raylet, ray._raylet.ObjectRef
        ), "The output of SingleBiasedSim run method is not ray object"
        getdata = ray.get(raylet)
        assert isinstance(
            getdata[0], np.ndarray
        ), "After calling `ray.get`, the outputs are not instanced of Quantity"
        assert isinstance(
            getdata[1], np.ndarray
        ), "After calling `ray.get`, the outputs are not instanced of Quantity"
        assert (
            getdata[0].ndim == 2
        ), "Dimension of the output's first index should be bigger than 1"
        assert (
            getdata[1].ndim == 3
        ), "Dimension of the output's second index is different than 2"
        assert isinstance(
            getdata[2], tuple
        ), "The type of the output's third index, should be tuple"
        assert len(getdata[2]) == 2, "The third output should be tuple of len 2"
        return None
    for inst in instance_list:
        ray.kill(inst)


def teardown_module(module):
    """whole test run finishes."""
    print("\n ### Test finishes, removing directories ###\n")
    shutil.rmtree("1")
