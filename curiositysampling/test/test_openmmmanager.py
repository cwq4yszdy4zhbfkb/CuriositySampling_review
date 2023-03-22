from curiositysampling.core import OpenMMManager
import ray
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmtools.testsystems import AlanineDipeptideVacuum
import numpy as np
import pytest
import time
import shutil


@pytest.fixture(scope="module")
def prepare():
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
    actor = OpenMMManager.remote(
        positions=testsystem.positions,
        system=system,
        topology=testsystem.topology,
        integrator=integrator,
        steps=5000,
        variables=[psi_bias, phi_bias],
    )
    # Give time for initialization
    time.sleep(0.5)
    return actor


def test_create_new_instance(prepare):
    raylet = prepare.create_new_instance.remote()
    assert isinstance(
        raylet, ray._raylet.ObjectRef
    ), "The output of OpenMMManager new instance should be ray object"
    sim_id = ray.get(raylet)
    print("simulation id is {}".format(sim_id))
    assert isinstance(sim_id, str), "The sim_id should be str"
    ray.kill(prepare)
    time.sleep(0.5)
