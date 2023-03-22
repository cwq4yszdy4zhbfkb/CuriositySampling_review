import openmm as mm
from openmm.app import DCDFile
from openmm.unit import nanometer
import numpy as np
import os
import gc


class TrajReporter:
    """Stores trajectory every N steps to an object."""

    def __init__(
        self, report_interval=10, enforcePeriodicBox=None, all_frames_dims=None
    ):
        self._reportInterval = report_interval
        self._append = False
        self._enforcePeriodicBox = enforcePeriodicBox
        # (frames, coordinates, 3)
        self.all_frames_dims = all_frames_dims
        # TODO: Store as float32/16, but gather statistics (mean, variance) as float64
        # then recover precision as in the mixed precision (normalize, denormalize)
        self.list_to_save = {
            "xyz": np.zeros(all_frames_dims, dtype=np.float64),
            "time": np.zeros(all_frames_dims[0], dtype=np.float64),
            "unitcell_vectors": np.zeros((all_frames_dims[0], 3, 3), dtype=np.float64),
        }
        self.energy_list = []
        self.energy_tors_list = []
        self.box_list = []
        self.first_position = False
        self.torsion_id = None
        self.timer = 0

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, True, self._enforcePeriodicBox)

    def report(self, simulation, state):

        positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
        self.list_to_save["xyz"][self.timer] = positions
        energy = state.getPotentialEnergy()
        if self.torsion_id is None and self.torsion_id != -1:
            for i, force in enumerate(simulation.system.getForces()):
                if force.__class__.__name__.count("Torsion"):
                    self.torsion_id = i
                    break
            if self.torsion_id is None:
                self.torsion_id = -1
        elif self.torsion_id != -1:
            energy_tors = simulation.context.getState(
                groups={self.torsion_id}, getEnergy=True
            ).getPotentialEnergy()
            self.energy_tors_list.append(energy_tors)
        self.energy_list.append(energy)

        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometer)
        self.list_to_save["unitcell_vectors"][self.timer] = box
        # self.box_list.append(box)
        self.list_to_save["time"][self.timer] = self.timer
        self.timer += 1

    def get_trajectory(self):
        return self.list_to_save

    def get_energy(self):
        return self.energy_list

    def get_torsenergy(self):
        return self.energy_tors_list

    def get_box(self):

        return self.list_to_save["unitcell_vectors"]

    def flush_reporter(self):
        # dont need to flush those
        # self.list_to_save["xyz"] *= 0.0
        # self.list_to_save["time"] *= 0.0
        # self.list_to_save["unitcell_vectors"] *= 0.0
        self.timer = 0
        self.energy_tors_list = []
        self.energy_list = []
        self.box_list = []


class DCDReporterMultiFile(object):
    """DCDReporterMultiFile outputs a series of frames from a Simulation to a file.
    To use it, create a DCDReporterMultiFile, then add it to the Simulation's list of reporters.
    The Reporter creates a trajectory file for every cycle of curiosity simulation.
    Arguments:
        file: The filename to write to, the files part named every cycle as
              number_filename.dcd, e.g. 0_niceprotein.dcd, 1_niceprotein.dcd.
              The filename should be given as a full path, e.g. /home/user/niceprotein.dcd
        reportInterval:
            The interval (in time steps) at which to write frames
        append: If True, open an existing DCD file to append to.  If False, create a new file.
        enforcePeriodicBox: Specifies whether particle positions should be translated so the center of every molecule
            lies in the same periodic box.  If None (the default), it will automatically decide whether
            to translate molecules based on whether the system being simulated uses periodic boundary
            conditions.
    """

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        self._reportInterval = reportInterval
        self._append = append
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            mode = "r+b"
        else:
            mode = "wb"
        self._mode = mode
        self._dcd = None
        self._path = os.path.dirname(file)
        self._name = os.path.basename(file)
        self._out = None
        self._counter = 0

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        Arguments:
            simulation : The Simulation to generate a report for
        Returns:
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, True, False, False, False, self._enforcePeriodicBox)

    def nextCycle(self, simulation):
        """Creates a new file, where trajectory is saved.
        The new file has the same name, with a number added at the end.
        """
        # close old file
        if self._out is not None:
            self._out.close()
            del self._dcd
        self._dcd = None
        # create a new name
        self._out = open(
            self._path + "/" + str(self._counter) + "_" + self._name, self._mode
        )
        self._dcd = DCDFile(
            self._out,
            simulation.topology,
            simulation.integrator.getStepSize(),
            simulation.currentStep,
            self._reportInterval,
            self._append,
        )
        self._counter += 1

    def report(self, simulation, state):
        """Generate a report.
        Arguments:
            simulation: The Simulation to generate a report for
            state: The current state of the simulation
        """

        if self._dcd is None:
            self._dcd = DCDFile(
                self._out,
                simulation.topology,
                simulation.integrator.getStepSize(),
                simulation.currentStep,
                self._reportInterval,
                self._append,
            )
        self._dcd.writeModel(
            state.getPositions(), periodicBoxVectors=state.getPeriodicBoxVectors()
        )

    def set_counter(self, counter=None):
        self._counter = counter

    def get_counter(self):
        return self._counter

    def __del__(self):
        self._out.close()
