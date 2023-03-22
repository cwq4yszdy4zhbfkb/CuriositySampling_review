# from __future__ import print_function, division
import numpy as np
import mdtraj as md
import parmed as prm
from openmm.app import *
from openmm import *
from openmm.unit import *


def strip_offsets(atom_names):
    """Convert a list of atom + offset strings into lists of atoms.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py#L269
    Arguments:
        atom_names A list of names of the atoms, whose offset prexifs you want to strip
    Notes
    -----
    For example, ["-C", "N", "CA", "C"] will be parsed as
    ["C","N","CA","C"]
    Returns:
        A list of atom names without offsets.
    """
    atoms = []
    for atom in atom_names:
        if atom[0] == "-":
            atoms.append(atom[1:])
        elif atom[0] == "+":
            atoms.append(atom[1:])
        else:
            atoms.append(atom)
    return atoms


def construct_atom_dict(topology):
    """Create dictionary to lookup indices by atom name, residue_id, and chain
    index.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py#L269
    Arguments:
        topology : An OpenMM topology object
    Returns:
        atom_dict : Tree of nested dictionaries such that
        `atom_dict[chain_index][residue_index][atom_name] = atom_index`
    """
    atom_dict = {}
    for chain in topology.chains():
        residue_dict = {}
        for residue in chain.residues():
            local_dict = {}
            for atom in residue.atoms():
                local_dict[atom.name] = atom.index
            residue_dict[residue.index] = local_dict
        atom_dict[chain.index] = residue_dict

    return atom_dict


def atom_sequence(top, atom_names, residue_offsets=None):
    """Find sequences of atom indices corresponding to desired atoms.
    This method can be used to find sets of atoms corresponding to specific
    dihedral angles (like phi or psi). It looks for the given pattern of atoms
    in each residue of a given chain. See the example for details.
    Source: https://github.com/mdtraj/mdtraj/blob/c0fa948f9bdc72c9d198c26ffacf8100a0b05638/mdtraj/geometry/dihedral.py
    Arguments:
    top : an OpenMM topology
        Topology for which you want dihedrals.
    atom_names : A numpy array with atom names used for calculating dihedrals
    residue_offsets :
        A numpy numpy array of integer offsets for each atom. These are used to refer
        to atoms forward or backward in the chain relative to the current
        residue
    Example:
    Here we calculate the phi torsion angles by specifying the correct
    atom names and the residue_id offsets (e.g. forward or backward in
    chain) for each atom.
    >>> traj = mdtraj.load("native.pdb")
    >>> atom_names = ["C" ,"N" , "CA", "C"]
    >>> residue_offsets = [-1, 0, 0, 0]
    >>> found_residue_ids, indices = _atom_sequence(traj, atom_names, residue_offsets)
    """

    atom_names = strip_offsets(atom_names)

    atom_dict = construct_atom_dict(top)

    atom_indices = []
    found_residue_ids = []
    atoms_and_offsets = list(zip(atom_names, residue_offsets))
    for chain in top.chains():
        cid = chain.index
        for residue in chain.residues():
            rid = residue.index
            # Check that desired residue_IDs are in dict
            if all([rid + offset in atom_dict[cid] for offset in residue_offsets]):
                # Check that we find all atom names in dict
                if all(
                    [
                        atom in atom_dict[cid][rid + offset]
                        for atom, offset in atoms_and_offsets
                    ]
                ):
                    # Lookup desired atom indices and and add to list.
                    atom_indices.append(
                        [
                            atom_dict[cid][rid + offset][atom]
                            for atom, offset in atoms_and_offsets
                        ]
                    )
                    found_residue_ids.append(rid)

    atom_indices = np.array(atom_indices)
    found_residue_ids = np.array(found_residue_ids)

    if len(atom_indices) == 0:
        atom_indices = np.empty(shape=(0, 4), dtype=np.int)

    return found_residue_ids, atom_indices


class EnergySelect:
    def __init__(self, topology_openmm, system, selection_md="protein"):
        """The class allows calculating MD energy for a given selection.
        To init the class it's requires to pass openmm topology and system.
        Arguments:
        topology_openmm: Topology of the full system from the openmm
        system: System object from the OpenMM
        selection_md: Selection for the calculated energy, according to the MDTraj syntax
        """
        self.selection_md = selection_md
        # create MDtraj topology
        topology = md.Topology.from_openmm(topology_openmm)
        # select atoms according to the selection
        self.sub_ind = topology.select(self.selection_md)
        # create a slice of topology
        sub_top = topology.subset(self.sub_ind)
        # save old topology to make a from positions
        self.old_topology = topology
        # convert sliced topology to the openmm format
        self.topology = sub_top.to_openmm()

        # Creating system only for protein
        struct = prm.openmm.load_topology(topology_openmm, system)
        # select structure
        struct = struct[self.sub_ind]
        # if there're hbond restrains, add parameters
        new_bond_type = prm.topologyobjects.BondType(k=400, req=1.0)
        constrained_bond_type = struct.bond_types.append(new_bond_type)
        struct.bond_types.claim()

        for bond in struct.bonds:
            if bond.type is None:
                bond.type = new_bond_type

        # crete a new system
        new_system = struct.createSystem(
            nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0 * prm.unit.nanometers
        )
        self.system = new_system
        # creating simulation object
        integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1 / picosecond, 0.004 * picoseconds
        )
        self.simulation = Simulation(self.topology, self.system, integrator)

    def calc_energy(self, positions):
        trajectory = md.Trajectory(positions, self.old_topology)
        new_positions = trajectory.atom_slice(self.sub_ind).xyz[0]
        self.simulation.context.setPositions(new_positions)
        state = self.simulation.context.getState(getEnergy=True)

        return state.getPotentialEnergy()


from mdtraj.utils import ensure_type
from mdtraj.utils.six import string_types
from mdtraj.utils.six.moves import xrange
from mdtraj.core import element
import itertools


##############################################################################
# Code
##############################################################################


def compute_contacts(
    traj,
    contacts="all",
    scheme="closest-heavy",
    ignore_nonprotein=True,
    periodic=True,
    soft_min=False,
    soft_min_beta=20,
):
    """Compute the distance between pairs of residues in a trajectory.
    Parameters
    ----------
    traj : md.Trajectory
        An mdtraj trajectory. It must contain topology information.
    contacts : array-like, ndim=2 or 'all'
        An array containing pairs of indices (0-indexed) of residues to
        compute the contacts between, or 'all'. The string 'all' will
        select all pairs of residues separated by two or more residues
        (i.e. the i to i+1 and i to i+2 pairs will be excluded).
    scheme : {'ca', 'closest', 'closest-heavy', 'sidechain', 'sidechain-heavy'}
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
            'sidechain' : distance is the closest distance between any
                two atoms in residue sidechains
            'sidechain-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in residue sidechains
    ignore_nonprotein : bool
        When using `contact==all`, don't compute contacts between
        "residues" which are not protein (i.e. do not contain an alpha
        carbon).
    periodic : bool, default=True
        If periodic is True and the trajectory contains unitcell information,
        we will compute distances under the minimum image convention.
    soft_min : bool, default=False
        If soft_min is true, we will use a diffrentiable version of
        the scheme. The exact expression used
         is d = \frac{\beta}{log\sum_i{exp(\frac{\beta}{d_i}})} where
         beta is user parameter which defaults to 20nm. The expression
         we use is copied from the plumed mindist calculator.
         http://plumed.github.io/doc-v2.0/user-doc/html/mindist.html
    soft_min_beta : float, default=20nm
        The value of beta to use for the soft_min distance option.
        Very large values might cause small contact distances to go to 0.
    Returns
    -------
    distances : np.ndarray, shape=(n_frames, n_pairs), dtype=np.float32
        Distances for each residue-residue contact in each frame
        of the trajectory
    residue_pairs : np.ndarray, shape=(n_pairs, 2), dtype=int
        Each row of this return value gives the indices of the residues
        involved in the contact. This argument mirrors the `contacts` input
        parameter. When `all` is specified as input, this return value
        gives the actual residue pairs resolved from `all`. Furthermore,
        when scheme=='ca', any contact pair supplied as input corresponding
        to a residue without an alpha carbon (e.g. HOH) is ignored from the
        input contacts list, meanings that the indexing of the
        output `distances` may not match up with the indexing of the input
        `contacts`. But the indexing of `distances` *will* match up with
        the indexing of `residue_pairs`
    Examples
    --------
    >>> # To compute the contact distance between residue 0 and 10 and
    >>> # residues 0 and 11
    >>> md.compute_contacts(t, [[0, 10], [0, 11]])
    >>> # the itertools library can be useful to generate the arrays of indices
    >>> group_1 = [0, 1, 2]
    >>> group_2 = [10, 11]
    >>> pairs = list(itertools.product(group_1, group_2))
    >>> print(pairs)
    [(0, 10), (0, 11), (1, 10), (1, 11), (2, 10), (2, 11)]
    >>> md.compute_contacts(t, pairs)
    See Also
    --------
    mdtraj.geometry.squareform : turn the result from this function
        into a square "contact map"
    Topology.residue : Get residues from the topology by index
    """
    if traj.topology is None:
        raise ValueError("contact calculation requires a topology")

    if isinstance(contacts, string_types):
        if contacts.lower() != "all":
            raise ValueError(
                "(%s) is not a valid contacts specifier" % contacts.lower()
            )

        residue_pairs = []
        for i in xrange(traj.n_residues):
            residue_i = traj.topology.residue(i)
            if ignore_nonprotein and not any(
                a for a in residue_i.atoms if a.name.lower() == "ca"
            ):
                continue
            for j in xrange(i + 3, traj.n_residues):
                residue_j = traj.topology.residue(j)
                if ignore_nonprotein and not any(
                    a for a in residue_j.atoms if a.name.lower() == "ca"
                ):
                    continue
                if residue_i.chain == residue_j.chain:
                    residue_pairs.append((i, j))

        residue_pairs = np.array(residue_pairs)
        if len(residue_pairs) == 0:
            raise ValueError("No acceptable residue pairs found")

    else:
        residue_pairs = ensure_type(
            np.asarray(contacts),
            dtype=int,
            ndim=2,
            name="contacts",
            shape=(None, 2),
            warn_on_cast=False,
        )
        if not np.all((residue_pairs >= 0) * (residue_pairs < traj.n_residues)):
            raise ValueError(
                "contacts requests a residue that is not in the permitted range"
            )

    # now the bulk of the function. This will calculate atom distances and then
    # re-work them in the required scheme to get residue distances
    scheme = scheme.lower()
    if scheme not in ["ca", "closest", "closest-heavy", "sidechain", "sidechain-heavy"]:
        raise ValueError(
            "scheme must be one of [ca, closest, closest-heavy, sidechain, sidechain-heavy]"
        )

    if scheme == "ca":
        if soft_min:
            import warnings

            warnings.warn(
                "The soft_min=True option with scheme=ca gives"
                "the same results as soft_min=False"
            )
        filtered_residue_pairs = []
        atom_pairs = []

        for r0, r1 in residue_pairs:
            ca_atoms_0 = [
                a.index for a in traj.top.residue(r0).atoms if a.name.lower() == "ca"
            ]
            ca_atoms_1 = [
                a.index for a in traj.top.residue(r1).atoms if a.name.lower() == "ca"
            ]
            if len(ca_atoms_0) == 1 and len(ca_atoms_1) == 1:
                atom_pairs.append((ca_atoms_0[0], ca_atoms_1[0]))
                filtered_residue_pairs.append((r0, r1))
            elif len(ca_atoms_0) == 0 or len(ca_atoms_1) == 0:
                # residue does not contain a CA atom, skip it
                if contacts != "all":
                    # if the user manually asked for this residue, and didn't use "all"
                    import warnings

                    warnings.warn(
                        "Ignoring contacts pair %d-%d. No alpha carbon." % (r0, r1)
                    )
            else:
                raise ValueError(
                    "More than 1 alpha carbon detected in residue %d or %d" % (r0, r1)
                )

        residue_pairs = np.array(filtered_residue_pairs)
        distances = md.compute_distances(traj, atom_pairs, periodic=periodic)

    elif scheme in ["closest", "closest-heavy", "sidechain", "sidechain-heavy"]:
        if scheme == "closest":
            residue_membership = [
                [atom.index for atom in residue.atoms]
                for residue in traj.topology.residues
            ]
        elif scheme == "closest-heavy":
            # then remove the hydrogens from the above list
            residue_membership = [
                [
                    atom.index
                    for atom in residue.atoms
                    if not (atom.element == element.hydrogen)
                ]
                for residue in traj.topology.residues
            ]
        elif scheme == "sidechain":
            residue_membership = [
                [atom.index for atom in residue.atoms if atom.is_sidechain]
                for residue in traj.topology.residues
            ]
        elif scheme == "sidechain-heavy":
            # then remove the hydrogens from the above list
            if "GLY" in [residue.name for residue in traj.topology.residues]:
                import warnings

                warnings.warn(
                    "selected topology includes at least one glycine residue, which has no heavy atoms in its sidechain. The distances involving glycine residues "
                    "will be computed using the sidechain hydrogen instead."
                )
            residue_membership = [
                [
                    atom.index
                    for atom in residue.atoms
                    if atom.is_sidechain and not (atom.element == element.hydrogen)
                ]
                if not residue.name == "GLY"
                else [atom.index for atom in residue.atoms if atom.is_sidechain]
                for residue in traj.topology.residues
            ]

        residue_lens = [len(ainds) for ainds in residue_membership]

        atom_pairs = []
        n_atom_pairs_per_residue_pair = []
        for pair in residue_pairs:
            atom_pairs.extend(
                list(
                    itertools.product(
                        residue_membership[pair[0]], residue_membership[pair[1]]
                    )
                )
            )
            n_atom_pairs_per_residue_pair.append(
                residue_lens[pair[0]] * residue_lens[pair[1]]
            )

        atom_distances = md.compute_distances(traj, atom_pairs, periodic=periodic)

        # now squash the results based on residue membership
        n_residue_pairs = len(residue_pairs)
        distances = np.zeros((len(traj), n_residue_pairs), dtype=np.float32)
        n_atom_pairs_per_residue_pair = np.asarray(n_atom_pairs_per_residue_pair)
        indexes = []
        atom_pairs = np.array(atom_pairs)
        for i in xrange(n_residue_pairs):
            index = int(np.sum(n_atom_pairs_per_residue_pair[:i]))
            n = n_atom_pairs_per_residue_pair[i]
            if not soft_min:
                distances[:, i] = atom_distances[:, index : index + n].min(axis=1)
                # returns indices in the space of index:index+n, which are atoms of residue
                g = atom_distances[:, index : index + n].argmin(axis=1)
                # returns indices in space of all residue atoms
                r = np.arange(index, index + n + 1)[g]
                # returns pair of trajectory
                atom_ind = atom_pairs[r]
            else:
                distances[:, i] = soft_min_beta / np.log(
                    np.sum(
                        np.exp(soft_min_beta / atom_distances[:, index : index + n]),
                        axis=1,
                    )
                )
            indexes.append(atom_ind)
    else:
        raise ValueError("This is not supposed to happen!")

    return distances, residue_pairs, np.transpose(np.array(indexes), [1, 0, -1])
