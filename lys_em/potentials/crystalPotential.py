import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
import copy

from lys_mat import CrystalStructure

from ..kinematical import structureFactors
from ..consts import m
from .interface import PotentialInterface
from ..space import FunctionSpace


class CrystalPotential(PotentialInterface):
    """
    Represents the potential of a crystal for electron microscopy simulations.

    Args:
        space (FunctionSpace): The simulation space object containing spatial parameters.
        crys (CrystalStructure): The crystal object containing unit cell and lattice information.
    """

    def __init__(self, space, crys, index=None):
        super().__init__(space)
        self._crys = crys
        self._index = index

    def replace(self, crys=None, unit=None, pos=None, Uani=None, params=None):
        """
        Replaces the crystal potential with a new one.

        Args:
            crys (CrystalStructure, optional): The new crystal structure for creating a new potential.
            unit (numpy.ndarray, optional): The new unit cell. If crys is not None, this will be ignored.
            pos (list of numpy.ndarray, optional): The new atomic positions. If crys is not None, this will be ignored.
            Uani (list of float, optional): The new isotropic displacement parameters. If crys is not None, this will be ignored.
            params (dict, optional): The new parameters for the crystal structure. Keys are the names of Sympy.Symbols and values are the new values.

        Returns:
            CrystalPotential: The new crystal potential.
        """
        if crys is None:
            crys = copy.deepcopy(self._crys)
            if unit is not None:
                crys.unit = unit
            if pos is not None:
                for at, p in zip(crys.atoms, pos):
                    at.position = p
            if Uani is not None:
                for at, u in zip(crys.atoms, Uani):
                    at.Uani = u
        if params is not None:
            crys = CrystalStructure(CrystalPotential._subs(self._crys.cell, params), [CrystalPotential._atomSubs(atom, params) for atom in self._crys.atoms])
        ncells, division = jnp.round(self.space.c / self._crys.unit[2][2]), jnp.round(self._crys.unit[2][2] / self.space.dz)
        space = FunctionSpace.fromCrystal(crys, self.space.N[0], self.space.N[1], ncells, division=division)
        return CrystalPotential(space, crys, index=self._index)

    def _sliceIndex(self, crys, sp):
        division = round(crys.unit[2][2] / sp.dz)
        zList = jnp.arange(division + 1) * crys.unit[2][2] / division
        u = jnp.array(crys.unit.T)
        pos = jnp.array([u.dot(at.position) for at in crys.atoms])
        return [[j for j, pos in enumerate(pos) if zList[i] <= pos[2] < zList[i + 1]] for i in range(len(zList) - 1)]

    def getPhase(self, beam):
        """
        Calculates the phase of the crystal potential.

        Args:
            beam (TEM): The TEM object representing the incident electron beam.

        Returns:
            float: The phase of the crystal potential.
        """
        @jax.jit
        def _calcPhase(V_k, mask, kd, n):
            return jnp.fft.ifft2(V_k * jnp.exp(1j * n * kd) * mask)

        sig = beam.wavelength * beam.relativisticMass / m
        V_ks = self.getSlicePotential(self._crys) * sig  # (division, Nx, Ny)
        n = jnp.arange(round(self.space.c / self._crys.unit[2][2]))  # (ncells)

        kd = self.space.kvec.dot(self._crys.unit[2][0:2])  # (Nx, Ny)
        f = jax.vmap(jax.vmap(_calcPhase, in_axes=(0, None, None, None)), in_axes=(None, None, None, 0))
        return f(V_ks, self.space.mask, kd, n).reshape(-1, *self.space.N[0:2]) / self.space.dV  # (ncells*division, Nx, Ny)

    def getSlicePotential(self, crys):
        """
        Calculates the potential of a crystal slice.

        Args:
            crys (CrystalStructure): The crystal object containing unit cell and lattice information.

        Returns:
            numpy.ndarray: The potential of the crystal slice.
        """
        k = self.space.kvec
        q = jnp.array([k[:, :, 0], k[:, :, 1], k[:, :, 1] * 0]).transpose(1, 2, 0)

        if self._index is None:
            self._index = self._sliceIndex(crys, self.space)

        slices = [_Crystal(crys.unit, [crys.atoms[i] for i in indices]) for indices in self._index]
        return jnp.array([structureFactors(c, q) for c in slices])

    @staticmethod
    def _subs(obj, dic):
        if not CrystalPotential._isSympyObject(obj):
            return obj

        if hasattr(obj, "__iter__"):
            if isinstance(obj, dict):
                res = {key: CrystalPotential._subs(value, dic) for key, value in obj.items()}
            else:
                res = [CrystalPotential._subs(value, dic) for value in obj]
                if type(obj) in (np.ndarray, jnp.ndarray):
                    res = jnp.array(res)
                else:
                    res = type(obj)(res)
        else:
            if isinstance(obj, sp.Basic):
                res = sp.lambdify(dic.keys(), obj)
                res = res(*dic.values())
            else:
                res = obj.subs(dic)
            if hasattr(res, "is_number"):
                if res.is_number:
                    return float(res)

        return res

    @staticmethod
    def _isSympyObject(obj):

        if hasattr(obj, "__iter__"):
            if type(obj) is str or len(obj) == 0:
                return False
            if isinstance(obj, dict):
                return any([CrystalPotential._isSympyObject(value) for value in obj.values()])
            else:
                return any([CrystalPotential._isSympyObject(y) for y in obj])

        if hasattr(obj, "free_symbols"):
            return len(obj.free_symbols) > 0

        return False

    @staticmethod
    def _atomSubs(atom, dic):
        res = atom.duplicate()
        for key, val in res.__dict__.items():
            if CrystalPotential._isSympyObject(val):
                setattr(res, key, CrystalPotential._subs(val, dic))
        return res


class _Crystal:
    def __init__(self, unit, atoms):
        self._unit = unit
        self._atoms = atoms

    @property
    def atoms(self):
        return self._atoms

    @property
    def unit(self):
        return self._unit
