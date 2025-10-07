import jax
import jax.numpy as jnp
import copy

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
        if index is None:
            self._index = self._sliceIndex(crys, space)
        else:
            self._index = index

    def replace(self, crys=None, unit=None, pos=None, Uani=None):
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
        ncells, division = jnp.round(self._space.c/self._crys.unit[2][2]), jnp.round(self._crys.unit[2][2]/self._space.dz)
        space = FunctionSpace.fromCrystal(crys, self._space.N[0], self._space.N[1], ncells, division=division)
        return CrystalPotential(space, crys, index=self._index)

    def _sliceIndex(self, crys, sp):
        division = round(crys.unit[2][2]/sp.dz)
        zList = jnp.arange(division + 1) * crys.unit[2][2] / division
        u = jnp.array(crys.unit.T)
        pos = jnp.array([u.dot(at.position) for at in crys.atoms])
        return [[j for j, pos in enumerate(pos) if zList[i] <= pos[2] < zList[i + 1]] for i in range(len(zList) - 1)]

    def getPhase(self, beam):
        @jax.jit
        def _calcPhase(V_k, mask, kd, n):
            return jnp.fft.ifft2(V_k * jnp.exp(1j*n*kd) * mask)

        sig =beam.wavelength * beam.relativisticMass / m
        V_ks = self.getSlicePotential(self._crys)*sig # (division, Nx, Ny)
        n = jnp.arange(round(self.space.c / self._crys.unit[2][2])) # (ncells)

        kd = self.space.kvec.dot(self._crys.unit[2][0:2]) # (Nx, Ny)
        f = jax.vmap(jax.vmap(_calcPhase, in_axes=(0, None, None, None)), in_axes=(None, None, None, 0))
        return f(V_ks, self.space.mask, kd, n).reshape(-1, *self.space.N[0:2])/self.space.dV # (ncells*division, Nx, Ny)
    
    def getSlicePotential(self, crys):
        k = self.space.kvec
        q = jnp.array([k[:, :, 0], k[:, :, 1], k[:, :, 1]*0]).transpose(1, 2, 0)
        slices = [_Crystal(crys.unit, [crys.atoms[i] for i in indices]) for indices in self._index]
        return jnp.array([structureFactors(c, q) for c in slices])


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