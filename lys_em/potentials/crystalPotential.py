import jax
import jax.numpy as jnp

from lys_mat import CrystalStructure

from ..kinematical import structureFactorFunc 
from ..consts import m
from .interface import PotentialInterface


class CrystalPotential(PotentialInterface):
    """
    Represents the potential of a crystal for electron microscopy simulations.

    Args:
        space (FunctionSpace): The simulation space object containing spatial parameters.
        crys (CrystalStructure): The crystal object containing unit cell and lattice information.
    """

    def __init__(self, space, crys):
        super().__init__(space)
        self._crys = crys
        self._sp = space

    def getPhase(self, beam):
        return self._getPhaseFromParameters(beam, self._crys.unit, self._crys.getAtomicPositions(), [at.Uani for at in self._crys.atoms])

    def getFromParameters(self, beam, unit, position, Uani):
        phase = self._getPhaseFromParameters(beam, unit, position, Uani)
        return self.phaseToTrans(phase)

    def _getPhaseFromParameters(self, beam, unit, position, Uani):
        @jax.jit
        def _calcPhase(V_k, mask, kd, n):
            return jnp.fft.ifft2(V_k * jnp.exp(1j*n*kd) * mask)

        V_ks = _Slices(self._crys, self._sp).getPotential(beam, unit ,position, Uani) # (division, Nx, Ny)
        n = jnp.arange(round(self._sp.c / self._crys.unit[2][2])) # (ncells)

        kd = self._sp.kvec.dot(self._crys.unit[2][0:2]) # (Nx, Ny)
        f = jax.vmap(jax.vmap(_calcPhase, in_axes=(0, None, None, None)), in_axes=(None, None, None, 0))
        return f(V_ks, self._sp.mask, kd, n).reshape(-1, *self._sp.N[0:2])/self._sp.dV # (ncells*division, Nx, Ny)


class _Slices:
    def __init__(self, crys, sp):
        self._sp = sp

        division = round(crys.unit[2][2]/sp.dz)
        zList = jnp.arange(division + 1) * crys.unit[2][2] / division
        self._index = [[j for j, pos in enumerate(crys.getAtomicPositions()) if zList[i] <= pos[2] < zList[i + 1]] for i in range(len(zList) - 1)]
        self._slices = [CrystalStructure(crys.cell, [crys.atoms[i] for i in indices]) for indices in self._index]

    def getPotential(self, beam, unit ,position, Uani):
        sig =beam.wavelength * beam.relativisticMass / m
        positions = [jnp.array([position[i] for i in indices]) for indices in self._index]
        Uanis = [jnp.array([Uani[i] for i in indices]) for indices in self._index]
        return jnp.array([self._calcFromParameters(c, unit, pos, U) for c, pos, U in zip(self._slices, positions, Uanis)]) * sig

    def _calcFromParameters(self, c, unit ,position, Uani):
        k = self._sp.kvec
        q = jnp.array([k[:, :, 0], k[:, :, 1], k[:, :, 1]*0]).transpose(1, 2, 0)
        return structureFactorFunc(c, q)(unit, position, Uani)
