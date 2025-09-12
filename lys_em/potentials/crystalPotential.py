import numpy as np
from lys_mat import CrystalStructure

from ..kinematical import structureFactors 
from ..consts import m

from .interface import PotentialInterface

import jax
import jax.numpy as jnp



class CrystalPotential(PotentialInterface):
    """
    Represents the potential of a crystal for electron microscopy simulations.

    Args:
        space (FunctionSpace): The simulation space object containing spatial parameters.
        crys (CrystalStructure): The crystal object containing unit cell and lattice information.
    """

    def __init__(self, space, crys):
        self._sp = space
        self._crys = crys

    def getPhase(self, beam):
        @jax.jit
        def _calcPhase(V_k, mask, kd, n):
            return jnp.fft.ifft2(V_k * jnp.exp(1j*n*kd) * mask)

        V_ks = _Slices(self._crys, self._sp).getPotentialTerms(beam) # (division, Nx, Ny)
        n = jnp.arange(round(self._sp.c / self._crys.unit[2][2])) # (ncells)

        kd = self._sp.kvec.dot(self._crys.unit[2][0:2]) # (Nx, Ny)
        f = jax.vmap(jax.vmap(_calcPhase, in_axes=(0, None, None, None)), in_axes=(None, None, None, 0))
        return f(V_ks, self._sp.mask, kd, n).reshape(-1, *self._sp.N[0:2])/self._sp.dV # (ncells*division, Nx, Ny)


class _Slices:
    def __init__(self, crys, sp):
        self._sp = sp
        self._slices = []

        division = round(crys.unit[2][2]/sp.dz)
        zList = np.arange(division + 1) * crys.unit[2][2] / division
        positionList = crys.getAtomicPositions()
        for i in range(len(zList) - 1):
            atomsList = [at for pos, at in zip(positionList, crys.atoms) if zList[i] <= pos[2] < zList[i + 1]]
            self._slices.append(CrystalStructure(crys.cell, atomsList))

    def getPotentialTerms(self, beam):
        sig =beam.wavelength * beam.relativisticMass / m
        return jnp.array([sig * self._calculatePotential(c) for c in self._slices])
    
    def _calculatePotential(self, crys):
        if len(crys.atoms) == 0:
            return self._sp.kvec[:,:,0]*0
        else:
            k = self._sp.kvec
            q = np.array([k[:, :, 0], k[:, :, 1], k[:, :, 1]*0]).transpose(1, 2, 0)
            return structureFactors(crys, q)

