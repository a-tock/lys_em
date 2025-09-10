import numpy as np
from lys_mat import CrystalStructure

from ..FFT import fft, ifft
from ..kinematical import structureFactors 
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
        self._sp = space
        self._crys = crys

    def getPhase(self, beam):
        V_ks = _Slices(self._crys, self._sp).getPotentialTerms(beam)
        return _calcPhase(V_ks, self._sp, self._sp.kvec, self._crys.unit[2][0], self._crys.unit[2][1], int(self._sp.c / self._crys.unit[2][2]))
    

def _calcPhase(V_ks, sp, kvec, dx, dy, numOfSlices):
    phase1 = np.exp(1j*kvec.dot([dx, dy]))
    potentials = []
    for n in range(numOfSlices):
        phase = sp.mask if n == 0 else phase * phase1
        for V_k in V_ks:
            potentials.append(ifft(V_k * phase / sp.dV))
    return np.array(potentials)


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
        return [sig * self._calculatePotential(c) for c in self._slices]
    
    def _calculatePotential(self, crys):
        if len(crys.atoms) == 0:
            return 0
        else:
            k = self._sp.kvec
            q = np.array([k[:, :, 0], k[:, :, 1], k[:, :, 1]*0]).transpose(1, 2, 0)
            return structureFactors(crys, q)

