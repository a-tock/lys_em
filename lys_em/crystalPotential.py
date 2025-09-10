import numpy as np
from lys_mat import CrystalStructure

from . import fft, ifft
from .kinematical import structureFactors 
from .consts import m


class CrystalPotential:
    def __init__(self, space, crys):
        self._sp = space
        self._crys = crys

    def get(self, beam):
        V_rs = _Slices(self._crys, self._sp).getPotentialTerms(beam)
        phase = _calcPhase(V_rs, self._sp.kvec, self._crys.unit[2][0], self._crys.unit[2][1], int(self._sp.c / self._crys.unit[2][2]))
        return np.array([ifft(fft(p) * self._sp.mask) for p in np.exp(1j*phase)])
    

def _calcPhase(V_rs, kvec, dx, dy, numOfSlices):
    phase1 = np.exp(1j*kvec.dot([dx, dy]))
    potentials = []
    for n in range(numOfSlices):
        phase = 1 if n == 0 else phase * phase1
        for V_r in V_rs:
            potentials.append(ifft(fft(V_r) * phase))
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
        res = []
        sig =beam.wavelength * beam.relativisticMass / m
        for c in self._slices:
            V_k = self._calculatePotential(c)  # A^2
            V_r = ifft(V_k * self._sp.mask) / self._sp.dV 
            res.append(sig *V_r)
        return res
    
    def _calculatePotential(self, crys):
        if len(crys.atoms) == 0:
            return 0
        else:
            k = self._sp.kvec
            q = np.array([k[:, :, 0], k[:, :, 1], k[:, :, 1]*0]).transpose(1, 2, 0)
            return structureFactors(crys, q)

