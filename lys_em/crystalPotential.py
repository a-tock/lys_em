import numpy as np
from lys_mat import CrystalStructure

from . import fft, ifft
from .kinematical import structureFactors 

m = 9.10938356e-31  # kg

class CrystalPotential:
    def __init__(self, space, beam, crys, numOfCells):
        self._sp = space
        self._beam = beam
        self._crys = crys
        self._cells = numOfCells

    def __iter__(self):
        V_rs = _Slices(self._crys, self._sp).getPotentialTerms(self._beam)
        self.pot = _Potentials(V_rs, self._sp.kvec, self._crys.unit[2][0], self._crys.unit[2][1], self._cells)
        return self.pot.__iter__()


class _Potentials:
    def __init__(self, V_rs, kvec, dx, dy, numOfSlices, type="precalc"):
        phase1 = np.exp(1j*kvec.dot([dx, dy]))
        if type == "precalc":
            self._pots = self.__calc_potential(V_rs, phase1, numOfSlices)

    def __calc_potential(self, V_rs, phase1, numOfSlices):
        potentials = []
        for n in range(numOfSlices):
            phase = 1 if n == 0 else phase * phase1
            for V_r in V_rs:
                potentials.append(ifft(fft(V_r) * phase))
        return potentials

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n == len(self._pots):
            raise StopIteration()
        res = self._pots[self._n]
        self._n += 1
        return res


class _Slices:
    def __init__(self, crys, sp):
        self._sp = sp
        self._slices = []
        zList = np.arange(sp.division + 1) * sp.dz
        positionList = crys.getAtomicPositions()
        for i in range(len(zList) - 1):
            atomsList = [at for pos, at in zip(positionList, crys.atoms) if zList[i] <= pos[2] < zList[i + 1]]
            self._slices.append(CrystalStructure(crys.cell, atomsList))

    def _calculatePotential(self, crys):
        if len(crys.atoms) == 0:
            return 0
        else:
            k = self._sp.kvec
            q = np.array([k[:, :, 0], k[:, :, 1], self._sp.getArray() * 0]).transpose(1, 2, 0)
            return structureFactors(crys, q)

    def getPotentialTerms(self, beam):
        res = []
        for c in self._slices:
            V_k = self._calculatePotential(c) * beam.getWavelength() * beam.getRelativisticMass() / m  # A^2
            V_r = self._sp.IFT(V_k * self._sp.mask)
            V_z = np.exp(1j * V_r)
            V_z_lim = self._sp.IFT(self._sp.FT(V_z) * self._sp.mask)
            res.append(V_z_lim)
        return res