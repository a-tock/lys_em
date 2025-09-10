import numpy as np

from ..FFT import fft, ifft
from ..consts import h, e, mu_0

from .interface import PotentialInterface


class MagneticPotential(PotentialInterface):
    def __init__(self, space, M):
        self._sp = space
        self._M = M

    def getPhase(self, beam):
        mx, my = fft(self._m[0]) * self._sp.dV, fft(self._m[1]) * self._sp.dV
        kx, ky = self._sp.kvec
        k2 = self._sp.k2
        const = 1j * np.pi * mu_0 * self._sp.dz / (h/e)
        V_k = const * (mx*ky-my*kx)/k2
        return ifft(V_k * self._sp.mask / self._sp.dV)
    
