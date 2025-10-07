import numpy as np
import jax.numpy as jnp 

from ..consts import h, e, mu_0, eps
from .interface import PotentialInterface


class MagneticPotential(PotentialInterface):
    def __init__(self, space, M):
        super().__init__(space)
        self._sp = space
        self._M = jnp.array(M) # in A/m

    def replace(self, M):
        return MagneticPotential(self._sp, M)

    def getPhase(self, beam):
        mx, my = jnp.fft.fft2(self._M[:,:,:,0]) * self._sp.dV, jnp.fft.fft2(self._M[:,:,:,1]) * self._sp.dV
        k = self._sp.kvec
        const = 1j * np.pi * mu_0 * self._sp.dz / (h/e)
        V_k = const * (mx*k[:,:,1]*0-my*k[:,:,0])/(jnp.linalg.norm(k, axis=2)**2+eps)
        return jnp.fft.ifft2(V_k * 1e-20 / self._sp.dV * self._sp.mask)
    
