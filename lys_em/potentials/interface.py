import numpy as np
import jax
import jax.numpy as jnp

class PotentialInterface:
    def __add__(self, obj):
        return _CombinedPotential(self, obj)

    def getPhase(self, beam):
        raise NotImplementedError("Potential class should implement this method.")
    
    def get(self, sp, beam):
        phase = jnp.array(self.getPhase(beam))
        return jax.vmap(_phaseToPot, in_axes=[0, None])(phase, sp.mask).block_until_ready()


class _CombinedPotential(PotentialInterface):
    def __init__(self, *objs):
        self._objs = objs

    def getPhase(self, beam):
        return np.sum([obj.getPhase(beam) for obj in self._objs], axis=0)


@jax.jit
def _phaseToPot(phase, mask):
    return jnp.fft.ifft2(jnp.fft.fft2(jnp.exp(1j*phase)) * mask)


