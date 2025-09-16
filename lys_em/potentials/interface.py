import numpy as np
import jax
import jax.numpy as jnp


class PotentialInterface:
    def __init__(self, space):
        @jax.jit
        def _phaseToPot(phase):
            return jnp.fft.ifft2(jnp.fft.fft2(jnp.exp(1j*phase)) * space.mask)
        self.phaseToTrans = jax.vmap(_phaseToPot)

    def __add__(self, obj):
        return _CombinedPotential(self, obj)

    def getPhase(self, beam):
        raise NotImplementedError("Potential class should implement this method.")
    
    def get(self, beam):
        return self.phaseToTrans(self.getPhase(beam))


class _CombinedPotential(PotentialInterface):
    def __init__(self, *objs):
        self._objs = objs

    def getPhase(self, beam):
        return np.sum([obj.getPhase(beam) for obj in self._objs], axis=0)




