import jax
import jax.numpy as jnp


class PotentialInterface:
    def __init__(self, space):
        self._space = space
    
    def getTransmissionFunction(self, beam):
        @jax.jit
        def phaseToTrans(phase):
            return jnp.fft.ifft2(jnp.fft.fft2(jnp.exp(1j*phase)) * self._space.mask)

        return  jax.vmap(phaseToTrans)(self.getPhase(beam))

    def getPhase(self, beam):
        raise NotImplementedError("Potential class should implement this method.")
    
    @property
    def space(self):
        return self._space





