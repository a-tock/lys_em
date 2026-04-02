from .interface import PotentialInterface
from ..space import FunctionSpace
import jax.numpy as jnp
import jax


class GeneralPotential(PotentialInterface):
    def __init__(self, space, potential, isTransFunc=False):
        super().__init__(space)
        self._sp = space
        if isTransFunc:
            self._pot = None
            self._transFunc = potential
        else:
            self._pot = potential
            self._transFunc = None

    def replace(self, potential, isTransFunc=False):
        return GeneralPotential(self._sp, potential, isTransFunc=isTransFunc)

    def getPhase(self, beam):
        if self._pot is not None:
            return self._pot
        else:
            return jnp.unwrap(jnp.unwrap(jnp.angle(self._transFunc), axis=-2), axis=-1)

    def getTransmissionFunction(self, beam):
        @jax.jit
        def mask(tf):
            return jnp.fft.ifft2(jnp.fft.fft2(tf) * self._space.mask)

        if self._transFunc is not None:
            return jax.vmap(mask)(self._transFunc)
        else:
            return super().getTransmissionFunction(beam)
