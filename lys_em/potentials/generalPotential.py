from .interface import PotentialInterface
from ..space import FunctionSpace
import jax.numpy as jnp


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
            return self._transFunc

    def getTransmissionFunction(self, beam):
        if self._transFunc is not None:
            return self._transFunc
        else:
            return super().getTransmissionFunction(beam)
