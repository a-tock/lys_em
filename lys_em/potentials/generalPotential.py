from .interface import PotentialInterface
from ..space import FunctionSpace
import jax.numpy as jnp
import jax


class GeneralPotential(PotentialInterface):
    def __init__(self, space, potential, isTransFunc=False):
        super().__init__(space)
        self._sp = space
        if isTransFunc:
            @jax.jit
            def mask(tf):
                return jnp.fft.ifft2(jnp.fft.fft2(tf) * space.mask)

            self._pot = None
            self._transFunc = jax.vmap(mask)(potential)
        else:
            self._pot = potential
            self._transFunc = None

    def replace(self, potential):
        """
        Return a copy of the GeneralPotential object with the provided potential.

        Args:
            potential (array-like): The potential to replace the existing potential.

        Returns:
            GeneralPotential: A new GeneralPotential object with the provided potential.
        """
        return GeneralPotential(self._sp, potential, isTransFunc=self._transFunc is not None)

    def getPhase(self, beam):
        """
        Calculate the phase of the potential.

        Args:
            beam (TEM): The TEM object representing the incident electron beam.

        Returns:
            jax.numpy.ndarray: The phase of the potential.
        """
        if self._pot is not None:
            return self._pot
        else:
            return jnp.unwrap(jnp.unwrap(jnp.angle(self._transFunc), axis=-2), axis=-1)

    def getTransmissionFunction(self, beam):
        """
        Calculate the transmission function of the potential.

        Args:
            beam (TEM): The TEM object representing the incident electron beam.

        Returns:
            jax.numpy.ndarray: The transmission function of the potential.
        """
        if self._transFunc is not None:
            return self._transFunc
        else:
            return super().getTransmissionFunction(beam)
