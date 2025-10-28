import jax
import jax.numpy as jnp


class PotentialInterface:
    """
    Initializes the PotentialInterface.

    Args:
        space (FunctionSpace): The simulation space object containing spatial parameters.
    """

    def __init__(self, space):

        self._space = space

    def getTransmissionFunction(self, beam):
        """
        Returns the transmission function of the potential.

        Args:
            beam (TEM): The beam object representing the incident electron beam.

        Returns:
            A 2D array of shape (Nx, Ny) where each element is
            the transmission function at the respective grid point in
            real space.
        """
        @jax.jit
        def phaseToTrans(phase):
            return jnp.fft.ifft2(jnp.fft.fft2(jnp.exp(1j * phase)) * self._space.mask)

        return jax.vmap(phaseToTrans)(self.getPhase(beam))

    def getPhase(self, beam):
        """
        Returns the phase of the potential.

        Args:
            beam (TEM): The beam object representing the incident electron beam.

        Returns:
            A 2D array of shape (Nx, Ny) where each element is
            the phase of the potential at the respective grid point in
            real space.
        """
        raise NotImplementedError("Potential class should implement this method.")

    @property
    def space(self):
        """
        Returns the simulation space object.

        Returns:
            FunctionSpace: The simulation space object containing spatial parameters.
        """
        return self._space
