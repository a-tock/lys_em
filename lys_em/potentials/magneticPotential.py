import numpy as np
import jax.numpy as jnp

from ..consts import h, e, mu_0, eps
from .interface import PotentialInterface


class MagneticPotential(PotentialInterface):
    """
    Creates a MagneticPotential object.

    Args:
        space (FunctionSpace): The simulation space object containing spatial parameters.
        M (numpy.ndarray): A 3D array of shape (Nx, Ny, Nz) containing the magnetic field strength in A/m at each grid point.
    """

    def __init__(self, space, M):
        super().__init__(space)
        self._sp = space
        self._M = jnp.array(M)  # in A/m

    def replace(self, M):
        """
        Replaces the magnetic field strength with a new array.

        Args:
            M (numpy.ndarray): A 3D array of shape (Nx, Ny, Nz) containing the magnetic field strength in A/m at each grid point.

        Returns:
            MagneticPotential: A new MagneticPotential object with the replaced magnetic field strength.
        """
        return MagneticPotential(self._sp, M)

    def getPhase(self, beam):
        """
        Calculates the phase shift of the electron beam due to the magnetic field.

        Args:
            beam (TEM): The TEM object.

        Returns:
            numpy.ndarray: A 2D array of shape (Nx, Ny) containing the phase shift of the electron beam at each grid point in radians.
        """
        mx, my = jnp.fft.fft2(self._M[:, :, :, 0]) * self._sp.dV, jnp.fft.fft2(self._M[:, :, :, 1]) * self._sp.dV
        k = self._sp.kvec
        const = 1j * np.pi * mu_0 * self._sp.dz / (h / e)
        V_k = const * (mx * k[:, :, 1] * 0 - my * k[:, :, 0]) / (jnp.linalg.norm(k, axis=2)**2 + eps)
        return jnp.fft.ifft2(V_k * 1e-20 / self._sp.dV * self._sp.mask)
