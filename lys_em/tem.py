import numpy as np
from .consts import m, e, hbar, kB, h, c


class TEM(object):
    """
    TEM parameters used for simulations.

    Args:
        acc (float): The acceleration voltage in kV.
        convergence (float, optional): The convergence angle of the incident electron beam in radians. Default is 0.
        divergence (float, optional): The divergence angle of the electron beam in radians. Default is np.inf.
        defocus (float, optional): The defocus value in angstroms. Default is 0.
        Cs (float, optional): The spherical aberration coefficient in millimeters. Default is 0.
        direction (list or array-like, optional): The direction vector of the electron beam. Default is [0, 0, -1].
    """

    def __init__(self, acc, convergence=0, divergence=np.inf, defocus=0, Cs=0, direction=[0, 0, -1]):
        self.__acc = acc
        self._convergence = convergence
        self._divergence = divergence
        self._defocus = defocus
        self._Cs = Cs
        self._direction = np.array(direction) / np.linalg.norm(direction)

    @property
    def wavelength(self):
        """
        Return a wavelength of probe electron in angstrom.
        """
        return 1.2264e-9 / np.sqrt(self.__acc * (1 + 9.7846e-7 * self.__acc)) * 1e10

    @property
    def k_in(self):
        """
        Return a wavenumber of incident electron (2pi/wavelength) in rad/angstrom.
        """
        return 2 * np.pi / self.wavelength

    @property
    def k_max(self):
        """
        Return a lateral wavenumber of most-tilted incident electron determined by convergence angle. 
        """
        return 2 * np.pi * self._convergence / self.wavelength

    def chi(self, k):
        """
        Return net phase error from spherical aberration and defocus.

        Args:
            k(sequence of length 2 array): The wavenumber.
        """
        alpha = self.wavelength * k / (2 * np.pi)
        return 2 * np.pi / self.wavelength * (0.25 * self._Cs * alpha**4 - 0.5 * self._defocus * alpha**2)

    @property
    def relativisticMass(self):
        '''
        Get relativistic mass of electron in kg.

        Returns:
            float:Relativistic mass
        '''
        return m + e * self.__acc / c**2

    @property
    def beamDirection(self):
        """
        Return the direction vector of the electron beam.

        Returns:
            array: The direction vector with shape (3,).
        """
        return self._direction
