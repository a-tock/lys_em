import numpy as np
from .consts import m, e, hbar, kB, h, c

class TEM(object):
    """
    TEM parameters used for simulations.

    Args:
        acc (float): The acceleration voltage.
        convergence (float): The convergence angle of incident electron beam.
    """
    def __init__(self, acc, convergence=0, divergence=np.inf, defocus=0, Cs=0, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        self.__acc = acc
        self._convergence = convergence
        self._divergence = divergence
        self._defocus = defocus
        self._Cs = Cs
        self.__direction = np.array(directions)

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
        alpha = self.wavelength*k/(2*np.pi)
        return 2*np.pi/self.wavelength * (0.25*self._Cs*alpha**4 - 0.5*self._defocus*alpha**2)

    @property
    def relativisticMass(self):
        '''
        Get relativistic mass of electron in kg.

        Returns:
            float:Relativistic mass 
        '''
        return m + e * self.__acc / c**2

    def getDirectionVectors(self):
        return self.__direction

