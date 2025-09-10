import numpy as np
from .FFT import fft, ifft
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

    def __init__(self, acc, convergence=0, divergence=np.inf, Cs=0):
        self.__acc = acc
        self._convergence = convergence
        self._divergence = divergence
        self._Cs = Cs

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

    @property
    def Cs(self):
        '''
        Get spherical aberration coefficient in angstrom.

        Returns:
            float: Spherical aberration coefficient
        '''
        return self._Cs

    @property
    def relativisticMass(self):
        '''
        Get relativistic mass of electron in kg.

        Returns:
            float:Relativistic mass
        '''
        return m + e * self.__acc / c**2


class TEMParameter:
    def __init__(self, defocus=0, tilt=[0, 0], position=[0, 0]):
        self._defocus = defocus
        self._tilt = np.radians(tilt)
        self._position = position

    @property
    def beamDirection(self):
        """
        Return the direction vector of the electron beam.

        Returns:
            array: The direction vector with shape (3,).
        """
        return np.array([np.sin(self._tilt[0]) * np.cos(self._tilt[1]), np.sin(self._tilt[0]) * np.sin(self._tilt[1]), -np.cos(self._tilt[0])])

    def beamTilt(self, type="polar"):
        """
        Return the tilt angles of the electron beam in radians.

        Returns:
            array: The tilt angles with shape (2,).
        """

        if type == "polar":
            return self._tilt
        elif type == "cartesian":
            x, y, z = self.beamDirection
            return np.degrees([np.arctan2(x, -z), np.arctan2(y, -z)])

    def chi(self, tem, k):
        """
        Return net phase error from spherical aberration and defocus.

        Args:
            k(sequence of length 2 array): The wavenumber.
        """
        alpha = tem.wavelength * k / (2 * np.pi)
        return 2 * np.pi / tem.wavelength * (0.25 * tem.Cs * alpha**4 - 0.5 * self._defocus * alpha**2)

    def getWaveFunction(self, sp, tem):
        """
        Generate a normalized 2D array representing the function space grid.

        Returns:
            numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is
            initialized to the value 1/(Nx*Ny), representing a uniform distribution
            over the function space.
        """

        kwave = np.where(sp.k2 <= tem.k_max**2, np.exp(1j * sp.kvec.dot(self._position)), 0)
        wave = ifft(kwave)
        return wave / np.sum(np.abs(wave)**2)
