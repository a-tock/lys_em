import numpy as np
import copy
from .consts import m, e, hbar, kB, h, c


class TEM(object):
    """
    TEM parameters used for simulations.

    Args:
        acc (float): The acceleration voltage in V.
        convergence (float, optional): The convergence angle of the incident electron beam in radians. Default is 0.
        divergence (float, optional): The divergence angle of the electron beam in radians. Default is np.inf.
        Cs (float, optional): The spherical aberration coefficient in angstrom. Default is 0.
        params (list, optional): A list of TEMParameter objects. Default is None.
    """

    def __init__(self, acc, convergence=0, divergence=np.inf, Cs=0, params=None):
        self.__acc = acc
        self._convergence = convergence
        self._divergence = divergence
        self._Cs = Cs
        if params is None:
            params = TEMParameter()
        if isinstance(params, TEMParameter):
            params = [params]
        self._parameters = params

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

    @property
    def params(self):
        '''
        Return a list of TEMParameter objects.

        Returns:
            list: List of TEMParameter objects
        '''
        return self._parameters

    def devide_params(self, n):
        n = int(np.min([n, len(self.params)]))
        tems = []
        chunks = np.array_split(np.arange(len(self._parameters)), n)
        for chunk in chunks:
            tems.append(self.replace(params=[self._parameters[j] for j in chunk]))

        return tems, chunks

    def replace(self, acc=None, convergence=None, divergence=None, Cs=None, params=None):
        if acc is None:
            acc = self.__acc
        if convergence is None:
            convergence = self._convergence
        if divergence is None:
            divergence = self._divergence
        if Cs is None:
            Cs = self._Cs
        if params is None:
            params = self._parameters
        return TEM(self.__acc, convergence=self._convergence, divergence=self._divergence, Cs=self._Cs, params=params)


class TEMParameter:
    """
    TEMParameter has properties that are parameters that can usually be changed during TEM measurement.

    Parameters:
        defocus (float, optional): The defocus value in angstroms. Default is 0.
        tilt (list or array-like, optional): The tilt angles of the electron beam in radians.
            It has the format tilt=[theta, phi], where theta is the angle with respect to the optical axis,
            and phi is the rotation angle in a plane perpendicular to the optical axis. Default is [0, 0].
        position (list or array-like, optional): The position of the electron beam. Default is [0, 0].
    """

    def __init__(self, defocus=0, tilt=None, position=None, tiltType="polar"):
        self._defocus = defocus
        if tilt is None:
            tilt = [0, 0]
        if position is None:
            position = [0, 0]
        if tiltType == "polar":
            self._tilt = np.radians(tilt)
        elif tiltType == "cartesian":
            self._tilt = self._cartesianToPolar(tilt)
        self._position = position

    @property
    def beamDirection(self):
        """
        Return the direction vector of the electron beam.

        Returns:
            array: The direction vector with shape (3,).
        """
        return np.array([np.sin(self._tilt[0]) * np.cos(self._tilt[1]), np.sin(self._tilt[0]) * np.sin(self._tilt[1]), -np.cos(self._tilt[0])])

    @property
    def position(self):
        """
        Return the position of the electron beam.

        Returns:
            array: The position of the electron beam with shape (2,).
        """
        return np.array(self._position)

    @property
    def defocus(self):
        """
        Return the defocus value in angstroms.

        Returns:
            float: The defocus value in angstroms.
        """
        return self._defocus

    def beamTilt(self, type="polar"):
        """
        Return the tilt angles of the electron beam in degrees.

        Parameters:
            type (str, optional): The type of the tilt angles. Possible values are "polar" and "cartesian". Default is "polar".

        Returns:
            array: The tilt angles with shape (2,).
        """
        if type == "polar":
            return np.degrees(self._tilt)
        elif type == "cartesian":
            x, y, z = self.beamDirection
            return np.degrees([np.arctan2(x, -z), np.arctan2(y, -z)])

    def replace(self, defocus=None, tilt=None, position=None, tiltType="polar"):
        if defocus is None:
            defocus = self._defocus
        if tilt is None:
            tilt = np.degrees(self._tilt)
        if position is None:
            position = self._position
        return TEMParameter(defocus=defocus, tilt=tilt, position=position, tiltType=tiltType)

    def _cartesianToPolar(self, tilt):
        theta_x = np.radians(tilt[0])
        theta_y = np.radians(tilt[1])
        theta = np.arccos(1 / np.sqrt(1 + np.tan(theta_x)**2 + np.tan(theta_y)**2))
        phi = np.arctan2(np.tan(theta_y), np.tan(theta_x))
        return np.array([theta, phi])
