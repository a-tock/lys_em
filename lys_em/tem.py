import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from .consts import m, e, c


@register_pytree_node_class
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

    def tilt(self, type="polar"):
        """
        Return the tilt angles of the electron beam in degrees.

        Args:
            type (str, optional): The type of the tilt angles.
                Possible values are "polar" and "cartesian". Default is "polar".

        Returns:
            jax.numpy.array: The tilt angles with shape (2,) in degrees.
        """
        return jnp.array([p.beamTilt(type=type) for p in self.params])

    @property
    def defocus(self):
        """
        Return the defocus values in angstroms.

        Returns:
            jax.numpy.array: An array of defocus values in angstroms.
        """
        return jnp.array([p.defocus for p in self.params])

    @property
    def position(self):
        """
        Return the position values in angstroms.

        Returns:
            jax.numpy.array: An array of position values in angstroms.
        """
        return jnp.array([p.position for p in self.params])

    def divide_params(self, n):
        """
        Devide parameters into n groups.

        Args:
            n (int): The number of groups.

        RReturns:
        tuple: (tems, indices)
            - tems (list): A list of TEM objects partitioned into `n` groups.
            - indices (list of ndarray): The original indices of `self.params`
                corresponding to each TEM object in `tems`.
        """
        n = int(np.min([n, len(self.params)]))
        tems = []
        chunks = np.array_split(np.arange(len(self._parameters)), n)
        for chunk in chunks:
            tems.append(self.replace(params=[self._parameters[j] for j in chunk]))

        return tems, chunks

    def replace(self, acc=None, convergence=None, divergence=None, Cs=None, params=None):
        """
        Return a copy of the TEM object with specific parameters replaced by the provided values.

        Args:
            acc (float, optional): The acceleration voltage in V.
            convergence (float, optional): The convergence angle of the incident electron beam in radians.
            divergence (float, optional): The divergence angle of the electron beam in radians.
            Cs (float, optional): The spherical aberration coefficient in angstrom.
            params (list, optional): A list of TEMParameter objects.

        Returns:
            TEM: A new TEM object with specific parameters replaced by the provided values.
        """
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

        new_obj = TEM(acc, convergence=convergence, divergence=divergence, Cs=Cs, params=params)

        return new_obj

    def tree_flatten(self):
        """
        Flatten the TEM object into a tuple of children and auxiliary data.

        Returns:
            tuple: A tuple containing the children and auxiliary data.
        """
        extracted = [(p._tilt, p.defocus, p.position) for p in self._parameters]
        tilts_raw, defocuses_raw, positions_raw = zip(*extracted)

        tilts = jnp.array(tilts_raw)
        defocuses = jnp.array(defocuses_raw)
        positions = jnp.array(positions_raw)

        children = (tilts, defocuses, positions, self.__acc, self._Cs, self._convergence, self._divergence)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a tuple of children and auxiliary data into a TEM object.

        Args:
            aux_data (dict): A dictionary of auxiliary data.
            children (tuple): A tuple containing the children.

        Returns:
            TEM: A new TEM object.
        """
        tilts, defocuses, positions, acc, Cs, convergence, divergence = children

        obj = cls.__new__(cls)
        obj.__acc = acc
        obj._Cs = Cs
        obj._convergence = convergence
        obj._divergence = divergence

        obj._parameters = [TEMParameter(tilt=t, defocus=d, position=p) for t, d, p in zip(tilts, defocuses, positions)]
        return obj

    def asdict(self, n_devices):
        """
        Convert the TEM object to a dictionary of arrays.

        Args:
            n_devices (int): The number of devices.

        Returns:
            tuple (dict, jax.numpy.array): 
                A tuple containing the dictionary of arrays and a boolean array.
        """
        num = len(self.params)

        q, r = num // n_devices, num % n_devices
        max_block = q + (1 if r > 0 else 0)
        total = n_devices * max_block

        res = {"wavelength": jnp.repeat(jnp.array(self.wavelength), num),
               "Cs": jnp.repeat(jnp.array(self.Cs), num),
               "k_max": jnp.repeat(jnp.array(self.k_max), num),
                "tilt": self.tilt(),
                "defocus": self.defocus,
                "position": self.position}

        idx = jnp.arange(total)
        safe_idx = jnp.where(idx < num, idx, 0)
        padded = jnp.where(idx < num, False, True)

        params = {key: jnp.take(val, safe_idx, axis=0) for key, val in res.items()}

        return params, padded


@register_pytree_node_class
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
        self._defocus = jnp.array(defocus, dtype=jnp.float32)
        if tilt is None:
            tilt = [0, 0]
        if position is None:
            position = [0, 0]
        tilt = jnp.array(tilt, dtype=jnp.float32)
        if tiltType == "polar":
            self._tilt = jnp.radians(tilt)
        elif tiltType == "cartesian":
            self._tilt = self._cartesianToPolar(tilt)
        self._position = jnp.array(position, dtype=jnp.float32)

    @property
    def beamDirection(self):
        """
        Return the direction vector of the electron beam.

        Returns:
            array: The direction vector with shape (3,).
        """
        return jnp.array([jnp.sin(self._tilt[0]) * jnp.cos(self._tilt[1]), jnp.sin(self._tilt[0]) * jnp.sin(self._tilt[1]), -jnp.cos(self._tilt[0])])

    @property
    def position(self):
        """
        Return the position of the electron beam.

        Returns:
            array: The position of the electron beam with shape (2,).
        """
        return self._position

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
            return jnp.degrees(self._tilt)
        elif type == "cartesian":
            x, y, z = self.beamDirection
            return jnp.degrees(jnp.array([jnp.arctan2(x, -z), jnp.arctan2(y, -z)]))

    def replace(self, defocus=None, tilt=None, position=None, tiltType="polar"):
        """
        Return a copy of the TEMParameter object with specific parameters replaced by the provided values.

        Args:
            defocus (float, optional): The defocus value in angstroms. Default is None.
            tilt (list or array-like, optional): The tilt angles of the electron beam in degrees.
                It has the format tilt=[theta, phi], where theta is the angle with respect to the optical axis,
                and phi is the rotation angle in a plane perpendicular to the optical axis. Default is None.
            position (list or array-like, optional): The position of the electron beam. Default is None.
            tiltType (str, optional): The type of the tilt angles. Possible values are "polar" and "cartesian". Default is "polar".

        Returns:
            TEMParameter: A new TEMParameter object with the replaced parameters.
        """
        if defocus is None:
            defocus = self._defocus
        if tilt is None:
            tilt = jnp.degrees(self._tilt)
        if position is None:
            position = self._position
        return TEMParameter(defocus=defocus, tilt=tilt, position=position, tiltType=tiltType)

    def _cartesianToPolar(self, tilt):
        theta_x = jnp.radians(tilt[0])
        theta_y = jnp.radians(tilt[1])
        theta = jnp.arccos(1 / jnp.sqrt(1 + jnp.tan(theta_x)**2 + jnp.tan(theta_y)**2))
        phi = jnp.arctan2(jnp.tan(theta_y), jnp.tan(theta_x))
        return jnp.array([theta, phi])

    def tree_flatten(self):
        """
        Flatten the TEMParameter object into a tuple of children and auxiliary data.

        Returns:
            tuple: A tuple containing the children and auxiliary data.
        """
        children = (self._defocus, self._tilt, self._position)
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten a tuple of children and auxiliary data into a TEMParameter object.

        Args:
            aux_data (dict): A dictionary of auxiliary data.
            children (tuple): A tuple containing the children.

        Returns:
            TEMParameter: A new TEMParameter object.
        """
        defocus, tilt, position = children

        obj = cls.__new__(cls)
        obj._defocus = defocus
        obj._tilt = tilt
        obj._position = position
        return obj
