import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import copy
from .consts import m, e, hbar, kB, h, c


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
        self._cached_arrays = None

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
    
    @property
    def tilt(self):
        return jnp.array([p.beamTilt() for p in self.params])
    
    @property
    def defocus(self):
        return jnp.array([p.defocus for p in self.params])
    
    @property
    def position(self):
        return jnp.array([p.position for p in self.params])

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

        new_obj = TEM(acc, convergence=convergence, divergence=divergence, Cs=Cs, params=params)

        # params がリストではなく、単一の TEMParameter で、かつ内部が配列（バッチ）の場合
        if isinstance(params, TEMParameter):
            if hasattr(params._defocus, "ndim") and params._defocus.ndim >= 1:
                new_obj._cached_arrays = {
                    "tilt": params._tilt,
                    "defocus": params._defocus,
                    "position": params._position
                }
                # multislice内の len(tem.params) 対策として、
                # 正しい長さを持つダミーリストをセットしておく
                batch_size = params._defocus.shape[0]
                new_obj._parameters = [None] * batch_size

        return new_obj

    @property
    def params_array(self):
        if self._cached_arrays is not None:
            return self._cached_arrays

        extracted = [(p._tilt, p.defocus, p.position) for p in self._parameters]
        tilts_raw, defocuses_raw, positions_raw = zip(*extracted)

        tilts = jnp.array(tilts_raw)
        defocuses = jnp.array(defocuses_raw)
        positions = jnp.array(positions_raw)

        self._cached_arrays = {"tilt": tilts, "defocus": defocuses, "position": positions}
        return self._cached_arrays

    @property
    def num_params(self):
        return self.params_array["defocus"].shape[0]

    def tree_flatten(self):
        if self._cached_arrays is None:
            extracted = [(p._tilt, p.defocus, p.position) for p in self._parameters]
            tilts_raw, defocuses_raw, positions_raw = zip(*extracted)

            tilts = jnp.array(tilts_raw)
            defocuses = jnp.array(defocuses_raw)
            positions = jnp.array(positions_raw)
        else:
            tilts = self._cached_arrays["tilt"]
            defocuses = self._cached_arrays["defocus"]
            positions = self._cached_arrays["position"]

        children = (tilts, defocuses, positions, self.__acc, self._Cs, self._convergence, self._divergence)
        aux_data = {}  # 全て数値なので空
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        tilts, defocuses, positions, acc, Cs, convergence, divergence = children

        obj = cls.__new__(cls)
        obj.__acc = acc
        obj._Cs = Cs
        obj._convergence = convergence
        obj._divergence = divergence

        # 配列をそのまま保持（List[Param]は作らない）
        obj._cached_arrays = {
            "tilt": tilts,
            "defocus": defocuses,
            "position": positions
        }
        obj._parameters = None
        return obj

    def asdict(self, index):
        return {"wavelength": self.wavelength, "Cs": self.Cs, "k_max": self.k_max, "tilt": self.tilt[index], "defocus": self.defocus[index], "position": self.position[index]}


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

    def split(self):
        n = max(len(jnp.atleast_1d(self._defocus)),
                len(jnp.atleast_1d(self._tilt[:, 0])),
                len(jnp.atleast_1d(self._tilt[:, 1])),
                len(jnp.atleast_1d(self._position[:, 0])),
                len(jnp.atleast_1d(self._position[:, 1])))

        defocuses = jnp.broadcast_to(jnp.atleast_1d(self._defocus), (n,))
        tilts_0 = jnp.broadcast_to(jnp.atleast_1d(self._tilt[:, 0]), (n,))
        tilts_1 = jnp.broadcast_to(jnp.atleast_1d(self._tilt[:, 1]), (n,))
        positions_0 = jnp.broadcast_to(jnp.atleast_1d(self._position[:, 0]), (n,))
        positions_1 = jnp.broadcast_to(jnp.atleast_1d(self._position[:, 1]), (n,))

        params = []
        for i in range(n):
            new_param = TEMParameter(
                defocus=defocuses[i],
                tilt=[jnp.degrees(tilts_0[i]), jnp.degrees(tilts_1[i])],
                position=[positions_0[i], positions_1[i]],
                tiltType="polar"
            )
            params.append(new_param)

        return params

    def tree_flatten(self):
        children = (self._defocus, self._tilt, self._position)
        aux_data = ()  # 全て数値なので空
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        defocus, tilt, position = children

        obj = cls.__new__(cls)
        obj._defocus = defocus
        obj._tilt = tilt
        obj._position = position
        return obj
