import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map

from . import TEMParameter


def multislice(sp, pot, tem, params, probe="TEM"):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    if isinstance(params, TEMParameter):
        return multislice(sp, pot, tem, [params], probe=probe)[0]

    mesh = jax.make_mesh((len(jax.devices()), ), ('i'))
    fft2 = shard_map(lambda u: jnp.fft.fft2(u, axes=(-2,-1)), mesh=mesh, in_specs=P("i",None,None), out_specs=P("i",None,None))
    ifft2 = shard_map(lambda u: jnp.fft.ifft2(u, axes=(-2,-1)), mesh=mesh, in_specs=P("i",None,None), out_specs=P("i",None,None))

    P_k = getPropagationTerm(sp, tem, params)  # shape (len(params), Nx, Ny)
    phi = getWaveFunction(sp, tem, params, probe=probe)  # shape (len(params), Nx, Ny)

    @jax.jit
    def _apply(phi, pots, P_k):
        def body(phi, V_r):
            return ifft2(fft2(phi * V_r) * P_k), None
        phi, _ = jax.lax.scan(body, phi, pots)
        return phi
    
    phi = _apply(phi, pot, P_k)

    if probe=="TEM":
        H = getAberrationFunction(sp, tem, params)  # shape (len(params), Nx, Ny)
        phi = ifft2(fft2(phi) * H * sp.mask)

    return phi


def getPropagationTerm(sp, tem, params):
    """
    Return the propagation term of the wave transfer function.

    The propagation term is calculated from the wave number k and the
    propagation distance dz. The wave number k is calculated from the
    crystal structure and the wavelength lamb. The propagation distance dz
    is given in Angstrom.

    The wave number k is represented as a 2D array of shape (Nx, Ny) where
    each element is the wave number at the respective grid point in
    reciprocal space. The unit of k is rad/A.

    The propagation term is calculated as exp(1j * k * dz).

    Args:
        lamb (float): The wavelength of the electron beam in Angstrom.
        theta_x (float, optional): The tilt angle of the incident beam along the x-axis in degree. Defaults to 0.
        theta_y (float, optional): The tilt angle of the incident beam along the y-axis in degree. Defaults to 0.

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is
        the propagation term at the respective grid point in reciprocal
        space.
    """

    @jax.vmap
    @jax.jit
    def _propagationTerm(theta):
        tilt = 1j * (sp.kvec.dot(jnp.tan(theta))) - 1j * tem.wavelength * sp.k**2 / 4 / jnp.pi
        return jnp.exp(sp.dz * tilt) * sp.mask

    thetas = jnp.array([jnp.radians(p.beamTilt(type="cartesian")) for p in params])
    return _propagationTerm(thetas)


def getWaveFunction(sp, tem, params, probe="TEM"):
    """
    Generate a normalized 2D array representing the function space grid.

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is
        initialized to the value 1/(Nx*Ny), representing a uniform distribution
        over the function space.
    """

    @jax.vmap
    @jax.jit
    def _shiftProbe(probe_func, pos):
        wave = jnp.fft.ifft2(probe_func * jnp.exp(1j * sp.kvec.dot(pos)))
        return wave / jnp.sum(jnp.abs(wave)**2)

    @jax.vmap
    @jax.jit
    def _probe(chi_n):
        return jnp.where(jnp.linalg.norm(sp.kvec, axis=2) <= tem.k_max, jnp.exp(-1j*chi_n), 0) # Nx, Ny

    defocus = jnp.array([p.defocus for p in params])
    chi = getChi(sp, tem)
    if probe in ["TEM","STEM"]:
        probe = _probe(chi(defocus)) # len(params), Nx, Ny
    else:
        probe = jax.vmap(lambda df: probe)(defocus)

    pos = jnp.array([p.position for p in params])
    return _shiftProbe(probe, pos)


def getAberrationFunction(sp, tem, params):
    chi = getChi(sp, tem)
    defocus = jnp.array([p.defocus for p in params])
    return jnp.exp(1j * chi(defocus))


def getChi(sp, tem):
    k = sp.k
    l = tem.wavelength
    Cs = tem.Cs

    @jax.vmap
    @jax.jit
    def _chi(df):
        return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - df * ((l * k)**2) / 2)

    return _chi