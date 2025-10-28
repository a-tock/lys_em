import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map


def multislice(pot, tem, probe="TEM"):
    """
    Calculate the multislice simulation.

    Args:
        pot (CrystalPotential): The CrystalPotential object.
        tem (TEM) : The TEM object.
        probe (str or array) : The probe type or the probe wave function. Probe type can be "TEM" or "STEM".
            The probe wave function should be a 2D array of shape (Nx, Ny). Default is "TEM".

    Returns:
        phi (numpy.ndarray) : The multislice simulation result.
    """
    mesh = jax.make_mesh((len(jax.devices()), ), ('i'))
    fft2 = shard_map(lambda u: jnp.fft.fft2(u, axes=(-2, -1)), mesh=mesh, in_specs=P("i", None, None), out_specs=P("i", None, None))
    ifft2 = shard_map(lambda u: jnp.fft.ifft2(u, axes=(-2, -1)), mesh=mesh, in_specs=P("i", None, None), out_specs=P("i", None, None))

    sp = pot.space
    t_r = pot.getTransmissionFunction(tem)
    P_k = getPropagationTerm(sp, tem)  # shape (len(params), Nx, Ny)
    phi = getWaveFunction(sp, tem, probe=probe)  # shape (len(params), Nx, Ny)

    @jax.jit
    def _apply(phi, pots, P_k):
        def body(phi, V_r):
            return ifft2(fft2(phi * V_r) * P_k), None
        phi, _ = jax.lax.scan(body, phi, pots)
        return phi

    phi = _apply(phi, t_r, P_k)

    if probe == "TEM":
        H = getAberrationFunction(sp, tem)  # shape (len(params), Nx, Ny)
        phi = ifft2(fft2(phi) * H * sp.mask)

    return jnp.squeeze(phi)


def getPropagationTerm(sp, tem):
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
        sp (FunctionSpace): The function space object.
        tem (TEM): The TEM object.

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is
        the propagation term at the respective grid point in reciprocal space.
    """
    @jax.vmap
    @jax.jit
    def _propagationTerm(theta):
        tilt = 1j * (sp.kvec.dot(jnp.tan(theta))) - 1j * tem.wavelength * sp.k**2 / 4 / jnp.pi
        return jnp.exp(sp.dz * tilt) * sp.mask

    thetas = jnp.array([jnp.radians(p.beamTilt(type="cartesian")) for p in tem.params])
    return _propagationTerm(thetas)


def getWaveFunction(sp, tem, probe="TEM"):
    """
    Calculate the wave function of the multislice simulation.

    Args:
        sp (FunctionSpace): The function space object.
        tem (TEM) : The TEM object.
        probe (str or array) : The probe type or the probe wave function. Probe type can be "TEM" or "STEM".
            The probe wave function should be a 2D array of shape (Nx, Ny). Default is "TEM".

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is the wave function at the respective grid point in real space.
    """
    @jax.vmap
    @jax.jit
    def _shiftProbe(probe_func, pos):
        wave = jnp.fft.ifft2(probe_func * jnp.exp(1j * sp.kvec.dot(pos)))
        return wave / jnp.sum(jnp.abs(wave)**2)

    @jax.vmap
    @jax.jit
    def _probe(chi_n):
        return jnp.where(jnp.linalg.norm(sp.kvec, axis=2) <= tem.k_max, jnp.exp(-1j * chi_n), 0)  # Nx, Ny

    defocus = jnp.array([p.defocus for p in tem.params])
    chi = getChi(sp, tem)
    if probe in ["TEM", "STEM"]:
        probe = _probe(chi(defocus))  # len(params), Nx, Ny
    else:
        probe = jax.vmap(lambda df: probe)(defocus)

    pos = jnp.array([p.position for p in tem.params])
    return _shiftProbe(probe, pos)


def getAberrationFunction(sp, tem):
    """
    Return the aberration function of the multislice simulation.

    The aberration function is calculated from the chi function and the defocus values.
    The chi function is calculated from the wave number k and the spherical
    aberration coefficient Cs. The defocus values are given in Angstrom.

    Args:
        sp (FunctionSpace): The function space object.
        tem (TEM) : The TEM object.

    Returns:
        numpy.ndarray: A 2D array of shape (len(params), Nx, Ny) where each element is the aberration function at the respective grid point in real space.
    """
    chi = getChi(sp, tem)
    defocus = jnp.array([p.defocus for p in tem.params])
    return jnp.exp(1j * chi(defocus))


def getChi(sp, tem):
    """
    Return the chi function of the multislice simulation.

    The chi function is calculated from the wave number k and the spherical
    aberration coefficient Cs. The wave number k is calculated from the function
    space object sp and the wavelength lambda of the TEM object tem. The spherical
    aberration coefficient Cs is given in millimeters.

    Args:
        sp (FunctionSpace): The function space object.
        tem (TEM) : The TEM object.

    Returns:
        callable: A function that takes a defocus value df in Angstrom and returns the chi function at that defocus value.
    """
    k = sp.k
    l = tem.wavelength
    Cs = tem.Cs

    @jax.vmap
    @jax.jit
    def _chi(df):
        return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - df * ((l * k)**2) / 2)

    return _chi


def diffraction(data):
    return jnp.abs(jnp.fft.fft2(data))**2
