import jax
import jax.numpy as jnp

from . import TEMParameter


def multislice(sp, pot, tem, params, probe=None):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    P_k = getPropagationTerm(sp, tem, params) # shape (len(params), Nx, Ny)
    phi = getWaveFunction(sp, tem, params, probe=probe) # shape (len(params), Nx, Ny)
    phi = jax.vmap(_apply, in_axes=[0, None, 0])(phi, pot, P_k)
    H = getAberrationFunction(sp, tem, params) # shape (len(params), Nx, Ny)
    return jnp.fft.ifft2(jnp.fft.fft2(phi) * H * sp.mask)


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
    @jax.jit
    def _propagationTerm(k, mask, lamb, theta, dz):
        tilt = 1j * (k.dot(jnp.tan(theta))) - 1j * lamb * jnp.linalg.norm(k, axis=2)**2 / 4 / jnp.pi
        return jnp.exp(dz * tilt) * mask
    
    if isinstance(params, TEMParameter):
        return getPropagationTerm(sp, tem, [params])[0]

    thetas = jnp.array([jnp.radians(p.beamTilt(type="cartesian")) for p in params])
    f = jax.vmap(_propagationTerm, in_axes=[None, None, None, 0, None])
    return f(sp.kvec, sp.mask, tem.wavelength, thetas, sp.dz)


def getWaveFunction(sp, tem, params, probe=None):
    """
    Generate a normalized 2D array representing the function space grid.

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is
        initialized to the value 1/(Nx*Ny), representing a uniform distribution
        over the function space.
    """
    @jax.jit
    def _shiftProbe(probe, pos):
        wave = jnp.fft.ifft2(probe * jnp.exp(1j * sp.kvec.dot(pos)))
        return wave / jnp.sum(jnp.abs(wave)**2)
    
    if isinstance(params, TEMParameter):
        return getWaveFunction(sp, tem, [params])[0]
    
    if probe is None:
        probe = jnp.where(jnp.linalg.norm(sp.kvec, axis=2) <= tem.k_max, 1, 0)

    pos = jnp.array([p.position for p in params])
    f = jax.vmap(_shiftProbe, in_axes=[None, 0])
    return f(probe, pos)


def getAberrationFunction(sp, tem, params):
    k = jnp.linalg.norm(sp.kvec, axis=2)
    l = tem.wavelength
    Cs = tem.Cs

    @jax.jit
    def _chi(df):
        return 2*jnp.pi/l * (Cs*((l*k)**4)/4 - df*((l*k)**2)/2)

    defocus = jnp.array([p.defocus for p in params])
    return jnp.exp(1j * jax.vmap(_chi)(defocus))


@jax.jit
def _apply(phi, pots, P_k):
    def body(phi, V_r):
        return jnp.fft.ifft2(jnp.fft.fft2(phi*V_r) * P_k), None
    phi, _ = jax.lax.scan(body, phi, pots)
    return phi
