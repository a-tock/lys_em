import jax
import jax.numpy as jnp

from . import TEM, TEMParameter, FunctionSpace, CrystalPotential


def calcMultiSliceDiffraction(c, numOfCells, V=60e3, Nx=128, Ny=128, division="Auto", theta_list=[[0, 0]], returnDepth=True):
    """
    Calculate the multi-slice diffraction pattern for a given crystal structure and transmission electron microscope (TEM) settings.

    Args:
        c (CrystalStructure): The crystal structure.
        numOfCells (int): The number of unit cells in the z-direction.
        V (float, optional): Accelerating voltage in volts. Default is 60e3.
        Nx (int, optional): Number of grid points along the x-direction. Default is 128.
        Ny (int, optional): Number of grid points along the y-direction. Default is 128.
        division (str, optional): Division strategy for calculating potential. Default is "Auto".
        theta_list (list, optional): List of theta angles for diffraction. Default is [[0, 0]].
        returnDepth (bool, optional): Whether to return depth information in the result. Default is True.

    Returns:
        DaskWave: The calculated diffraction pattern, optionally including depth information.
    """
    sp = FunctionSpace.fromCrystal(c, Nx, Ny, numOfCells, division=division)

    # prepare potential
    tem = TEM(V)
    pot = CrystalPotential(sp, c)

    # prepare list of thetas
    ncore = len(DaskWave.client.ncores()) if hasattr(DaskWave, "client") else 1
    shape = (int(len(theta_list) / ncore), Nx, Ny, sp.N[2]) if returnDepth else (int(len(theta_list) / ncore), Nx, Ny)
    thetas = [theta_list[shape[0] * i:shape[0] * (i + 1)] for i in range(ncore)]

    # Dstribute all tasks to each worker
    delays = [dask.delayed(__calc_single, traverse=False)(sp, pot, tem, t, returnDepth) for t in thetas]

    res = [da.from_delayed(d, shape, dtype=complex) for d in delays]
    x, y = np.linspace(0, c.a, Nx), np.linspace(0, c.b, Ny)
    z = np.linspace(0, sp.c, sp.N[2])

    if returnDepth:
        # shape is (mcore, thetas, thickness, nx, ny)
        res = DaskWave(da.stack(res).transpose(3, 4, 2, 0, 1).reshape(*sp.N, -1), x, y, z, None)
        return res
    else:
        # shape is (thetas, ncore, nx, ny)
        res = DaskWave(da.stack(res).transpose(2, 3, 0, 1).reshape(Nx, Ny, -1), x, y, None)
        return res


def multislice(sp, pot, tem, params):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    P_k = getPropagationTerm(sp, tem, params) # shape (len(params), Nx, Ny)
    phi = getWaveFunction(sp, tem, params) # shape (len(params), Nx, Ny)
    return jax.vmap(_apply, in_axes=[0, None, 0])(phi, pot, P_k)


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


@jax.jit
def _apply(phi, pots, P_k):
    def body(phi, V_r):
        return jnp.fft.ifft2(jnp.fft.fft2(phi*V_r) * P_k), None
    phi, _ = jax.lax.scan(body, phi, pots)
    return phi
