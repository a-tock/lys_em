import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

checkpoint = jax.checkpoint if hasattr(jax, 'checkpoint') else jax.ad_checkpoint.checkpoint
shard_map = jax.shard_map if hasattr(jax, 'shard_map') else jax.experimental.shard_map.shard_map


def multislice(pot, tem, probe="TEM", postprocess="square", sum=False, use_checkpoint=False):
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
    sp = pot.space
    postprocess = init_postprocess(postprocess)

    params = tem.asdict(len(jax.devices()))
    calc_phi = single_func(pot, tem, probe, use_checkpoint=use_checkpoint)
    
    init = jnp.zeros((1 if isinstance(probe, str) else len(probe),sp.N[0],sp.N[1]), dtype=jnp.float32)
    @jax.jit
    @checkpoint
    def local_dev(x):
        init_varying = jax.lax.pcast(init, "i", to="varying")
        def _post(carry, param):
            phi = calc_phi(param)
            post = postprocess(phi)
            return carry + post, None if sum else post
        
        summed, all = jax.lax.scan(_post, init_varying, x)
        if sum:
            return jax.lax.psum(summed, "i")
        else:
            return all

    # Parallelization: "calc" function perform multislice calculation for list of parameters in tem. 
    calc = shard_map(local_dev,
            mesh=jax.sharding.Mesh(jax.devices(), ('i',)), in_specs=P('i'), out_specs= P(None,None,None) if sum else P('i',None,None,None))

    # Move to main, remove padding and unnecessary axis
    phi = jax.device_get(calc(params))    # (nparams, nprobe, Nx, Ny)
    return phi


def init_postprocess(post):
    if post is None:
        return lambda x: x
    elif post == "square":
        return lambda x: abs(x)**2


def single_func(pot, tem, probe, use_checkpoint=False):
    sp = pot.space.asdict()
    t_r = pot.getTransmissionFunction(tem)
    probes = probe if not isinstance(probe, str) else jnp.array([jnp.fft.ifft2(jnp.ones(pot.space.N[:2]))])

    @jax.jit
    def run_single_param(phi, sp, tem, t_r):
        P_k = getPropagationTerm(sp, tem) # shape (Nx, Ny)
        step = lambda i, phi: jnp.fft.ifft2(jnp.fft.fft2(phi * t_r[i]) * P_k)
        step = checkpoint(step)
        return jax.lax.fori_loop(0, len(t_r), step, phi)
    run_single_param = checkpoint(run_single_param) if use_checkpoint else run_single_param

    @jax.jit
    @checkpoint
    def _loop(tem_i):
        phi = getWaveFunction(sp, tem_i, probes) # shape (nprobe, Nx, Ny)
        phi = run_single_param(phi,sp, tem_i, t_r)

        # Apply abberration for TEM
        if probe == "TEM":
            H = getAberrationFunction(sp, tem_i)  # shape (Nx, Ny)
            phi = jnp.fft.ifft2(jnp.fft.fft2(phi, axes=(-2,-1)) * H * sp["mask"], axes=(-2,-1))
        return phi

    return _loop


@jax.jit
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

    kvec, mask, dz , k= sp["kvec"], sp["mask"], sp["dz"], sp["k"]
    wavelength = tem["wavelength"]
    tilt = tem["tilt"]

    k2_term = -1j * wavelength * k**2 / (4 * jnp.pi)

    def _propagationTerm(theta):
        tilt = 1j * jnp.sum(kvec * jnp.tan(theta), axis=-1) + k2_term
        return jnp.exp(dz * tilt) * mask

    def _cartesianBeamTilt(t):
        return jnp.stack([t[0] * jnp.cos(t[1]), t[0] * jnp.sin(t[1])])

    theta = _cartesianBeamTilt(jnp.deg2rad(tilt))
    return _propagationTerm(theta)


@jax.jit
def getWaveFunction(sp, tem, probe="TEM"):
    """
    Calculate the wave function of the multislice simulation.

    Args:
        sp (FunctionSpace): The function space object.
        tem (TEM) : The TEM object.
        probe (str or array) : The probe type or the probe wave function.Probe type can be "TEM" or "STEM".
            The probe wave function should be a 2D array of shape (Nx, Ny). Default is "TEM".

    Returns:
        numpy.ndarray: A 3D array of shape (len(tem.params), Nx, Ny) where each element is the wave function at the respective grid point in real space.
    """

    kvec, k_max = sp["kvec"], tem["k_max"]
    defocus, position = tem["defocus"], tem["position"]
    k_norm = jnp.linalg.norm(kvec, axis=2)

    def _shiftProbe(probe_func, pos):
        wave = jnp.fft.ifft2(probe_func * jnp.exp(1j * kvec.dot(pos)))
        return wave / jnp.sqrt(jnp.sum(jnp.abs(wave)**2))

    def _mask(arr):
        return jnp.where(k_norm <= k_max, arr, 0j)

    chi = getChi(sp, tem)
    probes = _mask(jnp.fft.fft2(probe)*jnp.exp(-1j * chi(defocus)))
    phi = _shiftProbe(probes, position)
    return phi


@jax.jit
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
    defocus = tem["defocus"]
    chi = getChi(sp, tem)
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
    k = sp["k"] / 2 / jnp.pi  # rad/A
    l = tem["wavelength"]  # A
    Cs = tem["Cs"]  # A

    @jax.jit
    def _chi(df):  # df : A
        return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - df * ((l * k)**2) / 2)

    return _chi