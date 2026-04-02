import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.ad_checkpoint import checkpoint


def multislice(pot, tem, probe="TEM", returnDepth=False, use_checkpoint=False):
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

    # Calculation for single (i th) parameter
    t_r = pot.getTransmissionFunction(tem)
    def _loop(_, i):
        return _, run_single_param(pot.space, tem, t_r, probe, i, returnDepth, use_checkpoint)
    
    # Parallelization: "calc" function perform multislice calculation for list of parameters in tem. 
    indices_all = get_worker_indices(len(tem.params), len(jax.devices()))
    calc = jax.shard_map(jax.jit(lambda indices: jax.lax.scan(_loop, None, indices)[1]), 
                        mesh=jax.sharding.Mesh(jax.devices(), ('i',)), in_specs=P('i'), out_specs=P('i',None,None,None,None) if returnDepth else P('i',None,None,None))
    
    # Remove padding and unnecessary axis
    phi = calc(indices_all)[:tem.num_params].squeeze()
    return phi


def run_single_param(sp, tem, t_r, probe, index, returnDepth=False, use_checkpoint=False):
    """
    Peform multislice calculation for parameter set specified by index.

    Args:
        sp(FunctionSpace): FunctionSpace for calculation.
        tem(TEM): TEM setting including parameter set.
        t_r(Nx*Ny*Nz array): Transmission function for each slice
        probe: Probe function. Can be list of probe function for partial coherence calculation.
        index(int): It specifies the parameter set in tem.    
    """
    phi = getWaveFunction(sp, tem, tem.defocus[index], tem.position[index], probe=probe) # shape (nprobe, Nx, Ny)
    P_k = getPropagationTerm(sp, tem, tem.tilt[index]) # shape (Nx, Ny)

    # Apply multislice calculation
    def _body(phi_prev, V_r):
        phi_next = jnp.fft.ifft2(jnp.fft.fft2(phi_prev * V_r) * P_k)
        return phi_next, phi_next # Storing each wave function can be automatically avoided by optimization of jax if return depth is False
    if use_checkpoint:
        _body = checkpoint(_body)

    phi, phis_all = jax.lax.scan(_body, phi, t_r)
    if returnDepth:
        phi = phis_all

    # Apply abberration for TEM
    if probe == "TEM":
        H = getAberrationFunction(sp, tem, tem.defocus[index])  # shape (Nx, Ny)
        phi = jnp.fft.ifft2(jnp.fft.fft2(phi, axes=(-2,-1)) * H * sp.mask, axes=(-2,-1))

    return phi


def get_worker_indices(num_tasks, num_workers):
    q, r = num_tasks // num_workers, num_tasks % num_workers
    max_block = q + (1 if r > 0 else 0)
    total = num_workers * max_block

    arr = jnp.full((total,), 0, dtype=jnp.int32)
    arr = arr.at[:num_tasks].set(jnp.arange(num_tasks, dtype=jnp.int32))
    return arr


def getPropagationTerm(sp, tem, tilt):
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

    kvec, mask, dz = sp.kvec, sp.mask, sp.dz

    k2_term = -1j * tem.wavelength * sp.k**2 / (4 * jnp.pi)

    def _propagationTerm(theta):
        tilt = 1j * jnp.sum(kvec * jnp.tan(theta), axis=-1) + k2_term
        return jnp.exp(dz * tilt) * mask

    def _cartesianBeamTilt(t):
        return jnp.stack([t[0] * jnp.cos(t[1]), t[0] * jnp.sin(t[1])])

    theta = _cartesianBeamTilt(jnp.deg2rad(tilt))
    return _propagationTerm(theta)


def getWaveFunction(sp, tem, defocus, position, probe="TEM"):
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

    kvec, k_max = sp.kvec, tem.k_max
    k_norm = jnp.linalg.norm(kvec, axis=2)

    def _shiftProbe(probe_func, pos):
        wave = jnp.fft.ifft2(probe_func * jnp.exp(1j * kvec.dot(pos)))
        return wave / jnp.sqrt(jnp.sum(jnp.abs(wave)**2))

    def _mask(arr):
        return jnp.where(k_norm <= k_max, arr, 0j)

    chi = getChi(sp, tem)

    if type(probe) is str and probe in ["TEM", "STEM"]:
        probe = _mask(jnp.exp(-1j * chi(defocus)))
    else:
        probe = _mask(jnp.fft.fft2(probe)*jnp.exp(-1j * chi(defocus)))
        # probe = jax.vmap(lambda df: jnp.fft.fft2(probe))(defocus)

    phi = _shiftProbe(probe, position)
    if phi.ndim == 2:
        phi = jnp.expand_dims(phi, axis=0)    # Virtually consider multiple probes
    return phi


def getAberrationFunction(sp, tem, defocus):
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
    k = sp.k / 2 / jnp.pi  # rad/A
    l = tem.wavelength  # A
    Cs = tem.Cs  # A

    def _chi(df):  # df : A
        return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - df * ((l * k)**2) / 2)

    return _chi

