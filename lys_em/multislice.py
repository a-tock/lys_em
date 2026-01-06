import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
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

    # if len(tem.params) == 1:
    if tem.num_params == 1:
        ffts = (jnp.fft.fft2, jnp.fft.ifft2)
        tem_aligned = tem
    else:
        ffts = get_sharded_fft_operators(use_probemode=not isinstance(probe, str) and probe.ndim > 2)
        tem_aligned = tem.align_to_devices(len(jax.devices()))

    fft2, ifft2 = ffts

    sp = pot.space
    t_r = pot.getTransmissionFunction(tem_aligned)  # shape (ncells * division, Nx, Ny)
    P_k = getPropagationTerm(sp, tem_aligned)  # shape (len(params), Nx, Ny)
    phi = getWaveFunction(sp, tem_aligned, probe=probe)  # shape (len(params), modes, Nx, Ny)
    if phi.ndim == 4 and P_k.ndim == 3:
        P_k = jnp.expand_dims(P_k, axis=1)

    return_depth_jax = jnp.array(returnDepth, dtype=jnp.bool_)

    @jax.jit
    def core_func(phi, t_r, P_k):
        def body(phi_prev, V_r):
            phi_next = ifft2(fft2(phi_prev * V_r) * P_k)
            return phi_next, jnp.where(return_depth_jax, phi_next, jnp.zeros_like(phi_next))

        scan_body = checkpoint(body) if use_checkpoint else body

        phi_final, phis_all = jax.lax.scan(scan_body, phi, t_r)
        return phi_final, phis_all

    phi_final, phis_all = core_func(phi, t_r, P_k)

    if probe == "TEM":
        H = getAberrationFunction(sp, tem_aligned)  # shape (len(params), Nx, Ny)
        if H.ndim < phi.ndim:
            H = jnp.expand_dims(H, axis=1)
        phi = ifft2(fft2(phi) * H * sp.mask)

        if returnDepth:
            phis_all = jax.vmap(lambda p: ifft2(fft2(p) * H * sp.mask))(phis_all)

    # if returnDepth:
    #     res = jnp.swapaxes(phis_all[:, 0:len(tem.params)], 0, 1)
    # else:
    #     res = phi_final[0:len(tem.params)]
    if returnDepth:
        res = jnp.swapaxes(phis_all[:, 0:tem.num_params], 0, 1)
    else:
        res = phi_final[0:tem.num_params]

    if tem.num_params == 1:
        # if len(tem.params) == 1:
        res = res[0]

    return res


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

    kvec, mask, dz = sp.kvec, sp.mask, sp.dz

    k2_term = -1j * tem.wavelength * sp.k**2 / (4 * jnp.pi)

    # @jax.vmap
    def _propagationTerm(theta):
        tilt = 1j * jnp.sum(kvec * jnp.tan(theta), axis=-1) + k2_term
        return jnp.exp(dz * tilt) * mask

    # @jax.vmap
    def _cartesianBeamTilt(t):
        return jnp.stack([t[0] * jnp.cos(t[1]), t[0] * jnp.sin(t[1])])

    def _single_tilt_process(t):
        theta = _cartesianBeamTilt(t)
        return _propagationTerm(theta)

    # tilt = tem.params_array["tilt"]
    # return jax.vmap(_single_tilt_process, in_axes=0)(jnp.deg2rad(tilt))
    # return _propagationTerm(_cartesianBeamTilt(jnp.deg2rad(tilt)))
    return jax.vmap(_single_tilt_process, in_axes=0)(jnp.deg2rad(jnp.array([p.beamTilt() for p in tem.params])))


def getWaveFunction(sp, tem, probe="TEM", index=None):
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

    @jax.vmap
    def _shiftProbe(probe_func, pos):
        wave = jnp.fft.ifft2(probe_func * jnp.exp(1j * kvec.dot(pos)))
        return wave / jnp.sqrt(jnp.sum(jnp.abs(wave)**2))

    @jax.vmap
    def _probe(chi_n):
        return _mask(jnp.exp(-1j * chi_n))
        # return jnp.where(jnp.linalg.norm(kvec, axis=2) <= k_max, jnp.exp(-1j * chi_n), 0j)  # Nx, Ny

    def _mask(arr):
        return jnp.where(k_norm <= k_max, arr, 0j)

    # params = tem.params
    # if type(index) == int:
    #     params = [params[index]]
    # elif type(index) in (list, tuple, jnp.ndarray):
    #     params = [params[i] for i in index]

    # defocus = tem.params_array["defocus"]
    # positions = tem.params_array["position"]
    defocus, positions = map(jnp.array, zip(*[(p.defocus, p.position) for p in tem.params]))

    chi = getChi(sp, tem)

    if type(probe) is str and probe in ["TEM", "STEM"]:
        # probe = _probe(jax.vmap(getChi, in_axes=(None, None, 0))(sp, tem, defocus))
        # probe = _probe(chi(defocus))
        probe = jax.vmap(_mask)(jnp.exp(-1j * chi(defocus)))
    else:
        # probe = jax.vmap(_mask)(jax.vmap(lambda df: jnp.fft.fft2(probe))(defocus))
        probe = jax.vmap(lambda df: jnp.fft.fft2(probe))(defocus)

    return _shiftProbe(probe, positions)


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
    # defocus = tem.params_array["defocus"]

    # return jnp.exp(1j * jax.vmap(getChi, in_axes=(None, None, 0))(sp, tem, defocus))
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

    @jax.vmap
    def _chi(df):  # df : A
        return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - df * ((l * k)**2) / 2)

    return _chi

# def getChi(sp, tem, defocus):
#     """
#     Return the chi function of the multislice simulation.

#     The chi function is calculated from the wave number k and the spherical
#     aberration coefficient Cs. The wave number k is calculated from the function
#     space object sp and the wavelength lambda of the TEM object tem. The spherical
#     aberration coefficient Cs is given in millimeters.

#     Args:
#         sp (FunctionSpace): The function space object.
#         tem (TEM) : The TEM object.

#     Returns:
#         callable: A function that takes a defocus value df in Angstrom and returns the chi function at that defocus value.
#     """
#     k = sp.k / 2 / jnp.pi  # rad/A
#     l = tem.wavelength  # A
#     Cs = tem.Cs  # A

#     # @jax.vmap
#     # def _chi(df):  # df : A
#     return 2 * jnp.pi / l * (Cs * ((l * k)**4) / 4 - defocus * ((l * k)**2) / 2)

#     # return _chi


def get_sharded_fft_operators(use_probemode=False):
    devices = len(jax.devices())
    mesh = jax.make_mesh((devices, ), ('i'))
    # spec = P("i")
    spec = P("i", None, None, None) if use_probemode else P("i", None, None)

    fft2 = shard_map(lambda u: jnp.fft.fft2(u, axes=(-2, -1)), mesh=mesh, in_specs=spec, out_specs=spec)
    ifft2 = shard_map(lambda u: jnp.fft.ifft2(u, axes=(-2, -1)), mesh=mesh, in_specs=spec, out_specs=spec)

    return fft2, ifft2
