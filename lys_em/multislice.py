import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import NamedSharding
from jax._src.mesh import thread_resources
import contextlib
from jax.lax import with_sharding_constraint
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

    num_dev = len(jax.devices())
    use_parallel = tem.num_params > 1 and num_dev > 1

    tem_aligned = tem.align_to_devices(num_dev) if use_parallel else tem
    sp = pot.space
    t_r = pot.getTransmissionFunction(tem_aligned)
    P_k = getPropagationTerm(sp, tem_aligned)
    phi = getWaveFunction(sp, tem_aligned, probe=probe)

    if phi.ndim == 4 and P_k.ndim == 3:
        P_k = jnp.expand_dims(P_k, axis=1)

    if probe == "TEM":
        H = getAberrationFunction(sp, tem_aligned)  # shape (len(params), Nx, Ny)
        if H.ndim < phi.ndim:
            H = jnp.expand_dims(H, axis=1)
    else:
        H = jnp.zeros((num_dev, 1, 1), dtype=jnp.complex64)

    if use_parallel:
        mesh = jax.sharding.Mesh(jax.devices(), ('i',))
        use_probemode = not isinstance(probe, str) and probe.ndim > 2
        param_spec = P("i", None, None, None) if use_probemode else P("i", None, None)
        repl_spec = P(*(None,) * t_r.ndim)

        replicated = NamedSharding(mesh, P(*(None,) * phi.ndim))

        sharding_dist = NamedSharding(mesh, param_spec)

        phi = with_sharding_constraint(phi, sharding_dist)
        t_r = with_sharding_constraint(t_r, NamedSharding(mesh, repl_spec))
        P_k = with_sharding_constraint(P_k, sharding_dist)
        H = with_sharding_constraint(H, sharding_dist)

        fft2 = shard_map(lambda u: jnp.fft.fft2(u, axes=(-2, -1)), mesh=mesh, in_specs=param_spec, out_specs=param_spec)
        ifft2 = shard_map(lambda u: jnp.fft.ifft2(u, axes=(-2, -1)), mesh=mesh, in_specs=param_spec, out_specs=param_spec)
    else:
        fft2 = jnp.fft.fft2
        ifft2 = jnp.fft.ifft2

    @jax.jit
    def core_func(phi, t_r, P_k, H, mask):

        def body_with_depth(phi_prev, V_r):
            phi_next = ifft2(fft2(phi_prev * V_r) * P_k)
            return phi_next, phi_next

        def body_no_depth(phi_prev, V_r):
            phi_next = ifft2(fft2(phi_prev * V_r) * P_k)
            return phi_next, jnp.array(0.0)

        current_body = body_with_depth if returnDepth else body_no_depth

        if use_checkpoint:
            current_body = checkpoint(current_body)

        def run_full_process(p, t):
            phi_final, phis_all = jax.lax.scan(current_body, p, t)

            if probe == "TEM":
                phi_final = ifft2(fft2(phi_final) * H * mask)

                if returnDepth:
                    phis_all = jax.vmap(lambda layer: ifft2(fft2(layer) * H * mask))(phis_all)

            return phi_final, phis_all

        phi_final, phis_all = run_full_process(phi, t_r)

        # if use_checkpoint:
        #     checkpointed_scan = checkpoint(run_full_process)
        #     phi_final, phis_all = checkpointed_scan(phi, t_r)
        # else:
        #     phi_final, phis_all = run_full_process(phi, t_r)

        return phi_final, phis_all

    phi_final, phis_all = core_func(phi, t_r, P_k, H, sp.mask)

    if use_parallel:
        phi_final = with_sharding_constraint(phi_final, replicated)
        if returnDepth:
            phis_all = with_sharding_constraint(phis_all, replicated)

    res = jnp.swapaxes(phis_all[:, :tem.num_params], 0, 1) if returnDepth else phi_final[:tem.num_params]
    return res[0] if tem.num_params == 1 else res


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

    def _propagationTerm(theta):
        tilt = 1j * jnp.sum(kvec * jnp.tan(theta), axis=-1) + k2_term
        return jnp.exp(dz * tilt) * mask

    def _cartesianBeamTilt(t):
        return jnp.stack([t[0] * jnp.cos(t[1]), t[0] * jnp.sin(t[1])])

    def _single_tilt_process(t):
        theta = _cartesianBeamTilt(t)
        return _propagationTerm(theta)

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

    def _mask(arr):
        return jnp.where(k_norm <= k_max, arr, 0j)

    defocus, positions = map(jnp.array, zip(*[(p.defocus, p.position) for p in tem.params]))

    chi = getChi(sp, tem)

    if type(probe) is str and probe in ["TEM", "STEM"]:
        probe = jax.vmap(_mask)(jnp.exp(-1j * chi(defocus)))
    else:
        probe = jax.vmap(_mask)(jnp.fft.fft2(probe)*jnp.exp(-1j * chi(defocus)))
        # probe = jax.vmap(lambda df: jnp.fft.fft2(probe))(defocus)

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