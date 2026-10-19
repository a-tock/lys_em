import functools
import inspect
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .propagator import make_propagator

shard_map = jax.shard_map if hasattr(jax, 'shard_map') else jax.experimental.shard_map.shard_map


class Slices:
    """
    Transmission functions of all slices (Nz, Nx, Ny), and where they are placed.

    Args:
        pot, tem: The potential and the TEM object.
        mesh (jax.sharding.Mesh): The 1D mesh of the devices. All devices if None.
        mode (str): "device", "pinned" or "auto".
            "device": All slices are placed (replicated) on each device. Device memory ~ Nz.
            "pinned": Slices are placed in pinned host memory and copied to device chunk by chunk.
                Device memory does not depend on Nz.
            "auto": "pinned" if chunk is given and running on GPU, otherwise "device".
        chunk (int): Number of slices copied to device at once.
            If Nz is not a multiple of chunk, the slices are padded at the end (see nslices).

    Attributes:
        nslices (int or None): Number of slices if they are padded, otherwise None.
    """

    def __init__(self, pot, tem, mode="auto", chunk=None, mesh=None):
        if mode == "auto":
            mode = "pinned" if chunk is not None and jax.default_backend() == "gpu" else "device"
        if mode == "pinned":
            with jax.default_device(jax.devices("cpu")[0]):
                self._t_r = pot.getTransmissionFunction(tem)
            if chunk is None or chunk >= len(self._t_r):   # XLA fails for a single chunk with host offloading
                mode = "device"
        else:
            self._t_r = pot.getTransmissionFunction(tem)
        self.nslices = None
        nz = len(self._t_r)
        if chunk is not None and chunk < nz and nz % chunk:
            self.nslices = nz
            with jax.default_device(jax.devices("cpu")[0] if mode == "pinned" else None):
                self._t_r = jnp.pad(self._t_r, ((0, chunk - nz % chunk), (0, 0), (0, 0)), constant_values=1)
        self.mode = mode
        mesh = jax.sharding.Mesh(jax.devices(), ('i',)) if mesh is None else mesh
        self._sharding = jax.sharding.NamedSharding(mesh, P(), memory_kind="pinned_host" if mode == "pinned" else "device")

    def place(self):
        """
        Return t_r placed (replicated) on all devices, or on their pinned host memory.
        Called once per parameter group, so that the gradients of the groups are summed where t_r was calculated.
        """
        if self.mode != "pinned":
            return jax.device_put(self._t_r, self._sharding)
        cpu, sharding = jax.devices("cpu")[0], self._sharding

        @jax.custom_vjp
        def to_pinned(x):
            return jax.device_put(x, sharding)
        to_pinned.defvjp(lambda x: (to_pinned(x), None), lambda _, g: (jax.device_put(g, jax.sharding.SingleDeviceSharding(cpu)),))  # Device object keeps <host> type
        return to_pinned(self._t_r)


def multislice(pot, tem, probe="TEM", postprocess=None, sum=False, data=None, use_checkpoint=False, mesh=None, slices="auto", chunk="auto", batch="auto", group="auto"):
    """
    Calculate the multislice simulation.

    Args:
        pot (CrystalPotential): The CrystalPotential object.
        tem (TEM) : The TEM object.
        probe (str or array) : The probe type or the probe wave function. Probe type can be "TEM" or "STEM".
            The probe wave function should be a 3D array of shape (modes, Nx, Ny). Default is "TEM".
        postprocess (str or callable) : Specifies the post-processing method.
            It can be None (default), a string ("square" or "diffraction"), or a user-defined callable.
        sum (bool) : Whether to sum the data over all parameters in the tem.params list. Default is False.
        data (numpy.ndarray) : Additional data for post-processing.
            Required only if the postprocess method depends on external input; otherwise, None (default).
            Must have shape (len(tem.params), Nx, Ny) when provided.
        use_checkpoint (bool) : Not used (kept for compatibility). `jax.checkpoint` is always used so that
            device memory in the backward pass is reduced.
        mesh (jax.sharding.Mesh) : A 1D mesh for device parallelism.
            If None, a mesh is automatically generated. Default is None.
        slices (str): Where the transmission functions are placed ("auto", "device" or "pinned"). See Slices.
        chunk (int, None or "auto"): Number of slices copied to device at once. If given on GPU, device memory does not depend on Nz.
            None places all slices on device. "auto" uses None if they fit in device memory, otherwise 16.
        batch (int or "auto"): Number of parameters computed at once on each device. "auto" is 16.
        group (int, None or "auto"): Number of parameters per device in a single call. Device memory depends on group,
            not on nparams. None is all parameters. "auto" is the same as batch.

    Returns:
        phi (jax.numpy.ndarray) : The multislice simulation result.
    """
    sp = pot.space
    mesh = jax.sharding.Mesh(jax.devices(), ('i',)) if mesh is None else mesh
    ndev = mesh.size
    chunk, batch, group = _auto_options(sp, chunk, batch, group)
    t_r = Slices(pot, tem, slices, chunk, mesh)
    probe_type = probe if isinstance(probe, str) else None
    probes = probe if probe_type is None else jnp.array([jnp.fft.ifft2(jnp.ones(sp.N[:2]))])
    calc = _build_calc(probe_type, postprocess, sum, data is not None, chunk, batch, t_r.nslices, mesh)

    # Pad the parameters so that each device has the same number of groups of the same size.
    nparams = len(tem.params)
    params, _ = tem.asdict(1)
    if data is not None:
        params["data"] = jnp.asarray(data)
    n_local = -(-nparams // ndev)
    group = n_local if group is None else min(group, n_local)
    total = ndev * (-(-n_local // group) * group)
    params = jax.tree.map(lambda v: jnp.concatenate([v, jnp.repeat(v[:1], total - nparams, axis=0)]), params)
    params["mask"] = jnp.arange(total) < nparams    # False for padding

    spd = sp.asdict()
    def calc_group(p):
        return calc(p, t_r.place(), spd, probes)     # place() for each group, so that the gradient of t_r is summed on CPU
    phi = _map_groups(calc_group, params, ndev, group, sum)
    if not sum:
        phi = phi[:nparams]
    return phi.squeeze()


def _auto_options(sp, chunk, batch, group):
    """Default values of chunk, batch and group (see multislice)."""
    batch = 16 if batch == "auto" else batch
    group = batch if group == "auto" else group
    if chunk == "auto":
        chunk = None
        stats = jax.devices()[0].memory_stats() if jax.default_backend() == "gpu" else None
        if stats is not None:
            # Device memory for all slices is about 12 slices per slice in the backward pass (measured).
            slice_bytes = sp.N[0] * sp.N[1] * 8
            free = stats["bytes_limit"] - stats["bytes_in_use"]
            if 12 * sp.N[2] * slice_bytes > free / 2:
                chunk = 16
    return chunk, batch, group


@functools.lru_cache
def _build_calc(probe_type, postprocess, sum, use_data, chunk, batch, nslices, mesh):
    """
    Return the jitted function calc(params, t_r, spd, probes) for a group of parameters.
    It is cached so that it is compiled only once for the same options and shapes.
    """
    prop = make_propagator(chunk=chunk, batch=batch, nslices=nslices)
    post = _init_postprocess(postprocess)
    post_with_data = use_data and len(inspect.signature(post).parameters) == 2
    axis = mesh.axis_names[0]

    def local_dev(x, t_r, spd, probes):
        x = dict(x)
        mask, data = x.pop("mask"), x.pop("data", None)
        phi0 = jax.vmap(lambda p: getWaveFunction(spd, p, probes))(x)     # (nparams, modes, Nx, Ny)
        Pk = jax.vmap(lambda p: getPropagationTerm(spd, p))(x)            # (nparams, Nx, Ny)
        phi = prop(phi0, Pk, t_r)
        if probe_type == "TEM":
            aberration = lambda phi, p: jnp.fft.ifft2(jnp.fft.fft2(phi) * getAberrationFunction(spd, p) * spd["mask"])
            phi = jax.vmap(aberration)(phi, x)
        phi = jnp.sum(phi, axis=1)    # sum over modes, (nparams, Nx, Ny)
        res = jax.vmap(post)(phi, data) if post_with_data else jax.vmap(post)(phi)
        if sum:
            mask = mask.reshape(-1, *([1] * (res.ndim - 1)))
            return jax.lax.psum(jnp.where(mask, res, 0).sum(axis=0), axis)    # padding is excluded
        return res

    # Parallelization: parameters are distributed to the devices.
    out_specs = P() if sum else P(axis)
    return jax.jit(jax.checkpoint(shard_map(local_dev, mesh=mesh, in_specs=(P(axis), P(), P(), P()), out_specs=out_specs)))


def _map_groups(f, params, ndev, group, sum):
    """
    Apply f to the parameters, `group` parameters per device at a time.
    params are sharded device-major (ndev * n_local), where n_local is a multiple of group.
    Device memory depends on group, not on nparams.
    The results are summed (sum=True) or concatenated in the original order on CPU (sum=False).
    """
    n_local = len(jax.tree.leaves(params)[0]) // ndev
    local = jax.tree.map(lambda v: v.reshape(ndev, n_local, *v.shape[1:]), params)
    cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    res = []
    for start in range(0, n_local, group):
        p = jax.tree.map(lambda v: v[:, start:start + group].reshape(-1, *v.shape[2:]), local)
        r = f(p)
        res.append(r if sum else jax.device_put(r, cpu).reshape(ndev, group, *r.shape[1:]))   # only one group stays on device
    if sum:
        return functools.reduce(jnp.add, res)
    return jnp.concatenate(res, axis=1).reshape(ndev * n_local, *res[0].shape[2:])


def _init_postprocess(post):
    if post is None:
        return lambda x: x
    elif post == "square":
        return lambda x: jnp.abs(x)**2
    elif post == "diffraction":
        return lambda x: jnp.abs(jnp.fft.fft2(x))**2
    else:
        return post


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
        sp (dict): A FunctionSpace object serialized into a dictionary via asdict.
        tem (dict): A TEM object serialized into a dictionary via asdict.

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
        sp (dict): A FunctionSpace object serialized into a dictionary via asdict.
        tem (dict): A TEM object serialized into a dictionary via asdict.
        probe (str or array) : The probe type or the probe wave function. Probe type can be "TEM" or "STEM".
            The probe wave function should be a 3D array of shape (modes, Nx, Ny). Default is "TEM".

    Returns:
        numpy.ndarray: A 3D array of shape (modes, Nx, Ny) where each element is the wave function at the respective grid point in real space.
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
        sp (dict): A FunctionSpace object serialized into a dictionary via asdict.
        tem (dict): A TEM object serialized into a dictionary via asdict.

    Returns:
        numpy.ndarray: A 2D array of shape (Nx, Ny) where each element is the aberration function at the respective grid point in real space.
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
        sp (dict): A FunctionSpace object serialized into a dictionary via asdict.
        tem (dict): A TEM object serialized into a dictionary via asdict.

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