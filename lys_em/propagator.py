import math
import inspect
import jax
import jax.numpy as jnp

# Host offloading (jax.memory) is available from JAX 0.7.1. In older versions, all slices are placed on device.
HOST_OFFLOAD = hasattr(jax, "memory")
if HOST_OFFLOAD:
    Host, Dev = jax.memory.Space.Host, jax.memory.Space.Device


def _vma(x):
    """Mesh axes over which x is device-varying inside shard_map (aval.vma was replaced by manual_axis_type in new JAX)."""
    aval = jax.typeof(x)
    return aval.vma if hasattr(aval, "vma") else aval.manual_axis_type.varying


def _map(f, xs, batch):
    """jax.lax.map(f, xs, batch_size=batch). batch_size is not available before JAX 0.4.31."""
    if "batch_size" in inspect.signature(jax.lax.map).parameters:
        return jax.lax.map(f, xs, batch_size=batch)
    tree = jax.tree_util
    n = tree.tree_leaves(xs)[0].shape[0]
    batch = min(batch, n)
    m = n // batch * batch
    ys = jax.lax.map(jax.vmap(f), tree.tree_map(lambda x: x[:m].reshape(n // batch, batch, *x.shape[1:]), xs))
    ys = tree.tree_map(lambda y: y.reshape(m, *y.shape[2:]), ys)
    if m < n:
        rest = jax.vmap(f)(tree.tree_map(lambda x: x[m:], xs))
        ys = tree.tree_map(lambda a, b: jnp.concatenate([a, b]), ys, rest)
    return ys


def _pvary(x, axes):
    if not axes:
        return x
    if hasattr(jax.lax, "pcast"):
        return jax.lax.pcast(x, axes, to="varying")
    return jax.lax.pvary(x, axes)


def _propagate(phi, t, Pk, nvalid=None):
    """
    Propagate phi of one parameter (nprobe, Nx, Ny) through the slices t (nslices, Nx, Ny). Pk: (Nx, Ny)
    If nvalid is given, only the first nvalid slices are used (the rest is padding).
    The slices are split into about sqrt(nslices) segments, and each segment is checkpointed, so that
    the backward pass keeps about 2*sqrt(nslices) wave functions instead of nslices.
    """
    def step(i, phi, t, offset):
        new = jnp.fft.ifft2(jnp.fft.fft2(phi * t[i]) * Pk)
        return new if nvalid is None else jnp.where(offset + i < nvalid, new, phi)

    def segment(phi, t, offset):
        return jax.lax.fori_loop(0, t.shape[0], lambda i, phi: step(i, phi, t, offset), phi)
    n = t.shape[0]
    size = max(s for s in range(1, math.isqrt(n) + 1) if n % s == 0)
    if size == 1:
        return segment(phi, t, 0)
    segments = t.reshape(n // size, size, *t.shape[1:])
    return jax.lax.fori_loop(0, n // size, lambda j, phi: jax.checkpoint(segment)(phi, segments[j], j * size), phi)


def _propagate_all(phi, t, Pk, batch, nvalid=None):
    """_propagate for all parameters. phi: (nparams, nprobe, Nx, Ny), Pk: (nparams, Nx, Ny). batch parameters are computed at once."""
    f = jax.checkpoint(lambda a: _propagate(a[0], t, a[1], nvalid))  # in backward, recompute the propagation for each batch
    return _map(f, (phi, Pk), batch)


def _scan_chunks(f, carry, t_r, size, xs=None, reverse=False):
    """
    jax.lax.scan of f over chunks of slices: carry, y = f(carry, t, x, start), where t is the next `size` slices
    of t_r (Nz, Nx, Ny) moved to device, start is the index of its first slice, and x is the corresponding element of xs (if given).
    """
    t_chunks = t_r.reshape(-1, size, *t_r.shape[1:])
    starts = jnp.arange(len(t_chunks)) * size
    to_dev = (lambda t: jax.device_put(t, Dev)) if HOST_OFFLOAD else (lambda t: t)
    carry, ys = jax.lax.scan(lambda c, a: f(c, to_dev(a[0]), a[1], a[2]), carry, (t_chunks, xs, starts), reverse=reverse)
    return carry, ys


def _forward(phi0, Pk, t_r, size, batch, nslices, save=False):
    """Propagate through all slices t_r. If save, phi at the beginning of each chunk is stored on host."""
    def over_chunk(phi, t, _, start):
        new = _propagate_all(phi, t, Pk, batch, None if nslices is None else nslices - start)
        return new, (jax.device_put(phi, Host) if save else None)
    return _scan_chunks(over_chunk, phi0, t_r, size)


def make_propagator(chunk=None, batch=1, nslices=None):
    """
    Return prop(phi0, Pk, t_r) -> phi, which propagates the incident waves through all slices.

    Args:
        chunk (int): Number of slices transferred to device at once. Default (None) is all slices.
        batch (int): Number of parameters computed at once.
        nslices (int): Number of slices. If given, t_r may be padded at the end to a multiple of chunk.

    prop arguments (per device):
        phi0: (nparams, nprobe, Nx, Ny), Pk: (nparams, Nx, Ny), t_r: (Nz, Nx, Ny) on device or on (pinned) host.

    If t_r is on host, the gradient is calculated by custom_vjp, in which phi at the beginning of each chunk is
    stored on host and the inside of the chunk is recomputed, so that device memory does not depend on Nz.
    The gradient of t_r is also placed on host.
    """
    size = lambda t_r: len(t_r) if chunk is None else min(chunk, len(t_r))

    @jax.custom_vjp
    def prop(phi0, Pk, t_r):
        return _forward(phi0, Pk, t_r, size(t_r), batch, nslices)[0]

    def prop_fwd(phi0, Pk, t_r):
        phi, ck = _forward(phi0, Pk, t_r, size(t_r), batch, nslices, save=True)   # ck: phi at the beginning of each chunk (on host)
        return phi, (ck, Pk, t_r)

    def prop_bwd(res, g):
        ck, Pk, t_r = res
        def over_chunk(carry, t, phi, start):
            dphi, dPk = carry
            # t is replicated over devices. Make it device-varying here, so that dt is summed over devices
            # only once per chunk (below), not for each batch inside vjp.
            axes = tuple(_vma(dphi) - _vma(t))
            t = _pvary(t, axes)
            nvalid = None if nslices is None else nslices - start
            _, vjp = jax.vjp(lambda phi, t, pk: _propagate_all(phi, t, pk, batch, nvalid), jax.device_put(phi, Dev), t, Pk)
            dphi, dt, dpk = vjp(dphi)   # dt is summed over parameters
            return (dphi, dPk + dpk), jax.device_put(jax.lax.psum(dt, axes), Host)
        (dphi0, dPk), dt = _scan_chunks(over_chunk, (g, jnp.zeros_like(Pk)), t_r, size(t_r), xs=ck, reverse=True)
        return dphi0, dPk, dt.reshape(t_r.shape)

    prop.defvjp(prop_fwd, prop_bwd)

    def propagate(phi0, Pk, t_r):
        if HOST_OFFLOAD and jax.typeof(t_r).memory_space == Host:
            return prop(phi0, Pk, t_r)
        return _forward(phi0, Pk, t_r, size(t_r), batch, nslices)[0]
    return propagate
