import jax
import jax.numpy as jnp

Host, Dev = jax.memory.Space.Host, jax.memory.Space.Device


def _propagate(phi, t, Pk):
    """Propagate phi of one parameter (nprobe, Nx, Ny) through the slices t (nslices, Nx, Ny). Pk: (Nx, Ny)"""
    step = lambda i, phi: jnp.fft.ifft2(jnp.fft.fft2(phi * t[i]) * Pk)
    return jax.lax.fori_loop(0, t.shape[0], step, phi)


def _propagate_all(phi, t, Pk, batch):
    """_propagate for all parameters. phi: (nparams, nprobe, Nx, Ny), Pk: (nparams, Nx, Ny). batch parameters are computed at once."""
    f = jax.checkpoint(lambda a: _propagate(a[0], t, a[1]))  # in backward, recompute the propagation for each batch
    return jax.lax.map(f, (phi, Pk), batch_size=batch)


def _scan_chunks(f, carry, t_r, size, xs=None, reverse=False):
    """
    jax.lax.scan of f over chunks of slices: carry, y = f(carry, t, x), where t is the next `size` slices
    of t_r (Nz, Nx, Ny) moved to device, and x is the corresponding element of xs (if given).
    If f returns slices as y, they are concatenated back to (Nz, Nx, Ny).
    """
    t_chunks = t_r.reshape(-1, size, *t_r.shape[1:])
    carry, ys = jax.lax.scan(lambda c, a: f(c, jax.device_put(a[0], Dev), a[1]), carry, (t_chunks, xs), reverse=reverse)
    return carry, ys


def _forward(phi0, Pk, t_r, size, batch, save=False):
    """Propagate through all slices t_r. If save, phi at the beginning of each chunk is stored on host."""
    def over_chunk(phi, t, _):
        new = _propagate_all(phi, t, Pk, batch)
        return new, (jax.device_put(phi, Host) if save else None)
    return _scan_chunks(over_chunk, phi0, t_r, size)


def make_propagator(chunk=None, batch=1):
    """
    Return prop(phi0, Pk, t_r) -> phi, which propagates the incident waves through all slices.

    Args:
        chunk (int): Number of slices transferred to device at once. Default (None) is all slices.
        batch (int): Number of parameters computed at once.

    prop arguments (per device):
        phi0: (nparams, nprobe, Nx, Ny), Pk: (nparams, Nx, Ny), t_r: (Nz, Nx, Ny) on device or on (pinned) host.

    If t_r is on host, the gradient is calculated by custom_vjp, in which phi at the beginning of each chunk is
    stored on host and the inside of the chunk is recomputed, so that device memory does not depend on Nz.
    The gradient of t_r is also placed on host.
    """
    size = lambda t_r: len(t_r) if chunk is None else chunk

    @jax.custom_vjp
    def prop(phi0, Pk, t_r):
        return _forward(phi0, Pk, t_r, size(t_r), batch)[0]

    def prop_fwd(phi0, Pk, t_r):
        phi, ck = _forward(phi0, Pk, t_r, size(t_r), batch, save=True)   # ck: phi at the beginning of each chunk (on host)
        return phi, (ck, Pk, t_r)

    def prop_bwd(res, g):
        ck, Pk, t_r = res
        def over_chunk(carry, t, phi):
            dphi, dPk = carry
            _, vjp = jax.vjp(lambda phi, t, pk: _propagate_all(phi, t, pk, batch), jax.device_put(phi, Dev), t, Pk)
            dphi, dt, dpk = vjp(dphi)   # dt is summed over parameters (and devices, since t is replicated)
            return (dphi, dPk + dpk), jax.device_put(dt, Host)
        (dphi0, dPk), dt = _scan_chunks(over_chunk, (g, jnp.zeros_like(Pk)), t_r, size(t_r), xs=ck, reverse=True)
        return dphi0, dPk, dt.reshape(t_r.shape)

    prop.defvjp(prop_fwd, prop_bwd)

    def propagate(phi0, Pk, t_r):
        if jax.typeof(t_r).memory_space == Host:
            return prop(phi0, Pk, t_r)
        return _forward(phi0, Pk, t_r, size(t_r), batch)[0]
    return propagate
