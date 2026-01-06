import itertools

import numpy as np
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import itertools
import scipy.optimize as optimize
from scipy.special import j0, jn_zeros
from . import TEM, TEMParameter, FunctionSpace, CrystalPotential, multislice, GeneralPotential


def calcSADiffraction(V, crys, numOfCells, Nx=128, Ny=128, division="Auto", tilt=[0, 0]):
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    tem = TEM(V, params=TEMParameter(tilt=tilt))
    return diffraction(multislice(pot, tem))


def calcPrecessionDiffraction(V, crys, numOfCells, theta, nphi, Nx=128, Ny=128, division="Auto"):
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    tem = TEM(V, params=[TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, 360 / nphi)])
    return diffraction(multislice(pot, tem)).sum(axis=0)


def calcCBED(V, convergence, crys, numOfCells, ndisk=30, Nx=128, Ny=128, division="Auto", sum=True):
    # convergence is in radians
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)

    c = np.arange(-convergence, convergence, 2 * convergence / ndisk)
    tem = TEM(V, params=[TEMParameter(tilt=[tx, ty]) for tx, ty in itertools.product(c, c)])

    res = abs(np.fft.fft2(multislice(pot, tem), axes=(1, 2)))**2
    res.reshape(ndisk, ndisk, Nx, Ny)
    return res


def calc4DSTEM_Crystal(V, convergence, crys, numOfCells, Nx=256, Ny=256, division="Auto", scanx=16, scany=16):
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)

    c, r = np.arange(0, crys.a, crys.a / scanx), np.arange(0, crys.b, crys.b / scany)
    tem = TEM(V, convergence=convergence, params=[TEMParameter(beamPosition=[x, y]) for x, y in itertools.product(c, r)])

    res = abs(np.fft.fft2(multislice(pot, tem), axes=(1, 2)))**2
    res.reshape(scanx, scany, Nx, Ny)
    return res


def diffraction(data):
    return jnp.abs(jnp.fft.fft2(data))**2


def center_of_mass(data, axes=(-2, -1)):
    data = np.asarray(data, float)
    ndim = data.ndim
    axes = tuple(sorted(axes))

    total = np.sum(data, axis=axes, keepdims=False)
    total[total == 0] = np.nan

    coms = []
    for ax in axes:
        coords = np.arange(data.shape[ax], dtype=float)
        num = np.sum(data * np.expand_dims(coords, tuple(i for i in range(ndim) if i != ax)), axis=axes)
        com = num / total
        coms.append(com)

    return np.stack(coms, axis=-1)


def generic_ravel_pytree(params_tree):
    def complex_to_real_dict(leaf):
        if jnp.iscomplexobj(leaf):
            return {"re": leaf.real, "im": leaf.imag}
        return leaf

    real_tree = jax.tree_util.tree_map(complex_to_real_dict, params_tree)
    flat_params, unravel_fn_raw = ravel_pytree(real_tree)

    def unravel_fn(flat):
        unraveled_real_tree = unravel_fn_raw(flat)

        def real_dict_to_complex(leaf):
            if isinstance(leaf, dict) and "re" in leaf and "im" in leaf:
                return leaf["re"] + 1j * leaf["im"]
            return leaf

        return jax.tree_util.tree_map(real_dict_to_complex, unraveled_real_tree, is_leaf=lambda x: isinstance(x, dict) and "re" in x)

    return flat_params, unravel_fn


def expand_in_Bessel(data, K, K_max=100):
    K_max += (K - K_max % K) % K

    h, w = data.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w // 2)**2 + (y - h // 2)**2)

    alphas = jn_zeros(0, K_max) / r.max()
    bases = j0(alphas[:, None] * r.ravel())

    Q, _ = np.linalg.qr(bases.T)
    data_rolled = np.roll(data, (h // 2, w // 2), axis=(0, 1)).ravel()
    coefs = Q.T @ data_rolled

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    components = coefs[sorted_idx, None] * Q.T[sorted_idx]

    groups = components.reshape(-1, K, h * w).sum(axis=0).reshape(K, h, w)

    return np.roll(groups, (-h // 2, -w // 2), axis=(1, 2))
