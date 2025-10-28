import numpy as np
from . import scatteringFactor
from .space import FunctionSpace

import jax
import jax.numpy as jnp


def _safe_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _scatteringFactors(c, k):
    k4p = _safe_norm(k, axis=-1) / (4 * np.pi)
    Z = [at.Z for at in c.atoms]
    N = [at.Occupancy for at in c.atoms]
    F = {z: scatteringFactor(z, k4p) for z in np.unique(Z)}
    return jnp.array([F[z] * n for z, n in zip(Z, N)]).transpose(*(jnp.array(range(k4p.ndim)) + 1), 0)


def debyeWallerFactors(c, k):
    """
    Calculate the Debye-Waller factor for each atom in the crystal.

    Args:
        c (CrystalStructure): The crystal structure.
        k (array_like): The k vectors.

    Returns:
        array_like: The Debye-Waller factors for each atom in the crystal.
    """

    k = np.array(k)
    Uani = np.array([at.Uani for at in c.atoms])
    return _debyeWallerFactor_impl(c.unit, Uani, k)


@jax.jit
def _debyeWallerFactor_impl(unit, Uani, k):
    inv = jnp.linalg.inv(unit.T)
    R = jnp.diag(jnp.linalg.norm(inv, axis=1)).dot(unit)
    U = jnp.einsum("ij,njk,kl->nil", R.T, Uani, R)
    kUk = jnp.einsum("...i,nij,...j->...n", k, U, k)
    return jnp.exp(-kUk / 2)


def structureFactors(c, k):
    """
    Calculate the structure factors for a given crystal structure and k vectors.

    Args:
        c (CrystalStructure): The crystal structure.
        k (array_like): The k vectors.
        sum (str): The type of summation to be performed. Can be "atoms" (default) or "elements".

    Returns:
        array_like: The structure factors for the given crystal structure and k vectors.
    """
    if len(c.atoms) == 0:
        return jnp.zeros(k.shape[:-1], dtype=complex)
    f_i = jnp.array(_scatteringFactors(c, k))

    @jax.jit
    def _func(unit, position, Uani):
        T_i = _debyeWallerFactor_impl(unit, Uani, k)
        kr_i = jnp.tensordot(k, position, (-1, -1))
        st = f_i * T_i * jnp.exp(1j * kr_i)
        return st.sum(axis=-1)

    u = jnp.array(c.unit.T)
    position = jnp.array([u.dot(at.position) for at in c.atoms])
    return _func(c.unit, position, np.array([at.Uani for at in c.atoms]))


def formFactors(c, N, K):
    """
    Calculate the form factors for a given crystal structure and k vectors.

    Args:
        c (CrystalStructure): The crystal structure.
        N (array_like): The number of unit cells in each direction.
        K (array_like): The k vectors.

    Returns:
        array_like: The form factors for the given crystal structure and k vectors.
    """
    def _sindiv(N, kR):
        a = np.sin(kR / 2)
        return np.where(np.abs(a) < 1e-5, N, np.sin(kR / 2 * N) / a)
    unit = c.unit
    kR = np.einsum("ijk,lk->ijl", K, unit)
    shelement = [_sindiv(N[i], kR[:, :, i]) for i in range(3)]
    sh = N[0] * N[1] * N[2] * shelement[0] * shelement[1] * shelement[2]
    return sh


def calcKinematicalDiffraction(crys, TEM, TEMParam, numOfCells, Nx=128, Ny=128):
    """
    Calculate the kinematical diffraction pattern for a given crystal structure and transmission electron microscope (TEM) settings.

    Args:
        crys (CrystalStructure): The crystal structure.
        TEM (TEM): The transmission electron microscope settings.
        TEMParam (TEMParameter): The transmission electron microscope parameters.
        numOfCells (int): The number of unit cells in the z-direction.
        Nx (int, optional): Number of grid points along the x-direction. Default is 128.
        Ny (int, optional): Number of grid points along the y-direction. Default is 128.

    Returns:
        float: The kinematical diffraction pattern for the given crystal structure and TEM settings.
    """
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells)
    kx, ky = sp.kvec[:, :, 0], sp.kvec[:, :, 1]

    dx, dy, dz = -TEMParam.beamDirection * TEM.k_in
    kz = dz - np.sqrt(TEM.k_in**2 - (kx - dx)**2 - (ky - dy)**2)
    k = np.concatenate((sp.kvec, kz[:, :, np.newaxis]), axis=2)

    return abs(structureFactors(crys, k) * formFactors(crys, [1, 1, numOfCells], k))**2
