import numpy as np
from . import scatteringFactor
from .space import FunctionSpace


def _scatteringFactors(c, k):
    k4p = np.linalg.norm(k, axis=-1) / (4 * np.pi)
    Z = [at.Z for at in c.atoms]
    N = [at.Occupancy for at in c.atoms]
    F = {z: scatteringFactor(z, k4p) for z in np.unique(Z)}
    return np.array([F[z] * n for z, n in zip(Z, N)]).transpose(*(np.array(range(k4p.ndim)) + 1), 0)


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
    inv = np.linalg.norm(c.inv / 2 / np.pi, axis=1)
    T = np.array([[inv[0], 0, 0], [0, inv[1], 0], [0, 0, inv[2]]])
    R = T.dot(c.unit)
    U = np.array([R.T.dot(at.Uani).dot(R) for at in c.atoms])
    if len(k.shape) == 1:
        kUk = np.einsum("i,nij,j->n", k, U, k)
    if len(k.shape) == 2:
        kUk = np.einsum("ki,nij,kj->kn", k, U, k)
    if len(k.shape) == 3:
        kUk = np.einsum("kqi,nij,kqj->kqn", k, U, k)
    return np.exp(-kUk / 2)


def structureFactors(c, k, sum="atoms"):
    """
    Calculate the structure factors for a given crystal structure and k vectors.

    Args:
        c (CrystalStructure): The crystal structure.
        k (array_like): The k vectors.
        sum (str): The type of summation to be performed. Can be "atoms" (default) or "elements".

    Returns:
        array_like: The structure factors for the given crystal structure and k vectors.
    """
    r_i = c.getAtomicPositions()
    f_i = _scatteringFactors(c, k)
    T_i = debyeWallerFactors(c, k)
    kr_i = np.tensordot(k, r_i, [-1, -1])
    st = f_i * T_i * np.exp(1j * kr_i)
    if sum == "atoms":
        st = np.sum(st, axis=-1)
    if sum == "elements":
        st = st.transpose(*([k.ndim - 1] + list(range(k.ndim - 1))))
        st = np.array([np.sum([s for s, at in zip(st, c.atoms) if at.Element == element], axis=0) for element in c.getElements()])
        st = st.transpose(*(list(range(k.ndim))[1:] + [0]))
    return st


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


def calcKinematicalDiffraction(c, TEM, numOfCells, Nx=128, Ny=128):
    """
    Calculate the kinematical diffraction pattern for a given crystal structure and transmission electron microscope (TEM) settings.

    Args:
        c (CrystalStructure): The crystal structure.
        TEM (TEM): The transmission electron microscope settings.
        numOfCells (int): The number of unit cells in the z-direction.
        Nx (int, optional): Number of grid points along the x-direction. Default is 128.
        Ny (int, optional): Number of grid points along the y-direction. Default is 128.

    Returns:
        array_like: The intensity of the kinematical diffraction pattern.
    """

    sp = FunctionSpace.fromCrystal(c, Nx, Ny)
    kx, ky = sp.kvec[:, :, 0], sp.kvec[:, :, 1]

    dx, dy, dz = -TEM.beamDirection * TEM.k_in
    kz = dz - np.sqrt(TEM.k_in**2 - (kx - dx)**2 + (ky - dy)**2)
    k = np.concatenate((sp.kvec, kz[:, :, np.newaxis]), axis=2)

    return abs(structureFactors(c, k) * formFactors(c, [1, 1, numOfCells], k))**2
