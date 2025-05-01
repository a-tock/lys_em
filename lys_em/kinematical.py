import numpy as np
from . import scatteringFactor


def _scatteringFactors(c, k):
    k4p = np.linalg.norm(k, axis=-1) / (4 * np.pi)
    Z = [at.Z for at in c.atoms]
    N = [at.Occupancy for at in c.atoms]
    F = {z: scatteringFactor(z, k4p) for z in np.unique(Z)}
    return np.array([F[z] * n for z, n in zip(Z, N)]).transpose(*(np.array(range(k4p.ndim)) + 1), 0)


def debyeWallerFactors(c, k):
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
    def _sindiv(N, kR):
        a = np.sin(kR / 2)
        return np.where(np.abs(a) < 1e-5, N, np.sin(kR / 2 * N) / a)
    unit = c.unit
    kR = np.einsum("ijk,lk->ijl", K, unit)
    shelement = [_sindiv(N[i], kR[:, :, i]) for i in range(3)]
    sh = N[0] * N[1] * N[2] * shelement[0] * shelement[1] * shelement[2]
    return sh
