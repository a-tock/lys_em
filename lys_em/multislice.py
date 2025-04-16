import numpy as np
import dask.array as da
import dask
import pyfftw

from lys import DaskWave
from lys_mat import CrystalStructure

from .kinematical import structureFactors
fft, ifft = pyfftw.interfaces.numpy_fft.fft2, pyfftw.interfaces.numpy_fft.ifft2
m = 9.10938356e-31  # kg

pyfftw.interfaces.cache.enable()


class FunctionSpace:
    """
    Create 2 dimensional rectangular grid function space that is defined by the crystal structure.

    The length of the rectangular space is defined by the crys.a and crys.b
    Each cell of the grid function space is (crys.a/Nx crys.b/Ny)

    """
    def __init__(self, crys, Nx, Ny, division="Auto"):
        self._unit = np.array([crys.a, crys.b])
        self._N = np.array([Nx, Ny])
        if division == "Auto":
            self._division = int(crys.unit[2][2] / 2)
        else:
            self._division = division
        self._dz = crys.unit[2][2] / self._division

    def getArray(self):
        return np.ones((self._N[0], self._N[1])) / self._N[0] / self._N[1]

    @property
    def mask(self):
        """
        Return mask pattern whose diameter is identical with 2/3 of min(kx, ky).
        """
        if not hasattr(self, "_mask"):
            k = np.sqrt(self.k2)
            max = self._getMax()
            self._mask = np.where(k > max * 2 / 3, 0, 1)
        return self._mask

    def _getMax(self):
        return min(max(abs(self.kx)), max(abs(self.ky)))

    @property
    def k2(self):
        """
        Return k^2 for respective grid point in reciprocal space. The unit is rad/A.
        """
        if not hasattr(self, "_k2"):
            k = self.kvec
            self._k2 = k[:, :, 0]**2 + k[:, :, 1]**2
        return self._k2

    @property
    def kvec(self):
        """
        Return 2 dimensional reciprocal space grid. The unit is rad/A.
        Each cell has (2pi/a, 2pi/b) length in reciprocal space.
        """
        if not hasattr(self, "_kvec"):
            self._kvec = np.array(np.meshgrid(self.kx, self.ky)).transpose(2, 1, 0)
        return self._kvec

    @property
    def kx(self):
        """
        Calculate kx array in reciprocal space. The unit is rad/A.
        The maximum kx of the reciprocal space is 2pi/a * Nx
        """
        return np.fft.fftfreq(self._N[0], self._unit[0] / 2 / np.pi) * self._N[0]

    @property
    def ky(self):
        """
        Calculate ky in reciprocal space. The unit is rad/A.
        The maximum kx of the reciprocal space is 2pi/b * Ny
        """
        return np.fft.fftfreq(self._N[1], self._unit[1] / 2 / np.pi) * self._N[1]

    @property
    def dz(self):
        """
        Calculate thickness of the slice in the unit of A.
        It is identical with the (z component of c axis)/division
        """
        return self._dz

    @property
    def division(self):
        return self._division

    @property
    def dV(self):
        return self._unit[0] * self._unit[1] / self._N[0] / self._N[1]

    def FT(self, data):
        return np.fft.fft2(data) * self.dV

    def IFT(self, data):
        return np.fft.ifft2(data) / self.dV

    def getPropagationTerm(self, lamb, theta_x=0, theta_y=0):
        k2 = self.k2
        tx, ty = np.array([theta_x, theta_y]) * np.pi / 180
        kx, ky = self.kvec.transpose(2, 1, 0)[0].T, self.kvec.transpose(2, 1, 0)[1].T
        # kx, ky = self.kvec[:, :, 0], self.kvec[:, :, 1]
        tilt = 1j * (kx * np.tan(tx) + ky * np.tan(ty)) - 1j * lamb * k2 / 4 / np.pi
        return np.exp(self.dz * tilt)


class Slices:
    def __init__(self, crys, sp):
        self._sp = sp
        self._slices = []
        zList = np.arange(sp.division + 1) * sp.dz
        positionList = crys.getAtomicPositions()
        for i in range(len(zList) - 1):
            atomsList = [at for pos, at in zip(positionList, crys.atoms) if zList[i] <= pos[2] < zList[i + 1]]
            self._slices.append(CrystalStructure(crys.cell, atomsList))

    def _calculatePotential(self, crys):
        if len(crys.atoms) == 0:
            return 0
        else:
            k = self._sp.kvec
            q = np.array([k[:, :, 0], k[:, :, 1], self._sp.getArray() * 0]).transpose(1, 2, 0)
            return structureFactors(crys, q)

    def getPotentialTerms(self, beam):
        res = []
        for c in self._slices:
            V_k = self._calculatePotential(c) * beam.getWavelength() * beam.getRelativisticMass() / m  # A^2
            V_r = self._sp.IFT(V_k * self._sp.mask)
            V_z = np.exp(1j * V_r)
            V_z_lim = self._sp.IFT(self._sp.FT(V_z) * self._sp.mask)
            res.append(V_z_lim)
        return res


def calcMultiSliceDiffraction(c, numOfSlices, V=60e3, Nx=128, Ny=128, division="Auto", theta_list=[[0, 0]], returnDepth=True):
    sp = FunctionSpace(c, Nx, Ny, division)
    c.saveAsCif(".crystal.cif")
    ncore = len(DaskWave.client.ncores()) if hasattr(DaskWave,"client") else 1
    shape = (int(len(theta_list)/ncore), Nx, Ny, numOfSlices * sp.division) if returnDepth else  (int(len(theta_list)/ncore), Nx, Ny)

    # Dstribute theta list to each worker
    thetas = [theta_list[shape[0]*i:shape[0]*(i+1)] for i in range(ncore)]
    delays = [dask.delayed(__calc_single, traverse=False)(".crystal.cif", numOfSlices, V, Nx, Ny, division, t, returnDepth) for t in thetas]
    # shape: (ncore, theta, thickness, nx, ny)
    res = [da.from_delayed(d, shape, dtype=complex) for d in delays]

    x, y = np.linspace(0, c.a, Nx), np.linspace(0, c.b, Ny)
    z = np.linspace(0, sp.dz * sp.division * numOfSlices, sp.division * numOfSlices)

    if returnDepth:
        # shape is (ncore, thetas, thickness, nx, ny)
        res = DaskWave(da.stack(res).transpose(3, 4, 2, 0, 1).reshape(Nx, Ny, sp.division * numOfSlices, -1), x, y, z, None)
        return res
    else:
        # shape is (ncore, thetas, nx, ny)
        res = DaskWave(da.stack(res).transpose(2, 3, 0, 1).reshape(Nx, Ny, -1), x, y, None)
        return res


def __calc_single(cif, numOfSlices, V, Nx, Ny, division, thetas, returnDepth):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    c = CrystalStructure.from_cif(cif)
    sp = FunctionSpace(c, Nx, Ny, division)
    b = ElectronBeam(V, 0)
    V_rs = Slices(c, sp).getPotentialTerms(b)
    pot = _Potentials(V_rs, sp.kvec, c.unit[2][0], c.unit[2][1], numOfSlices)
    res = []
    for tx, ty in thetas:
        P_k = sp.getPropagationTerm(b.getWavelength(), tx, ty)
        phi = _apply(sp.getArray(), pot, P_k*sp.mask, returnDepth)
        res.append(phi)
    return np.array(res)


class _Potentials:
    def __init__(self, V_rs, kvec, dx, dy, numOfSlices, type="precalc"):
        phase1 = np.exp(1j*kvec.dot([dx, dy]))
        if type == "precalc":
            self._pots = self.__calc_potential(V_rs, phase1, numOfSlices)

    def __calc_potential(self, V_rs, phase1, numOfSlices):
        potentials = []
        for n in range(numOfSlices):
            phase = 1 if n == 0 else phase * phase1
            for V_r in V_rs:
                potentials.append(ifft(fft(V_r) * phase))
        return potentials

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n == len(self._pots):
            raise StopIteration()
        res = self._pots[self._n]
        self._n += 1
        return res


def _apply(phi, pots, P_k, returnDepth=False):
    """
    Calculate multislice simulation.

    if returnDepth==True, ndarray of (thickness, Nx, Ny) dimension will returnd, otherwise the shape will be (Nx, Ny).

    Returns:
        numpy array: see above.
    """

    res = []
    for V_r in pots:
        phi = ifft(P_k * fft(phi * V_r))
        if returnDepth:
            res.append(phi)
    if returnDepth:
        return np.array(res)
    else:
        return phi


def makePrecessionTheta(alpha, N=90,min=0, max=360, unit="deg", angle_offset=[0,0]):
    """
    Create list of precesssion angles in degree
    Args:
        alpha(float): The precession angle in degree.
        N(int): The number of sampling points.
        unit('deg' or 'rad'): The unit of angles.
    Return:
        list of length 2 sequence: The list of angles in the form of [(theta_x1, theta_y1), (theta_x2, theta_y2), ...]
    """
    if unit == "deg":
        alpha = alpha * np.pi / 180
    theta = np.linspace(2 * np.pi / 360 * min, 2 * np.pi / 360 * max, N, endpoint=False)
    result = np.arctan(np.tan(alpha) * np.array([np.cos(theta), np.sin(theta)]))
    if unit == "deg":
        result = result * 180 / np.pi
    return result.T + np.array(angle_offset)