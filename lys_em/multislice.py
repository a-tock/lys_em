import numpy as np
import dask.array as da
import dask

from lys import DaskWave
from lys_mat import CrystalStructure
from . import fft, ifft, TEM, CrystalPotential


class FunctionSpace:
    """
    Create 2 dimensional rectangular grid function space that is defined by the crystal structure.
    The length of the rectangular space is defined by the crys.a and crys.b
    Each cell of the grid function space is (crys.a/Nx crys.b/Ny)
    """
    def __init__(self, crys, Nx, Ny, division="Auto"):
        self._unit = np.array([crys.a, crys.b, crys.gamma])
        self._unitall = crys.unit
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
            max = np.sqrt(self._getMax())
            self._mask = np.where(k > max * 2 / 3, 0, 1)
        return self._mask

    def _getMax(self):
        k2_row0 = self.k2[self._N[0]//2, :]
        k2_col0 = self.k2[:, self._N[1]//2]
        return min(min(k2_row0), min(k2_col0))

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
        matrix = self._unitall
        inverse_matrix = 2 * np.pi * np.linalg.inv(matrix)
        inverse_matrix_2d = inverse_matrix[:2, :2]
        grid = self._create_grid()
        if not hasattr(self, "_kvec"):
            self._kvec = np.dot(grid, inverse_matrix_2d.T)
        return self._kvec

    def _create_grid(self):
        x = np.arange(-self._N[0]//2, self._N[0]//2)
        shift_x = np.roll(x, self._N[0]//2)
        y = np.arange(-self._N[1]//2, self._N[1]//2)
        shift_y = np.roll(y, self._N[1]//2)
        grid = np.array(np.meshgrid(shift_x, shift_y)).transpose(2, 1, 0)
        return grid

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
        return np.sin(self._unit[2] * np.pi / 180) * self._unit[0] * self._unit[1] / self._N[0] / self._N[1]

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


def calcMultiSliceDiffraction(c, numOfSlices, V=60e3, Nx=128, Ny=128, division="Auto", theta_list=[[0, 0]], returnDepth=True):
    sp = FunctionSpace(c, Nx, Ny, division)

    # prepare potential
    tem = TEM(V)
    pot = CrystalPotential(sp, tem, c, numOfSlices)

    # prepare list of thetas
    ncore = len(DaskWave.client.ncores()) if hasattr(DaskWave,"client") else 1
    shape = (int(len(theta_list)/ncore), Nx, Ny, numOfSlices * sp.division) if returnDepth else  (int(len(theta_list)/ncore), Nx, Ny)
    thetas = [theta_list[shape[0]*i:shape[0]*(i+1)] for i in range(ncore)]

    # Dstribute all tasks to each worker
    delays = [dask.delayed(__calc_single, traverse=False)(sp, pot, tem, t, returnDepth) for t in thetas]

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


def __calc_single(sp, pot, tem, thetas, returnDepth):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    res = []
    for tx, ty in thetas:
        P_k = sp.getPropagationTerm(tem.wavelength, tx, ty)
        phi = _apply(sp.getArray(), pot, P_k*sp.mask, returnDepth)
        res.append(phi)
    return np.array(res)


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