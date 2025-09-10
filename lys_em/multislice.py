import numpy as np
import dask.array as da
import dask

from lys import DaskWave
from . import fft, ifft, TEM, CrystalPotential, FunctionSpace


def calcMultiSliceDiffraction(c, numOfCells, V=60e3, Nx=128, Ny=128, division="Auto", theta_list=[[0, 0]], returnDepth=True):
    sp = FunctionSpace.fromCrystal(c, Nx, Ny, numOfCells, division=division)

    # prepare potential
    tem = TEM(V)
    pot = CrystalPotential(sp, c)

    # prepare list of thetas
    ncore = len(DaskWave.client.ncores()) if hasattr(DaskWave,"client") else 1
    shape = (int(len(theta_list)/ncore), Nx, Ny, sp.N[2]) if returnDepth else (int(len(theta_list)/ncore), Nx, Ny)
    thetas = [theta_list[shape[0]*i:shape[0]*(i+1)] for i in range(ncore)]

    # Dstribute all tasks to each worker
    delays = [dask.delayed(__calc_single, traverse=False)(sp, pot, tem, t, returnDepth) for t in thetas]

    res = [da.from_delayed(d, shape, dtype=complex) for d in delays]
    x, y = np.linspace(0, c.a, Nx), np.linspace(0, c.b, Ny)
    z = np.linspace(0, sp.c, sp.N[2])

    if returnDepth:
        # shape is (mcore, thetas, thickness, nx, ny)
        res = DaskWave(da.stack(res).transpose(3, 4, 2, 0, 1).reshape(*sp.N, -1), x, y, z, None)
        return res
    else:
        # shape is (thetas, ncore, nx, ny)
        res = DaskWave(da.stack(res).transpose(2, 3, 0, 1).reshape(Nx, Ny, -1), x, y, None)
        return res


def calcMultiSliceDiffraction2(c, numOfCells, V=60e3, Nx=128, Ny=128, division="Auto", theta_list=[[0, 0]], returnDepth=True):
    sp = FunctionSpace.fromCrystal(c, Nx, Ny, numOfCells, division=division)
    tem = TEM(V)
    pot = CrystalPotential(sp, c)  
    return __calc_single(sp, pot, tem, theta_list, returnDepth).transpose(1,2,0)


def __calc_single(sp, pot, tem, thetas, returnDepth):
    """
    Caluclate multislice simulations for list of thetas.
    The shape of returned array will be (Thetas, Nx, Ny) if returnDepth is True, otherwise (Thetas, thickness, Nx, Ny)
    """
    res = []
    pot_data = pot.get(tem)
    for tx, ty in thetas:
        P_k = sp.getPropagationTerm(tem.wavelength, tx, ty)
        phi = _apply(sp.getArray(), pot_data, P_k*sp.mask, returnDepth)
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