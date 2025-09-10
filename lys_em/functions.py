import numpy as np
import itertools
from . import TEM, TEMParameter, FunctionSpace, CrystalPotential, multislice


def calcSADiffraction(V, crys, numOfCells, Nx=128, Ny=128, division="Auto", tilt=[0, 0]):
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    params = [TEMParameter(tilt=tilt)]
    return abs(np.fft.fft2(multislice(sp, pot, tem, params)[0]))**2


def calcPrecessionDiffraction(V, crys, numOfCells, theta, nphi, Nx=128, Ny=128, division="Auto", sum=True):
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    params = [TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, nphi / 360)]
    res = abs(np.fft.fft2(multislice(sp, pot, tem, params), axes=(1, 2)))**2
    if sum:
        return res.sum(axis=0)
    else:
        return res


def calcCBED(V, convergence, crys, numOfCells, ndisk=30, Nx=128, Ny=128, division="Auto", sum=True):
    # convergence is in radians
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    c = np.arange(-convergence, convergence, 2 * convergence / ndisk)
    params = [TEMParameter(tilt=[tx, ty]) for tx, ty in itertools.product(c, c)]
    res = abs(np.fft.fft2(multislice(sp, pot, tem, params), axes=(1, 2)))**2
    res.reshape(ndisk, ndisk, Nx, Ny)
    return res


def calc4DSTEM_Crystal(V, convergence, crys, numOfCells, Nx=256, Ny=256, division="Auto", scanx=16, scany=16):
    tem = TEM(V, convergence=convergence)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys)
    c = np.arange(0, crys.a, crys.a / scanx)
    r = np.arange(0, crys.b, crys.b / scany)
    params = [TEMParameter(beamPosition=[x, y]) for x, y in itertools.product(c, r)]
    res = abs(np.fft.fft2(multislice(sp, pot, tem, params), axes=(1, 2)))**2
    res.reshape(scanx, scany, Nx, Ny)
    return res
