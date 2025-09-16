import itertools

import numpy as np
import jax
import jax.numpy as jnp

from . import TEM, TEMParameter, FunctionSpace, CrystalPotential, multislice


def calcSADiffraction(V, crys, numOfCells, Nx=128, Ny=128, division="Auto", tilt=[0, 0]):
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys).get(tem)
    params = [TEMParameter(tilt=tilt)]

    return diffraction(multislice(sp, pot, tem, params))[0]


def calcPrecessionDiffraction(V, crys, numOfCells, theta, nphi, Nx=128, Ny=128, division="Auto", sum=True):
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(sp, crys).get(tem)
    params = [TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, 360 / nphi)]

    return diffraction(multislice(sp, pot, tem, params)).sum(axis=0)


def fitPrecessionDiffraction(V, crys, numOfCells, theta, nphi, Nx=128, Ny=128, division="Auto"):
    import time
    start = time.time()
    tem = TEM(V)
    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    cpot = CrystalPotential(sp, crys)
    params = [TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, 360 / nphi)]

    data = diffraction(multislice(sp, cpot.get(tem), tem, params)).sum(axis=0).block_until_ready()
    print(time.time()-start)
    return

    def R(unit, pos, Uani):
        p = cpot.getFromParameters(tem, unit, pos, Uani)
        return jnp.sum((data - diffraction(multislice(sp, p, tem, params)).sum(axis=0))**2)

    g = jax.grad(R, argnums=(0,1))
    gr = g(crys.unit, crys.getAtomicPositions(), np.array([at.Uani for at in crys.atoms]))
    print(gr)


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


def diffraction(data):
    return abs(jnp.fft.fft2(data))**2