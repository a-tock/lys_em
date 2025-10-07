import itertools

import numpy as np
import jax
import jax.numpy as jnp
from . import TEM, TEMParameter, FunctionSpace, CrystalPotential, multislice


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


def fitPrecessionDiffraction(V, crys, numOfCells, theta, nphi, Nx=128, Ny=128, division="Auto"):
    import time
    start = time.time()

    sp = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    cpot = CrystalPotential(sp, crys)
    tem = TEM(V, params=[TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, 360 / nphi)])

    data = diffraction(multislice(cpot, tem)).sum(axis=0).block_until_ready()
    print("Prec total: ", time.time() - start)

    def R(unit, pos, Uani):
        pot = cpot.replace(unit=unit, pos=pos, Uani=Uani)
        return jnp.sum((data - diffraction(multislice(pot, tem)).sum(axis=0))**2)

    g = jax.grad(R, argnums=(0, 1))
    gr = g(crys.unit, [at.position for at in crys.atoms] , [at.Uani for at in crys.atoms])
    print(gr)
    gr = g(crys.unit*0.999, [at.position for at in crys.atoms] , [at.Uani for at in crys.atoms])
    print(gr)
    print("Grad: ", time.time() - start)


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
