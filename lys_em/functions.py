import itertools

import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as optimize
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


def fitPrecessionDiffraction(I_hkl, crys, initials, V, theta, nphi, numOfCells, Nx=128, Ny=128, division="Auto", bounds=None, method="L-BFGS-B", tol=1e-2):
    """
    Fit the precession diffraction data with the given crystal structure and parameters.

    Args:
        I_hkl(dict): A dictionary where keys are indices (h, k, l) and values are the corresponding intensities to be fitted.
        crys (CrystalStructure) : The crystal structure to be used.
        paramnames (list of str) : The names of the parameters to be fitted.
        initials (list of float) : The initial values of the parameters to be fitted.
        V (float) : The acceleration voltage of the TEM.
        theta (float) : The precession angle of the TEM.
        nphi (int) : The number of divisions of the precession angle.
        numOfCells (int) : The number of cells in the multislice simulation.
        bounds (array_like of tuple, optional) : The bounds of the parameters to be fitted, if None, it will be set to [-inf, inf].
        method (str, optional) : The optimization method to be used, default is "L-BFGS-B".
        tol (float, optional) : The tolerance of the optimization, default is 1e-4.
        Nx (int, optional) : The number of points in the x direction of the multislice simulation, default is 128.
        Ny (int, optional) : The number of points in the y direction of the multislice simulation, default is 128.
        division (str or int, optional) : The division method of the multislice simulation, default is "Auto".

    Returns:
        all_params (dict) : The fitted parameters.
        res (OptimizeResult) : The result of the optimization.
    """

    spc = FunctionSpace.fromCrystal(crys, Nx, Ny, numOfCells, division=division)
    pot = CrystalPotential(spc, crys)
    tem = TEM(V, params=[TEMParameter(tilt=[theta, phi]) for phi in np.arange(0, 360, 360 / nphi)])

    def calc_intensity(param):
        pot_subs = pot.replace(params=param)
        return diffraction(multislice(pot_subs, tem)).sum(axis=0)

    def residual(values):
        calc = calc_intensity({name: value for name, value in zip(crys.free_symbols, values)})
        calc_arr = jnp.array([calc[int(indice[0]), int(indice[1])] for indice in I_hkl.keys()])

        data = jnp.array(list(I_hkl.values()))
        scale = jnp.sum(data * calc_arr) / jnp.sum(calc_arr**2)
        R = jnp.linalg.norm(data - scale * calc_arr, ord=1) / jnp.linalg.norm(data, ord=1)

        if not isinstance(scale, jax.core.Tracer):
            param = {name: float(f"{val:.4g}") for name, val in zip(crys.free_symbols, values)}
            print(f"Step R = {R:.4f}, scale = {scale:.4g}, params = {param}")
        return jnp.array(R)

    init = np.array([initials[str(name)] for name in crys.free_symbols])
    bnd = np.array([bounds[str(name)] if bounds is not None else (-np.inf, np.inf) for name in crys.free_symbols])
    res = optimize.minimize(residual, init, bounds=bnd, jac=jax.grad(residual), method=method, tol=tol)

    return {name: val for name, val in zip(crys.free_symbols, res.x)}


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
