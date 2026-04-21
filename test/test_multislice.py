import unittest
import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal, assert_allclose

from lys_mat import CrystalStructure, Atom
from lys_em import TEM, TEMParameter, FunctionSpace  # , calcSADiffraction, calcPrecessionDiffraction
from lys_em.consts import m, e, h, hbar
from lys_em.scatteringFactor import projectedPotential
from lys_em.multislice import getPropagationTerm, multislice, getChi
from lys_em.functions import diffraction
from lys_em.potentials.crystalPotential import CrystalPotential

jax.config.update("jax_enable_x64", True)

class MultiSlice_test(unittest.TestCase):
    path = "test/DataFiles"

    def setUp(self):
        self.Au = CrystalStructure.loadFrom(self.path + "/Au.cif")
        self.VTe2_ortho = CrystalStructure.loadFrom(self.path + "/VTe2_ortho.cif")
        self.VTe2_trigonal = CrystalStructure.loadFrom(self.path + "/VTe2_trigonal.cif")
        self.Au_single = CrystalStructure([20, 20, 20, 90, 90, 90], [Atom("Au", (0, 0, 0))])

    def test_FunctionSpace(self):
        sp = FunctionSpace.fromCrystal(self.Au, 10, 10, 1)
        ref = np.linalg.norm(self.Au.InverseLatticeVectors()[0])

        assert_array_almost_equal(sp.kvec[0, 0], [0, 0])
        assert_array_almost_equal(sp.kvec[0, 1], [0, ref])
        assert_array_almost_equal(sp.kvec[1, 0], [ref, 0])
        assert_array_almost_equal(sp.kvec[2, 0], [ref * 2, 0])

#        sp = FunctionSpace.fromCrystal(self.Au_single, 512 * 2, 512 * 2, 1)

    def test_Propagation(self):
        a = 3

        sp = FunctionSpace(a, a, a, Nz=10)
        tem = TEM(60e3)
        spdict = sp.asdict()
        temdict, _, _ = tem.asdict(len(jax.devices()))
        temdict = {key: item[0] for key, item in temdict.items()}

        phi = np.zeros((128, 128), dtype=np.complex64)
        phi[0, 0] = 1
        P_k = np.array(getPropagationTerm(spdict, temdict))

        # compared with theoretical solution (propagation after a in free space)
        for i in range(10):
            phi = np.fft.ifft2(np.fft.fft2(phi) * P_k)
        calcphi = np.fft.ifft2(sp.mask * np.exp(-1j * tem.wavelength * a * sp.k**2 / 4 / np.pi))

        assert_array_almost_equal(phi, calcphi)

    def test_TaTe2(self):
        # Compare result with pre-calculated results at 2023/12/14.
        sp = FunctionSpace.fromCrystal(self.Au, Nx=128, Ny=128, ncells=10)
        pot = CrystalPotential(sp, self.Au)
        tem = TEM(60e3)
        res = diffraction(multislice(pot, tem))
        test = abs(np.fft.fft2(np.load(self.path + "/Au.npy")))**2
        assert_array_almost_equal(res / res[0][0], test / test[0][0])

        TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        sp = FunctionSpace.fromCrystal(TaTe2, Nx=512, Ny=256, ncells=90)
        pot = CrystalPotential(sp, TaTe2)
        tem = TEM(0.75e6, params=TEMParameter(tilt=[-0.80437, 0]))
        res = diffraction(multislice(pot, tem))
        res = jnp.sqrt(res)[:13, 0]
        ans = np.array([0.5577267344048619, 0, 0.002756932398382368, 0, 0.010163465792031692, 0, 0.11391665208304297, 0, 0.0031967300829784428, 0, 0.0024738436644140016, 0, 0.21191082484442536])
        assert_array_almost_equal(res / res[0], ans / ans[0], decimal=4)

    def test_TaTe2_prec(self):
        TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
#        res = fitPrecessionDiffraction(0.75e6, TaTe2, 50, 2, 180, Nx=256, Ny=64, division=1)

    def test_nonorthogonal(self):
        # Relative error of intensity of a specific index in orthogonal and non-orthogonal systems
        tem = TEM(200e3)

        sp_ortho = FunctionSpace.fromCrystal(self.VTe2_ortho, Nx=216, Ny=216, ncells=50, division=1)
        pot_ortho = CrystalPotential(sp_ortho, self.VTe2_ortho)
        calc_ortho = diffraction(multislice(pot_ortho, tem))

        sp_trigonal = FunctionSpace.fromCrystal(self.VTe2_trigonal, Nx=216, Ny=216, ncells=50, division=1)
        pot_trigonal = CrystalPotential(sp_trigonal, self.VTe2_trigonal)
        calc_trigonal = diffraction(multislice(pot_trigonal, tem))

        relative_error = np.abs(calc_ortho[1, 1] - calc_trigonal[1, 0]) / np.abs(calc_ortho[1, 1])
        self.assertTrue(np.all(relative_error < 1e-4))

    def test_PED(self):
        # Compare result with pre-calculated results at 2025/9/12.
        sp = FunctionSpace.fromCrystal(self.Au, Nx=128, Ny=128, ncells=20, division=1)
        pot = CrystalPotential(sp, self.Au)
        tem = TEM(200e3, params=[TEMParameter(tilt=[2, phi]) for phi in np.arange(0, 360, 360 / 60)])
        res = diffraction(multislice(pot, tem)).sum(axis=0)
#        np.save(self.path + "/Au_PED.npy", res)
        ans = np.load(self.path + "/Au_PED.npy")
        assert_array_almost_equal(res / res[0][0], ans / ans[0][0], decimal=4)

    def test_data(self):
        # Compare result with pre-calculated results at 2025/9/12.
        sp = FunctionSpace.fromCrystal(self.Au, Nx=128, Ny=128, ncells=10, division=1)
        pot = CrystalPotential(sp, self.Au)
        tem = TEM(200e3, params=[TEMParameter(tilt=[2, phi]) for phi in np.arange(0, 360, 360 / 30)])
        data = np.load(self.path + "/Au_PED_for_using_data.npy")

        res = multislice(pot, tem, sum=True, postprocess="diffraction")
        assert_allclose(res, data.sum(axis=0), atol=1e-6, rtol=1e-6)

        res = multislice(pot, tem, sum=True, postprocess=lambda phi, dat: jnp.abs(diffraction(phi)-dat), data=data)/(data.sum(axis=0)+1e-10)
        assert_allclose(res, 0, atol=1e-5)

    def test_grad(self):

        Cs = 1e10*1e-6
        defocus = 1e10*1e-6
        crys = CrystalStructure.loadFrom(self.path + "/TiO2_rutile.cif")
        sp = FunctionSpace.fromCrystal(crys, Nx=128, Ny=128, ncells=30, division=1)
        pot = CrystalPotential(sp, crys)
        tem = TEM(200e3, Cs = 1e10*1e-6, params=[TEMParameter(tilt=[0., 0], defocus=1e10*1e-6)])

        ref = diffraction(multislice(pot, tem)).real

        def calc_intensity(y_O4, Cs, theta, defocus):
            # pot_subs = pot.replace(params={"y_O4": y_O4})
            tem_subs = tem.replace(Cs=Cs, params=[TEMParameter(tilt=[theta, 0], defocus=defocus)])
            return diffraction(multislice(pot, tem_subs)).real

        def func_for_grad(values):
            return jnp.sum((ref - calc_intensity(*values))**2)/jnp.sum(ref**2)

        grad = jax.grad(func_for_grad)
        print(jnp.array(func_for_grad([1.9505, Cs*1000, 0., defocus*10])))
        print(jnp.array(grad([1.9505, Cs*1000, 0., defocus*10])))   #[-2.888e-02  5.261e-10 -1.283e-03  1.292e-05] ?

    def test_mesh(self):
        sp = FunctionSpace.fromCrystal(self.Au, Nx=128, Ny=128, ncells=10, division=1)
        pot = CrystalPotential(sp, self.Au)
        tem = TEM(200e3, params=[TEMParameter(tilt=[2, phi]) for phi in np.arange(0, 360, 360 / 30)])
        mesh = jax.sharding.Mesh(jax.devices(), ('j',))
        res = multislice(pot, tem, mesh=mesh)


class CrystalPotential_test(unittest.TestCase):
    path = "test/DataFiles"

    def setUp(self):
        self.Au = CrystalStructure.loadFrom(self.path + "/Au.cif")
        _TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        self.TaTe2 = _TaTe2.createSupercell(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]]).T)
        self.Au_single = CrystalStructure([20, 20, 20, 90, 90, 90], [Atom("Au", (0, 0, 0))])

    def test_CrystalPotential(self):
        # check single atom potential
        N = 1024  # if this value is too small, this test fails because of window function in Function Space.

        sp = FunctionSpace.fromCrystal(self.Au_single, N, N, 1, division=2)
        pot = CrystalPotential(sp, self.Au_single)
        V_k = pot.getSlicePotential(self.Au_single)[0]
        V_r = np.fft.ifft2(V_k * sp.mask) / sp.dV

        r = np.linspace(0, self.Au_single.a, N, endpoint=False)
        sf = projectedPotential("Au", r)

        assert_allclose(V_r[0][0], sf[0])
        assert_allclose(V_r[0][1], sf[1])
        assert_allclose(V_r[0][2], sf[2])

        # check potential calculation
        sp = FunctionSpace.fromCrystal(self.Au, 10, 10, 1, division=1)
        pot = CrystalPotential(sp, self.Au).getSlicePotential(self.Au)[0]
        self.assertTrue(np.abs((pot[0, 0] / pot[0, 2]) - 1.725) < 1e-3)
        self.assertTrue(np.abs(pot[0, 1]) < 1e-5)
