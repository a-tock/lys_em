import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from lys_mat import CrystalStructure, Atom
from lys_em import TEM, TEMParameter, FunctionSpace, calcSADiffraction, fitPrecessionDiffraction, calcPrecessionDiffraction
from lys_em.consts import m, e, h, hbar
from lys_em.scatteringFactor import projectedPotential
from lys_em.multislice import getPropagationTerm
from lys_em.potentials.crystalPotential import _Slices, CrystalPotential


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

        sp = FunctionSpace.fromCrystal(self.Au_single, 512 * 2, 512 * 2, 1)

    def test_TaTe2(self):
        # Compare result with pre-calculated results at 2023/12/14.
        res = calcSADiffraction(60e3, self.Au, 10, Nx=128, Ny=128)
        test = abs(np.fft.fft2(np.load(self.path + "/Au.npy")))**2
        assert_array_almost_equal(res / res[0][0], test / test[0][0])

        TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        res = calcSADiffraction(0.75e6, TaTe2, 90, Nx=512, Ny=256, tilt=[-0.80437, 0])
        res = np.sqrt(res)[:13, 0]
        ans = np.array([0.5577267344048619, 0, 0.002756932398382368, 0, 0.010163465792031692, 0, 0.11391665208304297, 0, 0.0031967300829784428, 0, 0.0024738436644140016, 0, 0.21191082484442536])
        assert_array_almost_equal(res / res[0], ans / ans[0], decimal=4)

    def test_TaTe2_prec(self):
        TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        res = fitPrecessionDiffraction(0.75e6, TaTe2, 50, 2, 180, Nx=256, Ny=64, division=1)

    def test_Propagation(self):
        a = 3

        sp = FunctionSpace(a, a, a, Nz=10)
        tem = TEM(60e3)

        phi = np.zeros((128, 128), dtype=np.complex64)
        phi[0, 0] = 1
        P_k = np.array(getPropagationTerm(sp, tem, [TEMParameter()])[0])

        # compared with theoretical solution (propagation after a in free space)
        for i in range(10):
            phi = np.fft.ifft2(np.fft.fft2(phi) * P_k)
        calcphi = np.fft.ifft2(sp.mask * np.exp(-1j * tem.wavelength * a * sp.k2 / 4 / np.pi))

        assert_array_almost_equal(phi, calcphi)

    def test_nonorthogonal(self):
        # Relative error of intensity of a specific index in orthogonal and non-orthogonal systems
        calc_ortho = calcSADiffraction(200e3, self.VTe2_ortho, 50, Nx=216, Ny=216, division=1)
        calc_trigonal = calcSADiffraction(200e3, self.VTe2_trigonal, 50, Nx=216, Ny=216, division=1)
        relative_error = np.abs(calc_ortho[1, 1] - calc_trigonal[1, 0]) / np.abs(calc_ortho[1, 1])
        self.assertTrue(np.all(relative_error < 1e-4))

    def test_PED(self):
        # Compare result with pre-calculated results at 2025/9/12.
        res = calcPrecessionDiffraction(200e3, self.Au, 50, 2, 360, Nx=128, Ny=128, division=1)
        ans = np.load(self.path + "/Au_PED.npy")
        assert_array_almost_equal(res / res[0][0], ans / ans[0][0])


class CrystalPotential_test(unittest.TestCase):
    path = "test/DataFiles"

    def setUp(self):
        self.Au = CrystalStructure.loadFrom(self.path + "/Au.cif")
        _TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        self.TaTe2 = _TaTe2.createSupercell(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]]).T)
        self.Au_single = CrystalStructure([20, 20, 20, 90, 90, 90], [Atom("Au", (0, 0, 0))])

    def test_Slices(self):
        # check slicing
        sp = FunctionSpace.fromCrystal(self.Au, 10, 10, 1, division=9)
        slices = _Slices(self.Au, sp)
        n = np.sum([len(s.atoms) for s in slices._slices])
        self.assertEqual(n, len(self.Au.atoms))

        sp = FunctionSpace.fromCrystal(self.TaTe2, 10, 10, 1, division=9)
        slices = _Slices(self.TaTe2, sp)
        n = np.sum([len(s.atoms) for s in slices._slices])
        self.assertEqual(n, len(self.TaTe2.atoms))

        # check single atom potential
        N = 1024  # if this value is too small, this test fails because of window function in Function Space.
        b = TEM(60e3, 0)

        sp = FunctionSpace.fromCrystal(self.Au_single, N, N, 1, division=2)
        pot = CrystalPotential(sp, self.Au_single)
        V_k = _Slices(self.Au_single, sp).getPotential(b, self.Au_single.unit, self.Au_single.getAtomicPositions(), [at.Uani for at in self.Au_single.atoms])[0]
        V_r = np.fft.ifft2(V_k * sp.mask) / sp.dV

        r = np.linspace(0, self.Au_single.a, N, endpoint=False)

        M, lamb = b.relativisticMass, b.wavelength
        sigma = 2 * np.pi * M * e * lamb / h**2  # kg C m /eV^2 s^2
        sigma0 = 2 * np.pi * m * e / h**2  # kg C m /eV^2 s^2

        sf = projectedPotential("Au", r) / sigma0 * sigma
        self.assertAlmostEqual(V_r[0][0], sf[0], places=5)
        self.assertAlmostEqual(V_r[0][1], sf[1], places=5)
        self.assertAlmostEqual(V_r[0][2], sf[2], places=5)

        # check potential calculation
        sp = FunctionSpace.fromCrystal(self.Au, 10, 10, 1, division=1)
        slices = _Slices(self.Au, sp)
        pot = slices.getPotential(b, self.Au.unit, self.Au.getAtomicPositions(), [at.Uani for at in self.Au.atoms])[0]
        self.assertTrue(np.abs((pot[0, 0] / pot[0, 2]) - 1.725) < 1e-3)
        self.assertTrue(np.abs(pot[0, 1]) < 1e-5)
