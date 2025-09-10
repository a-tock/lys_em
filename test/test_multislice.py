import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from lys_mat import CrystalStructure,Atom
from lys_em import TEM, FunctionSpace
from lys_em.consts import m, e, h, hbar
from lys_em.scatteringFactor import projectedPotential
from lys_em.multislice import calcMultiSliceDiffraction, _apply
from lys_em.potentials.crystalPotential import _Slices


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
        res = calcMultiSliceDiffraction(self.Au, 10, returnDepth=False)
        assert_array_almost_equal(res.compute()[:,:,0].data, np.load(self.path+"/Au.npy"))
    
        TaTe2 = CrystalStructure.loadFrom(self.path + "/TaTe2.cif")
        res = calcMultiSliceDiffraction(TaTe2, 90, V=0.75e6, Nx=512, Ny=256, returnDepth=False, theta_list=[[-0.80437, 0]])
        res = abs(np.fft.fft2(res.compute().data[:, :, 0]))
        assert_array_almost_equal(res[:13,0], [0.5577267344048619, 0, 0.002756932398382368, 0, 0.010163465792031692, 0, 0.11391665208304297, 0, 0.0031967300829784428, 0, 0.0024738436644140016, 0, 0.21191082484442536])

    def test_Propagation(self):
        a = 3
        l = TEM(60e3).wavelength

        sp = FunctionSpace(a,a,a,Nz=10)
        phi = np.zeros((128, 128))
        phi[0][0] = 1

        P_k = sp.getPropagationTerm(l)
        phi = _apply(phi, [1]*10, P_k*sp.mask)

        # theoretical solution (propagation after a in free space)
        calcphi = np.fft.ifft2(sp._mask * np.exp(-1j * l * a * sp.k2 / 4 / np.pi))

        assert_array_almost_equal(phi, calcphi)

    def test_nonorthogonal(self):
        # Relative error of intensity of a specific index in orthogonal and non-orthogonal systems
        calc_ortho = calcMultiSliceDiffraction(self.VTe2_ortho, numOfCells=50, V=200e3, Nx=216, Ny=216, division=1, theta_list=[[0, 0]], returnDepth=True)
        fourior_ortho = np.fft.fft2(calc_ortho, axes=(0, 1))
        cal_ortho = (np.absolute(fourior_ortho)**2).sum(axis=3)

        calc_trigonal = calcMultiSliceDiffraction(self.VTe2_trigonal, numOfCells=50, V=200e3, Nx=216, Ny=216, division=1, theta_list=[[0, 0]], returnDepth=True)
        fourior_trigonal = np.fft.fft2(calc_trigonal, axes=(0, 1))
        cal_trigonal = (np.absolute(fourior_trigonal)**2).sum(axis=3)
        relative_error = np.abs(cal_ortho[1, 1]-cal_trigonal[1, 0]) / np.abs(cal_ortho[1, 1])
        self.assertTrue(np.all(relative_error < 1e-4))


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
        V_k = _Slices(self.Au_single, sp)._calculatePotential(self.Au_single) * b.wavelength * b.relativisticMass / m  # A^2
        V_r = np.fft.ifft2(V_k * sp.mask)/ sp.dV

        r = np.linspace(0, self.Au_single.a, N, endpoint=False)

        M, lamb = b.relativisticMass, b.wavelength
        sigma = 2 * np.pi * M * e * lamb / h**2  # kg C m /eV^2 s^2
        sigma0 = 2 * np.pi * m * e / h**2  # kg C m /eV^2 s^2

        sf = projectedPotential("Au", r) / sigma0 * sigma
        self.assertAlmostEqual(V_r[0][0], sf[0])
        self.assertAlmostEqual(V_r[0][1], sf[1])
        self.assertAlmostEqual(V_r[0][2], sf[2])

        # check potential calculation
        sp = FunctionSpace.fromCrystal(self.Au, 10, 10, 1, division=1)
        slices = _Slices(self.Au, sp)
        pot = slices._calculatePotential(slices._slices[0])
        self.assertTrue(np.abs((pot[0, 0] / pot[0, 2]) - 1.725) < 1e-3)
        self.assertTrue(np.abs(pot[0, 1]) < 1e-5)
