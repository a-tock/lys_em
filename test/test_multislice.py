import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from lys_mat import CrystalStructure, CrystalStructureIO, Atom
from lys_em.scatteringFactor import projectedPotential
from lys_em.multislice import FunctionSpace, Slices, calcMultiSliceDiffraction, _apply, m, _Potentials


class MultiSlice_test(unittest.TestCase):
    path = "test/DataFiles"

    def setUp(self):
        #self.b = ElectronBeam(60e3, 0)

        self.Au = CrystalStructureIO.loadFrom(self.path + "/Au.cif")
        _TaTe2 = CrystalStructureIO.loadFrom(self.path + "/TaTe2.cif")
        self.TaTe2 = _TaTe2.createSupercell(np.array([[1, 0, 0], [0, 1, 0], [1, 0, 3]]))
        _TaTe2_1T = CrystalStructureIO.loadFrom(self.path + "/TaTe2_1T.cif")
        self.TaTe2_1T = _TaTe2_1T.createSupercell(np.array([[-3, -6, 0], [1, 0, 0], [0, 0, 3]]))
        self.Au_single = CrystalStructure([20, 20, 20, 90, 90, 90], [Atom("Au", (0, 0, 0))])

    def test_FunctionSpace(self):
        sp = FunctionSpace(self.Au, 10, 10)
        ref = np.linalg.norm(self.Au.InverseLatticeVectors()[0])

        assert_array_almost_equal(sp.kvec[0, 0], [0, 0])
        assert_array_almost_equal(sp.kvec[0, 1], [0, ref])
        assert_array_almost_equal(sp.kvec[1, 0], [ref, 0])
        assert_array_almost_equal(sp.kvec[2, 0], [ref * 2, 0])

        sp = FunctionSpace(self.Au.createSupercell([2, 1, 1]), 5, 3)
        self.assertAlmostEqual(sp.kx[1], ref / 2)
        self.assertAlmostEqual(sp.ky[1], ref)

        sp = FunctionSpace(self.Au_single, 512 * 2, 512 * 2, division=1)
        sp.division * sp.dz == self.TaTe2.unit[2][2]

    def test_Slices(self):
        # check slicing
        sp = FunctionSpace(self.Au, 10, 10, division=9)
        slices = Slices(self.Au, sp)
        n = np.sum([len(s.atoms) for s in slices._slices])
        self.assertEqual(n, len(self.Au.atoms))

        sp = FunctionSpace(self.TaTe2, 10, 10, division=9)
        slices = Slices(self.TaTe2, sp)
        n = np.sum([len(s.atoms) for s in slices._slices])
        self.assertEqual(n, len(self.TaTe2.atoms))

        # check single atom potential
        N = 1024  # if this value is too small, this test fails because of window function in Function Space.
        sp = FunctionSpace(self.Au_single, N, N, division=1)
        V_k = Slices(self.Au_single, sp)._calculatePotential(self.Au_single) * self.b.getWavelength() * self.b.getRelativisticMass() / m  # A^2
        V_r = sp.IFT(V_k * sp.mask)

        r = np.linspace(0, self.Au_single.a, N, endpoint=False)
        sf = projectedPotential("Au", r) / self.b.getSigma0() * self.b.getSigma()
        self.assertAlmostEqual(V_r[0][0], sf[0])
        self.assertAlmostEqual(V_r[0][1], sf[1])
        self.assertAlmostEqual(V_r[0][2], sf[2])

        # check potential calculation
        sp = FunctionSpace(self.Au, 10, 10, division=1)
        slices = Slices(self.Au, sp)
        pot = slices._calculatePotential(slices._slices[0])
        self.assertTrue(np.abs((pot[0, 0] / pot[0, 2]) - 1.725) < 1e-3)
        self.assertTrue(np.abs(pot[0, 1]) < 1e-5)

    def test_TaTe2(self):
        # Compare result with pre-calculated results at 2023/12/14.
        res = calcMultiSliceDiffraction(self.Au, 10, returnDepth=False)
        assert_array_almost_equal(res.compute()[:,:,0].data, np.load(self.path+"/Au.npy"))

        res = calcMultiSliceDiffraction(self.TaTe2, 10, returnDepth=False)
        assert_array_almost_equal(res.compute()[:,:,0].data, np.load(self.path+"/TaTe2.npy"))

        res = calcMultiSliceDiffraction(self.TaTe2_1T, 10, returnDepth=False)
        assert_array_almost_equal(res.compute()[:,:,0].data, np.load(self.path+"/TaTe2_1T.npy"))

    def test_Propagation(self):
        crys = CrystalStructure(self.Au.cell, [])
        sp = FunctionSpace(crys, 128, 128, 10)
        b = ElectronBeam(60e3, 0)
        V_rs = Slices(crys, sp).getPotentialTerms(b)
        pot = _Potentials(V_rs, sp.kvec, crys.unit[2][0], crys.unit[2][1], 1)

        phi = np.zeros((128, 128))
        phi[0][0] = 1

        P_k = sp.getPropagationTerm(b.getWavelength())
        phi = _apply(phi, pot, P_k*sp.mask)

        # theoretical solution (propagation after a in free space)
        calcphi = np.fft.ifft2(sp._mask * np.exp(-1j * self.b.getWavelength() * self.Au.a * sp.k2 / 4 / np.pi))

        assert_array_almost_equal(phi, calcphi)
