import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from lys_mat import CrystalStructure
from lys_em import debyeWallerFactors


class kinematical_test(unittest.TestCase):
    path = "test/DataFiles"

    def test_debyeWallerFactor(self):
        cif = self.path + "/VTe2.cif"
        c = CrystalStructure.loadFrom(cif)
        for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
            q = np.array(hkl).dot(c.inv)
            assert_array_almost_equal(debyeWallerFactors(c, q), self._calc_DW(c, hkl))

    @staticmethod
    def _calc_DW(c, hkl):
        res = []
        for at in c.atoms:
            U = at.Uani
            abc = np.linalg.norm(c.InverseLatticeVectors(), axis=1)
            res.append(np.exp(-np.einsum("ij,i,j,i,j->", U, hkl, hkl, abc, abc) / 2))
        return np.array(res)
