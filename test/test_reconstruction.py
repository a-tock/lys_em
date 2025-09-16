import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from lys_mat import CrystalStructure
from lys_em import structureFactors, TEM, TEMParameter, calcKinematicalDiffraction


class kinematical(unittest.TestCase):
    path = "test/DataFiles"

    def test_WSe2(self):
        return

        cif = self.path + "/WSe2_AB.cif"
        c = CrystalStructure.loadFrom(cif)

        res = []
        for alpha in np.linspace(-70, 70, 140 * 3 + 1):
            tem = TEM(200e3)
            param = TEMParameter(tilt=[alpha, 0])
            I = calcKinematicalDiffraction(c, tem, param, 1, Nx=15, Ny=15)
            res.append(I[2, 0])

        i = np.array((0, 2, 0))
        q = i.dot(c.inv)
        alpha = np.linspace(-70, 70, 140 * 3 + 1)
        qz = np.linalg.norm(q) * np.tan(alpha / 180 * np.pi)
        q_list = [q + [0, 0, z] for z in qz]
        I = abs(structureFactors(c, q_list))**2
        print(alpha)
