import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.special import erf

from lys_em import TEM, TEMParameter, FunctionSpace, MagneticPotential, multislice


class MagneticPotential_test(unittest.TestCase):
    path = "test/DataFiles"

    def test_DW(self, size=3000, width=50, Nx=900, Ny=300, show=False):
        sp = FunctionSpace(size, size, 10, Nx=Nx, Ny=Ny, Nz=1)
        tem = TEM(200e3)
        params = [TEMParameter(defocus=1e-7/1e-10)]

        x = sp.rvec[:,:,0]
        my = (erf(x/width)-erf((x-size/2)/width)+erf((x-size)/width)) 
        M = 8e5*np.array([[my*0, my, np.sqrt(1-my**2)]]).transpose(0,2,3,1) # magnetization for permalloy (A/m)

        pot = MagneticPotential(sp, M).get(tem)
        res = multislice(sp, pot, tem, params)

        if show:
            import matplotlib.pyplot as plt
            plt.imshow(abs(res[0])**2)
            plt.show()

