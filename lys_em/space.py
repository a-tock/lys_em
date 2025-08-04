import numpy as np


class FunctionSpace:
    """
    Create 2 dimensional rectangular grid function space.
    The length of the rectangular space is defined by a and b. Each cell of the grid function space is (crys.a/Nx crys.b/Ny)
    """
    def __init__(self, a, b, gamma=90, Nx=128, Ny=128):
        self._unit = np.array([[a,0], [b*np.cos(gamma*np.pi/180),b*np.sin(gamma*np.pi/180)]])
        self._N = np.array([Nx, Ny])
        
    @staticmethod 
    def fromCrystal(crys, Nx, Ny):
        return FunctionSpace(crys.a, crys.b, crys.gamma,Nx, Ny)

    def getArray(self):
        return np.ones((self._N[0], self._N[1])) / self._N[0] / self._N[1]

    @property
    def mask(self):
        """
        Return mask pattern whose diameter is identical with 2/3 of min(kx, ky).
        """
        if not hasattr(self, "_mask"):
            k = np.sqrt(self.k2)
            max = np.sqrt(self._getMax())
            self._mask = np.where(k > max * 2 / 3, 0, 1)
        return self._mask

    def _getMax(self):
        k2_row0 = self.k2[self._N[0]//2, :]
        k2_col0 = self.k2[:, self._N[1]//2]
        return min(min(k2_row0), min(k2_col0))

    @property
    def k2(self):
        """
        Return k^2 for respective grid point in reciprocal space. The unit is rad/A.
        """
        if not hasattr(self, "_k2"):
            k = self.kvec
            self._k2 = k[:, :, 0]**2 + k[:, :, 1]**2
        return self._k2

    @property
    def kvec(self):
        """
        Return 2 dimensional reciprocal space grid. The unit is rad/A.
        Each cell has (2pi/a, 2pi/b) length in reciprocal space.
        """
        inverse_matrix = 2 * np.pi * np.linalg.inv(self._unit)
        grid = self._create_grid()
        if not hasattr(self, "_kvec"):
            self._kvec = np.dot(grid, inverse_matrix.T)
        return self._kvec

    def _create_grid(self):
        x = np.arange(-self._N[0]//2, self._N[0]//2)
        shift_x = np.roll(x, self._N[0]//2)
        y = np.arange(-self._N[1]//2, self._N[1]//2)
        shift_y = np.roll(y, self._N[1]//2)
        grid = np.array(np.meshgrid(shift_x, shift_y)).transpose(2, 1, 0)
        return grid

    @property
    def dV(self):
        return np.sqrt(np.linalg.norm(self._unit[0])**2*np.linalg.norm(self._unit[1])**2-self._unit[0].dot(self._unit[1])**2)  / self._N[0] / self._N[1]

    def FT(self, data):
        return np.fft.fft2(data) * self.dV

    def IFT(self, data):
        return np.fft.ifft2(data) / self.dV

    def getPropagationTerm(self, lamb, dz, theta_x=0, theta_y=0):
        k2 = self.k2
        tx, ty = np.array([theta_x, theta_y]) * np.pi / 180
        kx, ky = self.kvec.transpose(2, 1, 0)[0].T, self.kvec.transpose(2, 1, 0)[1].T
        tilt = 1j * (kx * np.tan(tx) + ky * np.tan(ty)) - 1j * lamb * k2 / 4 / np.pi
        return np.exp(dz * tilt)