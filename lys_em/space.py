import numpy as np
import jax
import jax.numpy as jnp

class FunctionSpace:
    """
    FunctionSpace represents a 2-dimensional parallelogram grid function space.
    The length of the rectangular space is defined by a and b. Each cell of the grid function space is (crys.a/Nx crys.b/Ny)

    Args:
        a (float): Length of the unit cell along the a-vector.
        b (float): Length of the unit cell along the b-vector.
        c (float): Length of the unit cell along the optical axis direction
        gamma (float, optional): Angle between the unit cell vectors in degrees. Default is 90.
        Nx (int, optional): Number of grid points along the a-vector. Default is 128.
        Ny (int, optional): Number of grid points along the b-vector. Default is 128.
        Nz (int, optional): Number of divisions along the optical axis direction.
    """

    def __init__(self, a, b, c, gamma=90, Nx=128, Ny=128, Nz=10):
        self._unit = np.array([[a, 0], [b * np.cos(gamma * np.pi / 180), b * np.sin(gamma * np.pi / 180)]])
        self._c = c
        self._N = np.array([Nx, Ny, Nz])

    @staticmethod
    def fromCrystal(crys, Nx, Ny, ncells, division="Auto"):
        """
        Create a FunctionSpace instance from a CrystalStructure instance.

        Args:
            crys (CrystalStructure): The CrystalStructure instance to create the FunctionSpace from.
            Nx (int): The number of grid points along the a-vector.
            Ny (int): The number of grid points along the b-vector.
            ncells (int): The number of unit cells along the optical axis direction.
            division ('Auto' or int): The number of divisions of unit cell along optical axis direction.

        Returns:
            FunctionSpace: The created FunctionSpace instance.
        """
        if division == "Auto":
            division = int(crys.unit[2][2] / 2)
        return FunctionSpace(crys.a, crys.b, crys.unit[2][2] * ncells, crys.gamma, Nx, Ny, division * ncells)

    @property
    def mask(self):
        """
        Return mask pattern whose diameter is identical with 2/3 of min(kx, ky).
        """
        if not hasattr(self, "_mask"):
            k = jnp.sqrt(self.k2)
            max = jnp.sqrt(self._getMax())
            self._mask = jnp.where(k > max * 2 / 3, 0, 1)
        return self._mask

    def _getMax(self):
        k2_row0 = self.k2[self._N[0] // 2, :]
        k2_col0 = self.k2[:, self._N[1] // 2]
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
        Return 2-dimensional reciprocal space grid. The unit is rad/A.
        Each cell has (2pi/a, 2pi/b) length in reciprocal space.
        """
        if not hasattr(self, "_kvec"):
            inverse_matrix = 2 * np.pi * jnp.linalg.inv(self._unit)
            grid = self._create_grid()
            self._kvec = jnp.dot(grid, inverse_matrix.T)
        return self._kvec

    def _create_grid(self):
        x = jnp.arange(-self._N[0] // 2, self._N[0] // 2)
        shift_x = jnp.roll(x, self._N[0] // 2)
        y = jnp.arange(-self._N[1] // 2, self._N[1] // 2)
        shift_y = jnp.roll(y, self._N[1] // 2)
        grid = jnp.array(jnp.meshgrid(shift_x, shift_y)).transpose(2, 1, 0)
        return grid

    def fft(self, data):
        return jnp.fft.fft2(data) * self.dV

    def ifft(self, data):
        return jnp.fft.ifft2(data*self.mask) / self.dV

    @property
    def N(self):
        return self._N

    @property
    def c(self):
        return self._c

    @property
    def dz(self):
        return self._c / self._N[2]

    @property
    def dV(self):
        """
        Return volume element of the function space grid in reciprocal space.

        The unit of dV is A^2.
        """
        return np.sqrt(np.linalg.norm(self._unit[0])**2 * np.linalg.norm(self._unit[1])**2 - self._unit[0].dot(self._unit[1])**2) / self._N[0] / self._N[1]

