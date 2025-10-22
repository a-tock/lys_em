import numpy as np
import jax
import jax.numpy as jnp


class FunctionSpace:
    """
    FunctionSpace represents a 2-dimensional parallelogram grid function space.
    The length of the rectangular space is defined by a and b. Each cell of the grid function space is (crys.a/Nx crys.b/Ny)

    Args:
        a (float): Length of the unit cell along the a-vector in angstrome.
        b (float): Length of the unit cell along the b-vector in angstrome.
        c (float): Length of the unit cell along the optical axis direction in angstrome.
        gamma (float, optional): Angle between the unit cell vectors in degrees. Default is 90.
        Nx (int, optional): Number of grid points along the a-vector. Default is 128.
        Ny (int, optional): Number of grid points along the b-vector. Default is 128.
        Nz (int, optional): Number of divisions along the optical axis direction.
    """

    def __init__(self, a, b, c, gamma=90, Nx=128, Ny=128, Nz=10):
        self._unit = jnp.array([[a, 0], [b * jnp.cos(gamma * jnp.pi / 180), b * jnp.sin(gamma * jnp.pi / 180)]])
        self._c = c
        self._N = (int(Nx), int(Ny), int(Nz))

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

        def angle(v1, v2): return jnp.rad2deg(jnp.arccos(jnp.clip(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2)), -1.0, 1.0)))
        a, b = jnp.linalg.norm(crys.unit[0]), jnp.linalg.norm(crys.unit[1])
        return FunctionSpace(a, b, crys.unit[2][2] * ncells, angle(crys.unit[0], crys.unit[1]), Nx, Ny, division * ncells)

    @property
    def mask(self):
        """
        Return mask pattern whose diameter is identical with 2/3 of min(kx, ky).
        """
        max = jnp.sqrt(self._getMax())
        return jnp.where(self.k > max * 2 / 3, 0, 1)

    def _getMax(self):
        k2 = self.k**2
        k2_row0 = k2[self._N[0] // 2, :]
        k2_col0 = k2[:, self._N[1] // 2]
        return jnp.minimum(jnp.min(k2_row0), jnp.min(k2_col0))

    @property
    def k(self):
        """
        Return k for respective grid point in reciprocal space. The unit is rad/A.
        """
        def safe_norm(x, axis=-1, eps=1e-12):
            return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)

        return safe_norm(self.kvec, axis=2)

    @property
    def kvec(self):
        """
        Return 2-dimensional reciprocal space grid. The unit is rad/A.
        Each cell has (2pi/a, 2pi/b) length in reciprocal space.
        """
        inverse_matrix = 2 * jnp.pi * jnp.linalg.inv(self._unit)
        grid = self._create_grid()
        return jnp.dot(grid, inverse_matrix.T)

    @property
    def rvec(self):
        grid = self._create_grid()
        unit = jnp.array([self._unit[0] / self._N[0], self._unit[1] / self._N[1]])
        return jnp.dot(grid, unit)

    def _create_grid(self):
        x = jnp.arange(-self._N[0] // 2, self._N[0] // 2)
        shift_x = jnp.roll(x, self._N[0] // 2)
        y = jnp.arange(-self._N[1] // 2, self._N[1] // 2)
        shift_y = jnp.roll(y, self._N[1] // 2)
        grid = jnp.array(jnp.meshgrid(shift_x, shift_y, indexing="ij")).transpose(1, 2, 0)
        return grid

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
        return jnp.sqrt(jnp.linalg.norm(self._unit[0])**2 * jnp.linalg.norm(self._unit[1])**2 - self._unit[0].dot(self._unit[1])**2) / self._N[0] / self._N[1]
