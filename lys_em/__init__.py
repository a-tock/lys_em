import jax
# Matrix products (e.g. kvec.dot) lose accuracy with the default precision (TF32) on GPU, which changes the results and gradients.
jax.config.update("jax_default_matmul_precision", "highest")

from .tem import TEM, TEMParameter
from .space import FunctionSpace
from .scatteringFactor import scatteringFactor
from .kinematical import debyeWallerFactors, formFactors, structureFactors, calcKinematicalDiffraction
from .potentials import CrystalPotential, MagneticPotential, GeneralPotential, interface
from .multislice import multislice, getWaveFunction, getPropagationTerm, getAberrationFunction
from .functions import calcSADiffraction, calcPrecessionDiffraction, calcCBED, calc4DSTEM_Crystal, diffraction, center_of_mass, generic_ravel_pytree, expand_in_Bessel
