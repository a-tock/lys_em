import numpy as np

m = 9.10938356e-31  # kg
e = 1.6021766208e-19  # C
hbar = 1.054571800e-34  # Js
kB = 1.38064852e-23  # m^2kg/s^2K
h = hbar * 2 * np.pi
c = 2.99792458e8  # m/s

class ElectronBeam(object):
    def __init__(self, acc, convergence, directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        self.__acc = acc
        self.__convergence = convergence
        self.__direction = np.array(directions)

    def getWavelength(self, unit="angstrom"):
        if unit == "angstrom":
            return 1.2264e-9 / np.sqrt(self.__acc * (1 + 9.7846e-7 * self.__acc)) * 1e10
        if unit == "MKSA":
            return 1.2264e-9 / np.sqrt(self.__acc * (1 + 9.7846e-7 * self.__acc))

    def getKMax(self, unit="angstrom"):
        kmax = 2 * np.pi * self.__convergence / self.getWavelength(unit)
        return kmax

    def getK(self, unit="angstrom"):
        return 2 * np.pi / self.getWavelength(unit)

    def getDirectionVectors(self):
        return self.__direction

    def getRelativisticMass(self):
        '''
        Get relativistic mass in kg.

        Returns:
            float:Relativistic mass 
        '''
        return m + e * self.__acc / c**2

    def getSigma(self):
        M, lamb = self.getRelativisticMass(), self.getWavelength()
        return 2 * np.pi * M * e * lamb / h**2  # kg C m /eV^2 s^2

    def getSigma0(self):
        M, lamb = self.getRelativisticMass(), self.getWavelength("MKSA")
        return 2 * np.pi * m * e / h**2  # kg C m /eV^2 s^2