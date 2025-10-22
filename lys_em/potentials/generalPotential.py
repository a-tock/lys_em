from .interface import PotentialInterface


class GeneralPotential(PotentialInterface):
    def __init__(self, space, potential):
        super().__init__(space)
        self._sp = space
        self._pot = potential

    def replace(self, potential):
        return GeneralPotential(self._sp, potential)

    def getPhase(self, beam):
        return self._pot
