class PotentialInterface:
    def __add__(self, obj):
        return _CombinedPotential(self, obj)

    def getPhase(self, beam):
        raise NotImplementedError("Potential class should implement this method.")


class _CombinedPotential(PotentialInterface):
    def __init__(self, *objs):
        self._objs = objs

    def getPhase(self, beam):
        res = np.sum([obj.getPhase(beam) for obj in self._objs], axis=0)