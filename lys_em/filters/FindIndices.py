# Required lys modules for making filter
from lys import DaskWave
from lys.filters import FilterSettingBase, filterGUI, addFilter, FilterInterface

# We recommend users to import Qt widgets from llys, which is alias to Qt5 libraries (PyQt5 or PySide2)
from lys.Qt import QtWidgets
from DiffSim import CrystalStructure

import numpy as np


class FindIndices(FilterInterface):
    def __init__(self, cif, a, b):
        self._cif = cif
        self._a = a
        self._b = b

    def _execute(self, wave, *args, **kwargs):
        c = CrystalStructure.from_cif(self._cif)
        indices = find_indices(c, np.array([wave.x, wave.data.compute()]).T - wave.note["DiffractionCenter"], ai=self._a, bi=self._b)
        wave.note["indices"] = np.array(indices).tolist()
        print(wave.note["indices"])
        return DaskWave(wave.data, *wave.axes, **wave.note)

    def getParameters(self):
        return {"cif": self._cif, "a": self._a, "b": self._b}


def find_indices(c, peaks, ai=(1, 0, 0), bi=(0, 1, 0)):
    a = np.dot(ai, c.InverseLatticeVectors())  # to check
    b = np.dot(bi, c.InverseLatticeVectors())  # to check
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(a)
    # calculate rotation matrix from a to b
    theta = abs(np.arccos(a.dot(b) / np.linalg.norm(b)))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # find minimum length vector in peaks, which corresponds to (normalized) a vector
    avec, bvec = peaks[1], R.dot(peaks[1])
    M = np.linalg.inv(np.array([avec, bvec]).T)

    # calculate indices
    indices_loc = [np.round(M.dot(d)).astype(int) for d in peaks]
    indices = [np.array(i[0]) * ai + np.array(i[1]) * bi for i in indices_loc]
    return indices


@filterGUI(FindIndices)
class _FindIndicesSetting(FilterSettingBase):
    def __init__(self, dimension):
        super().__init__(dimension)
        self._cif = QtWidgets.QLineEdit()
        self._a = [QtWidgets.QSpinBox() for i in range(3)]
        self._b = [QtWidgets.QSpinBox() for i in range(3)]
        for w in self._a + self._b:
            w.setRange(-100000, 100000)
        self._a[0].setValue(1)
        self._b[1].setValue(1)

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("cif file"), 0, 0)
        g.addWidget(QtWidgets.QLabel("index1"), 1, 0)
        g.addWidget(QtWidgets.QLabel("index2"), 2, 0)

        g.addWidget(self._cif, 0, 1, 1, 2)
        g.addWidget(QtWidgets.QPushButton("Select", clicked=self.__select), 0, 3)

        for i in range(3):
            g.addWidget(self._a[i], 1, i + 1)
            g.addWidget(self._b[i], 2, i + 1)

        self.setLayout(g)

    def getParameters(self):
        return {"cif": self._cif.text(), "a": [w.value() for w in self._a], "b": [w.value() for w in self._b]}

    def setParameters(self, cif, a, b):
        self._cif.setText(cif)
        for i in range(3):
            self._a[i].setValue(a[i])
            self._b[i].setValue(b[i])

    def __select(self):
        name, ok = QtWidgets.QFileDialog.getOpenFileName(None, "Open cif", "", "cif(*.cif);;All(*.*)")
        if ok:
            self._cif.setText(name)


# Add filte to lys. You can use new filter from MultiCut
addFilter(
    FindIndices,
    gui=_FindIndicesSetting,
    guiName="FindIndices",
    guiGroup="lys_em"
)
