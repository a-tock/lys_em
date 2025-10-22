from lys import DaskWave, Wave
from lys.filters import FilterSettingBase, filterGUI, addFilter, FilterInterface
from lys.filters.filter.FreeLine import FreeLineFilter

# We recommend users to import Qt widgets from llys, which is alias to Qt5 libraries (PyQt5 or PySide2)
from lys.Qt import QtWidgets
from lys.widgets import AxisSelectionLayout

import dask.array as da
import numpy as np
from scipy.optimize import curve_fit


class ExtractIntensity2(FilterInterface):
    def __init__(self, axes, peaks, box_size, type):
        self._axes = axes
        self._peaks = peaks
        self._box_size = box_size
        self._type = type

    def _execute(self, wave, *args, **kwargs):
        peaks = Wave(self._peaks)
        newData=[calc_intensity(FreeLineFilter(axes=self._axes, range=_range_maker(peaks,np.array([px, py]), self._box_size), width=self._box_size)._execute(wave), type=self._type) for px, py in zip(peaks.x[1:], peaks.data[1:])]
        newAxes = [ax for i, ax in enumerate(wave.axes) if i not in self._axes]+[None]
        wave.note["indices"] = peaks.note["indices"]
        return DaskWave(newData, *newAxes, **wave.note)

    def getParameters(self):
        return {"axes": self._axes, "peaks": self._peaks, "box_size": self._box_size, "type": self._type}

    def getRelativeDimension(self):
        return -1


def calc_intensity(data, type):
    def f_lor(x, A, dx, mu_x, a, b):
        return A * (dx**2 / ((x - mu_x)**2 + dx**2)) + a * x + b

    def f_gau(x, A, dx, mu_x):
        return np.sqrt(np.pi) * A * np.exp(-(x - mu_x)**2 / (2 * dx**2)) 

    init = [np.max(data), 2, data.shape[0]/2]
    x = np.arange(data.shape[0])
    f = f_gau if type == "gauss" else f_lor
    try:
        popt, pcov = curve_fit(f, x, data, p0=init,maxfev=100000)
    except RuntimeError:
        popt = [0,0]
    print(popt)
    return popt[0] * np.abs(popt[1]) * np.pi

def _range_maker(peaks, peak_position, box_size):
    relative_coord = peak_position - np.array(peaks.note["DiffractionCenter"])
    relative_coord = np.array([[0,1],[-1,0]]).dot(relative_coord)
    return [peak_position - relative_coord * box_size / (2 * np.linalg.norm(relative_coord, ord=2)), peak_position + relative_coord * box_size / (2 * np.linalg.norm(relative_coord, ord=2))]



@filterGUI(ExtractIntensity2)
class _ExtractIntensitySetting(FilterSettingBase):
    def __init__(self, dimension):
        super().__init__(dimension)
        self._kx = AxisSelectionLayout("kx", dimension, init=0)
        self._ky = AxisSelectionLayout("ky", dimension, init=1)
        self._peaks = QtWidgets.QLineEdit()
        self._box_size = QtWidgets.QSpinBox()
        self._box_size.setValue(8)
        self._type = QtWidgets.QComboBox()
        self._type.addItems(["gauss", "lorentz"])

        g = QtWidgets.QGridLayout()
        g.addWidget(QtWidgets.QLabel("peak file"), 0, 0)
        g.addWidget(QtWidgets.QLabel("box size"), 1, 0)
        g.addWidget(QtWidgets.QLabel("type"), 2, 0)
        g.addWidget(self._peaks, 0, 1)
        g.addWidget(QtWidgets.QPushButton("Select", clicked=self.__select), 0, 2)
        g.addWidget(self._box_size, 1, 1, 1, 2)
        g.addWidget(self._type, 2, 1, 1, 2)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self._kx)
        layout.addLayout(self._ky)
        layout.addLayout(g)

        self.setLayout(layout)

    def getParameters(self):
        return {"axes": [self._kx.getAxis(), self._ky.getAxis()], "peaks": self._peaks.text(), "box_size": self._box_size.value(), "type": self._type.currentText()}

    def setParameters(self, axes, peaks, box_size, type):
        self._kx.setAxis(axes[0])
        self._ky.setAxis(axes[1])
        self._peaks.setText(peaks)
        self._box_size.setValue(box_size)
        self._type.setCurrentText(type)

    def __select(self):
        name, ok = QtWidgets.QFileDialog.getOpenFileName(None, "Open npz", "", "npz(*.npz);;All(*.*)")
        if ok:
            self._peaks.setText(name)


# Add filte to lys. You can use new filter from MultiCut
addFilter(
    ExtractIntensity2,
    gui=_ExtractIntensitySetting,
    guiName="ExtractIntensity4",
    guiGroup="Precession Diffraction"
)
