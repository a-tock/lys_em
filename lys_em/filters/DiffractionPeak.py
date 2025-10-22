"""
The template for making new filter that can be used in MultiCut.
This file can be used without edit. The sample filter is registered in 'User Defined Filters' in MultiCut.
See Section 'Making a new filter' in menu (Help -> Open lys reference) for detail
"""


# Required lys modules for making filter
from lys import DaskWave
from lys.filters import FilterSettingBase, filterGUI, addFilter, FilterInterface

# We recommend users to import Qt widgets from llys, which is alias to Qt5 libraries (PyQt5 or PySide2)
from lys.Qt import QtWidgets
from lys.widgets import ScientificSpinBox

import numpy as np

from scipy.signal import correlate
import scipy.ndimage
if hasattr(scipy.ndimage, 'maximum_filter'):
    from scipy.ndimage import maximum_filter, median_filter
else:
    from scipy.ndimage.filters import maximum_filter, median_filter


class DiffractionPeak(FilterInterface):
    def __init__(self, size_max, size_mean, threshold):
        self._size_max = size_max
        self._size_mean = size_mean
        self._threshold = threshold

    def _execute(self, wave, *args, **kwargs):
        center, peaks = find_peaks(wave.data, size_max=self._size_max, size_mean=self._size_mean, threshold=self._threshold)
        wave.note["DiffractionCenter"] = center
        peaks = [p for p in center + peaks if p[0] > 0 and p[1] > 0 and p[0] < wave.shape[0] and p[1] < wave.shape[1]]
        peaks = np.array(peaks).T
        p1 = wave.pointToPos(peaks[1], axis=1)
        p0 = wave.pointToPos(peaks[0], axis=0)
        return DaskWave(p1, p0, **wave.note)

    def getParameters(self):
        return {"size_max": self._size_max, "size_mean": self._size_mean, "threshold": self._threshold}

    def getRelativeDimension(self):
        return -1


def find_peaks(data, **kwargs):
    data = np.array(data) / np.max(data) * 100
    acr = correlate(data, data, mode="same")
    ct = correlate(data, acr, mode="same")
    center = np.unravel_index(np.argmax(ct.data), ct.data.shape)

    peaks = _find_periodic_peaks(acr, **kwargs).astype(int).T
    argsort = np.argsort(np.linalg.norm(peaks, axis=1))
    peaks = peaks[argsort]

    return center, peaks


def _find_periodic_peaks(data, size_max=7, size_mean=13, threshold=1.8):
    local_max = maximum_filter(data, size=size_max)
    mean = median_filter(data, size_mean)
    detected_peaks = np.where(data == local_max, 1, 0)
    detected_peaks = np.where(data > mean * threshold, detected_peaks, 0)
    res = np.where(detected_peaks == 1)
    return np.array([res[0] - data.shape[0] / 2, res[1] - data.shape[1] / 2])


@filterGUI(DiffractionPeak)
class _DiffractionPeakSetting(FilterSettingBase):
    def __init__(self, dimension):
        super().__init__(dimension)
        self._size_max = QtWidgets.QSpinBox()
        self._size_max.setRange(1, 100000)
        self._size_max.setValue(3)
        self._size_mean = QtWidgets.QSpinBox()
        self._size_mean.setRange(1, 100000)
        self._size_mean.setValue(13)
        self._threshold = ScientificSpinBox()
        self._threshold.setValue(1.8)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("Size max"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("Size mean"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("Threshold"), 0, 2)
        layout.addWidget(self._size_max, 1, 0)
        layout.addWidget(self._size_mean, 1, 1)
        layout.addWidget(self._threshold, 1, 2)

        self.setLayout(layout)

    def getParameters(self):
        return {"size_max": self._size_max.value(), "size_mean": self._size_mean.value(), "threshold": self._threshold.value()}

    def setParameters(self, size_max, size_mean, threshold):
        self._size_max.setValue(size_max)
        self._size_mean.setValue(size_mean)
        self._threshold.setValue(threshold)


# Add filte to lys. You can use new filter from MultiCut
addFilter(
    DiffractionPeak,
    gui=_DiffractionPeakSetting,
    guiName="DiffractionPeak",
    guiGroup="lys_em"
)
