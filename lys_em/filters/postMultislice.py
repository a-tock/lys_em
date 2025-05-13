import numpy as np
import dask.array as da

from lys import DaskWave, filters
from lys.filters import FilterInterface, FilterSettingBase, filterGUI
from lys.Qt import QtWidgets


class PostMultiSlice(FilterInterface):
    def __init__(self, type, process):
        self._type = type
        self._proc = process

    def _execute(self, wave, *args, **kwargs):
        if self._type == "Image":
            return DaskWave(self._process(wave.data).sum(axis=3), wave.axes[0], wave.axes[1], wave.axes[2], wave.note)
        elif self._type == "Diffraction":
            if wave.ndim == 4:
                wave = DaskWave(wave, chunks=(wave.shape[0], wave.shape[1], 1, 1))
                wave = filters.FourierFilter([0, 1], process='complex').execute(wave)

                data = wave.data
                data = data[data.shape[0] // 2 - 50:data.shape[0] // 2 + 50, data.shape[1] // 2 - 50:data.shape[1] // 2 + 50].compute()
                zero = np.zeros([1000, 1000, *data.shape[2:]], dtype=complex)
                zero[::10, ::10] = data
                data = self._process(da.from_array(zero))

                x = np.linspace(wave.axes[0][wave.shape[0] // 2 - 50], wave.axes[0][wave.shape[0] // 2 + 50], 1000) * 2 * np.pi
                y = np.linspace(wave.axes[1][wave.shape[1] // 2 - 50], wave.axes[1][wave.shape[1] // 2 + 50], 1000) * 2 * np.pi
                return DaskWave(data, x, y, wave.axes[2], wave.note)

            elif wave.ndim == 3:
                wave = DaskWave(wave, chunks=(wave.shape[0], wave.shape[1], 1))
                wave = filters.FourierFilter([0, 1], process='complex').execute(wave)

                data = wave.data
                data = data[data.shape[0] // 2 - 50:data.shape[0] // 2 + 50, data.shape[1] // 2 - 50:data.shape[1] // 2 + 50].compute()
                zero = np.zeros([1000, 1000, *data.shape[2:]], dtype=complex)
                zero[::10, ::10] = data
                data = self._process(da.from_array(zero))

                x = np.linspace(wave.axes[0][wave.shape[0] // 2 - 50], wave.axes[0][wave.shape[0] // 2 + 50], 1000) * 2 * np.pi
                y = np.linspace(wave.axes[1][wave.shape[1] // 2 - 50], wave.axes[1][wave.shape[1] // 2 + 50], 1000) * 2 * np.pi
                return DaskWave(data, x, y, wave.axes[2], wave.note)

    def _process(self, data):
        if self._proc == "Absolute":
            return da.absolute(data)
        elif self._proc == "Square":
            return da.absolute(data)**2
        elif self._proc == "Complex":
            return data

    def getParameters(self):
        return {"type": self._type, "process": self._proc}

    def getRelativeDimension(self):
        return 0


@filterGUI(PostMultiSlice)
class _PostMultiSetting(FilterSettingBase):
    _types = ["Image", "Diffraction"]
    _process = ["Absolute", "Square", "Complex"]

    def __init__(self, dim):
        super().__init__(dim)
        self.type = QtWidgets.QComboBox()
        self.type.addItems(self._types)
        self.proc = QtWidgets.QComboBox()
        self.proc.addItems(self._process)

        lh = QtWidgets.QHBoxLayout()
        lh.addWidget(self.type)
        lh.addWidget(self.proc)
        self.setLayout(lh)

    def getParameters(self):
        return {"type": self.type.currentText(), "process": self.proc.currentText()}

    def setParameters(self, type, process):
        self.type.setCurrentText(type)
        self.proc.setCurrentText(process)


filters.addFilter(PostMultiSlice, gui=_PostMultiSetting, guiName="MultiSlice", guiGroup="lys_em")
