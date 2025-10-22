import dask.array as da
import numpy as np
from scipy import ndimage, interpolate
from scipy.signal import correlate

from lys import DaskWave
from lys.widgets import AxisSelectionLayout, ScientificSpinBox
from lys.filters import FilterSettingBase, filterGUI, addFilter, FilterInterface
from lys.Qt import QtWidgets


class DiffractionBackground(FilterInterface):
    def __init__(self, axes, center):
        self._axes = axes
        self._center = center

    def _execute(self, wave, *args, **kwargs):
        if self._center == "auto":
            sl = []
            for i in range(wave.ndim):
                if i in self._axes:
                    sl.append(slice(None))
                else:
                    sl.append(0)
            data = wave.data[tuple(sl)].compute()
            acr = correlate(data, data, mode="same")
            ct = correlate(data, acr, mode="same")
            center = np.unravel_index(np.argmax(ct.data), ct.data.shape)
        else:
            center = wave.posToPoint(self._center)

        def f(x):
            return remove_bg(x, *center)
        gumap = da.gufunc(f, signature="(i,j)->(i,j)", output_dtypes=float, vectorize=True, axes=[tuple(self._axes), tuple(self._axes)], allow_rechunk=True)
        newData = gumap(wave.data)
        return DaskWave(newData, *wave.axes, **wave.note)

    def getParameters(self):
        return {"axes": self._axes, "center": self._center}


def remove_bg(data, cx, cy, level=10):
    rt = translate_rt(data, cx, cy, n_angle=360)
    bg = np.percentile(rt, level, axis=1)
    bg = interpolate.interp1d(np.arange(len(bg)), bg, kind="linear", bounds_error=False, fill_value=0)
    x, y = np.arange(data.shape[0]) - cx, np.arange(data.shape[1]) - cy
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    bg_image = bg(r)
    return data.astype(float) - bg_image


def translate_rt(data, cx, cy, n_angle=360, order=1):
    r_max = np.max([cx, cy, data.shape[0] - cx, data.shape[1] - cy])
    theta = np.linspace(0, 360, num=n_angle, endpoint=False) / 360 * 2 * np.pi
    result = []
    for r in range(int(r_max)):
        coords = np.array([r * np.cos(theta) + cx, r * np.sin(theta) + cy])
        result.append(ndimage.map_coordinates(data, coords, order=order))
    return np.array(result)


@filterGUI(DiffractionBackground)
class _DiffractionBackgroundSetting(FilterSettingBase):
    def __init__(self, dimension):
        """
        __init__ must take an argument that indicate dimension of input data.
        Initialize widgets (see documentation of PyQt5 and PySide2) after calling super().__init__(dimension).
        """
        super().__init__(dimension)
        self._axis1 = AxisSelectionLayout("axis1", dimension, init=0)
        self._axis2 = AxisSelectionLayout("axis2", dimension, init=1)

        self._mode = QtWidgets.QComboBox()
        self._mode.addItems(["Auto", "Manual"])
        self._mode.currentTextChanged.connect(self.__mode)

        self._cx = ScientificSpinBox()
        self._cy = ScientificSpinBox()

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Center (x,y)"))
        h1.addWidget(self._mode)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self._cx)
        h.addWidget(self._cy)
        self._cval = QtWidgets.QFrame()
        self._cval.setLayout(h)
        self._cval.hide()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self._axis1)
        layout.addLayout(self._axis2)
        layout.addLayout(h1)
        layout.addWidget(self._cval)

        self.setLayout(layout)

    def __mode(self, text):
        if text == "Auto":
            self._cval.hide()
        else:
            self._cval.show()

    def getParameters(self):
        if self._mode.currentText() == "Auto":
            center = "auto"
        else:
            center = [self._cx.value(), self._cy.value()]
        return {"axes": [self._axis1.getAxis(), self._axis2.getAxis()], "center": center}

    def setParameters(self, axes, center):
        if center == "auto":
            self._mode.setCurrentIndex(0)
        else:
            self._mode.setCurrentIndex(1)
            self._cx.setValue(center[0])
            self._cy.setValue(center[1])
        self._axis1.setAxis(axes[0])
        self._axis2.setAxis(axes[1])


# Add filte to lys. You can use new filter from MultiCut
addFilter(
    DiffractionBackground,
    gui=_DiffractionBackgroundSetting,
    guiName="DiffractionBackground",
    guiGroup="lys_em"
)
