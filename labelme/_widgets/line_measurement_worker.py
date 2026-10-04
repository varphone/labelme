from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore

from .._line_measurement import MeasurementParameters
from .._line_measurement import measure_line_profile


class LineMeasurementWorker(QtCore.QObject):
    """QObject worker that never touches Canvas or other widgets."""

    progress = QtCore.Signal(int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    @QtCore.Slot(object, object, object)
    def run(
        self,
        *,
        image: object,
        points: object,
        parameters: object,
    ) -> None:
        self._cancelled = False
        try:
            self.progress.emit(5)
            if self._cancelled:
                self.canceled.emit()
                return
            if not isinstance(image, np.ndarray):
                raise TypeError("image must be a NumPy array")
            if not isinstance(parameters, MeasurementParameters):
                raise TypeError("parameters must be MeasurementParameters")
            point_array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            result = measure_line_profile(
                image=cast(NDArray[np.generic], image),
                points=point_array,
                parameters=parameters,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled or result is None:
                self.canceled.emit()
                return
            self.progress.emit(100)
            self.succeeded.emit(result)
        except Exception as e:
            if self._cancelled:
                self.canceled.emit()
            else:
                self.failed.emit(f"{type(e).__name__}: {e}")

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancelled = True
