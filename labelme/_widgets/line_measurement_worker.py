from __future__ import annotations

from PySide6 import QtCore

from .._line_measurement import measure_line_profile


class LineMeasurementWorker(QtCore.QObject):
    """QObject worker that never touches Canvas or other widgets."""

    progress = QtCore.Signal(int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    @QtCore.Slot(object, object, object)
    def run(
        self,
        image: object,
        points: object,
        parameters: object,
    ) -> None:
        self._cancelled = False
        try:
            self.progress.emit(5)
            if self._cancelled:
                return
            result = measure_line_profile(
                image,  # type: ignore[arg-type]
                points,  # type: ignore[arg-type]
                parameters=parameters,  # type: ignore[arg-type]
            )
            if self._cancelled:
                return
            self.progress.emit(100)
            self.succeeded.emit(result)
        except Exception as e:
            if not self._cancelled:
                self.failed.emit(f"{type(e).__name__}: {e}")

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancelled = True
