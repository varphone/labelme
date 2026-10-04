from __future__ import annotations

from collections.abc import Callable
from typing import cast

from PySide6 import QtCore

from .._annotation_batch import AnnotationBatchReport
from .._annotation_batch import rename_label_in_files
from .._annotation_batch import set_circle_radius_in_files


class AnnotationBatchWorker(QtCore.QObject):
    """Run annotation-file updates without touching widgets."""

    progress = QtCore.Signal(int, int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    @QtCore.Slot(object, object, object, object)
    def rename_label(
        self,
        *,
        filenames: object,
        old_label: object,
        new_label: object,
        save_image_data: object,
    ) -> None:
        self._run(
            operation=lambda: rename_label_in_files(
                filenames=self._filenames(value=filenames),
                old_label=self._string(value=old_label, name="old_label"),
                new_label=self._string(value=new_label, name="new_label"),
                save_image_data=self._bool(
                    value=save_image_data, name="save_image_data"
                ),
                cancel_check=lambda: self._cancelled,
                progress_callback=self.progress.emit,
            )
        )

    @QtCore.Slot(object, object, object, object)
    def set_circle_radius(
        self,
        *,
        filenames: object,
        label: object,
        radius: object,
        save_image_data: object,
    ) -> None:
        self._run(
            operation=lambda: set_circle_radius_in_files(
                filenames=self._filenames(value=filenames),
                label=self._string(value=label, name="label"),
                radius=self._float(value=radius, name="radius"),
                save_image_data=self._bool(
                    value=save_image_data, name="save_image_data"
                ),
                cancel_check=lambda: self._cancelled,
                progress_callback=self.progress.emit,
            )
        )

    def _run(self, *, operation: Callable[[], AnnotationBatchReport]) -> None:
        self._cancelled = False
        try:
            report = operation()
            assert isinstance(report, AnnotationBatchReport)
            if report.canceled:
                self.canceled.emit()
            else:
                self.succeeded.emit(report)
        except Exception as error:
            if self._cancelled:
                self.canceled.emit()
            else:
                self.failed.emit(f"{type(error).__name__}: {error}")

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancelled = True

    @staticmethod
    def _filenames(*, value: object) -> list[str]:
        if not isinstance(value, list) or not all(
            isinstance(filename, str) for filename in value
        ):
            raise TypeError("filenames must be a list of strings")
        return cast(list[str], value)

    @staticmethod
    def _string(*, value: object, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        return value

    @staticmethod
    def _float(*, value: object, name: str) -> float:
        if not isinstance(value, int | float):
            raise TypeError(f"{name} must be a number")
        return float(value)

    @staticmethod
    def _bool(*, value: object, name: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
        return value
