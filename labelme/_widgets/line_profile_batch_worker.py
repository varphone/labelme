from __future__ import annotations

import dataclasses

from PySide6 import QtCore

from .._line_profile_batch import BatchOptions
from .._line_profile_batch import measure_annotation_files


class LineProfileBatchWorker(QtCore.QObject):
    """Run batch profile measurement without blocking the main window."""

    progress = QtCore.Signal(int, int)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    @QtCore.Slot(object, object, object)
    def run(
        self,
        filenames: object,
        output_dir: object,
        options: object,
    ) -> None:
        self._cancelled = False
        try:
            assert isinstance(filenames, list)
            assert isinstance(options, BatchOptions)
            batch_options = dataclasses.replace(
                options,
                cancel_check=lambda: self._cancelled,
                progress_callback=lambda completed, total: self.progress.emit(
                    completed, total
                ),
            )
            report = measure_annotation_files(
                filenames,
                output_dir=output_dir if isinstance(output_dir, str) else None,
                options=batch_options,
            )
            self.succeeded.emit(report)
        except Exception as error:
            if not self._cancelled:
                self.failed.emit(f"{type(error).__name__}: {error}")

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancelled = True
