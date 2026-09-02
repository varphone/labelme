from __future__ import annotations

from typing import Final

from PySide6 import QtCore
from PySide6 import QtWidgets

_VALUE_MAX: Final = 1_000_000_000.0
_DECIMALS: Final = 3


class LineProfileWidthWidget(QtWidgets.QWidget):
    """Editor for the active linestrip width anchor.

    The widget only emits typed values; the MainWindow owns Shape/profile
    mutation so edits participate in the existing backup and dirty workflow.
    """

    profile_committed = QtCore.Signal(float, float, str, float, bool)
    position_committed = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._position = self._make_spin_box(0.0, 1.0, "")
        self._width = self._make_spin_box(0.0, _VALUE_MAX, " px")
        self._confidence = self._make_spin_box(0.0, 1.0, "")
        self._source = QtWidgets.QComboBox(self)
        self._source.addItem(self.tr("Automatic"), "auto")
        self._source.addItem(self.tr("Manual"), "manual")
        self._confirmed = QtWidgets.QCheckBox(self.tr("Confirmed"), self)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.tr("Width"), self._width)
        layout.addRow(self.tr("Source"), self._source)
        layout.addRow(self.tr("Confidence"), self._confidence)
        layout.addRow(self._confirmed)
        self._position.editingFinished.connect(self._on_position_finished)
        for widget in (self._width, self._confidence):
            widget.editingFinished.connect(self._on_editing_finished)
        self._source.currentIndexChanged.connect(self._on_editing_finished)
        self._confirmed.toggled.connect(self._on_editing_finished)
        self.set_profile(None)

    @property
    def position_widget(self) -> QtWidgets.QDoubleSpinBox:
        return self._position

    def set_position(self, position: float) -> None:
        blocker = QtCore.QSignalBlocker(self._position)
        self._position.setValue(position)
        del blocker

    def _on_position_finished(self) -> None:
        if self._position.isEnabled():
            self.position_committed.emit(self._position.value())

    def _make_spin_box(
        self, minimum: float, maximum: float, suffix: str
    ) -> QtWidgets.QDoubleSpinBox:
        spin_box = QtWidgets.QDoubleSpinBox(self)
        spin_box.setDecimals(_DECIMALS)
        spin_box.setRange(minimum, maximum)
        spin_box.setSuffix(suffix)
        spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        return spin_box

    def set_profile(
        self,
        values: tuple[float, float, str, float, bool] | None,
    ) -> None:
        enabled = values is not None
        self.setEnabled(enabled)
        if values is None:
            return
        position, width, source, confidence, confirmed = values
        blockers = [
            QtCore.QSignalBlocker(self._position),
            QtCore.QSignalBlocker(self._width),
            QtCore.QSignalBlocker(self._confidence),
            QtCore.QSignalBlocker(self._source),
            QtCore.QSignalBlocker(self._confirmed),
        ]
        self._position.setValue(position)
        self._width.setValue(width)
        self._confidence.setValue(confidence)
        self._source.setCurrentIndex(self._source.findData(source))
        self._confirmed.setChecked(confirmed)
        del blockers

    def _on_editing_finished(self) -> None:
        if not self.isEnabled():
            return
        self.profile_committed.emit(
            self._position.value(),
            self._width.value(),
            self._source.currentData(),
            self._confidence.value(),
            self._confirmed.isChecked(),
        )
