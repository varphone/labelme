from __future__ import annotations

from typing import Final

from PySide6 import QtCore
from PySide6 import QtWidgets

_DECIMALS: Final = 3


class LineProfileVisibilityWidget(QtWidgets.QWidget):
    """Editor for the active normalized visibility anchor."""

    visibility_committed = QtCore.Signal(float, float, str, float, bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._position = self._make_spin_box(0.0, 1.0)
        self._visibility = self._make_spin_box(0.0, 1.0)
        self._source = QtWidgets.QComboBox(self)
        self._source.addItems(["auto", "manual"])
        self._confirmed = QtWidgets.QCheckBox(self.tr("Confirmed"), self)
        self._confidence = self._make_spin_box(0.0, 1.0)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow(self.tr("Position"), self._position)
        layout.addRow(self.tr("Visibility"), self._visibility)
        layout.addRow(self.tr("Source"), self._source)
        layout.addRow(self.tr("Confidence"), self._confidence)
        layout.addRow(self._confirmed)
        for widget in (self._position, self._visibility, self._confidence):
            widget.editingFinished.connect(self._on_editing_finished)
        self._source.currentTextChanged.connect(self._on_editing_finished)
        self._confirmed.toggled.connect(self._on_editing_finished)
        self.set_anchor(None)

    def _make_spin_box(
        self, minimum: float, maximum: float
    ) -> QtWidgets.QDoubleSpinBox:
        spin_box = QtWidgets.QDoubleSpinBox(self)
        spin_box.setDecimals(_DECIMALS)
        spin_box.setRange(minimum, maximum)
        spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        return spin_box

    def set_anchor(
        self, values: tuple[float, float, str, float, bool] | None
    ) -> None:
        self.setEnabled(values is not None)
        if values is None:
            return
        position, visibility, source, confidence, confirmed = values
        blockers = [
            QtCore.QSignalBlocker(self._position),
            QtCore.QSignalBlocker(self._visibility),
            QtCore.QSignalBlocker(self._source),
            QtCore.QSignalBlocker(self._confidence),
            QtCore.QSignalBlocker(self._confirmed),
        ]
        self._position.setValue(position)
        self._visibility.setValue(visibility)
        self._source.setCurrentText(source)
        self._confidence.setValue(confidence)
        self._confirmed.setChecked(confirmed)
        del blockers

    def _on_editing_finished(self) -> None:
        if not self.isEnabled():
            return
        self.visibility_committed.emit(
            self._position.value(),
            self._visibility.value(),
            self._source.currentText(),
            self._confidence.value(),
            self._confirmed.isChecked(),
        )
