from __future__ import annotations

from typing import Final

from PySide6 import QtCore
from PySide6 import QtWidgets

_RADIUS_MAX: Final = 1_000_000_000.0
_RADIUS_DECIMALS: Final = 3
_RADIUS_SUFFIX: Final = " px"

# Tolerance for treating an unchanged spin-box value as a no-op; half of the
# smallest displayable increment keeps programmatic syncs from committing
# rounding noise back onto the shape.
_COMMIT_EPSILON: Final = 0.5 * 10 ** (-_RADIUS_DECIMALS)


class CircleRadiusWidget(QtWidgets.QWidget):
    """Toolbar control for editing the radius of the selected circle.

    Enabled only while exactly one circle shape is selected; emits
    radius_committed when the user commits a new value (Return or focus
    loss). The widget itself never mutates shapes.
    """

    radius_committed = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        label = QtWidgets.QLabel(self.tr("Radius"), self)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setToolTip(self.tr("Radius of the selected circle"))
        label.setStatusTip(self.tr("Radius of the selected circle"))

        self._spin_box = QtWidgets.QDoubleSpinBox(self)
        self._spin_box.setDecimals(_RADIUS_DECIMALS)
        self._spin_box.setRange(0.0, _RADIUS_MAX)
        self._spin_box.setSuffix(_RADIUS_SUFFIX)
        self._spin_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._spin_box.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self._spin_box.setToolTip(self.tr("Radius of the selected circle"))
        self._spin_box.setStatusTip(self.tr("Radius of the selected circle"))
        self._spin_box.editingFinished.connect(self._on_editing_finished)

        sample = f"{_RADIUS_MAX:.{_RADIUS_DECIMALS}f}"
        self._spin_box.setMinimumWidth(
            self._spin_box.fontMetrics().horizontalAdvance(sample)
        )

        # Keep the same layout geometry as the zoom control in the toolbar:
        # default contents margins and spacing (a bare QVBoxLayout, like the
        # one built for the zoom widget), so the label and the spin box line
        # up pixel-identically with theirs.
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self._spin_box)

        self._last_radius: float | None = None
        self.set_radius(None)

    @property
    def radius_spin_box(self) -> QtWidgets.QDoubleSpinBox:
        return self._spin_box

    def set_radius(self, radius: float | None) -> None:
        """Show the radius of the currently selected circle.

        ``None`` (no circle selected) disables the control.
        """
        if radius is None:
            self._last_radius = None
            self._spin_box.setEnabled(False)
            return
        self._last_radius = float(radius)
        self._spin_box.setEnabled(True)
        self._spin_box.setValue(self._last_radius)

    def _on_editing_finished(self) -> None:
        if not self._spin_box.isEnabled():
            return
        value = self._spin_box.value()
        if (
            self._last_radius is not None
            and abs(value - self._last_radius) <= _COMMIT_EPSILON
        ):
            return
        self.radius_committed.emit(value)
