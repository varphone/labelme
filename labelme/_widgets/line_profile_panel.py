from __future__ import annotations

from collections.abc import Iterable

from PySide6 import QtGui
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from .line_profile_visibility_widget import LineProfileVisibilityWidget
from .line_profile_width_widget import LineProfileWidthWidget


class LineProfilePanel(QtWidgets.QWidget):
    """Vertical editor and action panel for the selected linestrip profile."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("LineProfilePanel")
        self.setMinimumWidth(200)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        description = QtWidgets.QLabel(
            self.tr("Select a linestrip to edit its width and visibility profile."),
            self,
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(description)

        actions_group = QtWidgets.QGroupBox(self.tr("Profile Actions"), self)
        self._actions_layout = QtWidgets.QVBoxLayout(actions_group)
        self._actions_layout.setContentsMargins(6, 8, 6, 6)
        self._actions_layout.setSpacing(4)
        layout.addWidget(actions_group)

        self.width_widget = LineProfileWidthWidget(self)
        self.position_widget = self.width_widget.position_widget
        self.position_widget.setEnabled(False)
        anchor_group = QtWidgets.QGroupBox(self.tr("Anchor"), self)
        anchor_layout = QtWidgets.QFormLayout(anchor_group)
        anchor_layout.addRow(self.tr("Position"), self.position_widget)
        layout.addWidget(anchor_group)

        width_group = QtWidgets.QGroupBox(self.tr("Width"), self)
        width_layout = QtWidgets.QVBoxLayout(width_group)
        width_layout.setContentsMargins(6, 8, 6, 6)
        width_layout.addWidget(self.width_widget)
        layout.addWidget(width_group)

        self.visibility_widget = LineProfileVisibilityWidget(self)
        visibility_group = QtWidgets.QGroupBox(self.tr("Visibility"), self)
        visibility_layout = QtWidgets.QVBoxLayout(visibility_group)
        visibility_layout.setContentsMargins(6, 8, 6, 6)
        visibility_layout.addWidget(self.visibility_widget)
        layout.addWidget(visibility_group)

        layout.addStretch(1)

    def set_actions(
        self,
        actions: Iterable[QtGui.QAction | tuple[QtGui.QAction, str] | None],
    ) -> None:
        """Add shared actions as compact icon-and-text buttons."""
        while (item := self._actions_layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for action in actions:
            if action is None:
                line = QtWidgets.QFrame(self)
                line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
                self._actions_layout.addWidget(line)
                continue
            if isinstance(action, tuple):
                source_action, label = action
            else:
                source_action = action
                label = source_action.text()
            display_action = QtGui.QAction(source_action.icon(), label, self)
            display_action.setToolTip(source_action.toolTip())
            display_action.setCheckable(source_action.isCheckable())
            display_action.setChecked(source_action.isChecked())
            display_action.setEnabled(source_action.isEnabled())
            display_action.triggered.connect(
                lambda _checked=False,
                source_action=source_action: source_action.trigger()
            )

            def sync_display_action(
                source_action: QtGui.QAction = source_action,
                display_action: QtGui.QAction = display_action,
            ) -> None:
                display_action.setIcon(source_action.icon())
                display_action.setToolTip(source_action.toolTip())
                display_action.setEnabled(source_action.isEnabled())
                display_action.setChecked(source_action.isChecked())

            source_action.changed.connect(sync_display_action)
            button = QtWidgets.QToolButton(self)
            button.setProperty("lineProfileSourceAction", source_action)
            button.setDefaultAction(display_action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setText(label)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            self._actions_layout.addWidget(button)
