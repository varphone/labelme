from __future__ import annotations

from PySide6 import QtCore
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from pytestqt.qtbot import QtBot

from labelme._widgets.canvas import Canvas
from labelme._widgets.minimap import MinimapWidget


def test_minimap_click_is_safe_without_an_image(qtbot: QtBot) -> None:
    canvas = Canvas()
    viewport = canvas.parentWidget()
    assert viewport is None

    scroll_area = QtWidgets.QScrollArea()
    minimap = MinimapWidget(canvas=canvas, viewport=scroll_area.viewport())
    qtbot.addWidget(scroll_area)
    qtbot.addWidget(canvas)
    qtbot.addWidget(minimap)
    spy = QSignalSpy(minimap.center_requested)

    qtbot.mouseClick(minimap, Qt.MouseButton.LeftButton, pos=QtCore.QPoint(10, 10))

    assert spy.count() == 0
