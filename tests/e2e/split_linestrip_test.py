from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QPoint
from PySide6.QtCore import QPointF
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow
from labelme._widgets.canvas import Canvas

from .conftest import MainWinFactory
from .conftest import click_canvas_fraction
from .conftest import image_to_widget_pos
from .conftest import show_window_and_wait_for_imagedata
from .conftest import submit_label_dialog


def _draw_linestrip(
    qtbot: QtBot,
    win: MainWindow,
    vertices: list[tuple[float, float]],
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="linestrip")
    qtbot.wait(50)
    for xy in vertices:
        click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=xy)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label="road")
    qtbot.keyPress(canvas, Qt.Key.Key_Return)
    qtbot.wait(200)

    def committed() -> None:
        assert len(canvas.shapes) >= 1
        assert canvas.shapes[-1].shape_type == "linestrip"

    qtbot.waitUntil(committed)


def _select_linestrip(
    qtbot: QtBot,
    canvas: Canvas,
    shape_index: int = 0,
) -> None:
    """Select a linestrip by clicking the midpoint of its first segment."""
    shape = canvas.shapes[shape_index]
    mid = QPointF(
        float((shape.points[0][0] + shape.points[1][0]) / 2),
        float((shape.points[0][1] + shape.points[1][1]) / 2),
    )
    pos = image_to_widget_pos(canvas=canvas, image_pos=mid)
    qtbot.mouseMove(canvas, pos=pos)
    qtbot.wait(50)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pos)
    qtbot.wait(50)
    assert len(canvas.selected_shapes) == 1


def _hover_vertex(
    qtbot: QtBot,
    canvas: Canvas,
    vertex_pos: QPointF,
) -> None:
    pos = image_to_widget_pos(canvas=canvas, image_pos=vertex_pos)
    qtbot.mouseMove(canvas, pos=QPoint(0, 0))
    qtbot.mouseMove(canvas, pos=pos)
    qtbot.wait(50)


def test_split_linestrip_at_vertex(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    vertices = [(0.1, 0.1), (0.5, 0.3), (0.9, 0.2)]
    _draw_linestrip(qtbot=qtbot, win=win, vertices=vertices)
    shape = canvas.shapes[0]
    original_points = shape.points.copy()
    assert len(original_points) == 3

    win._switch_canvas_mode(edit=True, create_mode=None)
    _select_linestrip(qtbot=qtbot, canvas=canvas, shape_index=0)

    # Hover the middle vertex (index 1).
    mid = shape.points[1]
    _hover_vertex(qtbot=qtbot, canvas=canvas, vertex_pos=QPointF(mid[0], mid[1]))
    assert canvas._hovered_vertex == 1
    assert win._actions.split_linestrip.isEnabled()

    # Simulate the mouse leaving the canvas (as happens when the user moves
    # the cursor to a context-menu item). The action must stay enabled
    # because _last_hovered_vertex preserves the vertex index.
    canvas.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert canvas._hovered_vertex is None
    assert canvas._last_hovered_vertex == 1
    assert win._actions.split_linestrip.isEnabled()

    win.split_linestrip()
    qtbot.wait(50)

    # The original shape is replaced by two linestrips sharing the split vertex.
    assert len(canvas.shapes) == 2
    left, right = canvas.shapes
    assert left.shape_type == "linestrip"
    assert right.shape_type == "linestrip"
    assert len(left.points) == 2
    assert len(right.points) == 2
    assert left.label == right.label == "road"
    assert np.allclose(left.points[0], original_points[0])
    assert np.allclose(left.points[1], original_points[1])
    assert np.allclose(right.points[0], original_points[1])
    assert np.allclose(right.points[1], original_points[2])

    # Undo restores the single original shape.
    win.undo_shape_edit()
    qtbot.wait(50)
    assert len(canvas.shapes) == 1
    assert canvas.shapes[0].shape_type == "linestrip"
    assert len(canvas.shapes[0].points) == 3


def test_split_linestrip_refuses_at_endpoint(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    _draw_linestrip(
        qtbot=qtbot,
        win=win,
        vertices=[(0.1, 0.1), (0.5, 0.3), (0.9, 0.2)],
    )
    shape = canvas.shapes[0]
    num_before = len(canvas.shapes)

    win._switch_canvas_mode(edit=True, create_mode=None)
    _select_linestrip(qtbot=qtbot, canvas=canvas, shape_index=0)

    # Hover the first endpoint (index 0); the action is disabled because
    # splitting at an endpoint would produce a one-point degenerate linestrip.
    first = shape.points[0]
    _hover_vertex(qtbot=qtbot, canvas=canvas, vertex_pos=QPointF(first[0], first[1]))
    assert canvas._hovered_vertex == 0
    assert not win._actions.split_linestrip.isEnabled()

    # Even if invoked programmatically, the canvas refuses to split.
    win.split_linestrip()
    qtbot.wait(50)
    assert len(canvas.shapes) == num_before


def test_split_linestrip_refuses_for_polygon(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "annotated/2011_000003.json"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=True, create_mode=None)

    # The annotated file contains polygon shapes; the action stays disabled
    # even when a vertex is hovered, because the selected shape is not a
    # linestrip.
    polygon_idx = next(
        i for i, s in enumerate(canvas.shapes) if s.shape_type == "polygon"
    )
    shape = canvas.shapes[polygon_idx]
    mid = QPointF(
        float((shape.points[0][0] + shape.points[1][0]) / 2),
        float((shape.points[0][1] + shape.points[1][1]) / 2),
    )
    pos = image_to_widget_pos(canvas=canvas, image_pos=mid)
    qtbot.mouseMove(canvas, pos=pos)
    qtbot.wait(50)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pos)
    qtbot.wait(50)

    _hover_vertex(
        qtbot=qtbot,
        canvas=canvas,
        vertex_pos=QPointF(float(shape.points[1][0]), float(shape.points[1][1])),
    )
    assert not win._actions.split_linestrip.isEnabled()
    num_before = len(canvas.shapes)
    win.split_linestrip()
    qtbot.wait(50)
    assert len(canvas.shapes) == num_before
