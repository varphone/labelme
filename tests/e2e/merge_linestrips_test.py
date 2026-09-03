from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow
from labelme._widgets.canvas import Canvas
from labelme._widgets.label_list_widget import LabelListWidget

from .conftest import MainWinFactory
from .conftest import click_canvas_fraction
from .conftest import show_window_and_wait_for_imagedata
from .conftest import submit_label_dialog


def _draw_linestrip(
    qtbot: QtBot,
    win: MainWindow,
    vertices: list[tuple[float, float]],
    label: str = "road",
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="linestrip")
    qtbot.wait(50)
    for xy in vertices:
        click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=xy)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label=label)
    qtbot.keyPress(canvas, Qt.Key.Key_Return)
    qtbot.wait(200)

    def committed() -> None:
        assert canvas.shapes[-1].shape_type == "linestrip"
        assert canvas.shapes[-1].label == label

    qtbot.waitUntil(committed)


def _draw_line(
    qtbot: QtBot,
    win: MainWindow,
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str = "road",
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="line")
    qtbot.wait(50)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p1)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label=label)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p2)

    def committed() -> None:
        assert canvas.shapes[-1].shape_type == "line"
        assert canvas.shapes[-1].label == label

    qtbot.waitUntil(committed)


def _select_label_list_items(
    canvas: Canvas, label_list: LabelListWidget, indices: list[int]
) -> None:
    """Select items in the label list by their indices."""
    label_list.clearSelection()
    for idx in indices:
        item = label_list.find_item_by_shape(shape=canvas.shapes[idx])
        label_list.select_item(item=item)


def test_merge_two_linestrips(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    # Draw two linestrips.
    _draw_linestrip(qtbot=qtbot, win=win, vertices=[(0.1, 0.1), (0.3, 0.2)])
    _draw_linestrip(qtbot=qtbot, win=win, vertices=[(0.3, 0.2), (0.5, 0.3)])
    assert len(canvas.shapes) == 2
    p0 = canvas.shapes[0].points.copy()
    p1 = canvas.shapes[1].points.copy()

    # Select both in the label list and merge.
    label_list = win._docks.label_list
    _select_label_list_items(canvas, label_list, [0, 1])
    assert len(label_list.selected_items()) == 2

    win.merge_linestrips()
    qtbot.wait(50)

    assert len(canvas.shapes) == 1
    merged = canvas.shapes[0]
    assert merged.shape_type == "linestrip"
    assert merged.label == "road"
    assert len(merged.points) == 4
    assert np.allclose(merged.points[0], p0[0])
    assert np.allclose(merged.points[1], p0[1])
    assert np.allclose(merged.points[2], p1[0])
    assert np.allclose(merged.points[3], p1[1])

    # Undo restores both original shapes.
    win.undo_shape_edit()
    qtbot.wait(50)
    assert len(canvas.shapes) == 2
    assert canvas.shapes[0].shape_type == "linestrip"
    assert canvas.shapes[1].shape_type == "linestrip"


def test_merge_line_and_linestrip(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    _draw_line(qtbot=qtbot, win=win, p1=(0.1, 0.1), p2=(0.3, 0.2))
    _draw_linestrip(qtbot=qtbot, win=win, vertices=[(0.3, 0.2), (0.5, 0.3), (0.7, 0.2)])
    assert len(canvas.shapes) == 2

    label_list = win._docks.label_list
    _select_label_list_items(canvas, label_list, [0, 1])
    win.merge_linestrips()
    qtbot.wait(50)

    assert len(canvas.shapes) == 1
    merged = canvas.shapes[0]
    assert merged.shape_type == "linestrip"
    # line (2 points) + linestrip (3 points) = 5 points
    assert len(merged.points) == 5


def test_merge_reverses_shapes_to_avoid_backtracking(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    # First shape: A → B
    _draw_linestrip(
        qtbot=qtbot,
        win=win,
        vertices=[(0.1, 0.1), (0.3, 0.2)],
        label="seg",
    )
    # Second shape: C → D, where D is closer to B than C is.
    # Without reversal the merge would go A→B→C→D (backtrack at B→C).
    # With reversal it should go A→B→D→C.
    _draw_linestrip(
        qtbot=qtbot,
        win=win,
        vertices=[(0.7, 0.4), (0.35, 0.22)],
        label="seg",
    )
    assert len(canvas.shapes) == 2
    p0 = canvas.shapes[0].points.copy()
    p1 = canvas.shapes[1].points.copy()

    label_list = win._docks.label_list
    _select_label_list_items(canvas, label_list, [0, 1])
    win.merge_linestrips()
    qtbot.wait(50)

    merged = canvas.shapes[0]
    assert len(merged.points) == 4
    # The first shape's points are kept as-is.
    assert np.allclose(merged.points[0], p0[0])
    assert np.allclose(merged.points[1], p0[1])
    # The second shape is reversed: its last point (closer to p0[1]) comes next.
    assert np.allclose(merged.points[2], p1[1])
    assert np.allclose(merged.points[3], p1[0])


def test_merge_refuses_for_polygon(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "annotated/2011_000003.json"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas
    num_before = len(canvas.shapes)

    # The annotated file has polygon shapes — merge should refuse.
    label_list = win._docks.label_list
    polygon_indices = [
        i for i, s in enumerate(canvas.shapes) if s.shape_type == "polygon"
    ]
    if len(polygon_indices) >= 2:
        _select_label_list_items(canvas, label_list, polygon_indices[:2])
        win.merge_linestrips()
        qtbot.wait(50)
        assert len(canvas.shapes) == num_before
