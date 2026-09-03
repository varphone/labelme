from __future__ import annotations

from pathlib import Path

import numpy as np
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

# Click offsets in image pixels. Must stay inside the snap threshold
# (epsilon / scale) for every canvas scale used by the tests.
_SNAP_OFFSET = 4.0


def _draw_rectangle(
    qtbot: QtBot,
    win: MainWindow,
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> None:
    canvas = win._canvas_widgets.canvas
    num_before = len(canvas.shapes)
    win._switch_canvas_mode(edit=False, create_mode="rectangle")
    qtbot.wait(50)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p1)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label="rect")
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p2)

    def shape_committed() -> None:
        assert len(canvas.shapes) == num_before + 1
        assert canvas.shapes[-1].shape_type == "rectangle"

    qtbot.waitUntil(shape_committed)


def _expected_landing_pos(canvas: Canvas, image_pos: QPointF) -> QPointF:
    """Image position where a click lands after widget-coordinate rounding."""
    widget_pos = image_to_widget_pos(canvas=canvas, image_pos=image_pos)
    return canvas.transform_widget_point_to_image(QPointF(widget_pos))


def _draw_line(
    qtbot: QtBot,
    win: MainWindow,
    p1: QPointF,
    p2: QPointF,
) -> None:
    canvas = win._canvas_widgets.canvas
    num_before = len(canvas.shapes)
    win._switch_canvas_mode(edit=False, create_mode="line")
    qtbot.wait(50)

    def click(image_pos: QPointF) -> None:
        pos = image_to_widget_pos(canvas=canvas, image_pos=image_pos)
        qtbot.mouseMove(canvas, pos=pos)
        qtbot.wait(50)
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=pos)
        qtbot.wait(50)

    click(p1)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label="line")
    click(p2)

    def shape_committed() -> None:
        assert len(canvas.shapes) == num_before + 1
        assert canvas.shapes[-1].shape_type == "line"

    qtbot.waitUntil(shape_committed)


def test_snap_to_point_snaps_cursor_to_annotation_points(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    # An existing annotation provides the snap targets.
    _draw_rectangle(qtbot=qtbot, win=win, p1=(0.2, 0.2), p2=(0.4, 0.4))
    corner_a = canvas.shapes[0].points[0]
    corner_b = canvas.shapes[0].points[1]

    # Snap is off by default.
    assert not canvas.snap_to_point
    assert not win._actions.toggle_snap_to_point.isChecked()

    # Enable snap through the toolbar action.
    win._actions.toggle_snap_to_point.setChecked(True)
    assert canvas.snap_to_point
    assert win._config["snap_to_point"]

    # A click near a corner lands exactly on the corner.
    _draw_line(
        qtbot=qtbot,
        win=win,
        p1=QPointF(corner_a[0] + _SNAP_OFFSET, corner_a[1] + _SNAP_OFFSET),
        p2=_expected_landing_pos(
            canvas, QPointF(0.8 * canvas.pixmap.width(), 0.5 * canvas.pixmap.height())
        ),
    )
    line = canvas.shapes[1]
    assert np.allclose(line.points[0], corner_a, atol=1e-6)

    # The preview line snaps on the closing click as well.
    _draw_line(
        qtbot=qtbot,
        win=win,
        p1=_expected_landing_pos(
            canvas, QPointF(0.1 * canvas.pixmap.width(), 0.9 * canvas.pixmap.height())
        ),
        p2=QPointF(corner_b[0] + _SNAP_OFFSET, corner_b[1] - _SNAP_OFFSET),
    )
    line = canvas.shapes[2]
    assert np.allclose(line.points[1], corner_b, atol=1e-6)

    # Disabling snap leaves the click position untouched.
    win._actions.toggle_snap_to_point.setChecked(False)
    assert not canvas.snap_to_point
    assert not win._config["snap_to_point"]
    click_target = QPointF(corner_a[0] + _SNAP_OFFSET, corner_a[1] + _SNAP_OFFSET)
    _draw_line(
        qtbot=qtbot,
        win=win,
        p1=click_target,
        p2=_expected_landing_pos(
            canvas, QPointF(0.8 * canvas.pixmap.width(), 0.5 * canvas.pixmap.height())
        ),
    )
    line = canvas.shapes[3]
    assert not np.allclose(line.points[0], corner_a, atol=0.5)
    assert np.allclose(
        line.points[0],
        [
            _expected_landing_pos(canvas, click_target).x(),
            _expected_landing_pos(canvas, click_target).y(),
        ],
        atol=1e-6,
    )
