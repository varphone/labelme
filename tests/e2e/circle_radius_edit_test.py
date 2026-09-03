from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow
from labelme._shape import Shape
from labelme._widgets.circle_radius_widget import CircleRadiusWidget

from .conftest import MainWinFactory
from .conftest import click_canvas_fraction
from .conftest import select_shape
from .conftest import show_window_and_wait_for_imagedata
from .conftest import submit_label_dialog


def _radius_widget(win: MainWindow) -> CircleRadiusWidget:
    widget = win._actions.circle_radius_action.defaultWidget()
    assert isinstance(widget, CircleRadiusWidget)
    return widget


def _draw_circle(
    qtbot: QtBot,
    win: MainWindow,
    center: tuple[float, float],
    circumference: tuple[float, float],
    label: str = "circle",
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="circle")
    qtbot.wait(50)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=center)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label=label)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=circumference)

    def shape_committed() -> None:
        assert len(canvas.shapes) == 1
        assert canvas.shapes[0].label == label
        assert canvas.shapes[0].shape_type == "circle"

    qtbot.waitUntil(shape_committed)


def _draw_rectangle(
    qtbot: QtBot,
    win: MainWindow,
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str = "rect",
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="rectangle")
    qtbot.wait(50)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p1)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label=label)
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=p2)

    def shape_committed() -> None:
        assert len(canvas.shapes) == 2
        assert canvas.shapes[1].label == label
        assert canvas.shapes[1].shape_type == "rectangle"

    qtbot.waitUntil(shape_committed)


def _circle_radius(shape: Shape) -> float:
    return float(np.linalg.norm(shape.points[1] - shape.points[0]))


def test_edit_selected_circle_radius_via_toolbar(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw/2011_000003.jpg"),
        config_overrides={"auto_save": False},
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    _draw_circle(
        qtbot=qtbot,
        win=win,
        center=(0.5, 0.5),
        circumference=(0.7, 0.5),
    )

    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=True, create_mode=None)
    select_shape(qtbot=qtbot, canvas=canvas, shape_index=0)

    widget = _radius_widget(win)
    spin_box = widget.radius_spin_box
    shape = canvas.selected_shapes[0]
    assert shape.shape_type == "circle"
    assert spin_box.isEnabled()
    assert spin_box.value() == pytest.approx(_circle_radius(shape), abs=0.001)

    # Commit a new radius from the toolbar spin box.
    radius_before = _circle_radius(shape)
    new_radius = radius_before / 2
    center_before = shape.points[0].copy()
    spin_box.setFocus()
    spin_box.selectAll()
    qtbot.keyClicks(spin_box, f"{new_radius:.3f}")
    qtbot.keyClick(spin_box, Qt.Key.Key_Return)

    assert np.linalg.norm(shape.points[1] - shape.points[0]) == pytest.approx(
        new_radius, abs=0.001
    )
    assert np.allclose(shape.points[0], center_before)
    assert spin_box.value() == pytest.approx(new_radius, abs=0.001)
    assert win._is_changed
    assert win._actions.undo.isEnabled()

    # A zero radius is rejected: the circle keeps its radius and the control
    # snaps back to the real value.
    spin_box.setFocus()
    spin_box.selectAll()
    qtbot.keyClicks(spin_box, "0")
    qtbot.keyClick(spin_box, Qt.Key.Key_Return)
    assert np.linalg.norm(shape.points[1] - shape.points[0]) == pytest.approx(
        new_radius, abs=0.001
    )
    assert spin_box.value() == pytest.approx(new_radius, abs=0.001)

    # Undo restores the previous radius.
    win.undo_shape_edit()
    assert np.linalg.norm(
        canvas.shapes[0].points[1] - canvas.shapes[0].points[0]
    ) == pytest.approx(radius_before, abs=0.001)

    # Leave the window clean so closing it does not prompt to save.
    if win.save_labels(label_path=str(tmp_path / "2011_000003.json")):
        win.mark_clean()


def test_circle_radius_widget_follows_selection(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    _draw_circle(
        qtbot=qtbot,
        win=win,
        center=(0.25, 0.25),
        circumference=(0.35, 0.25),
    )
    _draw_rectangle(
        qtbot=qtbot,
        win=win,
        p1=(0.6, 0.6),
        p2=(0.8, 0.8),
    )

    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=True, create_mode=None)
    win._set_zoom(value=100.0, pos=None)
    qtbot.wait(50)

    # No selection: the radius control is disabled.
    widget = _radius_widget(win)
    spin_box = widget.radius_spin_box
    assert not spin_box.isEnabled()

    # Selecting the circle enables the control and shows its radius.
    select_shape(qtbot=qtbot, canvas=canvas, shape_index=0)
    shape = canvas.selected_shapes[0]
    assert spin_box.isEnabled()
    assert spin_box.value() == pytest.approx(_circle_radius(shape), abs=0.001)

    # Selecting the rectangle disables the control again.
    select_shape(qtbot=qtbot, canvas=canvas, shape_index=1)
    assert canvas.selected_shapes[0].shape_type == "rectangle"
    assert not spin_box.isEnabled()

    # Deselecting leaves it disabled.
    click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=(0.05, 0.05))
    assert len(canvas.selected_shapes) == 0
    assert not spin_box.isEnabled()


def test_circle_radius_widget_stays_disabled_until_circle_selected(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    """A raw image without annotations leaves the control disabled."""
    win = main_win(file_or_dir=str(data_path / "raw/2011_000003.jpg"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    widget = _radius_widget(win)
    assert not widget.radius_spin_box.isEnabled()
