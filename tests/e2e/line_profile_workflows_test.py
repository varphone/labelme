from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtCore import QSize
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QToolBar
from PySide6.QtWidgets import QToolButton
from pytestqt.qtbot import QtBot

from labelme import _app
from labelme._label_file import Annotation
from labelme._label_file import ShapeDict
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._line_profile import LineProfile
from labelme._line_profile import ProfileAnchor
from labelme._line_profile import ProfileValue
from labelme._shape import Shape

from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
def test_line_profile_panel_is_docked_before_annotation_panels(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw" / "2011_000003.jpg"),
        size=QSize(1200, 800),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    profile_dock = win._docks.line_profile_dock
    panel = win._docks.line_profile_panel
    assert profile_dock.isVisible()
    assert win.dockWidgetArea(profile_dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert panel.isAncestorOf(panel.width_widget)
    assert panel.isAncestorOf(panel.visibility_widget)
    assert not panel.position_widget.isEnabled()
    assert panel.position_widget.parentWidget() is not panel.width_widget
    assert panel.position_widget.parentWidget() is not panel.visibility_widget
    assert all(
        label.text() != "Position"
        for widget in (panel.width_widget, panel.visibility_widget)
        for label in widget.findChildren(QLabel)
    )
    existing_docks = (
        win._docks.flag_dock,
        win._docks.label_dock,
        win._docks.shape_dock,
        win._docks.file_dock,
    )
    assert all(
        profile_dock.geometry().right() < dock.geometry().left()
        for dock in existing_docks
    )
    assert profile_dock.geometry().top() <= min(
        dock.geometry().top() for dock in existing_docks
    )
    assert profile_dock.geometry().bottom() >= max(
        dock.geometry().bottom() for dock in existing_docks
    )
    assert [dock.geometry().top() for dock in existing_docks] == sorted(
        dock.geometry().top() for dock in existing_docks
    )
    assert all(not win.tabifiedDockWidgets(dock) for dock in existing_docks)

    file_menu = win._build_file_list_context_menu()
    batch_action = next(
        action
        for action in file_menu.actions()
        if action.menu() is not None and action.text() == "Batch Line Profile"
    )
    batch_menu = batch_action.menu()
    assert batch_menu is not None
    assert [action.text() for action in batch_menu.actions()] == [
        "Fill Missing Line Profiles",
        "Rebuild Line Profiles",
    ]
    file_menu.deleteLater()

    action_texts = {
        button.property("lineProfileSourceAction").text()
        for button in panel.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert "Measure Line Profile" in action_texts
    assert "Insert Profile Anchor" in action_texts
    assert "Clear Line Profile" in action_texts
    panel_buttons = {
        button.text(): button
        for button in panel.findChildren(QToolButton)
        if button.defaultAction() is not None
    }
    assert [
        button.text()
        for button in panel.findChildren(QToolButton)
        if button.defaultAction() is not None
    ] == [
        "Preview",
        "Measure",
        "Insert",
        "Delete",
        "Clear",
        "Copy Previous",
        "Parameters",
    ]
    assert {
        "Measure",
        "Preview",
        "Insert",
        "Delete",
        "Parameters",
        "Clear",
    } <= panel_buttons.keys()
    assert all(not button.icon().isNull() for button in panel_buttons.values())
    preview_button = panel_buttons["Preview"]
    assert win._actions.show_line_profile_preview.isChecked()
    preview_button.click()
    qtbot.wait(10)
    assert not win._actions.show_line_profile_preview.isChecked()
    preview_button.click()
    qtbot.wait(10)
    assert win._actions.show_line_profile_preview.isChecked()

    tools_toolbar = win.findChild(QToolBar, "ToolsToolBar")
    assert tools_toolbar is not None
    assert panel.width_widget not in tools_toolbar.findChildren(
        type(panel.width_widget)
    )
    assert panel.visibility_widget not in tools_toolbar.findChildren(
        type(panel.visibility_widget)
    )


@pytest.mark.gui
def test_line_profile_parameters_are_available_before_measurement(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw" / "2011_000003.jpg"),
        size=QSize(900, 700),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="line",
        shape_type="linestrip",
        points=np.array([[10.0, 20.0], [180.0, 20.0]], dtype=np.float64),
    )
    win._load_shapes([shape], replace=True)
    win._canvas_widgets.canvas.select_shapes(shapes=[shape])

    assert win._actions.line_profile_measurement_parameters.isEnabled()
    button = next(
        button
        for button in win._docks.line_profile_panel.findChildren(QToolButton)
        if button.text() == "Parameters"
    )
    assert button.isEnabled()


@pytest.mark.gui
def test_profile_preview_zoom_and_undo_preserve_centerline(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw" / "2011_000003.jpg"),
        size=QSize(900, 700),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="stripe",
        shape_type="linestrip",
        points=np.array([[10.0, 20.0], [180.0, 20.0]], dtype=np.float64),
        line_profile=LineProfile(
            anchors=tuple(
                sorted(
                    (
                        ProfileAnchor(
                            0.0,
                            width=ProfileValue(12.0, "manual", 1.0, True),
                            visibility=ProfileValue(1.0, "manual", 1.0, True),
                        ),
                        *(
                            ProfileAnchor(
                                position,
                                width=ProfileValue(20.0, "auto", 0.7, False),
                            )
                            for position in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
                        ),
                        ProfileAnchor(
                            0.5,
                            visibility=ProfileValue(0.5, "auto", 0.4, False),
                        ),
                        ProfileAnchor(
                            1.0,
                            width=ProfileValue(80.0, "auto", 0.8, False),
                        ),
                    ),
                    key=lambda anchor: anchor.position,
                )
            )
        ),
    )
    win._load_shapes([shape], replace=True)
    win._canvas_widgets.canvas.select_shapes(shapes=[shape])

    original_points = shape.points.copy()
    for zoom in (25.0, 100.0, 400.0):
        win._set_zoom(value=zoom, pos=None)
        assert (
            win._canvas_widgets.canvas._find_line_profile_anchor_at_point(
                QPointF(10.0, 20.0)
            )
            is not None
        )
        assert (
            win._canvas_widgets.canvas._find_line_profile_anchor_at_point(
                QPointF(95.0, 20.0)
            )
            is not None
        )

    win._active_line_profile_anchor_kind = "width"
    win._active_line_profile_anchor_index = 0
    win.delete_line_profile_anchor()
    assert shape.line_profile is not None
    assert shape.line_profile.anchors[0].width is None
    assert shape.line_profile.anchors[0].visibility is not None
    win._active_line_profile_anchor_kind = "visibility"
    win._active_line_profile_anchor_index = 0
    win.delete_line_profile_anchor()
    assert shape.line_profile is not None
    assert all(anchor.position != 0.0 for anchor in shape.line_profile.anchors)

    win.clear_selected_line_profile()
    assert shape.line_profile is None
    win.undo_shape_edit()
    restored = win._canvas_widgets.canvas.shapes[0]
    assert restored.line_profile is not None
    np.testing.assert_array_equal(restored.points, original_points)
    assert win._docks.label_list[0].shape().line_profile is not None


@pytest.mark.gui
def test_profile_anchor_drag_repaints_without_syncing_panel_each_move(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw" / "2011_000003.jpg"),
        size=QSize(900, 700),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="stripe",
        shape_type="linestrip",
        points=np.array([[10.0, 20.0], [180.0, 20.0]], dtype=np.float64),
        line_profile=LineProfile(
            anchors=(
                ProfileAnchor(
                    0.5, width=ProfileValue(12.0, "manual", 1.0, True)
                ),
            )
        ),
    )
    win._load_shapes([shape], replace=True)
    canvas = win._canvas_widgets.canvas
    canvas.select_shapes(shapes=[shape])

    sync_panel = Mock()
    update_canvas = Mock()
    build_centerline = Mock(wraps=_app.line_profile_points)
    monkeypatch.setattr(_app, "line_profile_points", build_centerline)
    monkeypatch.setattr(win, "_sync_line_profile_widgets", sync_panel)
    monkeypatch.setattr(canvas, "update", update_canvas)

    win._on_line_profile_anchor_dragged(
        "width", 0, QPointF(85.0, 30.0), "width"
    )
    win._on_line_profile_anchor_dragged(
        "width", 0, QPointF(85.0, 35.0), "width"
    )

    assert sync_panel.call_count == 0
    assert update_canvas.call_count == 2
    assert build_centerline.call_count == 1

    win._on_line_profile_anchor_drag_finished()

    assert sync_panel.call_count == 1


@pytest.mark.gui
def test_continuous_frame_profile_copy_requires_mapping_and_does_not_write_previous(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = read_label_file(
        filename=str(data_path / "annotated" / "2011_000003.json")
    )
    image_bytes = source.image_data
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    source_shape: ShapeDict = {
        "label": "stripe",
        "points": [[10.0, 20.0], [180.0, 20.0]],
        "shape_type": "linestrip",
        "flags": {},
        "description": "",
        "group_id": None,
        "mask": None,
        "other_data": {},
        "line_profile": LineProfile(
            anchors=(
                ProfileAnchor(
                    0.0,
                    width=ProfileValue(12.0, "manual", 1.0, True),
                ),
            )
        ),
    }
    target_shape: ShapeDict = {**source_shape}
    target_shape.pop("line_profile")
    for name, shapes in (("frame-01", [source_shape]), ("frame-02", [target_shape])):
        image_path = frames_dir / f"{name}.jpg"
        image_path.write_bytes(image_bytes)
        write_label_file(
            filename=str(frames_dir / f"{name}.json"),
            annotation=Annotation(
                image_path=image_path.name,
                image_data=image_bytes,
                shapes=shapes,
                flags={},
                other_data={},
            ),
            image_height=None,
            image_width=None,
            save_image_data=True,
        )
    previous_frame_bytes = (frames_dir / "frame-01.json").read_bytes()

    win = main_win(file_or_dir=frames_dir)
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    assert win._docks.file_list.count() == 2
    win._docks.file_list.setCurrentRow(1)
    qtbot.waitUntil(
        lambda: win._image_path is not None and "frame-02" in win._image_path
    )

    monkeypatch.setattr(
        _app.QMessageBox,
        "question",
        lambda *args, **kwargs: _app.QMessageBox.StandardButton.Yes,
    )
    win.copy_profiles_from_previous_frame()

    current_shape = win._canvas_widgets.canvas.shapes[0]
    assert current_shape.line_profile is not None
    assert (frames_dir / "frame-01.json").read_bytes() == previous_frame_bytes
