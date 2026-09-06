from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from labelme._app import MainWindow

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import click_canvas_fraction
from .conftest import show_window_and_wait_for_imagedata
from .conftest import submit_label_dialog


def _draw_triangle(
    qtbot: QtBot,
    win: MainWindow,
    vertices: tuple[tuple[float, float], ...] = (
        (0.2, 0.2),
        (0.8, 0.2),
        (0.5, 0.8),
    ),
) -> None:
    canvas = win._canvas_widgets.canvas
    win._switch_canvas_mode(edit=False, create_mode="polygon")
    qtbot.wait(50)
    for xy in vertices:
        click_canvas_fraction(qtbot=qtbot, canvas=canvas, xy=xy)
    submit_label_dialog(qtbot=qtbot, label_dialog=win._label_dialog, label="obj")
    qtbot.keyPress(canvas, Qt.Key.Key_Return)
    qtbot.wait(200)

    def committed() -> None:
        assert len(canvas.shapes) == 1
        assert canvas.shapes[0].label == "obj"

    qtbot.waitUntil(committed)


def test_copy_annotations_to_next_unannotated_file(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
) -> None:
    # Open the annotated directory: the first image has annotations, the
    # second does not (the raw directory has no .json files).
    win = main_win(file_or_dir=str(data_path / "raw"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    canvas = win._canvas_widgets.canvas

    assert len(win.image_list) == 3

    # Draw a shape on the first image so there is something to copy.
    _draw_triangle(qtbot=qtbot, win=win)
    assert len(canvas.shapes) == 1

    # The next files are unannotated (no .json next to the .jpg).
    item0 = win._docks.file_list.item(0)
    item1 = win._docks.file_list.item(1)
    assert item0 is not None and item1 is not None
    assert item0.checkState() == Qt.CheckState.Checked
    assert item1.checkState() == Qt.CheckState.Unchecked

    win.copy_annotations_to_next()
    qtbot.wait(100)

    # The second file is now marked as annotated and is the current file.
    assert item1.checkState() == Qt.CheckState.Checked
    assert win._docks.file_list.currentRow() == 1
    assert win._image_path == win.image_list[1]

    # The label file was created on disk with the copied shapes.
    label_path = Path(data_path / "raw" / (Path(win.image_list[1]).stem + ".json"))
    assert label_path.exists()

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


def test_copy_annotations_to_next_shows_message_when_none_left(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
) -> None:
    win = main_win(file_or_dir=str(data_path / "annotated"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)

    # All files in annotated/ already have labels — nothing to copy to.
    for row in range(win._docks.file_list.count()):
        item = win._docks.file_list.item(row)
        assert item is not None
        assert item.checkState() == Qt.CheckState.Checked

    # The action runs without error and shows a status message.
    win.copy_annotations_to_next()
    qtbot.wait(100)
    assert win._docks.file_list.currentRow() == 0

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
