from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import QItemSelectionModel
from pytestqt.qtbot import QtBot

from ..conftest import close_or_pause
from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


def test_file_list_multi_select_shows_count_in_status_bar(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    pause: bool,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    file_list = win._docks.file_list

    assert file_list.count() == 3

    # Single selection shows no count label.
    file_list.setCurrentRow(0)
    qtbot.wait(50)
    assert win._status_bar.file_count.text() == ""

    # Ctrl+click selects a second item → count label appears.
    item1 = file_list.item(1)
    assert item1 is not None
    index1 = file_list.indexFromItem(item1)
    file_list.selectionModel().select(
        index1,
        (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        ),
    )
    qtbot.wait(50)
    assert "2 files selected" in win._status_bar.file_count.text()

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


def test_delete_selected_files_removes_files(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    # Copy raw images into tmp so deletion is safe.
    src_dir = data_path / "raw"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".png"):
            shutil.copy(f, work_dir / f.name)

    win = main_win(file_or_dir=str(work_dir))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    file_list = win._docks.file_list
    assert file_list.count() == 3

    # Select first two files.
    file_list.setCurrentRow(0)
    qtbot.wait(50)
    item1 = file_list.item(1)
    index1 = file_list.indexFromItem(item1)
    file_list.selectionModel().select(
        index1,
        (
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows
        ),
    )
    qtbot.wait(50)
    assert len(file_list.selectedItems()) == 2

    monkeypatch.setattr(win, "_confirm_deletion", lambda *a, **k: True)
    win.delete_selected_files()
    qtbot.wait(100)

    # Two files deleted, one remains.
    assert file_list.count() == 1
    remaining = [f.name for f in work_dir.iterdir() if f.is_file()]
    assert len(remaining) == 1

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)


def test_export_selected_files_copies_to_target(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause: bool,
) -> None:
    win = main_win(file_or_dir=str(data_path / "raw"))
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    file_list = win._docks.file_list
    assert file_list.count() == 3

    # Select all three.
    file_list.selectAll()
    qtbot.wait(50)
    assert len(file_list.selectedItems()) == 3

    export_dir = tmp_path / "export"
    export_dir.mkdir()

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(export_dir),
    )

    win.export_selected_files()
    qtbot.wait(100)

    exported = sorted(f.name for f in export_dir.iterdir() if f.is_file())
    # At least the 3 images should be exported.
    assert len(exported) >= 3
    assert any("2011_000003" in f for f in exported)

    close_or_pause(qtbot=qtbot, widget=win, pause=pause)
