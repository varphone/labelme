from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from labelme._line_measurement import MeasurementParameters
from labelme._widgets import line_measurement_worker as worker_module
from labelme._widgets.line_measurement_worker import LineMeasurementWorker


def test_worker_emits_success_without_touching_widgets(qtbot: QtBot) -> None:
    worker = LineMeasurementWorker()
    qtbot.addWidget(QtWidgets.QWidget())
    results: list[object] = []
    worker.succeeded.connect(results.append)

    worker.run(
        np.zeros((16, 32), dtype=np.uint8),
        [[1.0, 8.0], [30.0, 8.0]],
        MeasurementParameters(sample_spacing=8.0),
    )

    assert len(results) == 1


def test_worker_drops_a_result_canceled_during_measurement(
    monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    worker = LineMeasurementWorker()
    qtbot.addWidget(QtWidgets.QWidget())
    results: list[object] = []
    worker.succeeded.connect(results.append)
    original = worker_module.measure_line_profile

    def cancel_inside_measurement(
        image: object,
        points: object,
        *,
        parameters: object,
        cancel_check: object = None,
    ) -> object:
        worker.cancel()
        return original(
            image,  # type: ignore[arg-type]
            points,  # type: ignore[arg-type]
            parameters=parameters,  # type: ignore[arg-type]
            cancel_check=cancel_check,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        worker_module, "measure_line_profile", cancel_inside_measurement
    )
    worker.run(
        np.zeros((16, 32), dtype=np.uint8),
        [[1.0, 8.0], [30.0, 8.0]],
        MeasurementParameters(sample_spacing=8.0),
    )

    assert results == []


def test_worker_emits_failure_without_touching_widgets(
    monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    worker = LineMeasurementWorker()
    qtbot.addWidget(QtWidgets.QWidget())
    failures: list[str] = []
    worker.failed.connect(failures.append)

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic measurement failure")

    monkeypatch.setattr(worker_module, "measure_line_profile", fail)
    worker.run(
        np.zeros((16, 32), dtype=np.uint8),
        [[1.0, 8.0], [30.0, 8.0]],
        MeasurementParameters(sample_spacing=8.0),
    )

    assert failures == ["RuntimeError: synthetic measurement failure"]
