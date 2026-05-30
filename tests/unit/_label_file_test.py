from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from labelme._label_file import LabelFile
from labelme._label_file import LabelFileReadError
from labelme._label_file import LabelFileWriteError
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file


def test_LabelFile_load_windows_path(data_path: Path, tmp_path: Path) -> None:
    """Test that LabelFile can load JSON with Windows-style backslash paths.

    Regression test for https://github.com/wkentaro/labelme/issues/1725
    """
    (tmp_path / "images").mkdir()
    shutil.copy(
        data_path / "annotated" / "2011_000003.jpg",
        tmp_path / "images" / "2011_000003.jpg",
    )

    json_file = tmp_path / "annotations" / "2011_000003.json"
    json_file.parent.mkdir()
    with open(data_path / "annotated" / "2011_000003.json") as f:
        json_data = json.load(f)
    json_data["imagePath"] = "..\\images\\2011_000003.jpg"
    with open(json_file, "w") as f:
        json.dump(json_data, f)

    label_file = LabelFile(str(json_file))
    assert label_file.image_path == "../images/2011_000003.jpg"
    assert label_file.image_data is not None


@pytest.mark.parametrize(
    ("snake_attr", "camel_attr", "initial", "updated"),
    [
        ("image_path", "imagePath", "foo.jpg", "bar.jpg"),
        ("image_data", "imageData", b"foo", b"bar"),
        ("other_data", "otherData", {"foo": 1}, {"bar": 2}),
    ],
)
def test_LabelFile_camelCase_attribute_is_deprecated(
    snake_attr: str,
    camel_attr: str,
    initial: object,
    updated: object,
) -> None:
    label_file = LabelFile()
    setattr(label_file, snake_attr, initial)

    with pytest.warns(DeprecationWarning, match=snake_attr):
        assert getattr(label_file, camel_attr) == initial

    with pytest.warns(DeprecationWarning, match=snake_attr):
        setattr(label_file, camel_attr, updated)
    assert getattr(label_file, snake_attr) == updated


@pytest.fixture()
def annotated_raw(data_path: Path) -> dict[str, Any]:
    src = data_path / "annotated" / "2011_000003.json"
    with open(src) as f:
        return json.load(f)


@pytest.fixture()
def annotated_dst(data_path: Path, tmp_path: Path) -> Path:
    shutil.copy(
        data_path / "annotated" / "2011_000003.jpg",
        tmp_path / "2011_000003.jpg",
    )
    return tmp_path / "2011_000003.json"


def _dump_json(path: Path, raw: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(raw, f)


def test_read_label_file_returns_label_data(data_path: Path) -> None:
    label_data = read_label_file(
        filename=str(data_path / "annotated" / "2011_000003.json")
    )

    assert label_data.image_path == "2011_000003.jpg"
    assert label_data.image_data
    assert label_data.shapes


def test_read_label_file_extracts_other_data(
    annotated_raw: dict[str, Any],
    annotated_dst: Path,
) -> None:
    annotated_raw["customField"] = {"reviewer": "alice"}
    _dump_json(path=annotated_dst, raw=annotated_raw)

    label_data = read_label_file(filename=str(annotated_dst))

    assert label_data.other_data == {"customField": {"reviewer": "alice"}}


def test_read_label_file_normalizes_legacy_rectangle_points(
    annotated_raw: dict[str, Any],
    annotated_dst: Path,
) -> None:
    annotated_raw["shapes"] = [
        {
            "label": "bbox",
            "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": "",
            "score": None,
        }
    ]
    _dump_json(path=annotated_dst, raw=annotated_raw)

    label_data = read_label_file(filename=str(annotated_dst))

    assert label_data.shapes[0]["shape_type"] == "rectangle"
    assert label_data.shapes[0]["points"] == [[10, 20], [30, 40]]


@pytest.mark.parametrize(
    "mutator,error_match",
    [
        (lambda raw: raw.pop("imagePath"), "imagePath"),
        (lambda raw: raw.pop("imageData"), "imageData"),
        (lambda raw: raw.pop("shapes"), "shapes"),
        (lambda raw: raw["shapes"].append({"label": "x"}), "points is required"),
        (lambda raw: raw.update({"imageHeight": 1}), "imageHeight mismatch"),
        (lambda raw: raw.update({"imageWidth": 1}), "imageWidth mismatch"),
    ],
    ids=[
        "missing_imagePath",
        "missing_imageData",
        "missing_shapes",
        "invalid_shape",
        "imageHeight_mismatch",
        "imageWidth_mismatch",
    ],
)
def test_read_label_file_raises_read_error_on_malformed(
    annotated_raw: dict[str, Any],
    annotated_dst: Path,
    mutator: Callable[[dict[str, Any]], Any],
    error_match: str,
) -> None:
    mutator(annotated_raw)
    _dump_json(path=annotated_dst, raw=annotated_raw)

    with pytest.raises(LabelFileReadError, match=error_match):
        read_label_file(filename=str(annotated_dst))


def test_write_label_file_round_trips(data_path: Path, tmp_path: Path) -> None:
    src = read_label_file(filename=str(data_path / "annotated" / "2011_000003.json"))
    dst = tmp_path / "out.json"

    shapes: list[dict[str, Any]] = [{**s} for s in src.shapes]
    write_label_file(
        filename=str(dst),
        shapes=shapes,
        image_path=src.image_path,
        image_height=None,
        image_width=None,
        image_data=src.image_data,
        other_data={"customField": 42},
        flags={"ok": True},
    )

    reloaded = read_label_file(filename=str(dst))
    assert reloaded.image_path == src.image_path
    assert reloaded.flags == {"ok": True}
    assert reloaded.other_data == {"customField": 42}
    assert [(s["label"], s["points"], s["shape_type"]) for s in reloaded.shapes] == [
        (s["label"], s["points"], s["shape_type"]) for s in src.shapes
    ]


@pytest.mark.parametrize(
    "reserved_key",
    [
        "version",
        "imagePath",
        "imageData",
        "shapes",
        "flags",
        "imageHeight",
        "imageWidth",
    ],
)
def test_write_label_file_rejects_reserved_other_data_key(
    tmp_path: Path, reserved_key: str
) -> None:
    with pytest.raises(LabelFileWriteError, match=f"reserved key.*{reserved_key}"):
        write_label_file(
            filename=str(tmp_path / "out.json"),
            shapes=[],
            image_path="foo.jpg",
            image_height=None,
            image_width=None,
            other_data={reserved_key: "x"},
        )


def test_write_label_file_raises_on_dimension_mismatch(
    data_path: Path, tmp_path: Path
) -> None:
    src = read_label_file(filename=str(data_path / "annotated" / "2011_000003.json"))

    with pytest.raises(LabelFileWriteError, match="imageHeight mismatch"):
        write_label_file(
            filename=str(tmp_path / "out.json"),
            shapes=[],
            image_path="foo.jpg",
            image_height=1,
            image_width=None,
            image_data=src.image_data,
        )


def test_write_label_file_raises_write_error_on_io_failure(tmp_path: Path) -> None:
    bad_path = tmp_path / "missing_dir" / "out.json"

    with pytest.raises(LabelFileWriteError, match="failed to write"):
        write_label_file(
            filename=str(bad_path),
            shapes=[],
            image_path="foo.jpg",
            image_height=None,
            image_width=None,
        )
