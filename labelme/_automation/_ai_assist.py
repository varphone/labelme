from __future__ import annotations

import dataclasses

import numpy as np
import osam
import skimage.transform
from loguru import logger
from numpy.typing import NDArray

from .._ai_models import supports_point_prompts
from .._shape import Shape
from ._ai_image import AiImageInput
from ._ai_image import prepare_ai_image
from ._geometry import _round_bbox_to_int
from ._osam_session import OsamSession
from ._shape_builders import Detection
from ._shape_builders import shapes_from_detections
from ._suppression import match_detections_to_existing_shapes
from ._suppression import suppress_detections_greedy
from ._types import AiOutputFormat
from ._types import AiPromptKind

_MASK_THRESHOLD: float = 0.5


@dataclasses.dataclass(frozen=True)
class AiAssistProposal:
    new_shapes: list[Shape]
    matching_existing_shapes: list[Shape]


class AiAssistSession:
    model_name: str
    output_format: AiOutputFormat
    polygon_detail: int
    downsample_scale: float
    denoise_strength: float
    _session: OsamSession | None

    def __init__(
        self,
        *,
        model_name: str = "sam2:latest",
        output_format: AiOutputFormat = "polygon",
        polygon_detail: int = 80,
        downsample_scale: float = 1.0,
        denoise_strength: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.output_format = output_format
        self.polygon_detail = polygon_detail
        self.downsample_scale = downsample_scale
        self.denoise_strength = denoise_strength
        self._session = None

    def _get_session(self) -> OsamSession:
        if self._session is None or self._session.model_name != self.model_name:
            self._session = OsamSession(model_name=self.model_name)
        return self._session

    def propose_shapes(
        self,
        *,
        image: NDArray[np.uint8],
        image_id: str,
        prompt_kind: AiPromptKind,
        points: NDArray[np.floating],
        point_labels: NDArray[np.intp],
        existing_shapes: list[Shape],
        image_size: tuple[int, int] | None = None,
    ) -> AiAssistProposal:
        if prompt_kind == "points" and not supports_point_prompts(
            model_name=self.model_name
        ):
            raise ValueError(f"{self.model_name} does not support point prompts")
        image_input: AiImageInput = prepare_ai_image(
            image=image,
            downsample_scale=self.downsample_scale,
            denoise_strength=self.denoise_strength,
        )
        processed_points = points * np.array(
            [image_input.scale_x, image_input.scale_y], dtype=points.dtype
        )
        processed_image_id = image_id
        if self.downsample_scale != 1.0 or self.denoise_strength != 0.0:
            processed_image_id = (
                f"{image_id}:downsample={self.downsample_scale!r}:"
                f"denoise={self.denoise_strength!r}"
            )
        response: osam.types.GenerateResponse = self._get_session().run(
            image=image_input.image,
            image_id=processed_image_id,
            points=processed_points,
            point_labels=point_labels,
        )
        # iou_threshold is hardcoded because the AI Assist flow has no
        # user-facing IoU control (unlike the AI Text Prompt flow); 0.5 matches
        # the AI Text Prompt widget default.
        detections = _detections_from_annotations(response.annotations)
        detections = _restore_detections_to_original_image(
            detections=detections,
            image_shape=image.shape,
            image_input=image_input,
        )
        if prompt_kind == "points" and detections:
            detections = [
                max(
                    detections,
                    key=lambda detection: (
                        _count_satisfied_prompt_points(
                            detection=detection,
                            points=points,
                            point_labels=point_labels,
                        ),
                        detection.score,
                    ),
                )
            ]
        detections = suppress_detections_greedy(
            detections=detections,
            iou_threshold=0.5,
        )
        matches = match_detections_to_existing_shapes(
            detections=detections,
            existing_shapes=existing_shapes,
        )
        return AiAssistProposal(
            new_shapes=shapes_from_detections(
                detections=matches.new_detections,
                shape_type=self.output_format,
                image_size=image_size,
                polygon_detail=self.polygon_detail,
            ),
            matching_existing_shapes=matches.matching_shapes,
        )


def _restore_detections_to_original_image(
    *,
    detections: list[Detection],
    image_shape: tuple[int, ...],
    image_input: AiImageInput,
) -> list[Detection]:
    if image_input.scale_x == 1.0 and image_input.scale_y == 1.0:
        return detections

    image_height, image_width = image_shape[:2]
    restored: list[Detection] = []
    for detection in detections:
        bbox = detection.bbox
        if bbox is not None:
            bbox = tuple(
                coordinate / scale
                for coordinate, scale in zip(
                    bbox,
                    (
                        image_input.scale_x,
                        image_input.scale_y,
                        image_input.scale_x,
                        image_input.scale_y,
                    ),
                    strict=True,
                )
            )
        mask = detection.mask
        if mask is not None:
            if bbox is None:
                mask_shape = (image_height, image_width)
            else:
                xmin, ymin, xmax, ymax = _round_bbox_to_int(bbox=bbox)
                mask_shape = (
                    max(1, ymax - ymin + 1),
                    max(1, xmax - xmin + 1),
                )
            mask = skimage.transform.resize(
                mask,
                output_shape=mask_shape,
                order=0,
                mode="edge",
                preserve_range=True,
                anti_aliasing=False,
            ) >= _MASK_THRESHOLD
        restored.append(dataclasses.replace(detection, bbox=bbox, mask=mask))
    return restored


def _count_satisfied_prompt_points(
    *,
    detection: Detection,
    points: NDArray[np.floating],
    point_labels: NDArray[np.intp],
) -> int:
    return sum(
        _is_point_inside_detection(detection=detection, point=point)
        == (point_label == 1)
        for point, point_label in zip(points, point_labels, strict=True)
    )


def _is_point_inside_detection(
    *,
    detection: Detection,
    point: NDArray[np.floating],
) -> bool:
    if detection.bbox is None:
        return False
    xmin, ymin, xmax, ymax = _round_bbox_to_int(bbox=detection.bbox)
    x, y = (int(round(coordinate)) for coordinate in point)
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return False
    if detection.mask is None:
        return True
    return bool(detection.mask[y - ymin, x - xmin])


def _detections_from_annotations(
    annotations: list[osam.types.Annotation],
    /,
) -> list[Detection]:
    if not annotations:
        logger.warning("No annotations returned")
        return []
    sorted_annotations = sorted(
        annotations,
        key=lambda a: a.score if a.score is not None else 0,
        reverse=True,
    )
    detections: list[Detection] = []
    for annotation in sorted_annotations:
        bbox: tuple[float, float, float, float] | None = None
        if annotation.bounding_box is not None:
            bb = annotation.bounding_box
            bbox = (bb.xmin, bb.ymin, bb.xmax, bb.ymax)
        detections.append(
            Detection(
                bbox=bbox,
                mask=annotation.mask,
                score=annotation.score if annotation.score is not None else 0.0,
            )
        )
    return detections
