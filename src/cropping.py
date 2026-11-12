"""Image boundary detection, cropping, and intensity normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


@dataclass
class CropResult:
    normalized_image: np.ndarray | None
    bounds: tuple[int, int, int, int] | None
    status: str
    error: str = ""
    diagnostics: dict[str, Any] | None = None


class ImageCropper:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.horizontal_fraction = float(parameters["horizontal_fraction"])
        self.vertical_fraction = float(parameters["vertical_fraction"])
        self.top_threshold_fraction = float(parameters["top_threshold_fraction"])
        self.bottom_threshold_fraction = float(parameters["bottom_threshold_fraction"])
        self.horizontal_threshold_fraction = float(
            parameters["horizontal_threshold_fraction"]
        )
        self.normalization_percentiles = tuple(
            float(value) for value in parameters["normalization_percentiles"]
        )

    def run(self, source: Path | np.ndarray | None) -> CropResult:
        try:
            if source is None:
                raise ValueError("No image was provided to the cropper.")
            if isinstance(source, Path):
                with Image.open(source) as image:
                    original = np.asarray(image.convert("L"), dtype=np.uint8)
            else:
                original = np.asarray(source)
            if original.ndim != 2 or not original.size:
                raise ValueError("Expected a non-empty 2-D grayscale image.")

            height, width = original.shape
            search_width = round(width * self.horizontal_fraction)
            search_height = round(height * self.vertical_fraction)
            search_left = (width - search_width) // 2
            search_top = (height - search_height) // 2
            search_right = search_left + search_width
            search_bottom = search_top + search_height
            search = original[search_top:search_bottom, search_left:search_right]
            row_profile = search.mean(axis=1)
            column_profile = search.mean(axis=0)

            if row_profile.max() <= 0 or column_profile.max() <= 0:
                raise ValueError("The search area has no positive image intensity.")

            top_threshold = row_profile.max() * self.top_threshold_fraction
            bottom_threshold = row_profile.max() * self.bottom_threshold_fraction
            horizontal_threshold = (
                column_profile.max() * self.horizontal_threshold_fraction
            )
            top_rows = np.flatnonzero(row_profile >= top_threshold)
            bottom_rows = np.flatnonzero(row_profile >= bottom_threshold)
            left_columns = np.flatnonzero(column_profile >= horizontal_threshold)
            right_columns = left_columns
            if not len(top_rows) or not len(bottom_rows) or not len(left_columns):
                raise ValueError("Intensity thresholds did not detect crop boundaries.")

            left = search_left + int(left_columns[0])
            right = search_left + int(right_columns[-1]) + 1
            top = search_top + int(top_rows[0])
            bottom = search_top + int(bottom_rows[-1]) + 1
            if right - left < 2 or bottom - top < 2:
                raise ValueError("Detected crop bounds are empty or too small.")

            crop = np.asarray(original[top:bottom, left:right], dtype=np.float32)
            low, high = np.percentile(crop, self.normalization_percentiles)
            normalized = np.clip((crop - low) / max(float(high - low), 1e-6), 0, 1)
            return CropResult(
                normalized_image=normalized,
                bounds=(left, top, right, bottom),
                status="ok",
                diagnostics={
                    "original": original,
                    "search_bounds": (
                        search_left,
                        search_top,
                        search_right,
                        search_bottom,
                    ),
                    "row_profile": row_profile,
                    "column_profile": column_profile,
                    "row_thresholds": (float(top_threshold), float(bottom_threshold)),
                    "column_threshold": float(horizontal_threshold),
                    "top_threshold_fraction": self.top_threshold_fraction,
                    "bottom_threshold_fraction": self.bottom_threshold_fraction,
                    "horizontal_threshold_fraction": self.horizontal_threshold_fraction,
                },
            )
        except Exception as error:
            LOGGER.exception("Image cropping failed")
            return CropResult(
                normalized_image=None,
                bounds=None,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
