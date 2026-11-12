"""Pixel-to-millimeter calibration from the image's right-edge scale."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    strip_image: np.ndarray | None = None
    strip_path: Path | None = None
    tick_positions_px: list[float] = field(default_factory=list)
    tick_spacing_px: float = float("nan")
    mm_per_pixel: float = float("nan")
    status: str = "failed"
    error: str = ""


class ResolutionCalibrator:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.strip_with_px = int(parameters["strip_with_px"])
        self.cutout_width_px = int(parameters["cutout_width_px"])
        self.tick_intensity_threshold = float(parameters["tick_intensity_threshold"])
        self.tick_spacing_mm = float(parameters["tick_spacing_mm"])
        self.minimum_tick_count = int(parameters["minimum_tick_count"])
        if (
            self.strip_with_px < self.cutout_width_px
            or self.cutout_width_px < 1
            or self.tick_intensity_threshold < 0
            or self.tick_spacing_mm <= 0
            or self.minimum_tick_count < 2
        ):
            raise ValueError("Invalid resolution-calibration parameters.")

    def run(
        self,
        source: Path | np.ndarray | None,
        output_path: Path,
    ) -> CalibrationResult:
        strip: np.ndarray | None = None
        annotated_strip: np.ndarray | None = None
        positions: list[float] = []
        try:
            if source is None:
                raise ValueError("No image was provided to the calibrator.")
            if isinstance(source, Path):
                with Image.open(source) as image:
                    original = np.asarray(image.convert("L"), dtype=np.uint8)
            else:
                original = np.asarray(source)
            if original.ndim != 2 or not original.size:
                raise ValueError("Expected a non-empty 2-D grayscale image.")

            if self.strip_with_px > original.shape[1]:
                raise ValueError(
                    "The image is narrower than the configured calibration offset."
                )
            strip_start = original.shape[1] - self.strip_with_px
            strip_end = strip_start + self.cutout_width_px
            strip = np.asarray(original[:, strip_start:strip_end], dtype=np.uint8)
            if strip.shape[1] != self.cutout_width_px:
                raise ValueError("The calibration cutout is not the configured width.")

            row_peaks = strip.max(axis=1)
            tick_rows = row_peaks >= self.tick_intensity_threshold
            edges = np.flatnonzero(np.diff(np.r_[False, tick_rows, False]))
            positions = [
                (float(start) + float(end - 1)) / 2
                for start, end in zip(edges[::2], edges[1::2], strict=True)
            ]
            annotated_strip = np.repeat(strip[:, :, np.newaxis], 3, axis=2)
            for position in positions:
                annotated_strip[int(round(position)), :, :] = (255, 0, 0)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotated_strip).save(output_path)

            if len(positions) < self.minimum_tick_count:
                raise ValueError(
                    f"Detected {len(positions)} scale ticks; "
                    f"at least {self.minimum_tick_count} are required."
                )

            spacing = float(np.median(np.diff(positions)))
            if not math.isfinite(spacing) or spacing <= 0:
                raise ValueError("Detected scale ticks have invalid spacing.")

            return CalibrationResult(
                strip_image=annotated_strip,
                strip_path=output_path,
                tick_positions_px=positions,
                tick_spacing_px=spacing,
                mm_per_pixel=self.tick_spacing_mm / spacing,
                status="ok",
            )
        except Exception as error:
            LOGGER.exception("Resolution calibration failed")
            if strip is not None and not output_path.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    annotated_strip if annotated_strip is not None else strip
                ).save(output_path)
            return CalibrationResult(
                strip_image=annotated_strip if annotated_strip is not None else strip,
                strip_path=(
                    output_path
                    if strip is not None and output_path.is_file()
                    else None
                ),
                tick_positions_px=positions,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
