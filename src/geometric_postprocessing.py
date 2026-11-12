"""Band-distance and fascicle-length measurements."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.band_detection import BandResult
from src.fascicle_angle import AngleResult
import fascile_orient as orientation


LOGGER = logging.getLogger(__name__)


@dataclass
class GeometryResult:
    band_distances_px: list[float] = field(default_factory=list)
    mean_band_distance_px: float = float("nan")
    min_band_distance_px: float = float("nan")
    max_band_distance_px: float = float("nan")
    fascicle_lengths_px: list[float] = field(default_factory=list)
    fascicle_lines: list[dict[str, float]] = field(default_factory=list)
    mean_fascicle_length_px: float = float("nan")
    status: str = "failed"
    error: str = ""


class GeometricPostprocessor:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.band_distance_scan_count = int(parameters["band_distance_scan_count"])
        self.fascicle_scan_count = int(parameters["fascicle_scan_count"])
        if self.band_distance_scan_count < 1 or self.fascicle_scan_count < 1:
            raise ValueError("Geometry scan counts must be positive.")

    def run(
        self,
        image_width: int | None,
        bands: BandResult,
        angle: AngleResult,
    ) -> GeometryResult:
        try:
            if image_width is None or image_width < 2:
                raise ValueError("Geometry processing requires a valid image width.")
            if bands.status != "ok":
                raise ValueError(f"Band detection unavailable: {bands.error}")
            upper, lower = orientation.select_pair(bands.bands, None, None)
            xs = np.linspace(0, image_width - 1, self.band_distance_scan_count)
            upper_inner = (
                orientation.band_y(upper, xs, image_width) + upper["width"] / 2
            )
            lower_inner = (
                orientation.band_y(lower, xs, image_width) - lower["width"] / 2
            )
            distances = lower_inner - upper_inner
            if np.any(distances <= 0):
                raise ValueError("The selected aponeurosis bands overlap.")

            lines: list[dict[str, float]] = []
            length_error = ""
            if angle.status == "ok":
                try:
                    lines = self._measure_fascicle_lengths(
                        upper,
                        lower,
                        image_width,
                        angle.absolute_angle_deg,
                    )
                except Exception as error:
                    LOGGER.exception("Fascicle-length measurement failed")
                    length_error = f"{type(error).__name__}: {error}"
            else:
                length_error = f"Fascicle angle unavailable: {angle.error}"

            return GeometryResult(
                band_distances_px=[float(value) for value in distances],
                mean_band_distance_px=float(np.mean(distances)),
                min_band_distance_px=float(np.min(distances)),
                max_band_distance_px=float(np.max(distances)),
                fascicle_lengths_px=[
                    line["length_px"] for line in lines
                ],
                fascicle_lines=lines,
                mean_fascicle_length_px=(
                    float(np.mean([line["length_px"] for line in lines]))
                    if lines
                    else float("nan")
                ),
                status="ok" if not length_error else "partial",
                error=length_error,
            )
        except Exception as error:
            LOGGER.exception("Geometric postprocessing failed")
            return GeometryResult(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )

    def _measure_fascicle_lengths(
        self,
        upper: dict[str, float],
        lower: dict[str, float],
        image_width: int,
        angle_deg: float,
    ) -> list[dict[str, float]]:
        upper_slope = upper["slope"]
        lower_slope = lower["slope"]
        midpoint = (image_width - 1) / 2
        upper_intercept = upper["y0"] - upper_slope * midpoint + upper["width"] / 2
        lower_intercept = lower["y0"] - lower_slope * midpoint - lower["width"] / 2
        fascicle_slope = float(np.tan(np.deg2rad(angle_deg)))
        denominator = fascicle_slope - upper_slope
        if np.isclose(denominator, 0, atol=1e-9):
            raise ValueError("Fascicle line is parallel to the upper aponeurosis.")
        end_slope = (fascicle_slope - lower_slope) / denominator
        end_intercept = (upper_intercept - lower_intercept) / denominator
        start_xs = np.linspace(
            image_width - 1, 0, self.fascicle_scan_count
        )
        end_xs = end_slope * start_xs + end_intercept
        start_ys = lower_slope * start_xs + lower_intercept
        end_ys = upper_slope * end_xs + upper_intercept
        if np.any(end_ys >= start_ys):
            raise ValueError("Fascicle intersections do not run lower-to-upper.")
        lengths = np.hypot(end_xs - start_xs, end_ys - start_ys)
        return [
            {
                "start_x_px": float(start_x),
                "start_y_px": float(start_y),
                "end_x_px": float(end_x),
                "end_y_px": float(end_y),
                "length_px": float(length),
            }
            for start_x, start_y, end_x, end_y, length in zip(
                start_xs, start_ys, end_xs, end_ys, lengths, strict=True
            )
        ]
