"""Gabor fascicle orientation estimation relative to the lower aponeurosis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

import fascile_orient as orientation
from src.band_detection import BandResult


LOGGER = logging.getLogger(__name__)


@dataclass
class AngleResult:
    absolute_angle_deg: float = float("nan")
    lower_aponeurosis_angle_deg: float = float("nan")
    relative_angle_deg: float = float("nan")
    prominence: float = float("nan")
    roi: np.ndarray | None = None
    status: str = "failed"
    error: str = ""


class FascicleAngleEstimator:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.margin = float(parameters["margin"])
        self.wavelengths = [float(value) for value in parameters["wavelengths"]]
        self.angle_step = float(parameters["angle_step"])
        self.minimum_roi_pixels = int(parameters["minimum_roi_pixels"])
        self.sigma_along_factor = float(parameters["sigma_along_factor"])
        self.sigma_across_factor = float(parameters["sigma_across_factor"])

    def run(self, image: np.ndarray | None, bands: BandResult) -> AngleResult:
        try:
            if image is None:
                raise ValueError("Fascicle-angle estimator received no image.")
            if bands.status != "ok":
                raise ValueError(f"Band detection unavailable: {bands.error}")
            upper, lower = orientation.select_pair(bands.bands, None, None)
            roi = orientation.muscle_roi(image.shape, upper, lower, self.margin)
            if np.count_nonzero(roi) < self.minimum_roi_pixels:
                raise ValueError("The fitted bands enclose too few ROI pixels.")
            estimate = orientation.gabor_orientation(
                image,
                roi,
                self.wavelengths,
                self.angle_step,
                self.sigma_along_factor,
                self.sigma_across_factor,
            )
            lower_angle = float(np.degrees(np.arctan(lower["slope"])))
            relative_angle = orientation.axial_degrees(
                estimate["angle"] - lower_angle
            )
            return AngleResult(
                absolute_angle_deg=float(estimate["angle"]),
                lower_aponeurosis_angle_deg=lower_angle,
                relative_angle_deg=relative_angle,
                prominence=float(estimate["prominence"]),
                roi=roi,
                status="ok",
            )
        except Exception as error:
            LOGGER.exception("Fascicle-angle estimation failed")
            return AngleResult(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
