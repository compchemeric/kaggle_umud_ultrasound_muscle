"""Aponeurosis band fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from band_fit import fit_candidates


LOGGER = logging.getLogger(__name__)


@dataclass
class BandResult:
    bands: list[dict[str, float]] = field(default_factory=list)
    count: float = float("nan")
    status: str = "failed"
    error: str = ""


class AponeurosisDetector:
    def __init__(self, parameters: dict[str, Any]) -> None:
        self.count = int(parameters["count"])
        self.seed = int(parameters["seed"])
        self.min_separation = float(parameters["min_separation"])
        self.max_iterations = int(parameters["max_iterations"])
        self.population_size = int(parameters["population_size"])
        self.tolerance = float(parameters["tolerance"])

    def run(self, image: np.ndarray | None) -> BandResult:
        try:
            if image is None:
                raise ValueError("Band detector received no normalized image.")
            bands = fit_candidates(
                image,
                n=self.count,
                seed=self.seed,
                min_separation=self.min_separation,
                maxiter=self.max_iterations,
                popsize=self.population_size,
                tolerance=self.tolerance,
            )
            if len(bands) < 2:
                raise ValueError(f"Only {len(bands)} aponeurosis band(s) were fitted.")
            return BandResult(bands=bands, count=float(len(bands)), status="ok")
        except Exception as error:
            LOGGER.exception("Aponeurosis band detection failed")
            return BandResult(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
