"""YAML-configured orchestration and experiment output management."""

from __future__ import annotations

import csv
import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.band_detection import AponeurosisDetector, BandResult
from src.calibration import CalibrationResult, ResolutionCalibrator
from src.cropping import CropResult, ImageCropper
from src.fascicle_angle import AngleResult, FascicleAngleEstimator
from src.geometric_postprocessing import (
    GeometricPostprocessor,
    GeometryResult,
)
from src.processing_chart import ProcessingChart


LOGGER = logging.getLogger(__name__)
RESULT_COLUMNS = (
    "image_id",
    "pa_deg",
    "fl_mm",
    "mt_mm",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping: {path}")
    return config


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


class PipelineOrchestrator:
    def __init__(self, config: dict[str, Any], config_path: Path) -> None:
        self.config = config
        self.config_path = config_path
        self.calibrator = ResolutionCalibrator(config["algorithms"]["calibration"])
        self.cropper = ImageCropper(config["algorithms"]["cropping"])
        self.band_detector = AponeurosisDetector(
            config["algorithms"]["band_detection"]
        )
        self.angle_estimator = FascicleAngleEstimator(
            config["algorithms"]["fascicle_angle"]
        )
        self.geometry = GeometricPostprocessor(
            config["algorithms"]["geometric_postprocessing"]
        )
        run_id = str(uuid.uuid4())
        output_root = Path(config["meta"]["output_directory"]).expanduser()
        if not output_root.is_absolute():
            output_root = config_path.parent / output_root
        self.output_dir = output_root.resolve() / run_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config_path, self.output_dir / config_path.name)

    def run(self) -> list[dict[str, Any]]:
        image_dir = Path(self.config["meta"]["image_directory"]).expanduser()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        allowed_extensions = {
            extension.lower()
            for extension in self.config["meta"]["image_extensions"]
        }
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_extensions
        )
        image_limit = int(self.config["meta"]["image_count"])
        image_paths = image_paths[:image_limit]
        if not image_paths:
            raise FileNotFoundError(f"No supported images found in {image_dir}")

        records: list[dict[str, Any]] = []
        errors: list[str] = []
        chart_enabled = bool(self.config["meta"]["save_processing_charts"])
        for index, image_path in enumerate(image_paths, start=1):
            LOGGER.info("Processing %d/%d: %s", index, len(image_paths), image_path.name)
            chart = ProcessingChart(
                self.output_dir / f"{image_path.stem}_processing.png",
                image_path.name,
                enabled=chart_enabled,
            )
            calibration_result: CalibrationResult = self.calibrator.run(
                image_path,
                self.output_dir / f"{image_path.stem}_calibration_strip.png",
            )

            crop_result: CropResult = self.cropper.run(image_path)
            chart.add_crop(crop_result)

            normalized = crop_result.normalized_image
            band_result: BandResult = self.band_detector.run(normalized)
            chart.add_bands(normalized, band_result)

            angle_result: AngleResult = self.angle_estimator.run(
                normalized, band_result
            )
            chart.add_angle(normalized, band_result, angle_result)

            width = normalized.shape[1] if normalized is not None else None
            geometry_result: GeometryResult = self.geometry.run(
                width, band_result, angle_result
            )
            chart.add_geometry(normalized, band_result, geometry_result)
            for stage, status, error in (
                (
                    "calibration",
                    calibration_result.status,
                    calibration_result.error,
                ),
                ("cropping", crop_result.status, crop_result.error),
                ("band_detection", band_result.status, band_result.error),
                ("fascicle_angle", angle_result.status, angle_result.error),
                ("geometry", geometry_result.status, geometry_result.error),
            ):
                if error:
                    errors.append(f"{image_path.name}\t{stage}\t{status}\t{error}")

            mm_per_pixel = calibration_result.mm_per_pixel
            records.append(
                {
                    "image_id": image_path.stem,
                    "pa_deg": _finite_or_nan(angle_result.relative_angle_deg),
                    "fl_mm": _finite_or_nan(
                        geometry_result.mean_fascicle_length_px * mm_per_pixel
                    ),
                    "mt_mm": _finite_or_nan(
                        geometry_result.mean_band_distance_px * mm_per_pixel
                    ),
                }
            )
            if index % int(self.config["meta"]["checkpoint_interval"]) == 0:
                self._write_results(records)
                self._write_errors(errors)
                print(f"Saved results for {index}/{len(image_paths)} images.")

        self._write_results(records)
        self._write_errors(errors)
        if bool(self.config["meta"]["save_histograms"]):
            self._save_histograms(records)
        print(f"Experiment {self.output_dir.name} saved to {self.output_dir.resolve()}")
        return records

    def _write_results(self, records: list[dict[str, Any]]) -> None:
        result_path = self.output_dir / self.config["meta"]["result_filename"]
        with result_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for record in records:
                output_record = dict(record)
                for column in ("pa_deg", "fl_mm", "mt_mm"):
                    value = float(record[column])
                    output_record[column] = (
                        f"{value:.1f}" if math.isfinite(value) else "nan"
                    )
                writer.writerow(output_record)

    def _write_errors(self, errors: list[str]) -> None:
        error_path = self.output_dir / "errors.tsv"
        error_path.write_text(
            "image_name\tstage\tstatus\terror\n"
            + ("\n".join(errors) + "\n" if errors else ""),
            encoding="utf-8",
        )

    def _save_histograms(self, records: list[dict[str, Any]]) -> None:
        metrics = (
            ("mt_mm", "Mean band distance (mm)"),
            ("pa_deg", "Fascicle angle relative to lower aponeurosis (deg)"),
            ("fl_mm", "Mean fascicle length (mm)"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        for axis, (key, title) in zip(axes, metrics, strict=True):
            values = np.asarray(
                [record[key] for record in records if math.isfinite(record[key])],
                dtype=np.float64,
            )
            if values.size:
                axis.hist(values, bins="auto", color="#2563eb", edgecolor="white")
                axis.set_title(f"{title} (n={values.size})")
                axis.set_ylabel("Image count")
                axis.grid(axis="y", alpha=0.25)
            else:
                axis.set_title(f"{title} (no valid measurements)")
            axis.set_xlabel(title)
        figure.savefig(self.output_dir / "measurement_histograms.png", dpi=150)
        plt.close(figure)


def run_from_config(config_path: Path) -> list[dict[str, Any]]:
    config = load_config(config_path)
    return PipelineOrchestrator(config, config_path).run()
