import math
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.band_detection import BandResult
from src.calibration import ResolutionCalibrator
from src.cropping import ImageCropper
from src.fascicle_angle import AngleResult, FascicleAngleEstimator
from src.geometric_postprocessing import GeometricPostprocessor
from src.orchestrator import PipelineOrchestrator, load_config
from src.processing_chart import ProcessingChart


class PipelineComponentTests(unittest.TestCase):
    def test_calibrator_saves_strip_and_computes_mm_per_pixel(self) -> None:
        image = np.zeros((100, 80), dtype=np.uint8)
        image[[20, 40, 60], -50:-40] = 100
        calibrator = ResolutionCalibrator(
            {
                "strip_with_px": 50,
                "cutout_width_px": 10,
                "tick_intensity_threshold": 50,
                "tick_spacing_mm": 5,
                "minimum_tick_count": 3,
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            strip_path = Path(temporary_directory) / "sample_calibration_strip.png"
            result = calibrator.run(image, strip_path)

            self.assertEqual(result.status, "ok")
            self.assertTrue(strip_path.is_file())
            self.assertEqual(result.strip_image.shape, (100, 10, 3))
            self.assertEqual(result.tick_positions_px, [20.0, 40.0, 60.0])
            self.assertAlmostEqual(result.mm_per_pixel, 0.25)
            self.assertEqual(result.strip_image[20, 0].tolist(), [255, 0, 0])

    def test_cropper_detects_and_normalizes_synthetic_bounds(self) -> None:
        config = load_config(Path("pipeline_config.yaml"))
        image = np.zeros((120, 200), dtype=np.uint8)
        image[20:100, 20:180] = 80

        result = ImageCropper(config["algorithms"]["cropping"]).run(image)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.bounds, (20, 20, 180, 100))
        self.assertEqual(result.normalized_image.shape, (80, 160))

    def test_geometry_measures_band_gaps_and_fascicle_lines(self) -> None:
        bands = BandResult(
            bands=[
                {"y0": 20.0, "slope": 0.0, "width": 4.0},
                {"y0": 80.0, "slope": 0.0, "width": 4.0},
            ],
            count=2,
            status="ok",
        )
        angle = AngleResult(absolute_angle_deg=-45, status="ok")

        result = GeometricPostprocessor(
            {"band_distance_scan_count": 10, "fascicle_scan_count": 1}
        ).run(100, bands, angle)

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.band_distances_px), 10)
        self.assertEqual(len(result.fascicle_lengths_px), 1)
        self.assertEqual(result.fascicle_lines[0]["start_x_px"], 99.0)
        self.assertAlmostEqual(result.mean_band_distance_px, 56.0)
        self.assertAlmostEqual(
            result.mean_fascicle_length_px, math.sqrt(56**2 + 56**2)
        )

    def test_failed_previous_input_produces_nan_measurements_and_chart(self) -> None:
        config = load_config(Path("pipeline_config.yaml"))
        with self.assertLogs("src.fascicle_angle", level="ERROR"):
            angle = FascicleAngleEstimator(
                config["algorithms"]["fascicle_angle"]
            ).run(None, BandResult())
        with self.assertLogs("src.geometric_postprocessing", level="ERROR"):
            geometry = GeometricPostprocessor(
                config["algorithms"]["geometric_postprocessing"]
            ).run(None, BandResult(), angle)

        self.assertEqual(angle.status, "failed")
        self.assertTrue(math.isnan(angle.relative_angle_deg))
        self.assertEqual(geometry.status, "failed")
        self.assertTrue(math.isnan(geometry.mean_band_distance_px))

        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "progress.png"
            chart = ProcessingChart(chart_path, "synthetic")
            chart.add_error("Band detection", "synthetic failure")
            self.assertTrue(chart_path.is_file())

    def test_orchestrator_writes_uuid_experiment_outputs(self) -> None:
        config = load_config(Path("pipeline_config.yaml"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_directory = root / "images"
            image_directory.mkdir()
            image = np.zeros((120, 200), dtype=np.uint8)
            image[10:110, 10:190] = 80
            image[10:110, 145:155] = 0
            image[[20, 40, 60, 80, 100], 145:155] = 200
            Image.fromarray(image).save(image_directory / "sample.tif")
            config["meta"]["image_directory"] = str(image_directory)
            config["meta"]["output_directory"] = str(root / "output")
            config["meta"]["image_count"] = 1
            config["meta"]["save_processing_charts"] = True
            config["meta"]["save_histograms"] = True

            fitted_bands = [
                {"y0": 25.0, "slope": 0.0, "width": 4.0},
                {"y0": 75.0, "slope": 0.0, "width": 4.0},
            ]
            gabor_result = {
                "angle": -45.0,
                "prominence": 5.0,
                "valid_pixels": 1000,
            }
            with (
                patch("src.band_detection.fit_candidates", return_value=fitted_bands),
                patch(
                    "src.fascicle_angle.orientation.gabor_orientation",
                    return_value=gabor_result,
                ),
            ):
                experiment = PipelineOrchestrator(
                    config, Path("pipeline_config.yaml").resolve()
                )
                records = experiment.run()

            self.assertEqual(len(records), 1)
            self.assertTrue((experiment.output_dir / "pipeline_config.yaml").is_file())
            self.assertTrue(
                (experiment.output_dir / "sample_calibration_strip.png").is_file()
            )
            self.assertTrue((experiment.output_dir / "results.csv").is_file())
            self.assertTrue((experiment.output_dir / "sample_processing.png").is_file())
            with Image.open(experiment.output_dir / "sample_processing.png") as chart:
                self.assertGreater(chart.width, chart.height)
            self.assertTrue(
                (experiment.output_dir / "measurement_histograms.png").is_file()
            )
            with (experiment.output_dir / "results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                csv_records = list(csv.DictReader(file))
            self.assertEqual(
                list(csv_records[0]),
                ["image_id", "pa_deg", "fl_mm", "mt_mm"],
            )
            self.assertEqual(csv_records[0]["image_id"], "sample")
            self.assertTrue(math.isfinite(float(csv_records[0]["pa_deg"])))
            self.assertGreater(float(csv_records[0]["fl_mm"]), 0)
            self.assertGreater(float(csv_records[0]["mt_mm"]), 0)
            self.assertEqual(csv_records[0]["pa_deg"], "-45.0")
            self.assertEqual(csv_records[0]["fl_mm"], "16.3")
            self.assertEqual(csv_records[0]["mt_mm"], "11.5")

            config["meta"]["save_processing_charts"] = False
            config["meta"]["save_histograms"] = False
            with (
                patch("src.band_detection.fit_candidates", return_value=fitted_bands),
                patch(
                    "src.fascicle_angle.orientation.gabor_orientation",
                    return_value=gabor_result,
                ),
            ):
                plots_disabled = PipelineOrchestrator(
                    config, Path("pipeline_config.yaml").resolve()
                )
                plots_disabled.run()

            self.assertFalse(
                (plots_disabled.output_dir / "sample_processing.png").exists()
            )
            self.assertFalse(
                (plots_disabled.output_dir / "measurement_histograms.png").exists()
            )
            self.assertTrue(
                (plots_disabled.output_dir / "sample_calibration_strip.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
