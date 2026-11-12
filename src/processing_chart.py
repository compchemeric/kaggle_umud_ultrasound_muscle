"""Progressively saved processing charts for one input image."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import fascile_orient as orientation
from src.band_detection import BandResult
from src.cropping import CropResult
from src.fascicle_angle import AngleResult
from src.geometric_postprocessing import GeometryResult


LOGGER = logging.getLogger(__name__)
Renderer = Callable[[Any], None]


class ProcessingChart:
    def __init__(self, path: Path, image_name: str, enabled: bool = True) -> None:
        self.path = path
        self.image_name = image_name
        self.enabled = enabled
        self._panels: list[tuple[str, Renderer]] = []

    def add_crop(self, result: CropResult) -> None:
        if result.status == "ok" and result.normalized_image is not None:
            image = result.normalized_image
            diagnostics = result.diagnostics or {}
            original = diagnostics["original"]
            left, top, right, bottom = result.bounds

            def draw_bounds(axis: Any) -> None:
                axis.imshow(original, cmap="gray", vmin=0, vmax=255)
                axis.axvline(left, color="#f97316", label="left")
                axis.axvline(right - 1, color="#16a34a", label="right")
                axis.axhline(top, color="#2563eb", label="top")
                axis.axhline(bottom - 1, color="#9333ea", label="bottom")
                axis.legend(fontsize=7)
                axis.axis("off")

            self._add("Detected crop bounds", draw_bounds)

            def draw(axis: Any) -> None:
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
                axis.axis("off")

            self._add(
                f"Cropped and normalized ({image.shape[1]}x{image.shape[0]} px)",
                draw,
            )
            if diagnostics:
                row_profile = diagnostics["row_profile"]
                column_profile = diagnostics["column_profile"]
                row_thresholds = diagnostics["row_thresholds"]
                column_threshold = diagnostics["column_threshold"]

                def draw_row_profile(axis: Any) -> None:
                    axis.plot(row_profile, color="#2563eb")
                    axis.axhline(
                        row_thresholds[0],
                        color="#dc2626",
                        linestyle="--",
                        label="top threshold",
                    )
                    axis.axhline(
                        row_thresholds[1],
                        color="#f59e0b",
                        linestyle=":",
                        label="bottom threshold",
                    )
                    axis.set_xlabel("Row")
                    axis.set_ylabel("Mean intensity")
                    axis.legend(fontsize=7)
                    axis.grid(alpha=0.2)

                def draw_column_profile(axis: Any) -> None:
                    axis.plot(column_profile, color="#7c3aed")
                    axis.axhline(
                        column_threshold,
                        color="#06b6d4",
                        linestyle="--",
                        label="horizontal threshold",
                    )
                    axis.set_xlabel("Column")
                    axis.set_ylabel("Mean intensity")
                    axis.legend(fontsize=7)
                    axis.grid(alpha=0.2)

                self._add("Horizontal average by row", draw_row_profile)
                self._add("Vertical average by column", draw_column_profile)
        else:
            self.add_error("Cropping and normalization", result.error)

    def add_bands(self, image: np.ndarray | None, result: BandResult) -> None:
        if result.status != "ok" or image is None:
            self.add_error("Aponeurosis band detection", result.error)
            return
        bands = result.bands

        def draw(axis: Any) -> None:
            self._draw_bands(axis, image, bands)

        self._add("Fitted aponeurosis bands", draw)

    def add_angle(
        self,
        image: np.ndarray | None,
        bands: BandResult,
        result: AngleResult,
    ) -> None:
        if result.status != "ok" or image is None or result.roi is None:
            self.add_error("Fascicle-angle estimation", result.error)
            return

        def draw(axis: Any) -> None:
            self._draw_bands(axis, image, bands.bands)
            orientation.draw_orientation_segment(
                axis,
                result.roi,
                result.absolute_angle_deg,
                "#ff65d5",
                f"Relative to lower band: {result.relative_angle_deg:+.1f}°",
                "--",
            )
            axis.legend(loc="lower left", fontsize=8, facecolor="white")

        self._add(
            f"Fascicle angle relative to lower aponeurosis: "
            f"{result.relative_angle_deg:+.1f}°",
            draw,
        )

    def add_geometry(
        self,
        image: np.ndarray | None,
        bands: BandResult,
        result: GeometryResult,
    ) -> None:
        if result.status == "failed" or image is None or bands.status != "ok":
            self.add_error("Geometric postprocessing", result.error)
            return

        def draw(axis: Any) -> None:
            self._draw_bands(axis, image, bands.bands)
            upper, lower = orientation.select_pair(bands.bands, None, None)
            width = image.shape[1]
            xs = np.linspace(0, width - 1, len(result.band_distances_px))
            y_upper = (
                orientation.band_y(upper, xs, width) + upper["width"] / 2
            )
            y_lower = (
                orientation.band_y(lower, xs, width) - lower["width"] / 2
            )
            for x, y1, y2 in zip(
                xs, y_upper, y_lower, strict=True
            ):
                axis.annotate(
                    "",
                    xy=(x, y2),
                    xytext=(x, y1),
                    arrowprops={"arrowstyle": "<->", "color": "#ff3131"},
                )
            for line in result.fascicle_lines:
                axis.annotate(
                    "",
                    xy=(line["end_x_px"], line["end_y_px"]),
                    xytext=(line["start_x_px"], line["start_y_px"]),
                    arrowprops={"arrowstyle": "<->", "color": "#ff65d5"},
                )
            axis.set_title(
                f"Mean band gap {result.mean_band_distance_px:.1f}px; "
                f"mean fascicle length {result.mean_fascicle_length_px:.1f}px"
            )

        self._add("Geometric measurements", draw)
        if result.error:
            self.add_error("Fascicle-length measurement", result.error)

    def add_error(self, stage: str, error: str) -> None:
        message = error or "No additional error details were provided."

        def draw(axis: Any) -> None:
            axis.text(
                0.5,
                0.5,
                message,
                color="#b91c1c",
                ha="center",
                va="center",
                wrap=True,
                transform=axis.transAxes,
            )
            axis.axis("off")

        self._add(f"{stage} failed", draw)

    def _draw_bands(
        self, axis: Any, image: np.ndarray, bands: list[dict[str, float]]
    ) -> None:
        _, width = image.shape
        xs = np.arange(-width, 2 * width)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        for band, color in zip(bands, ("#00c9ff", "#ffbd39", "#ff4f91")):
            center = orientation.band_y(band, xs, width)
            axis.fill_between(
                xs,
                center - band["width"] / 2,
                center + band["width"] / 2,
                color=color,
                alpha=0.3,
            )
            axis.plot(xs, center, color=color, linewidth=1.5)
        axis.axvline(-0.5, color="white", linestyle="--", linewidth=0.8)
        axis.axvline(width - 0.5, color="white", linestyle="--", linewidth=0.8)
        axis.set_xlim(-width - 0.5, 2 * width - 0.5)
        axis.set_ylim(image.shape[0] - 1, 0)
        axis.axis("off")

    def _add(self, title: str, renderer: Renderer) -> None:
        self._panels.append((title, renderer))
        if self.enabled:
            self.save()

    def save(self) -> None:
        if not self._panels:
            return
        columns = min(4, len(self._panels))
        rows = (len(self._panels) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(6 * columns, 5 * rows),
            squeeze=False,
            layout="constrained",
        )
        figure.suptitle(f"{self.image_name}: processing chart")
        for axis, (title, renderer) in zip(axes.flat, self._panels):
            try:
                renderer(axis)
                axis.set_title(title)
            except Exception as error:
                LOGGER.exception("Could not plot chart panel %s", title)
                axis.clear()
                axis.text(
                    0.5,
                    0.5,
                    f"{type(error).__name__}: {error}",
                    color="#b91c1c",
                    ha="center",
                    va="center",
                    wrap=True,
                    transform=axis.transAxes,
                )
                axis.set_title(f"{title} plotting failed")
                axis.axis("off")
        for axis in list(axes.flat)[len(self._panels):]:
            axis.axis("off")
        figure.savefig(self.path, dpi=140)
        plt.close(figure)
