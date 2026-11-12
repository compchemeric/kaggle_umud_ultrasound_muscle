# Ultrasound fascicle pipeline

The processing pipeline is split into class-based modules under `src/`:

- `cropping.py` detects image bounds and normalizes the crop.
- `calibration.py` saves a 10-pixel-wide scale cutout beginning
  `strip_with_px` pixels from the image's right edge, annotates detected tick
  lines, and estimates millimeters per pixel from the 5 mm intervals before
  cropping.
- `band_detection.py` fits the aponeurosis bands.
- `fascicle_angle.py` estimates Gabor orientation and reports its angle relative
  to the lower aponeurosis.
- `geometric_postprocessing.py` measures the average band gap and fascicle
  lengths.
- `orchestrator.py` runs the configured algorithms and creates a UUID-named
  experiment directory.

Run the first 10 images using the project defaults:

```powershell
uv run python run_pipeline.py
```

Change image selection, algorithm parameters, and output settings in
`pipeline_config.yaml`. Override the config file with `--config PATH`.
Geometric postprocessing uses 10 vertical band-gap scan lines and one
fascicle-length scan line, starting at the right edge of the lower band.
Plot generation is off by default. Set `meta.save_processing_charts` and/or
`meta.save_histograms` to `true` in the YAML to enable those outputs.

Each run saves a copy of the input YAML, `results.csv`, per-image calibration
strips and progressive processing charts, `errors.tsv`, and measurement
histograms under `output/<uuid>/`. The results CSV has the required columns
`image_id`, `pa_deg`, `fl_mm`, and `mt_mm`: the source filename stem, fascicle
angle in degrees, mean fascicle length in millimeters, and mean band distance
in millimeters. Unavailable measurements are written as `nan`; error messages
are retained in the processing charts and `errors.tsv`.
