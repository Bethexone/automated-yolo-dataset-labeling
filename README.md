# Automated YOLO Dataset Labeling

A command-line tool that uses an OpenAI-compatible vision API to create
bounding-box annotations and export an unsplit YOLO dataset.

## Features

- Uses any OpenAI-compatible provider configured in `config.local.toml`.
- Accepts a flat directory of `.jpg`, `.jpeg`, `.png`, or `.webp` images.
- Produces `images/`, `labels/`, `data.yaml`, and `classes.txt`.
- Supports Qwen relative bounding-box coordinates from 0 to 1000.
- Stops immediately on the first API or validation error.
- Optionally saves annotated visualization images.

## Installation

```powershell
C:\software\miniconda3\envs\dataset_label\python.exe -m pip install -e .
```

Copy `config.example.toml` to `config.local.toml` and set `base_url`,
`api_key`, and `model`. The local configuration is ignored by Git.

## Usage

```powershell
& C:\software\miniconda3\envs\dataset_label\Scripts\dataset-label.exe `
  "C:\path\to\images" `
  --classes person "safety helmet"
```

The command creates `yolo_output/` under the input directory:

```text
yolo_output/
├── images/
├── labels/
├── classes.txt
└── data.yaml
```

Use `--test` to annotate only the first image, print its YOLO rows, and save
one visualization to `yolo_output/preview/`.

Use `--preview` during a normal batch run to save a visualization for every
image to `yolo_output/preview/`.

`--test` and `--preview` cannot be used together. Existing output directories
are never overwritten.

## Dataset privacy

Dataset images, generated labels, previews, outputs, and local API settings
are ignored by Git and are not included in this repository.
