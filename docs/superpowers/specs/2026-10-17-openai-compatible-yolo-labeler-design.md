# OpenAI-Compatible YOLO Auto Labeler Design

## Goal

Build a local command-line tool for one-person use. It labels a flat folder
of images through an OpenAI-compatible vision API and creates a complete,
unsplit YOLO bounding-box dataset.

The first provider configuration targets Alibaba Cloud Model Studio with
Qwen3-VL-Plus. The implementation supports other providers only when they
offer a compatible OpenAI Chat Completions vision API.

## Confirmed scope

- The tool accepts one image directory and does not recurse into subfolders.
- Supported input formats are JPG, JPEG, PNG, and WebP.
- The first dataset classes are `0: 人员` and `1: 安全帽`.
- The tool creates `<input-dir>/yolo_output`.
- The output is not split into training and validation sets.
- The API configuration is held in a local configuration file excluded from
  Git.
- The tool stops immediately if any validation, API, parsing, or output write
  operation fails.
- Existing `yolo_output` directories are never overwritten.

## Non-goals

- Provider-specific or non-OpenAI-compatible API adapters.
- Recursive image discovery.
- Concurrent calls, retries, resume support, or automatic output cleanup.
- Dataset train/validation splitting.
- Visual review, annotation editing, or preview rendering.

## Interface

The command form is:

```text
dataset-label <input-image-directory> --classes 人员 安全帽
```

The class argument defines the class order. For the first dataset, the order
is `人员` followed by `安全帽`, yielding class IDs 0 and 1 respectively.

The command loads `config.local.toml` from the project root. It must be
ignored by Git. A tracked `config.example.toml` provides the required shape:

```toml
[provider]
base_url = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
api_key = "replace-with-a-local-secret"
model = "qwen3-vl-plus"
```

## Processing flow

1. Validate the configuration file, input directory, class argument, image
   files, and absence of `<input-dir>/yolo_output`.
2. Create the output directory and its `images` and `labels` subdirectories.
3. Process supported images in a deterministic filename order.
4. Convert each local image into a Base64 Data URL and send it with a prompt
   that permits only the requested categories and requires a JSON response.
5. Parse response objects containing a requested category and
   `[x1, y1, x2, y2]` box coordinates.
6. Validate every returned object. Categories outside the requested class list,
   malformed JSON, non-numeric coordinates, inverted boxes, and coordinates
   outside the accepted `[0, 999]` range fail the command.
7. Convert valid boxes into YOLO rows and write the matching label file.
   A picture with no detections receives an empty label file.
8. Copy the source image unchanged to the output `images` directory.
9. Generate `data.yaml` with the output path and class mapping, then report
   image and per-class box totals.

## Coordinate conversion

Qwen3-VL two-dimensional localization returns coordinates in a fixed relative
`[0, 999]` coordinate system; they are not source-image pixel coordinates.
For a box `[x1, y1, x2, y2]`, write YOLO values directly as:

```text
x_center = (x1 + x2) / 2 / 999
y_center = (y1 + y2) / 2 / 999
width    = (x2 - x1) / 999
height   = (y2 - y1) / 999
```

This works identically for 2K, 4K, and any other image dimensions. Original
dimensions are not needed to create YOLO labels.

## Output layout

```text
<input-dir>/yolo_output/
  data.yaml
  images/
    source-image.ext
  labels/
    source-image.txt
```

`data.yaml` has no train/validation split and records the output directory and
the ordered class names. Each label file contains rows in the standard form:

```text
<class-id> <x-center> <y-center> <width> <height>
```

## Error behavior

The tool fails fast. It prints the affected file and reason, returns a nonzero
exit code, and does not delete successfully written output from earlier files.
It never silently skips a malformed API response or invalid bounding box.

## Verification

Automated tests use mocked API transport and temporary image directories; no
test calls a real provider. They cover configuration loading, flat image
discovery, JSON parsing, coordinate conversion, YOLO serialization, empty
labels, malformed responses, invalid coordinates, API failures, and output
directory collision rejection.

An integration check can be run manually with a newly generated local API key
after implementation. Credentials must never be placed in tracked files or
shared in chat.
