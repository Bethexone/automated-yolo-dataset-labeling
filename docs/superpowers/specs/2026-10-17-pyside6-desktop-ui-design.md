# PySide6 Desktop UI Design

## Goal

Add a local PySide6 desktop interface for the existing YOLO bbox annotation
workflow. It reuses the current OpenAI-compatible detector and YOLO exporter.

## Main window

The window has a fixed 200px navigation sidebar and a larger annotation
workspace. Sidebar brand and navigation text use a larger size without
changing its width. The workspace uses a consistent type scale: panel titles
share one size, labels and paths share the body size, and helper text is the
only smaller tier.

The annotation workspace shows:

- Application version.
- An API configuration selector and a management action.
- Dataset-folder picker.
- Export-folder picker. The selected folder is the final YOLO dataset root.
- A checked/unchecked `Generate preview images` checkbox.
- A classes table with automatic sequential IDs, editable class names, add,
  and remove controls. Table order defines YOLO IDs.
- A single-image test panel and a Start labeling action.

## API configurations

Configurations are stored in ignored `config.local.toml`. The UI supports a
configuration selector even though the initial file contains only the Qwen
configuration. Each configuration contains a display name, `base_url`,
`api_key`, and `model`.

## Test flow

The Test action sends only the first supported image from the dataset folder.
It renders the returned detections into the test panel and displays YOLO rows
and errors in the UI. Double-clicking the rendered image opens a modal with a
larger view; clicking outside or pressing Escape closes it. The test does not
write a dataset export.

## Batch flow

Start labeling validates the folders, classes, and active configuration. It
creates `images/`, `labels/`, `data.yaml`, and `classes.txt` directly in the
chosen export directory. When preview is checked, it also creates `preview/`.
The application refuses a pre-existing export directory and stops at the first
error, displaying that error in the interface.

## Structure

- Keep provider configuration, detection, YOLO conversion, and rendering in
  reusable non-UI modules.
- Add a GUI entry point and PySide6 window/controller layer.
- Move export-root selection into the workflow API so both CLI and GUI can use
  it; preserve the CLI's existing default of `<input>/yolo_output`.

## Verification

- Unit tests cover export-root selection and classes ordering.
- GUI tests cover class add/remove renumbering, preview checkbox propagation,
  and test-image rendering state.
- Manual smoke test verifies folder picking, first-image test, image zoom, and
  a batch output with and without previews.
