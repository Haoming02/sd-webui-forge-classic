# Local Customizations Summary

This document describes the updates included in commit `8f7254f3`:

- Commit message: `chore: save current local forge customization state`
- Date: 2026-02-20

## What Was Added/Changed

### 1) Inpaint masked-padding visual overlay
- Added `javascript/inpaintPaddingOverlay.js`
- Purpose:
  - In `img2img > Inpaint`, draws a live dashed overlay showing the effective region for **Only masked padding, pixels**.
  - Updates as mask content and padding slider change.
- Notes:
  - Uses Forge canvas selectors and works with `#img2maskimg`.

### 2) Inpaint brush width hotkey update
- Updated:
  - `modules_forge/forge_canvas/canvas.js`
  - `modules_forge/forge_canvas/canvas.html`
- Behavior change:
  - Brush width adjustment changed from **`W + Mouse Wheel`** to **`Ctrl + Mouse Wheel`**.
  - Tooltip text in canvas toolbar updated accordingly.

### 3) Expanded image preview wheel zoom
- Updated `javascript/imageviewer.js`
- Purpose:
  - In lightbox/expanded image view, mouse wheel now zooms in/out.
- Behavior:
  - Zoom range is bounded (`0.2x` to `8x`).
  - Zoom centers around cursor position.
  - Zoom resets when opening/closing modal, switching images, and leaving tiling mode.

### 4) Legacy extension compatibility shim
- Added `modules/sd_hijack.py`
- Purpose:
  - Provides a minimal compatibility layer for extensions still importing:
    - `from modules.sd_hijack import model_hijack`
  - Includes fallback prompt-length estimation if the model API is unavailable.

### 5) XPU startup script adjustments
- Updated `start-xpu.bat`
- Changes:
  - Ensures `ui-config-clean.json` exists on startup.
  - Uses dedicated launch args including:
    - `--port 7863`
    - `--ui-config-file ui-config-clean.json`
  - VRAM reserve set to `1.0`.

### 6) Clean UI config snapshot
- Added `ui-config-clean.json`
- Purpose:
  - Stores a stable UI configuration profile for launches using `--ui-config-file ui-config-clean.json`.

## Files Included In Commit

- `javascript/imageviewer.js` (modified)
- `javascript/inpaintPaddingOverlay.js` (added)
- `modules/sd_hijack.py` (added)
- `modules_forge/forge_canvas/canvas.html` (modified)
- `modules_forge/forge_canvas/canvas.js` (modified)
- `start-xpu.bat` (modified)
- `ui-config-clean.json` (added)

