# Forge Neo Paint Tab Extension – Full Implementation Spec

This document defines **exactly** how the Forge Neo Paint Tab extension must be implemented. It is written for a coding agent and should be followed literally.

The goal is a **Photoshop/Krita-style canvas tab** inside **sd-webui-forge-classic-neo** that supports drawing, iterative img2img generation, SDXL, Z-Image-Turbo, LoRAs, LCM live generation, fullscreen tablet use, and LAN access.

---

## 1. Core Goals

- Add a **new top-level tab** called **Paint** in the Forge Neo UI
- Provide a **canvas-based drawing workflow** similar to Krita AI Diffusion
- Support **iterative generation**: draw → img2img → draw → img2img
- Fully support:
  - SDXL
  - Z-Image-Turbo
  - LoRAs (add/remove, per-LoRA strength)
  - Sampler overrides
  - Tiling
  - Hires fix
  - LCM + Live auto-generation
- Support **fullscreen tablet mode** for iPad + Apple Pencil via LAN

---

## 2. Extension Layout (Required)

```
extensions/
  forge_paint/
    metadata.ini
    scripts/
      paint_tab.py
    javascript/
      00_fabric.min.js
      10_paint_canvas.js
    style.css
```

- Use **Fabric.js** for the canvas engine
- Do not inline Fabric via CDN; vendor it locally

---

## 3. Architectural Rules (Important)

1. **Never load or switch models inside the extension**
   - Always use the currently loaded Forge Neo model
   - This ensures SDXL and Z-Image-Turbo work automatically

2. **Never reimplement sampling logic**
   - Use `StableDiffusionProcessingImg2Img`
   - Let Forge Neo handle samplers, schedulers, turbo models, etc.

3. **All JS–Python communication must use stable elem_id selectors**
   - No label-based or DOM-guess selectors

---

## 4. Canvas Requirements (Frontend)

### Canvas Engine

- Fabric.js canvas
- Must support:
  - Freehand brush
  - Eraser (destination-out)
  - Brush size
  - Color picker
  - Text tool
  - Object selection and transform
  - Undo stack (JSON snapshots)

### Layer Model

- Background image = last generated image
- User drawing stays on top
- On generate:
  - Flatten canvas
  - Send PNG to backend
  - Replace background with generated image

---

## 5. Fullscreen + Tablet Mode

- Add a **Fullscreen button**
- Use browser Fullscreen API
- When fullscreen:
  - Canvas resizes to viewport
  - UI switches to tablet-friendly spacing
  - Pointer events must support Apple Pencil

---

## 6. Backend Generation (Python)

### Required Backend Function

- Accept flattened canvas PNG (base64)
- Run img2img using:
  - Current model (SDXL / Z-Image-Turbo)
  - Prompt + negative
  - Steps, CFG, seed
  - Sampler override
  - Denoising strength
  - Tiling
  - Hires fix (optional)

### LCM Behavior

If **LCM enabled**:
- Clamp steps to fast range (1–12)
- Clamp CFG to ~0.5–4
- Clamp denoise to ~0.25–0.85
- Disable hires fix automatically in Live mode

---

## 7. Live Mode (LCM Required)

### Behavior

- Only active if:
  - LCM enabled
  - Live mode enabled

### Live Logic

- Detect end of drawing stroke
- Wait `Live idle delay (ms)`
- If user does not draw again:
  - Auto-generate image
- If user draws again:
  - Cancel pending generation
- Enforce minimum time between generations

### Implementation Notes

- Use Fabric `path:created` event
- Use JS debounce + timers
- JS triggers Generate button automatically

---

## 8. LoRA Support (Required)

### UI

- LoRA dropdown (from `models/Lora/`)
- Refresh LoRA list button
- Add LoRA button
- Remove LoRA button
- Active LoRA list displayed next to prompt
- Each LoRA has adjustable strength

### Data Model

```
[{ name: "lora_name", strength: 0.8 }]
```

### Prompt Injection

- Append tags to prompt:

```
<lora:name:strength>
```

- Do not modify negative prompt

### Warnings

- If selected LoRA name contains:
  - `sd15`, `sd-15`, `sd_15`, or `1.5`
- Show warning:

> This LoRA appears to be SD 1.5 and may not work correctly with SDXL or Z-Image-Turbo

---

## 9. Required elem_id Map (Do Not Change)

These IDs **must exist exactly** so JS selectors are stable:

- `fpt_canvas_in`
- `fpt_gen_out_b64`
- `fpt_prompt`
- `fpt_negative`
- `fpt_generate`
- `fpt_lcm`
- `fpt_live`
- `fpt_live_delay`
- `fpt_live_min_interval`
- `fpt_steps`
- `fpt_cfg`
- `fpt_denoise`
- `fpt_seed`
- `fpt_sampler`
- `fpt_tiling`
- `fpt_hr_enable`
- `fpt_hr_scale`
- `fpt_hr_steps`
- `fpt_hr_upscaler`

---

## 10. LAN + iPad Usage

- Extension must work when Forge Neo is run as a LAN server
- UI must be usable on mobile Safari
- No mouse-only interactions

---

## 11. Explicit Non-Goals (Do NOT Implement)

- ControlNet support
- Mask layers
- Refiner switching
- Model loading or unloading
- Cloud or remote inference

---

## 12. Quality Expectations

- No brittle DOM selectors
- No hardcoded model assumptions
- No blocking UI during generation
- Must handle repeated live generations safely
- Must not spam generation when user is drawing

---

## 13. Definition of Done

The extension is complete when:

- User can draw on canvas
- Auto-generate triggers correctly in Live mode
- Generated image replaces background
- LoRAs can be added, removed, and adjusted
- Works with SDXL and Z-Image-Turbo
- Fullscreen tablet drawing works smoothly

---

This document is authoritative. The coding agent should follow it exactly.


---

# Agent Task-by-Task Implementation Checklist

This section converts the specification above into an **ordered, atomic checklist** for a coding agent. Tasks should be completed **top to bottom**. Do not skip steps.

---

## Phase 0 – Pre-flight

- [ ] Clone or open the `sd-webui-forge-classic-neo` repository
- [ ] Confirm Forge Neo launches successfully before changes
- [ ] Confirm `extensions/` directory exists and is writable
- [ ] Create a new git branch for this extension

---

## Phase 1 – Extension Skeleton

- [ ] Create folder `extensions/forge_paint/`
- [ ] Create `metadata.ini` with a unique extension name
- [ ] Create subfolders:
  - [ ] `scripts/`
  - [ ] `javascript/`
- [ ] Create empty files:
  - [ ] `scripts/paint_tab.py`
  - [ ] `javascript/10_paint_canvas.js`
  - [ ] `style.css`
- [ ] Vendor Fabric.js as `javascript/00_fabric.min.js`

---

## Phase 2 – Register Paint Tab (Python)

- [ ] Import Forge/A1111 extension hooks
- [ ] Register a **top-level tab** named `Paint`
- [ ] Verify tab appears in UI with no errors

---

## Phase 3 – Define Stable elem_id Contract (Python)

Create all required Gradio components with the exact `elem_id` values below.

### Hidden IO
- [ ] `fpt_canvas_in` – hidden textbox (canvas PNG data URL input)
- [ ] `fpt_gen_out_b64` – hidden textbox (generated image output)

### Prompt
- [ ] `fpt_prompt`
- [ ] `fpt_negative`

### Generation Controls
- [ ] `fpt_generate` (button)
- [ ] `fpt_steps`
- [ ] `fpt_cfg`
- [ ] `fpt_denoise`
- [ ] `fpt_seed`
- [ ] `fpt_sampler`
- [ ] `fpt_tiling`

### Hires Fix
- [ ] `fpt_hr_enable`
- [ ] `fpt_hr_scale`
- [ ] `fpt_hr_steps`
- [ ] `fpt_hr_upscaler`

### LCM + Live
- [ ] `fpt_lcm`
- [ ] `fpt_live`
- [ ] `fpt_live_delay`
- [ ] `fpt_live_min_interval`

- [ ] Verify all IDs exist in DOM at runtime

---

## Phase 4 – Backend img2img Pipeline

- [ ] Implement base64 → PIL conversion
- [ ] Implement PIL → base64 conversion
- [ ] Build `StableDiffusionProcessingImg2Img` using:
  - [ ] current loaded model only
  - [ ] prompt + negative
  - [ ] steps, CFG, seed
  - [ ] sampler override (optional)
  - [ ] tiling
  - [ ] denoise strength
- [ ] Ensure SDXL and Z-Image-Turbo work without special casing

---

## Phase 5 – Hires Fix Support

- [ ] Apply hires fix parameters only when enabled
- [ ] Ensure hires fix is disabled automatically during Live + LCM

---

## Phase 6 – LCM Behavior Enforcement

When `fpt_lcm` is enabled:

- [ ] Clamp steps to fast range (≤12)
- [ ] Clamp CFG to ≤4
- [ ] Clamp denoise to ≤0.85
- [ ] Disable hires fix in Live mode

---

## Phase 7 – LoRA Support

### Backend
- [ ] Scan `models/Lora/` for available LoRAs
- [ ] Maintain LoRA state as list of `{name, strength}`
- [ ] Append `<lora:name:strength>` tags to prompt

### UI
- [ ] LoRA dropdown
- [ ] Refresh LoRA list button
- [ ] Add LoRA button
- [ ] Remove LoRA button
- [ ] Active LoRA list displayed near prompt
- [ ] Per-LoRA strength editable

### Warnings
- [ ] Detect LoRA names containing `sd15`, `sd-15`, `sd_15`, `1.5`
- [ ] Display SD 1.5 compatibility warning

---

## Phase 8 – Canvas Engine (Frontend)

- [ ] Initialize Fabric.js canvas
- [ ] Implement brush tool
- [ ] Implement eraser tool (destination-out)
- [ ] Implement color picker
- [ ] Implement brush size control
- [ ] Implement text tool
- [ ] Enable object select/move/scale
- [ ] Implement undo stack (JSON snapshots)

---

## Phase 9 – Iterative Generation Loop

- [ ] Flatten canvas to PNG
- [ ] Write PNG to `fpt_canvas_in`
- [ ] Trigger `fpt_generate`
- [ ] Receive generated image via `fpt_gen_out_b64`
- [ ] Replace canvas background with generated image
- [ ] Preserve user drawing layer

---

## Phase 10 – Live Mode (Frontend)

- [ ] Listen for Fabric `path:created`
- [ ] Debounce generation using idle delay
- [ ] Cancel pending gen if user draws again
- [ ] Enforce minimum interval between gens
- [ ] Only trigger if LCM + Live enabled
- [ ] Do not trigger if prompt is empty

---

## Phase 11 – Fullscreen + Tablet Mode

- [ ] Add fullscreen toggle button
- [ ] Use Fullscreen API
- [ ] Resize canvas to viewport
- [ ] Increase UI hit targets for touch
- [ ] Ensure pointer events work with Apple Pencil

---

## Phase 12 – LAN Validation

- [ ] Run Forge Neo with LAN access
- [ ] Access UI from iPad browser
- [ ] Verify drawing latency
- [ ] Verify Live mode stability

---

## Phase 13 – Final Validation

- [ ] No brittle DOM selectors
- [ ] No model loading logic
- [ ] No ControlNet code
- [ ] No blocking UI during generation
- [ ] Repeated live gens do not crash or stall

---

## Phase 14 – Done Criteria

The extension is complete when:

- [ ] Canvas drawing works
- [ ] Iterative img2img loop works
- [ ] LoRAs can be added, removed, adjusted
- [ ] Live LCM auto-generation works
- [ ] Fullscreen tablet drawing works
- [ ] SDXL and Z-Image-Turbo both function correctly

---

End of checklist.
