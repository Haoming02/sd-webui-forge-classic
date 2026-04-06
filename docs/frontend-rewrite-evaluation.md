# Frontend Rewrite Evaluation

## Summary

Forge's inference backend is not the main problem. The unstable layer is the Gradio UI shell plus the custom JavaScript that has to chase Gradio DOM changes. Replacing that shell is viable, but only if the rewrite treats Gradio as an integration boundary rather than trying to keep patching around it.

The main architectural constraint is the script and extension system. Today scripts build real Gradio controls in `Script.ui()` and can intercept component creation globally. That means a direct frontend swap without a compatibility layer will break a large share of built-in and third-party extensions.

## What Is Coupled To Gradio

- `webui.py` launches `ui.create_ui()` and then runs the app through Gradio.
- `modules/ui.py` builds the main application as a large `gr.Blocks()` tree.
- `modules/scripts.py` requires scripts to return actual Gradio controls from `Script.ui()`.
- `modules/gradio_extensions.py` monkeypatches Gradio component creation so scripts can hook before and after component construction.
- `modules_forge/forge_canvas/canvas.py` implements Forge canvas behavior on top of Gradio components.
- Many built-in extensions and external scripts import Gradio directly.

## Alternatives

### 1. React or Vue SPA on top of FastAPI

This is the recommended replacement.

Why it fits:

- The backend already runs on FastAPI and exposes useful generation, progress, options, and model endpoints through `modules/api/api.py`.
- A custom SPA can own gallery rendering, lightbox behavior, canvas tools, and state management without depending on Gradio's DOM.
- Long-term maintenance is materially better than continuing to patch Gradio output markup.

Costs:

- Requires a compatibility layer for extensions that currently emit Gradio controls.
- Requires explicit frontend state and event models for txt2img, img2img, extras, settings, and progress.

### 2. HTMX plus server-rendered FastAPI templates

This is viable only for a reduced interface.

Why it is weaker:

- Gallery interactions, progressive previews, canvas editing, and extension panels are already highly interactive.
- Server-rendered fragments would reduce frontend tooling, but not enough to justify the weaker client model for this product.

### 3. PySide or another desktop-native UI

This is not recommended as the main replacement.

Why it is weaker:

- Forge is already used as a local web application and often benefits from browser-based remote access.
- A desktop UI would add packaging and platform work while not solving the extension contract problem.

### 4. Continue on Gradio

This is the lowest engineering effort in the short term and the worst long-term option.

- The current lightbox issue is a symptom of the broader problem: custom behavior is being layered on DOM that Forge does not fully control.
- Each Gradio markup change creates regression risk in galleries, tabs, canvas overlays, and custom script interactions.

## Recommendation

Adopt a custom frontend, served by the existing FastAPI app, with a staged migration away from Gradio.

The target architecture should be:

- FastAPI remains the backend host and generation API.
- A custom SPA owns the main interface.
- Gradio stays only as a temporary compatibility surface for legacy script UIs.
- A new UI-neutral schema becomes the contract between scripts and the future frontend.

## Rewrite Strategy

### Phase 1: Add a UI-neutral contract

- Keep `Script.ui()` for compatibility.
- Introduce a framework-neutral control schema and generate API-visible script metadata from it.
- Stop adding new frontend features that depend directly on Gradio DOM assumptions.

### Phase 2: Build the new shell outside Gradio

- Start with txt2img, img2img, gallery, progress, and settings that already map well to the API.
- Mount the custom frontend from FastAPI and run it beside the existing Gradio UI during transition.

### Phase 3: Bridge extensions

- Add a compatibility adapter so extensions can expose either a neutral schema or a legacy Gradio UI.
- Move built-in extensions first so the migration path is proven before asking third-party extensions to follow.

### Phase 4: Remove Gradio from the primary path

- Make the custom frontend the default UI.
- Keep Gradio only as a fallback or developer compatibility mode until the extension surface is sufficiently migrated.

## Immediate Engineering Recommendation

Do not spend more time on one-off DOM fixes as the main strategy.

Instead:

1. Treat Gradio as legacy UI infrastructure.
2. Use FastAPI as the permanent backend surface.
3. Move new UI work behind a neutral schema so a replacement frontend can be built incrementally.

## Reload Impact

This document alone requires no reload.

The `modules/scripts.py` and `modules/ui_schema.py` changes that introduce the neutral control schema require a full Forge restart.