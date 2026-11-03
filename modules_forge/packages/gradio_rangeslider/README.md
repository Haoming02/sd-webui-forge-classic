# RangeSlider for Gradio 6

Vendored from [freddyaboulton/gradio-range-slider](https://github.com/freddyaboulton/gradio-range-slider), version `0.0.8`, under the included Apache-2.0 license.

The frontend is ported to Gradio 6's component interface and Svelte 5.
The Python port fixes pair validation, validates bounds and step, and returns a tuple from preprocessing.
The two handles, numeric inputs, reset button, and `change`, `input`, and `release` events remain available.

Forge adds the parent `packages` directory to `sys.path` at startup, so existing imports work:

```py
from gradio_rangeslider import RangeSlider
```

Compiled assets in `templates/component` are included; running Forge requires no Node.js or package installation for this component.
To rebuild, run `npm ci --ignore-scripts` and `npm run build` from `frontend`.
From the repository root, run `venv/Scripts/python.exe modules_forge/packages/gradio_rangeslider/test_rangeslider.py` for the Python checks.

<hr>

<p align="center">
Copyright 2024 freddyaboulton
</p>
