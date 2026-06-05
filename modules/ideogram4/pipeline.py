"""Lazy loader + defensive caller for the official Ideogram4Pipeline (spec §4.3, §4.6).

This module deliberately performs *all* heavy imports (torch, diffusers, the
``ideogram4`` package) lazily inside functions, so that importing this module
(and therefore loading the WebUI UI) never requires the model to be installed.

The exact ``Ideogram4Pipeline.__call__`` signature can differ between the
diffusers integration and the standalone ``ideogram4`` package, and cannot be
verified in this environment.  ``call_pipeline`` therefore introspects the real
signature and only passes keyword arguments the pipeline actually accepts —
unknown args are dropped (with a debug log) rather than crashing generation.
"""

import inspect
import logging
import os

logger = logging.getLogger("ideogram4")

_PIPELINE_CACHE: dict = {}


class Ideogram4Error(RuntimeError):
    """Raised for user-actionable problems (missing package, weights, token, HW)."""


def _import_pipeline_class():
    """Locate the Ideogram4Pipeline class from diffusers or the ideogram4 package."""
    errors = []
    for module, attr in (
        ("diffusers", "Ideogram4Pipeline"),
        ("ideogram4", "Ideogram4Pipeline"),
        ("ideogram4.pipeline", "Ideogram4Pipeline"),
    ):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as e:  # ImportError or AttributeError
            errors.append(f"{module}.{attr}: {e}")

    raise Ideogram4Error(
        "Could not import Ideogram4Pipeline. Install the official inference code "
        "(`pip install ideogram4`, github.com/ideogram-oss/ideogram4) or a diffusers "
        "build that ships Ideogram4Pipeline.\nTried:\n  " + "\n  ".join(errors)
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _looks_like_local_path(model_path: str) -> bool:
    return os.path.exists(model_path)


def _hf_token() -> str | None:
    token = None
    try:
        from modules import shared

        token = getattr(shared.opts, "ideogram4_hf_token", None)
    except Exception:
        pass
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_pipeline(model_path: str, quantization: str = "nf4"):
    """Load (and cache) the Ideogram4Pipeline for ``(model_path, quantization)``.

    ``model_path`` is a local diffusers folder (default) or a Hugging Face repo id.
    Raises ``Ideogram4Error`` with an actionable message on misconfiguration.
    """
    if not model_path:
        raise Ideogram4Error(
            "Ideogram 4.0 model path is not set. Point Settings → 'Ideogram 4.0' → "
            "'Model path' at a local diffusers folder (or a Hugging Face repo id)."
        )

    quantization = (quantization or "nf4").lower()
    cache_key = (model_path, quantization)
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    is_local = _looks_like_local_path(model_path)
    if not is_local and "/" not in model_path:
        raise Ideogram4Error(
            f'Ideogram 4.0 model path "{model_path}" is neither an existing folder nor a '
            "Hugging Face repo id (expected e.g. 'ideogram-ai/ideogram-4-nf4')."
        )

    if quantization == "nf4" and not _cuda_available():
        raise Ideogram4Error(
            "The nf4 weights require CUDA, which is not available here. Use a CUDA GPU, "
            "or switch quantization to fp8 (which needs the official non-diffusers loader)."
        )
    if quantization == "fp8":
        raise Ideogram4Error(
            "fp8 weights are not Diffusers-compatible, so they cannot be loaded through "
            "Ideogram4Pipeline. Use the nf4 weights, or run the official fp8 inference path."
        )

    PipelineClass = _import_pipeline_class()

    from_pretrained_kwargs = {}
    if not is_local:
        token = _hf_token()
        if token:
            from_pretrained_kwargs["token"] = token

    logger.info("Loading Ideogram 4.0 pipeline from %s (%s)", model_path, quantization)
    try:
        pipe = PipelineClass.from_pretrained(model_path, **from_pretrained_kwargs)
    except TypeError:
        # older diffusers used `use_auth_token` instead of `token`
        if "token" in from_pretrained_kwargs:
            from_pretrained_kwargs["use_auth_token"] = from_pretrained_kwargs.pop("token")
            pipe = PipelineClass.from_pretrained(model_path, **from_pretrained_kwargs)
        else:
            raise
    except Exception as e:
        name = type(e).__name__
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            raise Ideogram4Error(
                "Access to the Ideogram 4.0 weights was denied. Accept the license at "
                "https://huggingface.co/ideogram-ai/ideogram-4-nf4 and set an HF token "
                "(Settings → 'Ideogram 4.0' → 'HF token', or the HF_TOKEN env var)."
            ) from e
        raise

    if _cuda_available():
        try:
            pipe = pipe.to("cuda")
        except Exception:
            logger.warning("Could not move Ideogram 4.0 pipeline to CUDA", exc_info=True)

    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def clear_cache():
    _PIPELINE_CACHE.clear()


def _build_generator(seed):
    if seed is None or seed < 0:
        return None
    try:
        import torch

        device = "cuda" if _cuda_available() else "cpu"
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        logger.warning("Could not build a torch.Generator for seed %s", seed, exc_info=True)
        return None


def call_pipeline(
    pipe,
    prompt: str,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    guidance_schedule=None,
    mu=None,
    std=None,
    negative_prompt=None,
    transparent: bool = False,
    seed=None,
    num_images: int = 1,
):
    """Call the pipeline, passing only kwargs its ``__call__`` actually accepts.

    Returns a list of PIL images.
    """
    try:
        sig = inspect.signature(pipe.__call__)
        accepted = set(sig.parameters)
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (TypeError, ValueError):
        accepted = set()
        has_var_kw = True

    def pick(*names):
        """Return the first accepted parameter name, or the canonical one under **kwargs."""
        for n in names:
            if n in accepted:
                return n
        return names[0] if has_var_kw else None

    generator = _build_generator(seed)

    # logical arg -> (candidate parameter names in priority order, value)
    logical = [
        (("height",), height),
        (("width",), width),
        (("num_steps", "num_inference_steps"), steps),
        (("guidance_scale",), guidance_scale),
        (("guidance_schedule",), guidance_schedule),
        (("mu",), mu),
        (("std",), std),
        (("negative_prompt",), negative_prompt),
        (("transparent", "transparent_background"), True if transparent else None),
        (("num_images_per_prompt", "num_images"), num_images),
    ]

    kwargs = {}
    for names, value in logical:
        if value is None:
            continue
        key = pick(*names)
        if key is not None:
            kwargs[key] = value

    # seed: prefer a real generator, else fall back to a `seed`/`generator` kwarg
    if generator is not None and (("generator" in accepted) or has_var_kw):
        kwargs["generator"] = generator
    elif seed is not None and seed >= 0:
        key = pick("seed", "generator")
        if key == "seed":
            kwargs["seed"] = int(seed)

    logger.debug("Ideogram4 __call__ kwargs: %s", sorted(kwargs))

    result = pipe(prompt, **kwargs)

    images = getattr(result, "images", None)
    if images is None:
        images = result if isinstance(result, (list, tuple)) else [result]
    return list(images)
