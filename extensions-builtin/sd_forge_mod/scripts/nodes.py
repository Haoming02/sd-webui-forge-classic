import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .adapter import resolve_adapter_path, validate_adapter_for_model
from .anima_patch import register_modulation_wrapper


def _extract_pooled_output(conditioning, input_name):
    if conditioning is None or len(conditioning) == 0:
        raise RuntimeError(f"{input_name} is empty.")

    first_entry = conditioning[0]
    if not isinstance(first_entry, (list, tuple)) or len(first_entry) < 2:
        raise RuntimeError(
            f"{input_name} is invalid. Expected a CONDITIONING entry with metadata."
        )

    metadata = first_entry[1]
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{input_name} metadata is invalid.")

    pooled = metadata.get("pooled_output", None)
    if pooled is None:
        raise RuntimeError(
            f"{input_name} is missing pooled_output. Provide CLIP-based conditioning with pooled embeddings."
        )
    if not torch.is_tensor(pooled):
        raise RuntimeError(
            f"{input_name}.pooled_output must be a tensor, got {type(pooled)}."
        )

    if pooled.ndim == 1:
        pooled = pooled.unsqueeze(0)
    elif pooled.ndim != 2:
        raise RuntimeError(
            f"{input_name}.pooled_output must have rank 1 or 2, got rank {pooled.ndim}."
        )

    if pooled.shape[0] < 1:
        raise RuntimeError(f"{input_name}.pooled_output has invalid empty batch.")

    return pooled[:1].detach().float().cpu().contiguous()


def _validate_anima_model(model):
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("Invalid MODEL input: missing internal model object.")

    image_model = None
    model_config = getattr(base_model, "model_config", None)
    if model_config is not None:
        image_model = model_config.unet_config.get("image_model", None)

    if image_model != "anima":
        raise RuntimeError(
            "Anima Mod Guidance supports only Anima models (image_model='anima')."
        )

    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("Invalid Anima model: diffusion_model is missing.")
    return diffusion_model


class AnimaModGuidance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AnimaModGuidance",
            display_name="Anima Mod Guidance",
            category="advanced/guidance/anima",
            description=(
                "Strict official-style modulation guidance for Anima (Cosmos Predict2 architecture).\n"
                "Requires CLIP pooled conditionings for base/positive/negative directions."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("clip_base_conditioning"),
                io.Conditioning.Input("clip_positive_conditioning"),
                io.Conditioning.Input("clip_negative_conditioning"),
                io.Float.Input("w", default=3.0, min=-20.0, max=20.0, step=0.01),
                io.Int.Input("start_layer", default=0, min=0, max=1024, step=1),
                io.Int.Input("end_layer", default=-1, min=-1, max=1024, step=1),
                io.Combo.Input("adapter_mode", options=["auto_download", "local_file"]),
                io.String.Input("adapter_path", default="", multiline=False, advanced=True),
            ],
            outputs=[io.Model.Output(display_name="patched_model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip_base_conditioning,
        clip_positive_conditioning,
        clip_negative_conditioning,
        w,
        start_layer,
        end_layer,
        adapter_mode,
        adapter_path,
    ) -> io.NodeOutput:
        diffusion_model = _validate_anima_model(model)

        pooled_base = _extract_pooled_output(clip_base_conditioning, "clip_base_conditioning")
        pooled_positive = _extract_pooled_output(clip_positive_conditioning, "clip_positive_conditioning")
        pooled_negative = _extract_pooled_output(clip_negative_conditioning, "clip_negative_conditioning")

        resolved_adapter_path = resolve_adapter_path(adapter_mode, adapter_path)
        adapter_meta = validate_adapter_for_model(resolved_adapter_path, diffusion_model)

        pooled_dim = int(pooled_base.shape[1])
        if pooled_dim != adapter_meta["pooled_dim"]:
            raise RuntimeError(
                "clip_base_conditioning pooled dim mismatch: "
                f"got {pooled_dim}, expected {adapter_meta['pooled_dim']}."
            )
        if int(pooled_positive.shape[1]) != adapter_meta["pooled_dim"]:
            raise RuntimeError(
                "clip_positive_conditioning pooled dim mismatch: "
                f"got {int(pooled_positive.shape[1])}, expected {adapter_meta['pooled_dim']}."
            )
        if int(pooled_negative.shape[1]) != adapter_meta["pooled_dim"]:
            raise RuntimeError(
                "clip_negative_conditioning pooled dim mismatch: "
                f"got {int(pooled_negative.shape[1])}, expected {adapter_meta['pooled_dim']}."
            )

        patched_model = model.clone()
        register_modulation_wrapper(
            patched_model,
            adapter_path=resolved_adapter_path,
            clip_base_pooled=pooled_base,
            clip_positive_pooled=pooled_positive,
            clip_negative_pooled=pooled_negative,
            w=w,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        return io.NodeOutput(patched_model)


class AnimaModGuidanceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AnimaModGuidance,
        ]


async def comfy_entrypoint() -> AnimaModGuidanceExtension:
    return AnimaModGuidanceExtension()
