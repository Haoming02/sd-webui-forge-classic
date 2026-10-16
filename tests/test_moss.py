import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import Qwen2Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules_forge/packages"))
test_argv = sys.argv
sys.argv = [sys.argv[0], "--use-pytorch-cross-attention", "--disable-xformers", "--disable-sage", "--disable-flash"]

import huggingface_guess
from huggingface_guess import model_list

from backend.loader import load_huggingface_component, split_state_dict
from backend.nn.llm.qwen35_text import Qwen35Text, torch_chunk_gated_delta_rule
from backend.nn.moss import MOSSImageTransformer
from backend.operations import using_forge_operations
from backend.sampling.condition import compile_conditions
from backend.sampling.moss import moss_sigmas
from backend.text_processing.moss_engine import Qwen35MOSSImageEngine
from modules import prompt_parser
from modules_forge.presets import CFG, SCHEDULERS, SHIFT, STEPS, PresetArch
from tools.convert_moss import convert, read_component

sys.argv = test_argv


def tiny_qwen_config():
    return dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16, partial_rotary_factor=0.5, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, rope_parameters={"rope_theta": 10000000.0}, layer_types=["linear_attention"] * 3 + ["full_attention"])


def initialize(model):
    for name, value in model.named_parameters():
        torch.nn.init.normal_(value, std=0.02)
    return model.eval()


class MOSSImageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = dict(dim=32, ffn_dim=48, num_heads=2, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8)

    def test_mask_and_negative_skip_in_mixed_batch(self):
        model = MOSSImageTransformer(**self.config, uncond_skip_layers=(1,)).eval()
        x, context = torch.randn(2, 8, 2, 3), torch.randn(2, 7, 24)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        t = torch.tensor([750.0, 750.0])
        with torch.inference_mode():
            mixed = model(x, t, context, mask, transformer_options={"uncond_indices": [1]})
            positive = model(x[:1], t[:1], context[:1, :3])
            negative = model(x[1:], t[1:], context[1:], transformer_options={"uncond_indices": [0]})
        torch.testing.assert_close(mixed, torch.cat((positive, negative)))

    def test_fp8_storage_uses_compute_dtype(self):
        source = MOSSImageTransformer(**self.config).eval()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, manual_cast_enabled=True):
            quantized = MOSSImageTransformer(**self.config).eval()
        quantized.load_state_dict(source.state_dict())
        with torch.inference_mode():
            output = quantized(torch.randn(1, 8, 2, 3), torch.tensor([500.0]), torch.randn(1, 5, 24))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.dtype, torch.float32)

    def test_delta_rule_matches_token_recurrence(self):
        for length in (1, 13, 65):
            q, k = [torch.randn(1, length, 2, 8) for _ in range(2)]
            v = torch.randn(1, length, 2, 6)
            g = -torch.rand(1, length, 2)
            beta = torch.rand(1, length, 2)
            actual, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta)
            q, k = F.normalize(q, dim=-1) / 8**0.5, F.normalize(k, dim=-1)
            state = torch.zeros(1, 2, 8, 6)
            expected = []
            for index in range(length):
                state *= g[:, index, :, None, None].exp()
                delta = (v[:, index] - torch.einsum("bhk,bhkv->bhv", k[:, index], state)) * beta[:, index, :, None]
                state += k[:, index, :, :, None] * delta[:, :, None, :]
                expected.append(torch.einsum("bhk,bhkv->bhv", q[:, index], state))
            torch.testing.assert_close(actual, torch.stack(expected, dim=1), atol=2e-6, rtol=2e-5)

    def test_qwen_loader_accepts_hf_and_comfy_prefixes(self):
        config = tiny_qwen_config()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            source = initialize(Qwen35Text(config))
        ids = torch.randint(0, 64, (1, 13))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "text_encoder"
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"text_config": config}))
            for prefix in ("language_model.", "model.language_model.", "model."):
                state = {prefix + k.removeprefix("model."): v for k, v in source.state_dict().items()}
                state["visual.unused.weight"] = torch.ones(1)
                with patch("backend.loader.memory_management.should_use_bf16", return_value=False), patch("backend.loader.memory_management.text_encoder_dtype", return_value=torch.float32):
                    loaded = load_huggingface_component(None, "text_encoder", "transformers", "Qwen3_5Model", folder, state)
                with torch.inference_mode():
                    torch.testing.assert_close(loaded(ids)[0], source(ids)[0], atol=3e-4, rtol=3e-3)

    def test_checkpoint_detection_split_and_save_prefix(self):
        model = MOSSImageTransformer(**self.config)
        state = model.state_dict()
        guess = huggingface_guess.guess({"model.diffusion_model." + k: v for k, v in state.items()})
        self.assertIsInstance(guess, model_list.MOSSImage)
        self.assertEqual(guess.unet_config["num_layers"], 3)
        self.assertEqual(guess.latent_format.latent_channels, 128)
        with tempfile.TemporaryDirectory() as folder:
            dit, encoder, vae = [Path(folder) / name for name in ("moss.safetensors", "qwen.safetensors", "vae.safetensors")]
            save_file(state, dit)
            save_file({"language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.ones(2, 2)}, encoder)
            save_file({"decoder.conv_in.weight": torch.ones(2, 2)}, vae)
            split, _ = split_state_dict(str(dit), [str(encoder), str(vae)])
            split = {key: set(value) for key, value in split.items()}
        self.assertEqual(split["transformer"], set(state))
        self.assertIn("language_model.layers.0.linear_attn.in_proj_qkv.weight", split["text_encoder"])
        self.assertIn("decoder.conv_in.weight", split["vae"])

    def test_variable_prompt_schedules_keep_padding_masked(self):
        def row(length):
            return {"crossattn": torch.cat((torch.randn(length, 24), torch.zeros(1, 24))), "attention_mask": torch.tensor([True] * length + [False])}

        short, long = row(3), row(7)
        schedules = [[prompt_parser.ScheduledPromptConditioning(10, value)] for value in (short, long)]
        cond = prompt_parser.reconstruct_cond_batch(schedules, 0)
        self.assertEqual(cond["crossattn"].shape, (2, 8, 24))
        self.assertEqual(cond["attention_mask"].sum(1).tolist(), [3, 7])
        compiled = compile_conditions(cond)[0]["model_conds"]
        self.assertIn("attention_mask", compiled)
        self.assertNotIn("y", compiled)

    def test_tokenizer_template_and_truncation(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(ROOT / "backend/huggingface/OpenMOSS-Team/MOSS-Image-Base/tokenizer", local_files_only=True)

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1, 1)

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, ids):
                self.length = ids.shape[1]
                return torch.ones(1, self.length, 24), None

        encoder = Encoder()
        engine = Qwen35MOSSImageEngine(encoder, tokenizer)
        self.assertEqual(engine.tokenize("A cat"), [248045, 846, 198, 32, 7993, 248046, 198, 248045, 74455, 198, 248068, 198])
        result = engine(["cat " * 5000])
        self.assertEqual(encoder.length, 4096)
        self.assertEqual(result["attention_mask"][0].sum(), 4096)
        self.assertFalse(result["attention_mask"][0][-1])

    def test_preset_and_schedule(self):
        self.assertIn("moss", PresetArch.choices())
        self.assertEqual((STEPS[PresetArch.moss], CFG[PresetArch.moss], SHIFT[PresetArch.moss]), (50, 3.0, 5.0))
        self.assertEqual(SCHEDULERS[PresetArch.moss], "MOSS")
        sigmas = moss_sigmas(50)
        self.assertEqual(sigmas.shape, (51,))
        self.assertEqual(sigmas[0], 1)
        self.assertEqual(sigmas[-1], 0)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]))
        torch.testing.assert_close(sigmas[:-1] * 1000, (sigmas[:-1] * 1000).round(), atol=1e-4, rtol=0)

    def test_converter_combines_shards_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            source, models = Path(folder) / "source", Path(folder) / "models"
            for component in ("transformer", "text_encoder", "vae"):
                (source / component).mkdir(parents=True)
            config = dict(_class_name="MOSSImageTransformer", text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0)
            (source / "transformer/config.json").write_text(json.dumps(config))
            save_file({"a": torch.ones(2, dtype=torch.bfloat16)}, source / "transformer/one.safetensors")
            save_file({"b": torch.zeros(3)}, source / "transformer/two.safetensors")
            (source / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
            for component, stem in (("text_encoder", "model"), ("vae", "diffusion_pytorch_model")):
                save_file({"weight": torch.ones(2)}, source / component / f"{stem}.safetensors")
            targets = convert(source, models)
            self.assertEqual([path.parent.name for path in targets], ["Stable-diffusion", "text_encoder", "VAE"])
            state = read_component(targets[0].parent, "moss-image-dit")
            self.assertEqual(state["a"].dtype, torch.bfloat16)
            self.assertEqual(set(state), {"a", "b"})
            del state
            with self.assertRaises(FileExistsError):
                convert(source, models)

    def test_converter_rejects_shard_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
            with self.assertRaises(ValueError):
                read_component(path, "model")


if __name__ == "__main__":
    unittest.main()
