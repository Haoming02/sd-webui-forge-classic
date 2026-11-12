# Copyright Forge 2024
# AGPL V3 + CC-BY-SA

from abc import ABC, abstractmethod
from functools import cache

import torch
from gguf import quants as gguf_quants
from gguf.constants import GGML_QUANT_SIZES, QK_K, GGMLQuantizationType

from .quick_4bits_ops import (
    change_4bits_order,
    quick_unpack_4bits,
    quick_unpack_4bits_u,
)

quick_split = lambda x, p: torch.split(x, p + [x.shape[1] - sum(p)], dim=-1)


class TorchQuant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    def __init_subclass__(cls, qtype: GGMLQuantizationType):
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]

    @classmethod
    def quantize_pytorch(cls, data, parent) -> torch.Tensor:
        if not parent.baked:
            raise ValueError("GGUF Tensor is not baked...")

        block_size, type_size = cls.block_size, cls.type_size
        blocks = data.reshape(-1, block_size)
        parent.data = cls.quantize_blocks_pytorch(blocks, block_size, type_size, parent).contiguous()
        return parent

    @classmethod
    def bake(cls, parameter):
        if parameter.baked:
            return

        data = parameter.data
        rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        parameter.data = blocks.contiguous()
        cls.bake_inner(parameter)
        parameter.baked = True
        return

    @classmethod
    def bake_inner(cls, parameter):
        pass

    @classmethod
    def dequantize_pytorch(cls, x):
        if not x.baked:
            raise ValueError("GGUF Tensor is not baked...")

        blocks = cls.dequantize_blocks_pytorch(x.data, cls.block_size, cls.type_size, x)
        return blocks.view(x.shape)

    @classmethod
    @abstractmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        raise NotImplementedError


class BF16(TorchQuant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


class Q4_0(TorchQuant, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data
        d, x = quick_split(blocks, [2])
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        x = change_4bits_order(x).view(torch.uint8)
        parameter.data = torch.cat([d, x], dim=-1).contiguous()
        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        d, qs = quick_split(blocks, [2])
        d = d.view(parameter.computation_dtype)
        qs = quick_unpack_4bits(qs)
        return d * qs

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        n_blocks = blocks.shape[0]

        imax = torch.abs(blocks).argmax(dim=-1, keepdim=True)
        max_vals = torch.gather(blocks, -1, imax)

        d = max_vals / -8
        id = torch.where(d == 0, torch.tensor(0.0, device=d.device), 1.0 / d)

        qs = torch.trunc((blocks * id) + 8.5).clip(0, 15).to(torch.uint8)

        qs = qs.reshape((n_blocks, block_size // 2, 2))
        qs = qs[..., 0] | (qs[..., 1] << 4)

        d = d.to(parent.computation_dtype).view(torch.uint8)

        return torch.cat([d, qs], dim=-1)


class Q4_1(TorchQuant, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data

        d, m, qs = quick_split(blocks, [2, 2])
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        m = m.view(torch.float16).to(parameter.computation_dtype).view(torch.uint8)
        qs = change_4bits_order(qs).view(torch.uint8)

        parameter.data = torch.cat([d, m, qs], dim=-1).contiguous()

        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        d, m, qs = quick_split(blocks, [2, 2])
        d = d.view(parameter.computation_dtype)
        m = m.view(parameter.computation_dtype)
        qs = quick_unpack_4bits_u(qs)
        return (d * qs) + m


class Q5_0(TorchQuant, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        def to_uint32(x):
            # by. City96 - Apache-2.0
            x = x.view(torch.uint8).to(torch.int32)
            return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

        n_blocks = blocks.shape[0]

        d, qh, qs = quick_split(blocks, [2, 4])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        qh = to_uint32(qh)

        qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
        ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

        qh = (qh & 1).to(torch.uint8)
        ql = (ql & 0x0F).reshape(n_blocks, -1)

        qs = (ql | (qh << 4)).to(torch.int8) - 16
        return d * qs

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        n_blocks = blocks.shape[0]

        imax = torch.abs(blocks).argmax(dim=-1, keepdim=True)
        max_val = torch.gather(blocks, dim=-1, index=imax)

        d = max_val / -16
        id = torch.where(d == 0, torch.tensor(0.0, device=d.device), 1.0 / d)

        q = torch.trunc((blocks.float() * id.float()) + 16.5).clamp(0, 31).to(torch.uint8)

        qs = q.view(n_blocks, 2, block_size // 2)
        qs = (qs[..., 0, :] & 0x0F) | (qs[..., 1, :] << 4)

        qh = q.view(n_blocks, 32)
        qh_packed = torch.zeros((n_blocks, 4), dtype=torch.uint8, device=qh.device)

        for i in range(4):
            qh_packed[:, i] = (qh[:, i * 8 + 0] >> 4) | (qh[:, i * 8 + 1] >> 3 & 0x02) | (qh[:, i * 8 + 2] >> 2 & 0x04) | (qh[:, i * 8 + 3] >> 1 & 0x08) | (qh[:, i * 8 + 4] << 0 & 0x10) | (qh[:, i * 8 + 5] << 1 & 0x20) | (qh[:, i * 8 + 6] << 2 & 0x40) | (qh[:, i * 8 + 7] << 3 & 0x80)

        d = d.to(torch.float16).view(torch.uint8)

        return torch.cat([d, qh_packed, qs], dim=-1)


class Q5_1(TorchQuant, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        def to_uint32(x):
            # by. City96 - Apache-2.0
            x = x.view(torch.uint8).to(torch.int32)
            return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

        n_blocks = blocks.shape[0]

        d, m, qh, qs = quick_split(blocks, [2, 2, 4])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        m = m.view(torch.float16).to(parameter.computation_dtype)
        qh = to_uint32(qh)

        qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
        ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
        qh = (qh & 1).to(torch.uint8)
        ql = (ql & 0x0F).reshape((n_blocks, -1))

        qs = ql | (qh << 4)
        return (d * qs) + m


class Q8_0(TorchQuant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data
        d, x = quick_split(blocks, [2])
        x = x.view(torch.int8)
        d = d.view(torch.float16).to(parameter.computation_dtype).view(torch.int8)
        parameter.data = torch.cat([d, x], dim=-1).contiguous()
        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        d, x = quick_split(blocks, [2])
        d = d.view(parameter.computation_dtype)
        return x * d

    @classmethod
    def quantize_blocks_pytorch(cls, blocks, block_size, type_size, parent) -> torch.Tensor:
        d = torch.abs(blocks).max(dim=1, keepdim=True).values / 127
        ids = torch.where(d == 0, torch.zeros_like(d), 1 / d)
        qs = torch.round(blocks * ids)
        d = d.to(parent.computation_dtype).view(torch.int8)
        qs = qs.to(torch.int8)
        return torch.cat([d, qs], dim=1)


class Q2_K(TorchQuant, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        # by. City96 - Apache-2.0
        n_blocks = blocks.shape[0]
        scales, qs, d, dmin = quick_split(blocks, [QK_K // 16, QK_K // 4, 2])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        dmin = dmin.view(torch.float16).to(parameter.computation_dtype)

        dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
        ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
        shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
        qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
        qs = qs.reshape((n_blocks, QK_K // 16, 16))
        qs = dl * qs - ml
        return qs.reshape((n_blocks, -1))


class Q3_K(TorchQuant, qtype=GGMLQuantizationType.Q3_K):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        # by. City96 - Apache-2.0
        n_blocks = blocks.shape[0]
        hmask, qs, scales, d = quick_split(blocks, [QK_K // 8, QK_K // 4, 12])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        lscales, hscales = scales[:, :8], scales[:, 8:]
        lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
        scales = scales.to(torch.int8) - 32
        dl = (d * scales).reshape((n_blocks, 16, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
        qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
        q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
        return (dl * q).reshape((n_blocks, QK_K))


class Q4_K(TorchQuant, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = 12

    @staticmethod
    def get_scale_min_pytorch(scales):
        n_blocks = scales.shape[0]
        scales = scales.view(torch.uint8)
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
        sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
        min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
        return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

    @classmethod
    def bake_inner(cls, parameter):
        blocks = parameter.data
        n_blocks = blocks.shape[0]
        d, dmin, scales, qs = quick_split(blocks, [2, 2, cls.K_SCALE_SIZE])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
        sc, m = Q4_K.get_scale_min_pytorch(scales)
        d = (d * sc).reshape((n_blocks, -1, 1))
        dm = (dmin * m).reshape((n_blocks, -1, 1)).to(parameter.computation_dtype)

        qs = qs.reshape((n_blocks, -1, 1, 32))
        qs = change_4bits_order(qs)

        d = d.view(torch.uint8).reshape((n_blocks, -1))
        dm = dm.view(torch.uint8).reshape((n_blocks, -1))
        qs = qs.view(torch.uint8)

        parameter.data = torch.cat([d, dm, qs], dim=-1).contiguous()
        return

    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        n_blocks = blocks.shape[0]

        d, dm, qs = quick_split(blocks, [16, 16])
        d = d.view(parameter.computation_dtype).view((n_blocks, -1, 1))
        dm = dm.view(parameter.computation_dtype).view((n_blocks, -1, 1))
        qs = quick_unpack_4bits_u(qs).view((n_blocks, -1, 32))

        return (d * qs - dm).reshape((n_blocks, QK_K))


class Q5_K(TorchQuant, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        # by. City96 - Apache-2.0
        QK_K = 256
        K_SCALE_SIZE = 12
        n_blocks = blocks.shape[0]
        d, dmin, scales, qh, qs = quick_split(blocks, [2, 2, K_SCALE_SIZE, QK_K // 8])
        d = d.view(torch.float16).to(parameter.computation_dtype)
        dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
        sc, m = Q4_K.get_scale_min_pytorch(scales)
        d = (d * sc).reshape((n_blocks, -1, 1))
        dm = (dmin * m).reshape((n_blocks, -1, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
        ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
        qh = (qh & 0x01).reshape((n_blocks, -1, 32))
        q = ql | (qh << 4)
        return (d * q - dm).reshape((n_blocks, QK_K))


class Q6_K(TorchQuant, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter) -> torch.Tensor:
        n_blocks = blocks.shape[0]
        ql, qh, scales, d = quick_split(blocks, [QK_K // 2, QK_K // 4, QK_K // 16])
        scales = scales.view(torch.int8).to(parameter.computation_dtype)
        d = d.view(torch.float16).to(parameter.computation_dtype)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
        ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
        ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
        qh = (qh & 0x03).reshape((n_blocks, -1, 32))
        q = (ql | (qh << 4)).to(torch.int8) - 32
        q = q.reshape((n_blocks, QK_K // 16, -1))
        return (d * q).reshape((n_blocks, QK_K))


# region I-Quants


def _iq_bits(values, width, count):
    shifts = torch.arange(count, device=values.device, dtype=values.dtype) * width
    return (values.unsqueeze(-1) >> shifts) & ((1 << width) - 1)


def _iq_words(values):
    b0, b1, b2, b3 = values.reshape(values.size(0), -1, 4).to(torch.int32).unbind(-1)
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


def _iq_lookup(table, indices):
    return table.index_select(0, indices.flatten().to(torch.int32))


@cache
def _iq_grid(quant, device):
    quant.init_grid()
    return torch.tensor(quant.grid.reshape(quant.grid_shape), dtype=torch.int8, device=device)


@cache
def _iq_sign_table(device):
    return torch.tensor(list(gguf_quants.IQ2_XXS.ksigns), dtype=torch.uint8, device=device)


def _iq_signs(values):
    return 1 - 2 * _iq_bits(values, 1, 8).to(torch.int8)


def _iq_apply_scale(values, scales, dtype):
    return values.float().mul_(scales).flatten(1).to(dtype)


class IQ2_XXS(TorchQuant, qtype=GGMLQuantizationType.IQ2_XXS):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs = quick_split(blocks, [2])
        qs = qs.reshape(blocks.size(0), -1, 8)
        scale_signs = _iq_words(qs[..., 4:])
        scales = (d.view(torch.float16).float() * (0.5 + ((scale_signs >> 28) & 15)) * 0.25).unsqueeze(-1)
        sign_bytes = _iq_lookup(_iq_sign_table(blocks.device), _iq_bits(scale_signs, 7, 4))
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ2_XXS, blocks.device), qs[..., :4])
        values = grid.mul_(_iq_signs(sign_bytes)).reshape(blocks.size(0), -1, 32)
        return _iq_apply_scale(values, scales, parameter.computation_dtype)


class IQ2_XS(TorchQuant, qtype=GGMLQuantizationType.IQ2_XS):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs, scales = quick_split(blocks, [2, QK_K // 4])
        qs = qs.view(torch.int16)
        scales = (d.view(torch.float16).float() * (0.5 + _iq_bits(scales, 4, 2).flatten(1)) * 0.25).unsqueeze(-1)
        sign_bytes = _iq_lookup(_iq_sign_table(blocks.device), (qs >> 9) & 127)
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ2_XS, blocks.device), qs & 511)
        values = grid.mul_(_iq_signs(sign_bytes)).reshape(blocks.size(0), -1, 16)
        return _iq_apply_scale(values, scales, parameter.computation_dtype)


class IQ2_S(TorchQuant, qtype=GGMLQuantizationType.IQ2_S):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs, signs, qh, scales = quick_split(blocks, [2, QK_K // 8, QK_K // 8, QK_K // 32])
        scales = (d.view(torch.float16).float() * (0.5 + _iq_bits(scales, 4, 2).flatten(1)) * 0.25).unsqueeze(-1)
        indices = qs.to(torch.int16) | (_iq_bits(qh, 2, 4).flatten(1).to(torch.int16) << 8)
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ2_S, blocks.device), indices)
        values = grid.mul_(_iq_signs(signs.flatten())).reshape(blocks.size(0), -1, 16)
        return _iq_apply_scale(values, scales, parameter.computation_dtype)


class IQ3_XXS(TorchQuant, qtype=GGMLQuantizationType.IQ3_XXS):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs, scale_signs = quick_split(blocks, [2, QK_K // 4])
        scale_signs = _iq_words(scale_signs)
        scales = (d.view(torch.float16).float() * (0.5 + ((scale_signs >> 28) & 15)) * 0.5).unsqueeze(-1)
        sign_bytes = _iq_lookup(_iq_sign_table(blocks.device), _iq_bits(scale_signs, 7, 4))
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ3_XXS, blocks.device), qs).reshape(-1, 8)
        values = grid.mul_(_iq_signs(sign_bytes)).reshape(blocks.size(0), -1, 32)
        return _iq_apply_scale(values, scales, parameter.computation_dtype)


class IQ3_S(TorchQuant, qtype=GGMLQuantizationType.IQ3_S):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs, qh, signs, scales = quick_split(blocks, [2, QK_K // 4, QK_K // 32, QK_K // 8])
        scales = (d.view(torch.float16).float() * (2 * _iq_bits(scales, 4, 2).flatten(1) + 1)).unsqueeze(-1)
        indices = qs.to(torch.int16) | (_iq_bits(qh, 1, 8).flatten(1).to(torch.int16) << 8)
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ3_S, blocks.device), indices).reshape(-1, 8)
        values = grid.mul_(_iq_signs(signs.flatten())).reshape(blocks.size(0), -1, 32)
        return _iq_apply_scale(values, scales, parameter.computation_dtype)


class IQ1_S(TorchQuant, qtype=GGMLQuantizationType.IQ1_S):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs, qh = quick_split(blocks, [2, QK_K // 8])
        qh = qh.view(torch.int16)
        scales = d.view(torch.float16).float() * (2 * ((qh >> 12) & 7) + 1)
        indices = qs.to(torch.int16) | (_iq_bits(qh, 3, 4).flatten(1) << 8)
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ1_S, blocks.device), indices).reshape(blocks.size(0), -1, 32)
        delta = (1 - 2 * ((qh >> 15) & 1)).to(torch.int8).unsqueeze(-1)
        values = 8 * grid + delta
        return _iq_apply_scale(values, (scales * 0.125).unsqueeze(-1), parameter.computation_dtype)


class IQ1_M(TorchQuant, qtype=GGMLQuantizationType.IQ1_M):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        qs, qh, scales = quick_split(blocks, [QK_K // 8, QK_K // 16])
        scales = scales.view(torch.int16)

        s0, s1, s2, s3 = ((scales >> 12) & 15).unbind(1)
        d = (s0 | (s1 << 4) | (s2 << 8) | (s3 << 12)).view(torch.float16).float().unsqueeze(-1)
        scales = d * (2 * _iq_bits(scales, 3, 4).flatten(1) + 1)
        qh = _iq_bits(qh, 4, 2).flatten(1)
        indices = qs.to(torch.int16) | ((qh & 7).to(torch.int16) << 8)
        grid = _iq_lookup(_iq_grid(gguf_quants.IQ1_S, blocks.device), indices)
        delta = (1 - 2 * (qh >> 3).to(torch.int8)).reshape(-1, 1)
        values = (8 * grid + delta).reshape(blocks.size(0), -1, 16)
        return _iq_apply_scale(values, (scales * 0.125).unsqueeze(-1), parameter.computation_dtype)


@cache
def _iq4_table(device):
    values = torch.tensor(gguf_quants.IQ4_NL.kvalues, dtype=torch.int8)
    packed = torch.arange(256)
    return torch.stack((values[packed & 15], values[packed >> 4]), dim=-1).to(device)


def _iq4_values(qs):
    values = _iq_lookup(_iq4_table(qs.device), qs)
    return values.reshape(qs.size(0), -1, 16, 2).transpose(-2, -1).flatten(2)


class IQ4_NL(TorchQuant, qtype=GGMLQuantizationType.IQ4_NL):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, qs = quick_split(blocks, [2])
        return _iq_apply_scale(_iq4_values(qs), d.view(torch.float16).float().unsqueeze(-1), parameter.computation_dtype)


class IQ4_XS(TorchQuant, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def dequantize_blocks_pytorch(cls, blocks, block_size, type_size, parameter):
        d, scales_h, scales_l, qs = quick_split(blocks, [2, 2, QK_K // 64])
        scales_h = _iq_bits(scales_h.view(torch.int16), 2, 8).flatten(1).to(torch.int8)
        scales_l = _iq_bits(scales_l, 4, 2).flatten(1).to(torch.int8)
        scales = (scales_l | (scales_h << 4)) - 32
        d = d.view(torch.float16).float() * scales
        return _iq_apply_scale(_iq4_values(qs), d.unsqueeze(-1), parameter.computation_dtype)
