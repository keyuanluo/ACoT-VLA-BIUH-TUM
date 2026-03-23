# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_50m":
        return Config(
            width=128,
            depth=18,
            mlp_dim=1024,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_150m":
        # ~160M params
        return Config(
            width=768,
            depth=18,
            mlp_dim=3072,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_250m":
        return Config(
            width=896,
            depth=18,
            mlp_dim=3584,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_500m":
        return Config(
            width=1408,
            depth=18,
            mlp_dim=5632,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_600m":
        # 620M params
        return Config(
            width=1536,
            depth=18,
            mlp_dim=6144,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


# @at.typecheck
# class Block(nn.Module):
#     """Transformer block."""
#
#     configs: tuple[Config, ...]
#
#     dropout: float = 0.0
#     dropout_bdims: tuple[int, ...] = ()
#
#     @nn.compact
#     def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
#         xs = sharding.activation_sharding_constraint(xs)
#         drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x
#
#         attn = Attention(configs=self.configs, name="attn")
#
#         pre_attn = []
#         gates = []
#         for i, x in enumerate(xs):
#             if x is not None:
#                 x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
#             pre_attn.append(x)
#             gates.append(gate if x is not None else None)
#
#         pre_attn = sharding.activation_sharding_constraint(pre_attn)
#         post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
#         post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
#         post_attn = sharding.activation_sharding_constraint(post_attn)
#         xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
#         xs = sharding.activation_sharding_constraint(xs)
#
#         out = []
#         gates = []
#         for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
#             if x is not None:
#                 x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
#                 x = lora.FeedForward(  # noqa: PLW2901
#                     features=config.width,
#                     hidden_dim=config.mlp_dim,
#                     lora_config=config.lora_configs.get("ffn"),
#                     name=_name("mlp", i),
#                 )(x)
#             out.append(x)
#             gates.append(gate if x is not None else None)
#
#         out = sharding.activation_sharding_constraint(out)
#         out = jax.tree.map(lambda x: drop(x, deterministic), out)
#         xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
#         xs = sharding.activation_sharding_constraint(xs)
#
#         return xs, kv_cache

# @at.typecheck
# class Block(nn.Module):
#     configs: tuple[Config, ...]
#     dropout: float = 0.0
#     dropout_bdims: tuple[int, ...] = ()
#     block_size: int = 2  # 新增：每隔几层形成一个block
#
#     @nn.compact
#     def __call__(self, carry, kv_cache, positions, attn_mask, adarms_cond,
#                  bar_state,  # 新增：(blocks, partial_block, layer_number)
#                  deterministic=True):
#         xs, bar_state = carry
#         blocks, partial_block, layer_number = bar_state
#
#         # ---- 原有逻辑：attention ----
#         attn = Attention(configs=self.configs, name="attn")
#         pre_attn = []
#         gates = []
#         for i, x in enumerate(xs):
#             if x is not None:
#                 x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
#             pre_attn.append(x)
#             gates.append(gate if x is not None else None)
#
#         post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
#
#         # ---- 新增：BAR before attention residual ----
#         if partial_block is not None:
#             h = self._block_attn_res(blocks, partial_block,
#                                      name="attn_res_proj", norm_name="attn_res_norm")
#         else:
#             h = xs[0]  # fallback
#
#         xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates)]
#
#         # 更新 partial_block（attention之后）
#         partial_block = xs[0] if partial_block is None else partial_block + post_attn[0]
#
#         # ---- 检查是否到达block边界 ----
#         # block_size // 2 因为每层有 ATTN + MLP 两个子层
#         def update_blocks(args):
#             blocks, partial_block = args
#             return blocks + [partial_block], None
#
#         def keep_blocks(args):
#             return args
#
#         at_boundary = (layer_number % (self.block_size // 2) == 0)
#         # 注意：jax里条件判断需要用jax.lax.cond
#         blocks, partial_block = jax.lax.cond(
#             at_boundary,
#             update_blocks,
#             keep_blocks,
#             (blocks, partial_block)
#         )
#
#         # ---- 原有逻辑：MLP ----
#         out = []
#         gates = []
#         for i, (x, config) in enumerate(zip(xs, self.configs)):
#             if x is not None:
#                 x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])
#                 x = lora.FeedForward(...)(x)
#             out.append(x)
#             gates.append(gate if x is not None else None)
#
#         # ---- 新增：BAR before MLP residual ----
#         # (同上，在MLP之前也插入一次block_attn_res)
#
#         xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates)]
#
#         # 更新 partial_block（MLP之后）
#         partial_block = partial_block + out[0] if partial_block is not None else out[0]
#
#         new_bar_state = (blocks, partial_block, layer_number + 1)
#         return (xs, new_bar_state), kv_cache

@at.typecheck
class Block(nn.Module):
    """Transformer block with optional Block Attention Residuals (BAR)."""

    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adopt_bar: bool = False  # 是否启用 BAR
    block_size: int = 3  # 每隔几个 Transformer 层形成一个 block

    @nn.compact
    def __call__(self, carry, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        # ── 解包 carry ────────────────────────────────────────────────────────
        xs, bar_state = carry
        blocks_buf, partial_block, block_count, layer_idx = bar_state

        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # ── BAR: 到达 block 边界时保存当前 partial_block ──────────────────────
        if self.adopt_bar:
            def save_block(args):
                buf, pb, cnt = args
                buf = jax.lax.dynamic_update_slice(buf, pb[None], (cnt, 0, 0, 0))
                return buf, jnp.zeros_like(pb), cnt + 1

            def keep_block(args):
                return args

            blocks_buf, partial_block, block_count = jax.lax.cond(
                layer_idx % self.block_size == 0,
                save_block,
                keep_block,
                (blocks_buf, partial_block, block_count),
            )

        # ── BAR: 在 Attention 之前增强隐状态 ──────────────────────────────────
        if self.adopt_bar and xs[0] is not None:
            h = self._bar_transform(
                blocks_buf, block_count, partial_block,
                "attn_bar_proj", "attn_bar_norm", xs[0].dtype,
            )
            xs_in = [h if i == 0 else x for i, x in enumerate(xs)]
        else:
            xs_in = xs

        # ── Attention（原有逻辑，输入换成 xs_in）─────────────────────────────
        attn = Attention(configs=self.configs, name="attn")
        pre_attn, gates = [], []
        for i, x in enumerate(xs_in):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)

        xs_before_attn = xs[0]  # 保存更新前的 xs[0]，用于计算 delta
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # ── BAR: 更新 partial_block（记录 Attention 实际贡献的 delta）─────────
        if self.adopt_bar and xs[0] is not None and xs_before_attn is not None:
            partial_block = partial_block + (xs[0] - xs_before_attn)

        # ── BAR: 在 MLP 之前增强隐状态 ────────────────────────────────────────
        if self.adopt_bar and xs[0] is not None:
            h = self._bar_transform(
                blocks_buf, block_count, partial_block,
                "mlp_bar_proj", "mlp_bar_norm", xs[0].dtype,
            )
            xs_in = [h if i == 0 else x for i, x in enumerate(xs)]
        else:
            xs_in = xs

        # ── MLP（原有逻辑，输入换成 xs_in）───────────────────────────────────
        out, gates = [], []
        for i, (x, config) in enumerate(zip(xs_in, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    lora_config=config.lora_configs.get("ffn"),
                    name=_name("mlp", i),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)

        xs_before_mlp = xs[0]  # 保存更新前的 xs[0]
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        # ── BAR: 更新 partial_block（记录 MLP 实际贡献的 delta）──────────────
        if self.adopt_bar and xs[0] is not None and xs_before_mlp is not None:
            partial_block = partial_block + (xs[0] - xs_before_mlp)

        new_bar_state = (blocks_buf, partial_block, block_count, layer_idx + 1)
        return (xs, new_bar_state), kv_cache

    def _bar_transform(self, blocks_buf, block_count, partial_block, proj_name, norm_name, dtype):
        """
        Block Attention Residual: 用已完成的 blocks + 当前 partial_block
        做 softmax attention，返回增强后的隐状态 h。

        blocks_buf:    [max_blocks, B, T, D]  — 预分配的历史 block 缓冲区
        block_count:   scalar int32           — 已写入的有效 block 数量
        partial_block: [B, T, D]              — 当前 block 内的累积残差
        """
        max_blocks = blocks_buf.shape[0]

        # 拼接：[max_blocks + 1, B, T, D]（最后一个位置是 partial_block，始终有效）
        all_v = jnp.concatenate([blocks_buf, partial_block[None]], axis=0)
        N1, B, T, D = all_v.shape

        # RMSNorm：对所有 block 做归一化（Key）
        all_flat = all_v.reshape(N1 * B, T, D)
        K_flat, _ = RMSNorm(name=norm_name)(all_flat, None)
        K = K_flat.reshape(N1, B, T, D)  # [N+1, B, T, D]

        # 可学习的伪查询（pseudo-query）：D → 1，得到每个位置的 logit
        logits = nn.Dense(1, use_bias=False, name=proj_name)(K).squeeze(-1)  # [N+1, B, T]

        # 屏蔽未填充的 block slot（只有 0..block_count-1 和 partial_block 有效）
        valid = jnp.concatenate([
            jnp.arange(max_blocks) < block_count,  # [max_blocks]
            jnp.array([True]),  # partial_block 始终有效
        ])  # [N+1]
        logits = jnp.where(valid[:, None, None], logits, -1e9)

        # 在 block 维度上做 softmax，得到各 block 的权重
        probs = jax.nn.softmax(logits, axis=0)  # [N+1, B, T]

        # 加权求和，得到增强后的隐状态
        h = jnp.einsum("nbt,nbtd->btd", probs, all_v)  # [B, T, D]
        return h.astype(dtype)


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


# @at.typecheck
# class Module(nn.Module):
#     """Transformer model, supporting a mixture of different weights for different tokens."""
#
#     configs: Sequence[Config]  # list of configs, one for each expert
#     embed_dtype: str
#
#     dropout: float = 0.0
#     dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
#     adarms: bool = False
#
#     def setup(self):
#         # all experts must have the same depth
#         assert all(config.depth == self.configs[0].depth for config in self.configs)
#
#         self.embedder = Embedder(
#             vocab_size=PALIGEMMA_VOCAB_SIZE,
#             embed_dim=self.configs[0].width,  # embedder for first expert only
#             name="embedder",
#         )
#         block_cls = nn.remat(
#             Block,
#             prevent_cse=False,
#             static_argnums=(5,),  # 0=self, 6=deterministic
#             policy=jax.checkpoint_policies.nothing_saveable,
#         )
#         self.layers = nn.scan(
#             block_cls,
#             variable_axes={"params": 0},
#             split_rngs={"params": True, "dropout": True},
#             in_axes=(
#                 0,
#                 nn.broadcast,
#                 nn.broadcast,
#                 nn.broadcast,
#                 nn.broadcast,
#             ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
#             length=self.configs[0].depth,
#         )(
#             configs=self.configs,
#             dropout=self.dropout,
#             dropout_bdims=self.dropout_bdims,
#         )
#         self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]
#
#     @at.typecheck
#     def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
#         return self.embedder.encode(tokens).astype(self.embed_dtype)
#
#     @at.typecheck
#     def __call__(
#         self,
#         # list of token arrays, one for each expert, or None if that expert should not be run
#         embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
#         positions: at.Int[at.Array, "b t"],
#         mask: at.Bool[at.Array, "b t s"],
#         adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
#         *,
#         kv_cache: KVCache | None = None,
#         deterministic: bool = True,
#     ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
#         initial_bar_state = ([], None, 0)
#
#
#         embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
#         mask = jnp.asarray(mask)[:, None, :, :]
#         if adarms_cond is None:
#             adarms_cond = [None] * len(self.configs)
#
#         (embedded, bar_state) , kv_cache = self.layers((embedded, initial_bar_state), kv_cache, positions, mask, adarms_cond, deterministic)
#
#         assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
#
#         return [
#             f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
#         ], kv_cache
#
#     def init(self, use_adarms: Sequence[bool]):
#         """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
#         self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
#         self(
#             [jnp.zeros((1, 1, c.width)) for c in self.configs],
#             jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
#             jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
#             adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
#         )

@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    adopt_bar: bool = False   # ← 新增
    block_size: int = 3       # ← 新增

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # deterministic（carry 替换 xs 后位置不变，仍是第5位）carry=0, kv_cache=1, positions=2, attn_mask=3, adarms_cond=4, deterministic=5
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,             # kv_cache：每层各自的 cache，沿层扫描
                nn.broadcast,  # positions
                nn.broadcast,  # attn_mask
                nn.broadcast,  # adarms_cond
                nn.broadcast,  # deterministic
            ),
            # carry = (xs, bar_state)，是第一个参数，不在 in_axes 里声明
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            adopt_bar=self.adopt_bar,    # ← 传给 Block
            block_size=self.block_size,  # ← 传给 Block
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # ── 初始化 BAR carry ──────────────────────────────────────────────────
        # 取 xs[0]（PaliGemma 流）的 shape 来确定 B、T、D
        # 当使用 kv_cache 推理时 xs[0] 可能为 None，此时 BAR 自动 no-op
        ref = embedded[0]
        if ref is not None:
            B, T = ref.shape[0], ref.shape[1]
        else:
            B, T = 1, 1  # dummy，BAR 会因 xs[0] is None 而跳过
        D = self.configs[0].width
        depth = self.configs[0].depth
        max_blocks = depth // self.block_size + 2  # +2 留余量

        bar_state = (
            jnp.zeros((max_blocks, B, T, D), dtype=jnp.bfloat16),  # blocks_buf
            jnp.zeros((B, T, D),             dtype=jnp.bfloat16),  # partial_block
            jnp.array(0, dtype=jnp.int32),                          # block_count
            jnp.array(0, dtype=jnp.int32),                          # layer_idx
        )
        # ─────────────────────────────────────────────────────────────────────

        (embedded, _), kv_cache = self.layers(
            (embedded, bar_state),  # carry：(xs, bar_state) 合并为一个 tuple
            kv_cache, positions, mask, adarms_cond, deterministic,
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )

def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
