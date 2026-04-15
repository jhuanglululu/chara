import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


from .rope import RoPE
from ..configs import ModelConfig
from ..caches import DecoderCache


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE) -> None:
        super().__init__()

        self.d_model = config.d_model
        self.d_head = config.d_head
        self.d_latent = config.d_latent
        self.n_heads = config.n_heads
        self.d_rope = config.d_rope

        self.rope = rope

        # training parameters
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_dkv = nn.Linear(config.d_model, config.d_latent, bias=False)
        self.w_ukv = nn.Linear(config.d_latent, 2 * config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        full_d_rope = config.n_heads * config.d_rope
        self.w_q_rot = nn.Linear(config.d_model, full_d_rope, bias=False)
        self.w_k_rot = nn.Linear(config.d_model, config.d_rope, bias=False)

        self.need_weight_update = True
        # inference weights
        self.w_q_uk: Tensor
        self.w_uv_o: Tensor

        self.scale = 1.0 / math.sqrt(self.d_head + self.d_rope)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, cache: DecoderCache | None = None
    ) -> tuple[Tensor, DecoderCache | None]:
        if self.training:
            return self._train(x, mask), cache
        else:
            return self._inference(x, mask, cache)

    def _sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """attn_mask is a block mask: True marks blocked positions.

        pytorch has a bug with mps implementation of scaled dot product attention
        when v has a different head dim than q/k, so fall back to a manual impl.
        """
        if q.device.type == "mps":
            scores = (q @ k.transpose(-1, -2)) * self.scale
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            return torch.softmax(scores, dim=-1) @ v

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attn_mask is None else ~attn_mask,
            scale=self.scale,
        )

    def _absorb_weights(self):
        if not self.need_weight_update:
            return
        self.need_weight_update = False

        w_q = self.w_q.weight.reshape(self.n_heads, self.d_head, self.d_model)
        w_o = self.w_o.weight.reshape(self.d_model, self.n_heads, self.d_head)

        w = self.w_ukv.weight.reshape(self.n_heads, 2 * self.d_head, self.d_latent)
        w_uk, w_uv = w.chunk(2, dim=1)

        w_q_uk = torch.einsum("nhl,nhm->nlm", w_uk, w_q)
        w_uv_o = torch.einsum("mnh,nhl->mnl", w_o, w_uv)

        w_q_uk = w_q_uk.reshape(self.n_heads * self.d_latent, self.d_model)
        w_uv_o = w_uv_o.reshape(self.d_model, self.n_heads * self.d_latent)

        self.register_buffer("w_q_uk", w_q_uk, persistent=False)
        self.register_buffer("w_uv_o", w_uv_o, persistent=False)

    def _train(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape

        self.need_weight_update = True

        q = self.w_q(x)
        q = q.reshape(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        dkv = self.w_dkv(x)
        ukv = self.w_ukv(dkv)
        ukv = ukv.reshape(B, T, self.n_heads, 2 * self.d_head).permute(0, 2, 1, 3)
        k, v = ukv.chunk(2, dim=-1)

        q_rot = self.w_q_rot(x)
        q_rot = q_rot.reshape(B, T, self.n_heads, self.d_rope).permute(0, 2, 1, 3)

        k_rot = self.w_k_rot(x)
        k_rot = k_rot[:, None, :, :].expand(B, self.n_heads, T, -1)

        pos_q_rot = self.rope(q_rot)
        pos_k_rot = self.rope(k_rot)

        q = torch.concat([q, pos_q_rot], dim=-1)
        k = torch.concat([k, pos_k_rot], dim=-1)

        value = self._sdpa(q, k, v, attn_mask=mask)
        value = value.permute(0, 2, 1, 3).reshape(B, T, self.d_model)

        return self.w_o(value)

    def _inference(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cache: DecoderCache | None = None,
    ) -> tuple[Tensor, DecoderCache | None]:
        self._absorb_weights()

        B, T_X, _ = x.shape
        S = T_X if cache is None else (T_X + cache.atten_k.shape[1])

        # content
        q = F.linear(x, self.w_q_uk)
        q = q.reshape(B, T_X, self.n_heads, self.d_latent).permute(0, 2, 1, 3)

        dk = self.w_dkv(x)
        if cache is not None:
            dk = torch.concat([cache.atten_k, dk], dim=1)
            cache.atten_k = dk
        k = dk[:, None, :, :].expand(B, self.n_heads, S, -1)
        v = k

        # positional
        q_rot = self.w_q_rot(x)
        q_rot = q_rot.reshape(B, T_X, self.n_heads, self.d_rope).permute(0, 2, 1, 3)

        k_rot = self.w_k_rot(x)
        if cache is not None:
            k_rot = torch.concat([cache.atten_k_rot, k_rot], dim=1)
            cache.atten_k_rot = k_rot
        k_rot = k_rot[:, :, None, :].permute(0, 2, 1, 3).expand(B, self.n_heads, S, -1)

        q_rot = self.rope(q_rot, offset=S - T_X)
        k_rot = self.rope(k_rot)

        q = torch.concat([q, q_rot], dim=-1)
        k = torch.concat([k, k_rot], dim=-1)

        value = self._sdpa(q, k, v, attn_mask=mask)
        value = value.permute(0, 2, 1, 3).reshape(B, T_X, self.n_heads * self.d_latent)

        return F.linear(value, self.w_uv_o), cache
