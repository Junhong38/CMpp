"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

from typing import Optional

import torch
import torch.nn as nn
import flash_attn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class MyAdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embbedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        # self.emb = nn.Linear(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear(
            self.silu(self.timestep_embbedder(self.timestep_proj(timestep)))
        )  # (n_points, embedding_dim * 2)
        # (valid_P, embedding_dim), (valid_P, embedding_dim)
        scale, shift = emb.chunk(2, dim=1)
        # broadcast to (n_points, embedding_dim)
        scale = scale[batch]
        shift = shift[batch]

        return self.norm(x) * (1 + scale) + shift


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        #  1. self attention
        self.norm1 = MyAdaLayerNorm(dim, num_embeds_ada_norm)

        self.self_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.self_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # 2. global attention
        self.norm2 = MyAdaLayerNorm(dim, num_embeds_ada_norm)

        self.global_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.global_attn_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

        # 3. feed forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

    def pad_sequence(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ):
        seq_ranges = (
            torch.arange(max_seqlen, device=x.device)
            .unsqueeze(0)
            .expand(len(seqlens), -1)
        )
        valid_mask = seq_ranges < seqlens.unsqueeze(1)
        padded_x = torch.zeros(
            (seqlens.shape[0], max_seqlen, x.shape[-1]), device=x.device, dtype=x.dtype
        )
        padded_x[valid_mask] = x

        return padded_x, valid_mask

    def forward(
        self,
        hidden_states: torch.Tensor,  # (n_points, embed_dim)
        timestep: torch.Tensor,  # (valid_P,)
        batch: torch.Tensor,  # (valid_P,)
        self_attn_seqlens: torch.Tensor,
        self_attn_cu_seqlens: torch.Tensor,
        self_attn_max_seqlen: torch.Tensor,
        global_attn_seqlens: torch.Tensor,
        global_attn_cu_seqlens: torch.Tensor,
        global_attn_max_seqlen: torch.Tensor,
        # graph_mask: torch.Tensor,  # (B, global_attn_max_seqlen, global_attn_max_seqlen)
        coarse_seg_pred: Optional[torch.Tensor] = None,  # (n_points,)
    ):
        n_points, embed_dim = hidden_states.shape
        # we use ada_layer_norm
        # 1. self attention
        norm_hidden_states = self.norm1(hidden_states, timestep, batch)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.self_attn_to_qkv(norm_hidden_states).reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=self_attn_cu_seqlens,
            max_seqlen=self_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)

        attn_output = self.self_attn_to_out(attn_output)
        hidden_states = hidden_states + attn_output

        if coarse_seg_pred is not None:
            hidden_states = hidden_states * (1 + coarse_seg_pred.unsqueeze(1))

        # 2. global attention
        norm_hidden_states = self.norm2(hidden_states, timestep, batch)

        # attn_bias = graph_mask  # (B, global_attn_max_seqlen, global_attn_max_seqlen)
        # # Ensure attention_bias is a slice multiple of 8
        # nearest_8_multiple = ((global_attn_max_seqlen + 7) // 8) * 8
        # attn_bias_padded = torch.zeros(
        #     (attn_bias.shape[0], nearest_8_multiple, nearest_8_multiple),
        #     device=attn_bias.device,
        # ).half()
        # attn_bias_padded[:, :global_attn_max_seqlen, :global_attn_max_seqlen] = (
        #     attn_bias
        # )
        # attn_bias_padded = attn_bias_padded.unsqueeze(1).expand(
        #     -1, self.num_attention_heads, -1, -1
        # )

        # global_qkv = (
        #     self.global_attn_to_qkv(norm_hidden_states)
        #     .reshape(n_points, 3, self.num_attention_heads * self.attention_head_dim)
        #     .unbind(1)
        # )  # tuple of 3 tensors: (n_points, num_heads, head_dim)
        # q_padded, global_valid_mask = self.pad_sequence(
        #     global_qkv[0], global_attn_seqlens, global_attn_max_seqlen
        # )
        # k_padded, _ = self.pad_sequence(
        #     global_qkv[1], global_attn_seqlens, global_attn_max_seqlen
        # )
        # v_padded, _ = self.pad_sequence(
        #     global_qkv[2], global_attn_seqlens, global_attn_max_seqlen
        # )

        # global_out = xformers.ops.memory_efficient_attention(
        # query=q_padded.view(
        #     -1,
        #     global_attn_max_seqlen,
        #     self.num_attention_heads,
        #     self.attention_head_dim,
        # ),
        # key=k_padded.view(
        #     -1,
        #     global_attn_max_seqlen,
        #     self.num_attention_heads,
        #     self.attention_head_dim,
        # ),
        # value=v_padded.view(
        #     -1,
        #     global_attn_max_seqlen,
        #     self.num_attention_heads,
        #     self.attention_head_dim,
        # ),
        #     attn_bias=attn_bias_padded[
        #         :, :, :global_attn_max_seqlen, :global_attn_max_seqlen
        #     ],
        # )  # (B, max_seqlen, num_heads, head_dim)

        # global_out = torch.nn.functional.scaled_dot_product_attention(
        #     query=q_padded.view(
        #         -1,
        #         global_attn_max_seqlen,
        #         self.num_attention_heads,
        #         self.attention_head_dim,
        #     ).permute(0, 2, 1, 3),
        #     key=k_padded.view(
        #         -1,
        #         global_attn_max_seqlen,
        #         self.num_attention_heads,
        #         self.attention_head_dim,
        #     ).permute(0, 2, 1, 3),
        #     value=v_padded.view(
        #         -1,
        #         global_attn_max_seqlen,
        #         self.num_attention_heads,
        #         self.attention_head_dim,
        #     ).permute(0, 2, 1, 3),
        #     attn_mask=attn_bias_padded[
        #         :, :, :global_attn_max_seqlen, :global_attn_max_seqlen
        #     ],
        # ).permute(0, 2, 1, 3)
        # global_out = global_out.view(
        #     -1, global_attn_max_seqlen, embed_dim
        # )  # (B, max_seqlen, embed_dim)
        # # unpad global_out
        # global_out = global_out[global_valid_mask]
        # global_out = self.global_attn_to_out(global_out)
        # hidden_states = hidden_states + global_out

        global_out_flash = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=self.global_attn_to_qkv(norm_hidden_states).reshape(
                n_points, 3, self.num_attention_heads, self.attention_head_dim
            ),
            cu_seqlens=global_attn_cu_seqlens,
            max_seqlen=global_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim)
        global_out_flash = self.global_attn_to_out(global_out_flash)
        hidden_states = hidden_states + global_out_flash

        # 3. feed forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states  # (n_points, embed_dim)
