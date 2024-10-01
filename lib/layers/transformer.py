import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from lib.params.llama import Params
from lib.layers.pos_enc import apply_rotary_emb
from lib.layers.norm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class KVCache(nn.Module):
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv

class Attention(nn.Module):
    def __init__(self, args: Params):
        super().__init__()
        self.flash = args.use_flash # use flash attention?
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1 # AK: model parallel size is 1 for 1 GPU
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False )
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        # calculate query, key, value and split out heads
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # rotate query, keys (RoPE)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            xk, xv = self.cache.update(start_pos, xk, xv)
        # repeat k/v heads if n_kv_heads < n_heads (GQA)
        xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # make heads be a batch dim
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
        # attention
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        # concatenate all the heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # output projection
        proj = self.wo(output)
        return proj

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        # hidden dim gymnastics that Meta simplified only later
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, args: Params):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     '''torch.repeat_interleave(x, dim=2, repeats=n_rep)'''
#     bs, seq_len, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     output = x[:, :, :, None, :].expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
#     output = output.reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
#     return output
#
#
# class KVCache(nn.Module):
#     def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
#         super().__init__()
#         cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
#         # register_buffer keeps no-grad constants in state_dict, which can be loaded later
#         self.register_buffer('cache_k', torch.zeros(cache_shape, dtype=dtype, device=device))
#         self.register_buffer('cache_v', torch.zeros(cache_shape, dtype=dtype, device=device))
#
#     def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor):
#         seq_len = xk.size(1)
#         self.cache_k[:, start_pos: start_pos + seq_len] = xk
#         self.cache_v[:, start_pos: start_pos + seq_len] = xv
#         xk = self.cache_k[:, :start_pos + seq_len]
#         xv = self.cache_v[:, :start_pos + seq_len]
#         return xk, xv
#
#
# class Attention(nn.Module):
#     def __init__(self, args: Params):
#         super().__init__()
#         self.use_flash = args.use_flash
#         self.use_bias = args.use_att_linear_bias
#         self.n_kv_heads = args.n_kv_heads if args.n_kv_heads else args.n_heads
#         # model parallel
#         model_parallel_size = 1
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads
#         self.n_heads = args.n_heads
#         self.wq = nn.Linear(args.dim, args.dim, bias=self.use_bias)
#         self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=self.use_bias)
#         self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=self.use_bias)
#         self.wo = nn.Linear(args.dim, args.dim, bias=self.use_bias)
#         self.cache = None
#
#     def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, use_cache: bool,
#                 mask: Optional[torch.Tensor]):
#         bs, seq_len, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
#         xq = xq.view(bs, seq_len, self.n_local_heads, self.head_dim)
#         xk = xk.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bs, seq_len, self.n_local_kv_heads, self.head_dim)
#         xq = apply_rotary_emb(xq, freqs_cis)
#         xk = apply_rotary_emb(xk, freqs_cis)
#         # use kv cache when inference
#         if self.cache:
#             xk, xv = self.cache.update(start_pos, xk, xv)
#         xk = repeat_kv(xk, self.n_rep)  # (bs, cache_len + seq_len, n_local_heads, head_dim)
#         xv = repeat_kv(xv, self.n_rep)  # (bs, cache_len + seq_len, n_local_heads, head_dim)
#         # (bs, seq_len, n_local_heads, head_dim) -> (bs, n_local_heads, seq_len, head_dim)
#         xq = xq.transpose(1, 2)
#         xk = xk.transpose(1, 2)
#         xv = xv.transpose(1, 2)
#         if self.use_flash:
#             output = F.scaled_dot_product_attention(xq, xk, xv, mask)
#         else:
#             scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
#             if mask is not None:
#                 scores += mask
#             scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # (bs, n_local_heads, seq_len, seq_len)
#             output = torch.matmul(scores, xv)  # (bs, n_local_heads, seq_len, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
#         proj = self.wo(output)
#         return proj
#
#
# class FeedForward(nn.Module):
#     def __init__(self, args: Params):
#         super().__init__()
#         self.dim = args.dim
#         self.hidden_dim = 4 * args.dim  # originally in GPT2, up to 4*dim then down to dim
#         self.multiple_of = args.multiple_of
#         self.ffn_dim_multiplier = args.ffn_dim_multiplier
#         self.use_bias = args.use_mlp_linear_bias
#         # GPT2: 2*d*hd -> LLaMA3: 3*d*hd', so hd'=2*hd/3 can maintain the same amount ffn parameters
#         self.hidden_dim = int(2 * self.hidden_dim / 3)
#         if self.ffn_dim_multiplier:
#             self.hidden_dim = int(self.ffn_dim_multiplier * self.hidden_dim)
#         self.w1 = nn.Linear(self.dim, self.hidden_dim, self.use_bias)
#         self.w2 = nn.Linear(self.hidden_dim, self.dim, self.use_bias)
#         self.w3 = nn.Linear(self.dim, self.hidden_dim, self.use_bias)
#
#     def forward(self, x: torch.Tensor):
#         # silu GLU
#         output = self.w2(F.silu(self.w1(x)) * self.w3(x))
#         return output
#
#
# class Block(nn.Module):
#     def __init__(self, args: Params):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Attention(args)
#         self.ffn = FeedForward(args)
#         self.att_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
#
#     def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
#         # pre-norm can converge faster than post-norm, and don't forget the residual connection
#         h = x + self.attention(self.att_norm(x), start_pos, freqs_cis, False, mask)
#         output = h + self.ffn(self.ffn_norm(h))
#         return output
