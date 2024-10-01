import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Params:
    # model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 129256
    multiple_of: int = 1  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    max_batch_size: int = 491520
    micro_batch_size: int = 1
    max_seq_len: int = 2048  # 2048
    max_gen_len: int = 2048  # 2048
    use_flash: bool = True  # use flash attention or not
    use_att_linear_bias: bool = False
    use_mlp_linear_bias: bool = False
    use_compile: bool = False

    # training
    grad_accum_steps: int = 1
    use_ddp: bool = True
    device: str = 'cpu'
    device_type: str = 'cpu'
    ddp_world_size: int = 1
    ddp_local_rank: int = 0
    ddp_rank: int = 0
    master_process: bool = True
    model_parallel_size: int = 1
    seed: int = 2024
    max_steps: int = 10000
    val_freq: int = 100
    save_freq: int = 100000
    best_step: int = 0
    best_val_loss: float = float('inf')

    # lr schedule
    max_lr: float = 1e-5
    min_lr: float = 0.1 * max_lr
    warmup_steps: int = 0.05 * max_steps

    # infer
    temperature: float = 1.0
    top_p: float = 0.9

    # logging
    init_from: str = 'resume'  # scratch or resume
    data_dir: str = ''
    ckpt_dir: str = ''
    tokenizer_path: str = ''
    out_dir: str = ''

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if not self.n_kv_heads:
            self.n_kv_heads = self.n_heads

        assert self.n_kv_heads <= self.n_heads, \
            f'n_kv_heads: {self.n_kv_heads} should be smaller than n_heads: {self.n_heads}'
        assert self.n_heads % self.n_kv_heads == 0, \
            f'n_heads: {self.n_heads} should be divisible by n_kv_heads: {self.n_kv_heads}'
        assert self.dim % self.n_heads == 0, \
            f'dim: {self.dim} should be divisible by n_heads: {self.n_heads}'

        # self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda'
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     self.device = 'cpu'  # mac mps
        #
        # self.device_type = 'cuda' if self.device.startswith('cuda') else 'cpu'
        # self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


if __name__ == '__main__':
    params = Params(
        out_dir='out',
    )
    print(json.dumps(params.__dict__, indent=4))
