import numpy as np
from lib.params.llama import Params

# max_lr = 6e-4
# min_lr = max_lr * 0.1
# warmup_steps = 715
# max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


def get_lr(params: Params, step):
    max_lr = params.max_lr
    min_lr = params.min_lr
    warmup_steps = params.warmup_steps
    max_steps = params.max_steps

    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)
