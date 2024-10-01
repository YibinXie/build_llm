import torch
import os
import time
import numpy as np
from torch.utils.data import DataLoader
from lib.models.gpt import GPTConfig, GPT
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from lib.dataset.dataset import DataLoaderLite
import wandb

run = wandb.init(project="test")
config = run.config
config.learning_rate = 0.01

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(it):
    # 1) linear warmup for warmup_iters steps.
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def main():
    # init ddp
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # single process
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # mac
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if master_process:
        print(f'using device: {device}, device_type: {device_type}, dtype: {dtype}')
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2024)
    torch.set_float32_matmul_precision('high')
    is_compile = False
    init_from = 'scratch'  # 'scratch', 'resume', 'gpt2'
    out_dir = ""

    # Load model
    start_step = 0
    if init_from == 'scratch':
        print('initializing a new model from scratch')
        model = GPT(GPTConfig(vocab_size=50304))
    elif init_from == 'resume':
        print(f'resuming training from {out_dir}')
        ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = GPT(GPTConfig(vocab_size=50304))
        state_dict = checkpoint['model']
        key_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(key_prefix):
                state_dict[k[len(key_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_step = checkpoint['step']
        val_loss = checkpoint['val_loss']
        best_step = checkpoint['best_step']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f'initializing from OpenAI GPT-2 weights: {init_from}')
        dropout = 0
        model = GPT.from_pretrained(init_from)
        config = model.config
    model.to(device)
    if is_compile:
        print('compiling the model')
        raw_model = model
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if is_ddp else model
    # run.watch(model)

    # optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type,
                                               master_process=master_process)

    # data loader
    total_batch_size = 524288
    batch_size = 4
    seq_len = 1024
    train_loader = DataLoaderLite(B=batch_size, T=seq_len, process_rank=ddp_rank, num_processes=ddp_world_size,
                                  split='train', master_process=master_process)
    val_loader = DataLoaderLite(B=batch_size, T=seq_len, process_rank=ddp_rank, num_processes=ddp_world_size,
                                split="val", master_process=master_process)
    assert total_batch_size % (
            batch_size * seq_len * ddp_world_size) == 0, 'make sure total_batch_size can be divisible by B * L * world_size'
    grad_accum_steps = total_batch_size // (batch_size * seq_len * ddp_world_size)
    if master_process:
        print(
            f'total batch size: {total_batch_size}, world size: {ddp_world_size}, gradient accumulation steps: {grad_accum_steps}')

    # training loop
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        # evaluate validation loss
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                if is_ddp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f'validation loss: {val_loss_accum.item():.4f}')
            model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if is_ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if is_ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            run.log({"loss": loss_accum.item(), "lr": lr, "norm": norm, "dt": dt * 1000, "tok/sec": tokens_per_sec})
    if is_ddp:
        destroy_process_group()
    run.finish()


if __name__ == "__main__":
    main()
