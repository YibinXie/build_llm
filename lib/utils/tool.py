import os
import torch
import time
from lib.params.llama import *
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from lib.utils.lr import get_lr
from lib.dataset.hellaswag import render_example, iterate_examples
from lib.utils.utils import send_wx_message, get_most_likely_row


def train_one_step(
        step: int,
        params: Params,
        model: torch.nn.Module,
        train_loader,
        optimizer: torch.optim.Optimizer,
        log_file: str,
):
    t0 = time.time()
    grad_accum_steps = params.grad_accum_steps
    use_ddp = params.use_ddp
    device = params.device
    device_type = params.device_type
    ddp_world_size = params.ddp_world_size
    master_process = params.master_process

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if use_ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # loss = model.forward_loss(x, y)
            loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if use_ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)

    # determine and set the learning rate for this iteration
    lr = get_lr(params, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()  # wait for the GPU to finish work

    dt = time.time() - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        msg = f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")


def val_one_step(
        step: int,
        params: Params,
        model: torch.nn.Module,
        raw_model: torch.nn.Module,
        val_loader,
        optimizer: torch.optim.Optimizer,
        log_file: str,
):
    t0 = time.time()
    grad_accum_steps = params.grad_accum_steps
    use_ddp = params.use_ddp
    device = params.device
    device_type = params.device_type
    ddp_world_size = params.ddp_world_size
    master_process = params.master_process
    last_step = step == (params.max_steps - 1)

    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # bfloat16
                # loss = model.forward_loss(x, y)
                loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    if use_ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    if master_process:
        bese_val_loss = params.best_val_loss
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        if val_loss_accum < bese_val_loss:
            params.best_step = step
            params.best_val_loss = val_loss_accum
        # don't save at the first step
        if step > 0 and (step % params.save_freq == 0 or last_step):
            # optionally write model checkpoints
            checkpoint_path = os.path.join(params.out_dir, f"checkpoint_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'param': params,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'best_step': params.best_step,
                'best_val_loss': params.best_val_loss
            }
            torch.save(checkpoint, checkpoint_path)


def eval_hellaswag(
        step: int,
        params: Params,
        model: torch.nn.Module,
        log_file: str,
):
    use_ddp = params.use_ddp
    device = params.device
    device_type = params.device_type
    ddp_world_size = params.ddp_world_size
    ddp_rank = params.ddp_rank
    master_process = params.master_process

    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # logits = model.forward_inference(tokens, 0)
                logits = model(tokens, start_pos=0)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if use_ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()

    acc_norm = num_correct_norm / num_total
    if master_process:
        msg = f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}"
        print(msg)
        send_wx_message(name=msg, title="HellaSwag")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")
