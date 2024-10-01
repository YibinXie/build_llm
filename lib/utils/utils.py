import torch
import platform
import requests
from torch.nn import functional as F
from torch.distributed import init_process_group
import os
from lib.params.llama import *


def get_os_type():
    os_type = platform.system()
    if os_type == 'Darwin':
        return 'macOS'
    elif os_type == 'Linux':
        return 'Linux'
    else:
        raise ValueError(f"Unsupported OS type: {os_type}")


def send_wx_message(name, title="AutoDL"):
    headers = {
        "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjg4NjcsInV1aWQiOiJmMjhlMjVlNy1hMTc1LTQ4OTUtOTZjZS00NjQ5OWNjMGNjMjQiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.iCvp3MoqXuxHXl4CAtGVsKa2OLsdeqt18heKk-8Me6FvpzDh9NfT4QHvBTXXywAfoY7QJeFynYFRxes2I0Gi3g"}
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                         json={
                             "title": title,
                             "name": name
                         }, headers=headers)
    # print(resp.json())
    # print(resp.content.decode())


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def init_ddp(params: Params):
    params.use_ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if params.use_ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        params.ddp_rank = int(os.environ['RANK'])
        params.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        params.ddp_world_size = int(os.environ['WORLD_SIZE'])
        params.device = f'cuda:{params.ddp_local_rank}'  # each process gets a different GPU
        torch.cuda.set_device(params.device)
        params.master_process = params.ddp_rank == 0  # this process will do logging, checkpointing etc
        params.device_type = "cuda" if params.device.startswith("cuda") else "cpu"
        print(f'DDP run, '
              f'rank: {params.ddp_rank}, local_rank: {params.ddp_local_rank}, world_size: {params.ddp_world_size}')
        print(f'Using device: {params.device}, device_type: {params.device_type}')
    else:
        # vanilla, non-DDP run
        params.ddp_rank = 0
        params.ddp_local_rank = 0
        params.ddp_world_size = 1
        params.master_process = True
        # attempt to autodetect device
        params.device = "cpu"
        if torch.cuda.is_available():
            params.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            params.device = "cpu"  # mac mps
        params.device_type = "cuda" if params.device.startswith("cuda") else "cpu"
        print('non-DDP run')
        print(f'Using device: {params.device}, device_type: {params.device_type}')


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def cal_grad_accum(params: Params):
    assert params.max_batch_size % (params.micro_batch_size * params.max_seq_len * params.ddp_world_size) == 0, \
        f'make sure total_batch_size{params.max_batch_size} can be divisible by B{params.micro_batch_size} * L{params.max_seq_len} * world_size{params.ddp_world_size}'
    params.grad_accum_steps = params.max_batch_size // (params.micro_batch_size * params.max_seq_len * params.ddp_world_size)
    if params.master_process:
        print(
            f'total batch size: {params.max_batch_size}, world size: {params.ddp_world_size}, gradient accumulation steps: {params.grad_accum_steps}')


if __name__ == '__main__':
    params = Params()
    print(params)
    init_ddp(params)
    print(params)
