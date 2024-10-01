import os
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from lib.dataset.dataset import DataLoaderLite
from lib.models.llama import Llama
from lib.params.llama import Params
from lib.utils.tool import train_one_step, val_one_step, eval_hellaswag
from lib.utils.utils import set_seed, init_ddp, cal_grad_accum, get_os_type
from ori.train_gpt2 import last_step


# simple launch:
# python train/llama32/train_pre.py
# DDP launch for e.g. 2 GPUs:
# torchrun --standalone --nproc_per_node=2 train/llama32/train_pre.py
def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_dir", type=str, default=None)
    parse.add_argument("--ckpt_dir", type=str, default=None)
    parse.add_argument("--tokenizer_path", type=str, default=None)
    parse.add_argument("--out_dir", type=str, default=None)

    parse.add_argument("--max_steps", type=int, default=None)
    parse.add_argument("--max_lr", type=float, default=None)
    args = parse.parse_args()

    os_type = get_os_type()
    if os_type == 'macOS':
        local_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/project/data/fineweb_edu_mini_sample"
        data_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/project/data/fineweb_edu_10BT_llama3_tokenized"
        cache_dir = "~/.cache/huggingface/datasets"
        ckpt_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/models/Llama-3.2-1B-Instruct/original"
        tokenizer_path = "/Users/yibinxie/Documents/myspace/Work/LLM/models/Llama-3.2-1B-Instruct/original/tokenizer.model"
        out_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/output/llama32/fineweb_edu_10BT/test"
    else:
        local_dir = "/root/autodl-fs/data/fineweb_edu_10BT"
        data_dir = "/root/autodl-tmp/data/fineweb_edu_10BT_llama3_tokenized"
        cache_dir = "/root/autodl-tmp/.cache/huggingface/datasets"
        ckpt_dir = "/root/autodl-tmp/models/Llama-3.2-1B-Instruct/original"
        tokenizer_path = "/root/autodl-tmp/models/Llama-3.2-1B-Instruct/original/tokenizer.model"
        out_dir = "/root/autodl-tmp/output/llama32/fineweb_edu_10BT/test"

    # init
    params = Params(
        init_from="resume",
        data_dir=data_dir if args.data_dir is None else args.data_dir,
        ckpt_dir=ckpt_dir if args.ckpt_dir is None else args.ckpt_dir,
        tokenizer_path=tokenizer_path if args.tokenizer_path is None else args.tokenizer_path,
        out_dir=out_dir if args.out_dir is None else args.out_dir,
        temperature=1.0,
        top_p=0.9,
        max_seq_len=2048,
        micro_batch_size=5,
        max_batch_size=491520,
        max_gen_len=2048,
        max_steps=500 if args.max_steps is None else args.max_steps,  # 19531 about 1 epoch
        save_freq=100,
        max_lr=1e-5 if args.max_lr is None else args.max_lr,
        use_flash=True,
    )

    print(json.dumps(asdict(params), indent=4))

    init_ddp(params)
    cal_grad_accum(params)
    set_seed(params.seed)
    torch.set_float32_matmul_precision('high')

    print(f'Initializing model weights from {params.ckpt_dir}')
    llama = Llama.build(params)
    model = llama.model

    # Load model
    start_step = 0
    if params.init_from == 'resume':
        checkpoints = sorted(Path(out_dir).glob('*.pt'))
        if not checkpoints:
            print('No checkpoints found, initializing a new model from scratch')
        else:
            ckpt_path = checkpoints[0]
            print(f'Resuming training from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')  # load on CPU to save GPU memory
            state_dict = checkpoint['model']
            key_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(key_prefix):
                    state_dict[k[len(key_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            start_step = checkpoint['step']
            val_loss = checkpoint['val_loss']
            params.best_step = checkpoint['best_step']
            params.best_val_loss = checkpoint['best_val_loss']
            print(f'Resumed training from step {start_step}, val_loss: {val_loss}')
    else:
        print('Initializing a new model from scratch')

    model.to(params.device)
    if params.use_compile:
        print('Compiling the model')
        model = torch.compile(model)
    # print(params)
    if params.use_ddp:
        print(params.ddp_local_rank)
        model = DDP(model, device_ids=[params.ddp_local_rank], output_device=params.ddp_local_rank)
        print(f'Using DDP with local rank {params.ddp_local_rank}')
    # print(model)
    # print(model.module)
    raw_model = model.module if params.use_ddp else model  # always contains the "raw" unwrapped model

    optimizer = raw_model.configure_optimizers(
        learning_rate=params.max_lr,
        weight_decay=0.0,
        device_type=params.device_type,
        master_process=params.master_process
    )

    # optimizer = raw_model.configure_optimizers(
    #     weight_decay=0.1,
    #     learning_rate=params.max_lr,
    #     device_type=params.device_type,
    #     master_process=params.master_process
    # )

    os.makedirs(params.out_dir, exist_ok=True)
    log_file = os.path.join(params.out_dir, f"log_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    train_loader = DataLoaderLite(
        data_root=data_dir,
        B=params.micro_batch_size,
        T=params.max_seq_len,
        process_rank=params.ddp_rank,
        num_processes=params.ddp_world_size,
        split='train',
        master_process=params.master_process
    )

    val_loader = DataLoaderLite(
        data_root=data_dir,
        B=params.micro_batch_size,
        T=params.max_seq_len,
        process_rank=params.ddp_rank,
        num_processes=params.ddp_world_size,
        split='val',
        master_process=params.master_process
    )

    # training loop
    print('Start training')
    for step in range(start_step, params.max_steps):
        last_step = step == (params.max_steps - 1)
        # validation
        if step % params.val_freq == 0 or last_step:
            val_one_step(step, params, model, raw_model, val_loader, optimizer, log_file)
            eval_hellaswag(step, params, model, log_file)
        # training
        train_one_step(step, params, model, train_loader, optimizer, log_file)
    print('Training finished')

    if params.use_ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
