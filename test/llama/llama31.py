import os
import glob
import fire
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
import torch
from sympy.physics.vector.printing import params
from torch import nn
import torch.nn.functional as F
import numpy as np
from lib.params.llama import Params
from lib.dataset.dataset import DistributedShardedDataLoader
from lib.models.llama import Llama
from lib.utils.tool import train_one_step, val_one_step
from lib.utils.utils import set_seed, init_ddp, cal_grad_accum, get_os_type
from datetime import datetime


def main():
    os_type = get_os_type()
    if os_type == 'macOS':
        local_dir = ""
        data_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/project/data/tinystories/"
        cache_dir = "~/.cache/huggingface/datasets"
        ckpt_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/models/Llama-3.2-1B-Instruct/original"
        tokenizer_path = "/Users/yibinxie/Documents/myspace/Work/LLM/models/Llama-3.2-1B-Instruct/original/tokenizer.model"
        out_dir = "/Users/yibinxie/Documents/myspace/Work/LLM/output/llama32/tinystories/test"
    else:
        local_dir = ""
        data_dir = "/root/autodl-tmp/data/tinystories/"
        cache_dir = "/root/autodl-tmp/.cache/huggingface/datasets"
        ckpt_dir = "/root/autodl-tmp/models/Llama-3.2-1B-Instruct/original"
        tokenizer_path = "/root/autodl-tmp/models/Llama-3.2-1B-Instruct/original/tokenizer.model"
        out_dir = "/root/autodl-tmp/output/llama32/tinystories/test"

    # init
    params = Params(
        init_from="resume",
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        out_dir=out_dir,
        temperature=1.0,
        top_p=0.9,
        max_seq_len=2048,
        max_gen_len=1024,
        max_batch_size=5,
        use_flash=True
    )

    init_ddp(params)
    set_seed(2024)
    torch.set_float32_matmul_precision('high')

    # load the val data shard
    train_loader = DistributedShardedDataLoader(
        filename_pattern=os.path.join(data_dir, "*train.bin"),
        B=params.max_batch_size,
        T=params.max_seq_len,
        process_rank=0,
        num_processes=1,
    )

    val_loader = DistributedShardedDataLoader(
        filename_pattern=os.path.join(data_dir, "*val.bin"),
        B=params.max_batch_size,
        T=params.max_seq_len,
        process_rank=0,
        num_processes=1,
    )

    llama = Llama.build(params)
    print(params)

    total_batch_size = params.max_batch_size * params.max_seq_len
    print(f"total_batch_size: {total_batch_size}")

    # super simple training loop to start
    model = llama.model
    model.to(params.device)
    if params.use_compile:
        print('Compiling the model')
        model = torch.compile(model)

    optimizer = model.configure_optimizers(learning_rate=params.max_lr, weight_decay=0.0,
                                           device_type=params.device_type, master_process=params.master_process)

    os.makedirs(params.out_dir, exist_ok=True)
    log_file = os.path.join(params.out_dir, f"log_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    for step in range(params.max_steps):
        # optimizer.zero_grad()
        # x, y = train_loader.next_batch()
        # x, y = x.to(params.device), y.to(params.device)
        # loss = model.forward_loss(x, y)
        # loss.backward()
        # optimizer.step()
        # print(f"step {step}, loss: {loss.item()}")

        if step % params.val_freq == 0:
            val_one_step(step, params, model, model, val_loader, optimizer, log_file)

        train_one_step(step, params, model, train_loader, optimizer, log_file)

    # and now generate
    model.eval()
    prompts: List[str] = [
        "I'm a language model",
        # "Once upon a time",
        # "One day",
        # "Lily and George were best friends",
        # "On a dark and stormy night",
    ]

    sample_rng = torch.Generator(device=params.device)
    sample_rng.manual_seed(1337)
    t0 = time.time()
    results = llama.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=params.max_gen_len,
        temperature=params.temperature,
        top_p=params.top_p
    )
    t1 = time.time()
    print(f"Generated in {t1 - t0:.2f} seconds")
    for prompt, result in zip(prompts, results):
        print(prompt, end="")  # AK: change end="\n" to end=""
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
