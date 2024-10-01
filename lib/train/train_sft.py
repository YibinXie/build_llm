import os
import torch

from gpt import GPT, GPTConfig
from dataset import SFTDataset
from lib.layers.lora import replace_linear_with_lora

if __name__ == "__main__":
    finetune_type = 'lora'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2 if device == 'cpu' else 16
    # print(device, batch_size)
    if batch_size <= 10:
        log_dir = '../log/sft'
        data_path = '../data/alpaca_gpt4_data_zh.json'
        model_path = None
    else:
        log_dir = "/data/oss_bucket_0/xx/LLm/gpt2/log/sft"
        data_path = "/data/oss_bucket_0/xx/11m/data/sft/alpaca_gpt4_data_zh.json"
        model_path = "/data/oss_bucket_0/xx/12m/gpt2/log/model_19072.pt"

    os.makedirs(log_dir, exist_ok=True)

    config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])

    if finetune_type == 'lora':
        replace_linear_with_lora(model)

    model.to(device)
    print(model)

    train_ds = SFTDataset(data_path, prompt_max_len=128, answer_max_len=1024 - 128, max_len=1024)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=0)

    max_steps = len(train_ds) // batch_size
    trainer = SFTTrainerV2(model, train_loader, device, log_dir)
    trainer.train(max_steps)