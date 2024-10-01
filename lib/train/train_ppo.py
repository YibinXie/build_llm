import os
import torch
from torch.utils.data import DataLoader
from trainer import PPOTrainer
from gpt import GPTConfig, GPTActor, GPTRewardModel, GPTCritic
from dataset import STDataset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        log_dir = '../log/ppo'
        data_path = './data/alpaca_gpt4_data_zh.json'
        model_path = None
        batch_size = 2
        num_workers = 0
    else:
        log_dir = "/data/oss_bucket_0/xx/12m/gpt2/log/ppo"
        data_path = "/data/oss_bucket_0/xx/11m/data/sft/alpaca_gpt4_data_zh.json"
        model_path = "/data/oss_bucket_0/xx/12m/gpt2/log/model_19072.pt"
        batch_size = 16
        num_workers = 4

    os.makedirs(log_dir, exist_ok=True)

    config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)
    sft_model = GPTActor(config).to(device)
    rm_model = GPTRewardModel(config).to(device)
    actor = GPTActor(config).to(device)
    critic = GPTCritic(config).to(device)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        state_dict = checkpoint['model']
        for model in [sft_model, rm_model, actor, critic]:
            model.load_state_dict(state_dict, strict=False)
        print(f'Finished loading from {model_path}')

    train_ds = STDataset(data_path, prompt_max_len=128, answer_max_len=1024 - 128, max_len=1024)
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    trainer = PPOTrainer(sft_model, rm_model, actor, critic, train_loader, train_loader, device, log_dir)
    trainer.train(max_epochs=1)