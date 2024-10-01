import torch
import os
from torch.utils.data import DataLoader
from trainer import RewardModelTrainer
from gpt import GPTConfig, GPTRewardModel
from dataset import RewardDataset

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        log_dir = '../log/rm'
        data_path = './data/rm_static/data/'
        model_path = None
        batch_size = 2
        num_workers = 0
        config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=2, n_head=12, n_embd=24)
    else:
        log_dir = "/data/oss_bucket_0/xx/12m/gpt2/log/rm"
        data_path = "Dahoas/rm-static"
        model_path = "/data/oss_bucket_0/xx/LLm/gpt2/log/model_19072.pt"
        batch_size = 16
        num_workers = 8
        config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)

    os.makedirs(log_dir, exist_ok=True)
    model = GPTRewardModel(config)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        print(f'Finished loading from {model_path}')

    model.to(device)
    print(model)

    train_ds = RewardDataset(data_path, block_size=config.block_size, split='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    test_ds = RewardDataset(data_path, block_size=config.block_size, split='test')
    test_loader = DataLoader(test_ds, batch_size=batch_size * 4, pin_memory=True, num_workers=num_workers)

    max_epochs = 10
    trainer = RewardModelTrainer(model, train_loader, test_loader, device, log_dir)
    trainer.train(max_epochs)