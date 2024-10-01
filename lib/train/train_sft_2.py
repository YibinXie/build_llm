import click
import torch
from trainer import SFTTrainer
from gpt import GPT, GPTConfig
from dataset import EYLSFTStaticDataset
from lib.params.config import get_configs


def train(pretrain, batch_size, exp_name):
    device = 'cpu'
    cfg = get_configs("gpt2-medium/dropout")
    cfg.max_steps = 200000 // batch_size
    cfg.batch_size = batch_size
    cfg.pretrain = pretrain
    assert pretrain == "huggingface"
    cfg.exp_name = exp_name

    # model = GPT.from_pretrained(cfg)
    config = GPTConfig()
    model = GPT(config)
    train_ds = EYLSFTStaticDataset(block_size=1024,
                                   split='train')
    test_ds = EYLSFTStaticDataset(block_size=1024,
                                  split='test')
    trainer = SFTTrainer(cfg, device, model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
@click.option('--pretrain', '-p', default="huggingface")
@click.option('--batch-size', '-b', default=2)
@click.option('--exp-name', '-n', default="default")
def main(strategy, pretrain, batch_size, exp_name):
    torch.manual_seed(1234)
    train(pretrain, batch_size, exp_name)


if __name__ == "__main__":
    main()
