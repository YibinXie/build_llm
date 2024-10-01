import click
import torch
from torch import nn
from trainer import SFTTrainer
from gpt import GPT, GPTConfig
from dataset import EYLSFTStaticDataset
from config import get_configs
from functools import partial


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # self.lora_a = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        # self.lora_b = nn.Parameter(torch.zeros(rank, output_dim))

        self.lora_a = nn.Linear(input_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, output_dim, bias=False)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.lora_a @ self.lora_b)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank=8, alpha=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            print(name, module)
            # lora_layer = LoRALayer(module.in_features, module.out_features, rank, alpha)
            lora_layer = LinearWithLoRA(module, rank, alpha)
            setattr(model, name, lora_layer)
        else:
            replace_linear_with_lora(module, rank)


if __name__ == '__main__':
    config = GPTConfig()
    model = GPT(config)
    for param in model.parameters():
        param.requires_grad = False
    # print(model)

    assign_lora = partial(LinearWithLoRA, rank=8, alpha=16)

    # model.lm_head = assign_lora(model.lm_head)

    # for layer in model.modules():
    #     if isinstance(layer, nn.Linear):
    #         print(layer)
    #
    #         layer.children()
    #         layer = assign_lora(layer)

    # replace_linear_with_lora(model)
    #
    # print(model)

    # print(list(model.named_children()))
    print(model.transformer['h'][0].attn.c_attn)
