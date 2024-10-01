import os
import torch
import tiktoken
from lib.models.gpt import GPT, GPTConfig

if __name__ == "__main__":
    finetune_type = 'lora'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2 if device == 'cpu' else 16
    # print(device, batch_size)
    if batch_size <= 10:
        log_dir = '../log/sft'
        data_path = '../data/alpaca_gpt4_data_zh.json'
        model_path = '../log/log124m_40b/model_19072.pt'
    else:
        log_dir = "/data/oss_bucket_0/xx/12m/gpt2/log/sft"
        data_path = "/data/oss_bucket_0/xx/11m/data/sft/alpaca_gpt4_data_zh.json"
        model_path = "/data/oss_bucket_0/xx/12m/gpt2/log/model_19072.pt"

    os.makedirs(log_dir, exist_ok=True)

    config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)
    model = GPT(config)

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        print(f'Finished loading from {model_path}')

    model.to(device)
    # print(model)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a Language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0)

    tokens = model.batch_generate(tokens, 1024)
    tokens = tokens[0, :1024].tolist()
    decoded = enc.decode(tokens)
    print(decoded)