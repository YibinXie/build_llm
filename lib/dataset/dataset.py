import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from tqdm import tqdm
import random
import json
import tiktoken
import numpy as np
import os
import glob


class SFTDataset(Dataset):
    def __init__(self, file_path, prompt_max_len=128, answer_max_len=128, max_len=256):
        super().__init__()
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokens = []
        self.prompt_max_len = prompt_max_len
        self.answer_max_len = answer_max_len
        self.max_len = max_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token
        self.pad = self.eot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        prompt = self.enc.encode(sample['instruction'] + sample['input'])
        prompt.append(self.eot)
        answer = self.enc.encode(sample['output'])
        answer.append(self.eot)
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len - 1]
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len - 1]
        inputs = prompt + answer
        context_len = len(prompt)
        mask_pos = context_len - 1
        pad_len = self.max_len - len(inputs)
        inputs = inputs + [self.pad] * pad_len
        assert len(inputs) == self.max_len, 'pad len error'
        loss_mask = [0] * context_len + [1] * (len(inputs[mask_pos + 1:])) + [0] * pad_len
        x = torch.tensor(inputs[:-1], dtype=torch.long)
        y = torch.tensor(inputs[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return x, y, loss_mask


class RewardDataset(Dataset):
    def __init__(self, file_path, block_size, split):
        dataset = load_dataset(file_path, split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token
        self.pad = self.eot
        self.pairs = []
        print(f'loaded reward dataset {split} split')
        for data in dataset:
            prompt = data['prompt']
            pos_text = prompt + data['chosen']
            pos_tokens = self.enc.encode(pos_text)
            pos_tokens.append(self.eot)
            neg_text = prompt + data['rejected']
            neg_tokens = self.enc.encode(neg_text)
            neg_tokens.append(self.eot)
            if len(pos_tokens) > block_size:
                pos_tokens = pos_tokens[:block_size]
            else:
                pad_len = block_size - len(pos_tokens)
                pos_tokens.extend([self.pad] * pad_len)
            if len(neg_tokens) > block_size:
                neg_tokens = neg_tokens[:block_size]
            else:
                pad_len = block_size - len(neg_tokens)
                neg_tokens.extend([self.pad] * pad_len)
            pair = torch.tensor([pos_tokens, neg_tokens], dtype=torch.long)
            self.pairs.append(pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


class RLHFDataset(Dataset):
    def __init__(self, file_path, block_size, split):
        super().__init__()
        dataset = load_dataset(file_path, split=split)
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token
        self.pad = self.eot
        self.pairs = []
        print(f'loaded RLHF dataset {split} split')
        for data in dataset:
            pos_text = data['chosen']
            pos_tokens = self.enc.encode(pos_text)
            pos_tokens.append(self.eot)
            neg_text = data['rejected']
            neg_tokens = self.enc.encode(neg_text)
            neg_tokens.append(self.eot)
            if len(pos_tokens) > block_size:
                pos_tokens = pos_tokens[:block_size]
            else:
                pad_len = block_size - len(pos_tokens)
                pos_tokens.extend([self.pad] * pad_len)
            if len(neg_tokens) > block_size:
                neg_tokens = neg_tokens[:block_size]
            else:
                pad_len = block_size - len(neg_tokens)
                neg_tokens.extend([self.pad] * pad_len)
            pair = torch.tensor([pos_tokens, neg_tokens], dtype=torch.long)
            self.pairs.append(pair)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

    @classmethod
    def save(cls, split, tar_path):
        dataset = load_dataset('Anthropic/hh-rlhf', split=split)
        res = []
        for data in tqdm(dataset):
            res.append(data)
        with open(tar_path, 'w') as fp:
            json.dump(res, fp)


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, data_root, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


class EYLSFTStaticDataset(Dataset):
    def __init__(self, block_size, split):
        super().__init__()

        if split == 'train':
            with open("../data/sft_train.json") as fp:
                data = json.load(fp)
        elif split == 'test':
            with open("../data/sft_test.json") as fp:
                data = json.load(fp)

        self.tokens = []
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token

        cnt = 0
        print(f'Loading {split} dataset...')
        for chosen in tqdm(data):
            cnt += 1
            response_text = chosen
            response = self.enc.encode(response_text)
            response.append(self.eot)

            self.tokens += response

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]  # next token prediction
        return x, y


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 7, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        assert header[1] == 7, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.int32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it iterates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


if __name__ == '__main__':
    dataset = EYLSFTStaticDataset(1024, 'train')
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
