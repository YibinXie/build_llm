import os
import json
from lib.dataset.dataset import load_dataset

def save_data(data_path, split, tar_path):
    dataset = load_dataset(data_path, split=split)
    res = []
    for data in dataset:
        res.append(data)
    with open(tar_path, 'w') as fp:
        json.dump(res, fp)

# RLHF dataset hh-rlhf
# RLHFDataset.save('train', os.path.join(tar_dir, 'train.json'))
# RLHFDataset.save('test', os.path.join(tar_dir, 'test.json'))

# RLHF dataset awesome-chatgpt-prompts
data_path = 'fka/awesome-chatgpt-prompts'
tar_dir = "/data/oss_bucket_0/xx/LLm/data/awesome-chatgpt-prompts"
os.makedirs(tar_dir, exist_ok=True)
save_data(data_path, 'train', os.path.join(tar_dir, 'train.json'))
# save_data(data_path, 'test', os.path.join(tar_dir, 'test.json'))