# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
This script collects shards of mined triplets and combines them into a single file

Assumes you have run create_deterministic_samples.py in shard mode first to create the triplets.
"""
import torch
import argparse
import os
from tqdm import tqdm

import config as cfg
root_path = cfg.cc3m_deterministic_root_path[:-3]       # Get rid of the .pt at the end of the file name, root_path should be a directory
parser = argparse.ArgumentParser('DetermCombine', add_help=False)

# Data
parser.add_argument('--root_path', default=root_path, type=str, help='Root path where all shards are stored')
args = parser.parse_args()

# List dir
paths = [os.path.join(args.root_path, x) for x in os.listdir(args.root_path)]
final_save_path = args.root_path + '.pt'
print(f'Combining {len(paths)} shards...')
print(f'Saving final examples to {final_save_path}...')

# Load paths
all_examples = []
print('Loading shards...')
for i in tqdm(paths):
    examples_i = torch.load(i)
    all_examples.append(examples_i)

# Make flat list
all_examples = [item for sublist in all_examples for item in sublist]
print(f'Saving total of {len(all_examples)} samples...')
torch.save(all_examples, final_save_path)


