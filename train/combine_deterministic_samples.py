import torch
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser('DetermCombine', add_help=False)

# Data
parser.add_argument('--root_path', default='/checkpoint/sgvaze/conditional_similarity/cc3m_deterministic_samples/CCConditionalDistractor_2.5E+04_4.8_nico_no_filter_concreteness', type=str, help='Root path where all shards are stored')
parser.add_argument('--n_shards', default=400, type=int, help='Num shards to combine')

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


