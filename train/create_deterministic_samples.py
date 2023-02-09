"""
This script mines deterministic triplets. 
If shard_index = None is specified, all triplets will be mined in a single process

A shard index can also be mined to parse a subset of the total number of triplets you wish to mine
            (this is useful on SLURM based systems)
"""
import os

from datasets import cc_3m_dataset
from datasets.cc_3m_dataset import CCConditionalDistractor
from utils.gen_utils import none_flag
import argparse

import config as cfg

def get_args_parser():

    parser = argparse.ArgumentParser('CC3MDatasetCreate', add_help=False)

    parser.add_argument('--num_samples', default=cfg.num_deterministic_samples, type=float, help='How many deterministic samples to create')
    parser.add_argument('--concreteness_threshold', default=cfg.cc3m_concreteness_threshold, type=float, help='Discard triplets which dont have a minimum concreteness score')
    parser.add_argument('--shard_index', default=None, type=none_flag, help='Which shard to do (out of NUM_SHARDS in config)')

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    
    print('Mining triplets for training...')
    print(f'Using scene graph annotations from {cfg.cc3m_tsg_path}')

    # Hard code path for now
    if args.shard_index is None:

        print('Processing all samples in a single process...')
        args.save_path = cfg.cc3m_deterministic_root_path

    else:
        print(f'Processing in shards, processing shard: {args.shard_index}')
        print(f'Shard Index Type is {type(args.shard_index)}')
        # If processing in shards, create directory in which to save all shards
        save_dir = cfg.cc3m_deterministic_root_path[:-3]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        args.save_dir = save_dir

        args.save_path = os.path.join(args.save_dir, f'shard_{args.shard_index}.pt')

    dataset = CCConditionalDistractor(min_images_with_subject=cfg.cc3m_min_images_with_subject, transform=None, tokenizer=None, cc3m_annots_path=cfg.cc3m_tsg_path)
    cc_3m_dataset.create_deterministic_subset(dataset, args.num_samples, save_path=args.save_path, concreteness_threshold=args.concreteness_threshold, shard_index=args.shard_index)