import os

from datasets import cc_3m_dataset
from datasets.cc_3m_dataset import CCConditionalDistractor
from utils.gen_utils import none_flag
import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('CC3MDatasetCreate', add_help=False)

    # Data
    parser.add_argument('--num_samples', default=1.6e6, type=float, help='How many deterministic samples to create')

    # Facilitate sharding
    parser.add_argument('--shard_index', default=None, type=none_flag, help='Which shard to do (out of 40)')

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    # Hard code path for now
    if args.shard_index is None:

        args.save_path = cfg.cc3m_deterministic_root_path

    else:
        print(f'Processing in shards, processing shard: {args.shard_index}')
        print(f'Shard Index Type is {type(args.shard_index)}')
        # If processing in shards, create directory in which to save all shards
        save_dir = cfg.cc3m_deterministic_root_path

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        args.save_dir = save_dir

        args.save_path = os.path.join(args.save_dir, f'shard_{args.shard_index}.pt')


    dataset = CCConditionalDistractor(min_images_with_subject=cfg.cc3m_min_images_with_subject, transform=None, tokenizer=None, cc3m_annots_path=cfg.cc3m_tsg_path)
    cc_3m_dataset.create_deterministic_subset(dataset, args.num_samples, save_path=args.save_path, concreteness_threshold=args.concreteness_threshold, shard_index=args.shard_index)