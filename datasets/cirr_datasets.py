# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/data_utils.py

import json

import PIL.Image
from torch.utils.data import Dataset
import torch
import os
import numpy as np

from config import cirr_root

"""
Code adapted from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/data_utils.py
"""

class CIRRDataset(Dataset):

    def __init__(self, split: str, preprocess: callable, image_base_path: str = cirr_root, tokenizer = None):
        """
        :param split: dataset split, should be in ['test', 'val']
        :param preprocess: function which preprocess the image
        """
        self.preprocess = preprocess
        self.split = split
        self.image_base_path = image_base_path
        self.tokenizer = tokenizer

        if split not in ['test1', 'val', 'train']:
            raise ValueError("split should be in ['test1', 'val', 'train']")

        # get a mapping from image name to relative path
        img_meta_path = os.path.join(image_base_path, 'cirr', 'image_splits', f'split.rc2.{split}.json')
        with open(img_meta_path) as f:
            self.name_to_relpath = json.load(f)

        # Load captions
        text_meta_path = os.path.join(image_base_path, 'cirr', 'captions', f'cap.rc2.{split}.json')
        with open(text_meta_path) as f:
            self.caption_data = json.load(f)

        self.name_to_global_index = {name: global_index for global_index, name in enumerate(self.name_to_relpath.keys())}
        print(f"CIRR {split} dataset initialized")

    def __len__(self):
        return len(self.name_to_relpath)

    def get_image_from_name(self, image_name):

        rel_path = self.name_to_relpath[image_name].strip('./')
        image_path = os.path.join(self.image_base_path, rel_path)
        im = PIL.Image.open(image_path)
        image = self.preprocess(im)
        return image

class CIRRValGlobalDataset(CIRRDataset):

    """
    Returns triplets of (ref, caption, target_idx)
    Where target_idx is the target image's index in the whole dataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        try:
            
            sample = self.caption_data[index]

            ref_name = sample['reference']
            ref_img = self.get_image_from_name(ref_name)
            caption = sample['caption']
            target_name = sample['target_hard']

            if self.tokenizer is not None:
                caption = self.tokenizer(caption)

            ref_global_rank = self.name_to_global_index[ref_name]
            target_global_rank = self.name_to_global_index[target_name]

            return ref_img, caption, ref_global_rank, target_global_rank

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.caption_data)

class CIRRImageDataset(CIRRDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_index_to_name = {v: k for k, v in self.name_to_global_index.items()}

    def __getitem__(self, index):

        ref_name = self.global_index_to_name[index]
        ref_img = self.get_image_from_name(ref_name)

        return ref_img, index

    def __len__(self):
        return len(self.global_index_to_name)
# ---------------------------------------------------------
