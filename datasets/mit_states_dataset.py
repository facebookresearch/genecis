# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ecom-research/ComposeAE/blob/master/datasets.py

"""
Code adapted from: https://github.com/ecom-research/ComposeAE/blob/master/datasets.py
"""

from PIL import Image
import torch
import torch.utils.data
import random

from config import mit_states_root

class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.test_queries = []

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError

class MITStates(BaseDataset):
    """MITStates dataset."""

    def __init__(self, path=mit_states_root, split='train', transform=None):
        super(MITStates, self).__init__()
        self.path = path
        self.transform = transform
        self.split = split

        self.imgs = []
        test_nouns = [
            u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
            u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
            u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
            u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
            u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
            u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
            u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
            u'wheel', u'window', u'wool'
        ]

        from os import listdir
        for f in listdir(path + '/images'):
            if ' ' not in f:
                continue
            adj, noun = f.split()
            if adj == 'adj':
                continue
            if split == 'train' and noun in test_nouns:
                continue
            if split == 'test' and noun not in test_nouns:
                continue

            for file_path in listdir(path + '/images/' + f):
                assert (file_path.endswith('jpg'))
                self.imgs += [{
                    'file_path': path + '/images/' + f + '/' + file_path,
                    'captions': [f],
                    'adj': adj,
                    'noun': noun
                }]

        self.caption_index_init_()
        if split == 'test':
            self.generate_test_queries_()

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            texts += img['captions']
        return texts

    def __getitem__(self, idx):
        try:
            self.saved_item
        except:
            self.saved_item = None
        if self.saved_item is None:
            while True:
                idx, target_idx1 = self.caption_index_sample_(idx)
                idx, target_idx2 = self.caption_index_sample_(idx)
                if self.imgs[target_idx1]['adj'] != self.imgs[target_idx2]['adj']:
                    break
            idx, target_idx = [idx, target_idx1]
            self.saved_item = [idx, target_idx2]
        else:
            idx, target_idx = self.saved_item
            self.saved_item = None

        mod_str = self.imgs[target_idx]['adj']

        return {
            'source_img_id': idx,
            'source_img_data': self.get_img(idx),
            'source_caption': self.imgs[idx]['captions'][0],
            'target_img_id': target_idx,
            'target_img_data': self.get_img(target_idx),
            'noun': self.imgs[idx]['noun'],
            'target_caption': self.imgs[target_idx]['captions'][0],
            'mod': {
                'str': mod_str
            }
        }

    def caption_index_init_(self):
        self.caption2imgids = {}
        self.noun2adjs = {}
        for i, img in enumerate(self.imgs):
            cap = img['captions'][0]
            adj = img['adj']
            noun = img['noun']
            if cap not in self.caption2imgids.keys():
                self.caption2imgids[cap] = []
            if noun not in self.noun2adjs.keys():
                self.noun2adjs[noun] = []
            self.caption2imgids[cap].append(i)
            if adj not in self.noun2adjs[noun]:
                self.noun2adjs[noun].append(adj)
        for noun, adjs in self.noun2adjs.items():
            assert len(adjs) >= 2

    def caption_index_sample_(self, idx):
        noun = self.imgs[idx]['noun']
        # adj = self.imgs[idx]['adj']
        target_adj = random.choice(self.noun2adjs[noun])
        target_caption = target_adj + ' ' + noun
        target_idx = random.choice(self.caption2imgids[target_caption])
        return idx, target_idx

    def generate_test_queries_(self):
        self.test_queries = []
        for idx, img in enumerate(self.imgs):
            adj = img['adj']
            noun = img['noun']
            for target_adj in self.noun2adjs[noun]:
                if target_adj != adj:
                    mod_str = target_adj
                    self.test_queries += [{
                        'source_img_id': idx,
                        'source_caption': adj + ' ' + noun,
                        'target_caption': target_adj + ' ' + noun,
                        'noun': self.imgs[idx]['noun'],
                        'mod': {
                            'str': mod_str
                        }
                    }]
        print(len(self.test_queries), 'test queries')

    def __len__(self):
        return len(self.imgs)

    def get_img(self, idx, raw_img=False):
        img_path = self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

class MITStatesImageOnly(MITStates):

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.get_img(idx), idx

class MITStatesGlobalTestSet(MITStates):

    def __init__(self, path=mit_states_root, split='test', transform=None, use_complete_text_query=False, tokenizer=None):
        
        assert split == 'test'
        super().__init__(path, split, transform)
        self.use_complete_text_query = use_complete_text_query
        self.tokenizer = tokenizer

        # For every test query, record which other images are relevant
        # Could be more efficient?
        self.captions_to_idx = {}
        max_size = 0
        for idx, test_img in enumerate(self.imgs):
            
            caption = test_img['captions'][0]
            if caption in self.captions_to_idx.keys():
                self.captions_to_idx[caption].append(idx)
            else:
                self.captions_to_idx[caption] = [idx]

        # Pad captions_to_idx with dummy values to ensure constant size return
        max_size = max([len(v) for k, v in self.captions_to_idx.items()])
        for k, v in self.captions_to_idx.items():
            self.captions_to_idx[k] += [len(self.imgs) + 1] * (max_size - len(v))       # Pad the list with an invalid image index

    def __len__(self):
        return len(self.test_queries)

    def __getitem__(self, idx):
        
        test_query = self.test_queries[idx]
        ref_global_rank = test_query['source_img_id']

        ref_img = self.get_img(ref_global_rank)

        if self.use_complete_text_query:
            caption = test_query['target_caption']
        else:
            caption = test_query['mod']['str']

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        relevant_target_imgs = self.captions_to_idx[test_query['target_caption']]

        return ref_img, caption, ref_global_rank, torch.Tensor(relevant_target_imgs)
        

if __name__ == '__main__':

    dataset = MITStatesGlobalTestSet(path=mit_states_root, split='test')
    debug = 0