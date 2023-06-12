# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
import inflect
from copy import deepcopy

from tqdm import tqdm

from config import cc3m_root, cc3m_deterministic_root_path, cc3m_tsg_path, noun_concreteness_score_path, NUM_SHARDS

CC3M_MIN_IMGS_WITH_SUBJECT = 5

def create_deterministic_subset(dataset, num_samples=10e6, save_path=None, concreteness_threshold=None, shard_index=None):

    dataset_type = type(dataset).__name__
    num_samples = int(num_samples)

    if save_path is None:
        save_path = cc3m_deterministic_root_path

    determ_dataset = DeterministicDatasetWithConcreteness(dataset, num_samples, concreteness_threshold=concreteness_threshold, shard_index=shard_index)
    loader = torch.utils.data.DataLoader(determ_dataset, shuffle=False, num_workers=4, batch_size=8, collate_fn=DeterministicDatasetWithConcreteness.collate_fn)
    print(f'Deterministically creating {len(determ_dataset)} samples and saving to {save_path}...')

    all_samples = []
    for batch_idx, batch in enumerate(tqdm(loader)):
        
        all_samples.extend(batch)
    
    torch.save(all_samples, save_path)

class CCConditionalBaseDataset(Dataset):

    def __init__(self, cc3m_deterministic_root_path: str = None,       # Can either be key into predefined paths or an entire path in itself
                        min_images_with_subject: int = CC3M_MIN_IMGS_WITH_SUBJECT, 
                        num_samples_per_epoch: int = 100, 
                        transform = None, 
                        tokenizer = None,
                        cc3m_path: str = cc3m_root,
                        cc3m_annots_path : str = cc3m_tsg_path) -> None:

        super().__init__()

        print(f'Loading scene graphs from {cc3m_annots_path}...')

        self.cc_path = cc3m_path
        self.cc3m_annots_path = cc3m_annots_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.deterministic_dataset = True if cc3m_deterministic_root_path is not None else False
        self.num_samples_per_epoch = num_samples_per_epoch

        annotated_samples = torch.load(cc3m_annots_path)

        # --------------------------
        # Create dataframe
        # --------------------------
        print('Constructing CC3M dataset...')
        all_samples = []
        for img in annotated_samples:
            for relation in img['relations']:
                sample = {
                    'image_id': img['image_id'],
                    'captions': img['captions'],
                    'subject': relation['subject'],
                    'relation': relation['relation'],
                    'object': relation['object'],
                }
                all_samples.append(sample)

        # All samples
        self.annotated_samples_df = pd.DataFrame(all_samples)

        # -------------
        # SUBJECTS TO IMAGE ID
        # var dict uq_subjects
        # Parse all unique subjects (aka entities, eg 'dog' 'donkey' 'evening) in the dataset
        # Keep a subject only if it:
        #       Appears in at least min_images_with_subject images
        #       Is paired with at least two objects
        # To this end, also store:
        #       Which images the subject appears in
        #       The index of the sample with the subject in the global dataframe
        #       Which indices contain images from each other object
        # -------------

        print('------Finding candidate subjects...')
        uq_subjects = {}
        for idx, sample in enumerate(self.annotated_samples_df.values):

            sub = sample[2]
            obj = sample[4]
            img_id = sample[0]

            if sub in uq_subjects.keys():
                uq_subjects[sub]['image_ids'].append(img_id)
                uq_subjects[sub]['df_idxs'].append(idx)

                if obj in uq_subjects[sub]['objects'].keys():
                    uq_subjects[sub]['objects'][obj].append(idx)
                else:
                    uq_subjects[sub]['objects'][obj] = [idx]

            else:
                uq_subjects[sub] = {}
                uq_subjects[sub]['image_ids'] = [img_id]
                uq_subjects[sub]['df_idxs'] = [idx]
                uq_subjects[sub]['objects'] = {
                    obj: [idx]
                }

        for k, v in uq_subjects.items():
            uq_subjects[k]['image_ids'] = set(v['image_ids'])

        self.uq_subjects = uq_subjects

        # Subjects which are permissible (have more than min_images_with_subject images)
        self.candidate_subjects = [k for k, v in uq_subjects.items() if len(v['image_ids']) >= min_images_with_subject and len(v['objects']) >= 2]
        
        # Print total images considered
        cands_ = set(self.candidate_subjects)
        print(f'------Considering subset of {len(set.union(*[v["image_ids"] for k,v in self.uq_subjects.items() if k in cands_]))}')

        p_candidate_subjects = np.array([len(v['image_ids']) for k, v in uq_subjects.items() if len(v['image_ids']) >= min_images_with_subject and len(v['objects']) >= 2]).astype('float')
        p_candidate_subjects /= p_candidate_subjects.sum()
        self.p_candidate_subjects = p_candidate_subjects   

        # --------------------------
        # Load deterministic samples if deterministic
        # --------------------------
        if self.deterministic_dataset:

            print(f'------Using deterministic samples, loading from {cc3m_deterministic_root_path}...')
            self.samples = torch.load(cc3m_deterministic_root_path)

            # Retrieve deterministic sample with index
            self.get_sample = lambda index: self.samples[index]

            # Set length
            print(f'------Running deterministic dataset with {len(self)} samples...')

        else:

            # Generate random sample on the fly
            self.get_sample = self.get_random_sample

            print(f'------Generating samples on the fly with {len(self)} samples per epoch...')

    def __len__(self):

        if self.deterministic_dataset:
            return len(self.samples)
        else:
            return self.num_samples_per_epoch

    def load_single_sample(self, sample):

        img_id = sample['image_id']
        path = os.path.join(self.cc_path, 'training', str(img_id))
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def split_to_train_val(self, val_preprocess=None, val_split: float = 0.1, val_start_idx: int = None):

        """
        Function to split the instantiated dataset object into training and validation splits
        """

        if val_start_idx is None:
            val_start_idx = 0
        num_val_samples = int(val_split * len(self))

        assert self.deterministic_dataset
        assert val_start_idx + num_val_samples <= len(self)

        val_dataset = deepcopy(self)    

        if val_preprocess is not None:
            val_dataset.transform = val_preprocess

        # Now reserve some samples for training and the rest for testing
        val_dataset.samples = val_dataset.samples[val_start_idx:val_start_idx + num_val_samples]
        self.samples = self.samples[:val_start_idx] + self.samples[val_start_idx + num_val_samples:]

        print(f'Split train dataset into train and val sets of length {len(self)} and {len(val_dataset)} (starting at index {val_start_idx})')

        return val_dataset

class CCConditionalDistractor(CCConditionalBaseDataset):

    """
    Instead of only returning a (image1, image2, condition) triplet, 
    this class also returns a 'distractor' image which acts as a confounder for a text only retrieval
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        uq_objects = {}
        for idx, sample in enumerate(self.annotated_samples_df.values):

            img_id = sample[0]
            obj = sample[4]

            # Objects
            if obj in uq_objects.keys():
                uq_objects[obj]['image_ids'].append(img_id)
                uq_objects[obj]['df_idxs'].append(idx)
            else:
                uq_objects[obj] = {}
                uq_objects[obj]['image_ids'] = [img_id]
                uq_objects[obj]['df_idxs'] = [idx]

        # Objects
        for k, v in uq_objects.items():
            uq_objects[k]['image_ids'] = set(v['image_ids'])

        self.uq_objects = uq_objects

    def get_random_sample(self, index):

        # Sample random batch - never use index
        # Choose a reference object
        ref_subject = np.random.choice(self.candidate_subjects, p=self.p_candidate_subjects)

        # Reference and target indices
        ref_img_idx = np.random.choice(self.uq_subjects[ref_subject]['df_idxs'])
        ref_img = self.annotated_samples_df.iloc[ref_img_idx]

        # Get target image, make sure it's not the same as the reference
        target_img = ref_img
        attempts = 0
        while target_img.object == ref_img.object or target_img.image_id == ref_img.image_id:
            target_object = np.random.choice(list(self.uq_subjects[ref_subject]['objects'].keys()))
            target_img_idx = np.random.choice(self.uq_subjects[ref_subject]['objects'][target_object])
            target_img = self.annotated_samples_df.iloc[target_img_idx]
            attempts += 1
            if attempts > 100:
                print('Difficult sample. Ref and target image are the same')
                break

        # Get text distractor (has the same object)
        target_object = target_img.object
        text_distractor_idx = np.random.choice(self.uq_objects[target_object]['df_idxs'])

        sample = {
            'ref_img_idx': ref_img_idx,
            'target_img_idx': target_img_idx,
            'text_distractor_img_idx': text_distractor_idx
        }

        return sample
    
    def __getitem__(self, index):

        sample = self.get_sample(index)

        ref_img_idx = sample['ref_img_idx']
        target_img_idx = sample['target_img_idx']
        text_distractor_img_idx = sample.get('text_distractor_img_idx', ref_img_idx)    # Not all deterministically sampled triplets have a text distractors. Hack for now

        ref_img = self.annotated_samples_df.iloc[ref_img_idx]
        target_img = self.annotated_samples_df.iloc[target_img_idx]
        text_distractor_img = self.annotated_samples_df.iloc[text_distractor_img_idx]
        
        # Get condition
        condition = ' '.join([target_img.relation, target_img.object])

        ref_img = self.load_single_sample(ref_img)
        target_img = self.load_single_sample(target_img)
        text_distractor_img = self.load_single_sample(text_distractor_img)

        if self.tokenizer is not None:
            condition = self.tokenizer(condition)

        return ref_img, target_img, text_distractor_img, condition

class DeterministicDatasetWithConcreteness(Dataset):

        def __init__(self, base_dataset: CCConditionalBaseDataset,
                           n_samples, 
                           concreteness_threshold=None, 
                           shard_index: int = None     # Specify if wanting to process a subset of the indices (e.g if processing in shards). Hard code n_shards = 40
                           ) -> None:
            super().__init__()
            self.base_dataset = base_dataset
            self.n_samples = n_samples
            self.concreteness_threshold = concreteness_threshold
            self.max_tries = 1000        # How many times to try and generate a sample with sufficient concreteness given a seed
            self.NUM_SHARDS = NUM_SHARDS

            if shard_index is None:
                self.sample_seeds = np.arange(n_samples)
                print(f'Processing without shards with {len(self)} samples...')
            else:
                print(shard_index)
                self.sample_seeds = np.array_split(np.arange(n_samples), self.NUM_SHARDS)[int(shard_index)]
                print(f'Processing shard: {shard_index} with {len(self)} samples...')

            if concreteness_threshold is not None:
                self.concreteness_scores = self.get_conreteness_scores()
            else:
                self.concreteness_scores = None

        def __getitem__(self, index):
            
            seed = self.sample_seeds[index]
            np.random.seed(seed)
            sample = self.base_dataset.get_random_sample(None)
            
            if self.concreteness_threshold is not None:
                
                num_tries = 0
                sample_concreteness = self.compute_sample_concreteness(sample)

                while sample_concreteness <= self.concreteness_threshold:

                    sample = self.base_dataset.get_random_sample(None) 
                    sample_concreteness = self.compute_sample_concreteness(sample)
                    num_tries += 1
                    if num_tries == self.max_tries:
                        print(f'Sample at seed {index} is a difficult sample')

            return sample

        def __len__(self):
            return len(self.sample_seeds)
        
        @staticmethod
        def collate_fn(batch):
            return [x for x in batch]

        def compute_image_concreteness(self, img):

            concs = []
            for entity in (img.subject, img.object):
                for word in entity.split(' '):
                    concs.append(self.conreteness_scores.get(word, -1))
                
            return np.mean(concs)

        def get_conreteness_scores(self):
        
            print('Loading database of concreteness scores...')

            concreteness_scores = pd.read_csv(noun_concreteness_score_path, sep='\\t')
            concreteness_scores_dict = {}
            for row in tqdm(concreteness_scores.values):
                
                if str(row[0]) == 'nan':
                    word = 'null'
                else: 
                    word = row[0].lower()
                
                conc = row[2]
                concreteness_scores_dict[word] = conc

            # Deal with plurals
            engine = inflect.engine()
            single_keys = list(concreteness_scores_dict.keys())
            for k in tqdm(single_keys):
                k_plural = engine.plural(k)
                concreteness_scores_dict[k_plural] = concreteness_scores_dict[k]

            return concreteness_scores_dict

        def compute_sample_concreteness(self, sample):

            """
            Assume that sample contains some indices into the global CC3M annotated dataset from which we can compute visual concreteness
            """
            sample_concreteness = []
            for _, img_idx in sample.items():
                
                img = self.base_dataset.annotated_samples_df.iloc[img_idx]
                for entity in ('subject', 'object'):
                    
                    concreteness_score = self.concreteness_scores.get(img[entity], -1)
                    sample_concreteness.append(concreteness_score)
            
            return np.mean(sample_concreteness)