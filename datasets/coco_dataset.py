from torch.utils.data import Dataset
import os

from PIL import Image
import pickle
import torch
import numpy as np

import config as cfg

class COCODataset(Dataset):

    def __init__(self, transform=None, root_dir=cfg.coco_root) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.transform = transform

    def load_sample(self, sample):

        fname = sample.filepath.split('/')[-1]
        fpath = os.path.join(self.root_dir, 'val2017', fname)
        img = Image.open(fpath)
        
        if self.transform is not None:
            img = self.transform(img)

        return img

class COCOValSubset(COCODataset):

    def __init__(self, val_split_path, tokenizer=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path, 'rb') as handle:
            val_samples = pickle.load(handle)

        # TODO: This is a hack, and should not be needed for the final version
        gallery_lens, counts = np.unique([len(x['gallery']) for x in val_samples], return_counts=True)
        modal_gallery_len = gallery_lens[np.argmax(counts)]
        refined_val_samples = [s for s in val_samples if len(s['gallery']) == modal_gallery_len]
        self.val_samples = refined_val_samples

        print(f'Evaluating COCO with gallery length {modal_gallery_len + 1} (including true target)')
        print(f'Discarding {len(val_samples) - len(refined_val_samples)} samples...')

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        if self.transform is not None:
            gallery = torch.stack(gallery)
            gallery_and_target = torch.cat([target.unsqueeze(0), gallery])
        else:
            gallery_and_target = [target] + gallery

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        # By construction, target_rank = 0
        return reference, caption, gallery_and_target, 0  

    def __len__(self):
        return len(self.val_samples)

if __name__ == '__main__':
    
    dataset = COCOValSubset(
        val_split_path=os.path.join(cfg.genecis_root, 'change_object.pkl')
        )
    y = dataset[0]