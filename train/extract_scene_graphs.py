# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Script extracts scene graphs from all samples in Conceptual Captions 3M
"""

import os
import numpy as np
import pandas as pd
import sng_parser
from tqdm import tqdm
import torch

import config as cfg

def parse_single_annot(annot):

    graph = sng_parser.parse(annot)

    entities = {i: x['head'] for i, x in enumerate(graph['entities'])}
    relations = [{'subject': entities[x['subject']], 'object': entities[x['object']], 'relation': x['relation']}
        for x in graph['relations']
    ]

    return entities, relations

if __name__ == '__main__': 

    print(f'Parsing scene graphs for CC3M, saving scene graph annotations to {cfg.cc3m_tsg_path}')
    print('Loading CC3M data...')
    images_npy = np.load(os.path.join(cfg.cc3m_root, 'train_all.npy'), allow_pickle=True)
    images = pd.DataFrame(images_npy.tolist())

    annotated_samples = []

    for i in tqdm(range(len(images))):

        sample = images_npy[i]
        annot = sample['captions'][0]
        graph_entities, graph_relations = parse_single_annot(annot)
        sample['entities'] = graph_entities.values()
        sample['relations'] = graph_relations

        annotated_samples.append(sample)

    annotated_samples = np.array(annotated_samples)
    for i in range(len(annotated_samples)):
        annotated_samples[i]['entities'] = list(annotated_samples[i]['entities'])

    torch.save(annotated_samples, cfg.cc3m_tsg_path)