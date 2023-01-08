import re
import pandas as pd
import os
import numpy as np

from matplotlib import pyplot as plt
pd.options.display.width = 0

rx_dict = {
    'model_dir': re.compile(r'model_dir=\'(.*?)\''),
    'lr': re.compile(" lr=(\d*\.\d+|\d+)")
}

def get_file(path):

    file = []
    with open(path, 'r') as myfile:
        for myline in myfile:  # For each line, read to a string,
            file.append(myline)

    return file


def parse_out_file(path, rx_dict, verbose=True):

    file = get_file(path=path)

    data = []
    for i, line in enumerate(file):

        if line.find('batch_size_per_gpu') != -1:
            sample = {}
            for line2 in file[i:]:
                
                # if 'dilate_crop' in line2:
                #     sample['dilate'] = line2.split(':')[-1].strip()
                
                # if 'pad_crop' in line2:
                #     sample['pad'] = line2.split(':')[-1].strip()

                if 'dataset:' in line2:
                    dataset = line2.split(':')[-1].strip()
                    # if dataset in ('CIRR', 'mit_states'):
                    #     break
                    sample['dataset'] = dataset

                if 'model:' in line2:
                    sample['model'] = line2.split(':')[-1].strip()
                
                if 'combiner_mode:' in line2:
                    sample['combiner_mode'] = line2.split(':')[-1].strip()

                if 'combiner_pretrain_path:' in line2:
                    sample['combiner_pretrain_path'] = line2.split(':')[-1].strip().split('/')[-1]

                if 'Recall @ 1 =' in line2:
                    sample[f'Recall @ 1'] = line2.split('=')[-1].strip()     
                
                if 'Recall @ 5 =' in line2 or 'Recall @ 2 =' in line2:
                    sample[f'Recall @ 2'] = line2.split('=')[-1].strip()
                    
                if 'Recall @ 10 =' in line2 or 'Recall @ 3 =' in line2:
                    sample[f'Recall @ 3'] = line2.split('=')[-1].strip()
                    break

            data.append(sample)
    return pd.DataFrame(data)

base_path = '/checkpoint/sgvaze/slurm_outputs/myLog-65052770_{}.out'
all_data = []
for i in range(7, 8):
    # data = parse_out_file(base_path.format(i), rx_dict=rx_dict)
    data = parse_out_file('/checkpoint/sgvaze/slurm_outputs/myLog-812078_0.out', rx_dict=rx_dict)
    # data = parse_out_file(f'/checkpoint/sgvaze/slurm_outputs/myLog-65519175_{i}.out', rx_dict=rx_dict)
    all_data.append(data)

print(pd.concat(all_data))
