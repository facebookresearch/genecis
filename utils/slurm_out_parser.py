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
    with open(path, 'rt') as myfile:
        for myline in myfile:  # For each line, read to a string,
            file.append(myline)

    return file


def parse_out_file(path, rx_dict, verbose=True):

    file = get_file(path=path)

    models = []
    for i, line in enumerate(file):

        if line.find('log_dir') != -1:
            print(line)


def parse_multiple_files(all_paths, rx_dict):

    all_data = []
    for path in all_paths:
        print(path)
        try:
            data = parse_out_file(path, rx_dict)
        except:
            print('No log file ...')

base_path = '/checkpoint/sgvaze/slurm_outputs/myLog-721202_0.out'
all_paths = [base_path.format(i) for i in range(10)]
data = parse_multiple_files(all_paths, rx_dict)