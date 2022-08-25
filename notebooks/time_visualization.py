import csv
import json
import os
import numpy as np
from glob import glob

def get_finished_time_perf(weights_dir):
    """
    Get the list of time perf csv, return a list.
    """
    if not os.path.exists(weights_dir):
        raise ValueError('The directory {} does not exist.'.format(weights_dir))

    perf_files = []
    for dir in os.listdir(weights_dir):
        for file in glob(os.path.join(weights_dir, dir, '*.csv')):
            perf_files.append(file)
    return perf_files

def get_avg_perf_dict(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader, None)

        perfs = []
        for row in reader:
            row = [float(item) for item in row]
            perfs.append(row)
    perfs = np.asarray(perfs, dtype=float)
    perfs = np.mean(perfs, axis=0)
    assert perfs.size == 5

    perf_dict = dict(zip(headers, perfs))
    return perf_dict


weights_dir = '/home/xiayan/testdir/MinkLoc3D-SI/weights'
perf_files = get_finished_time_perf(weights_dir)

assert len(perf_files) > 0

file = perf_files[0]

perf_dict = get_avg_perf_dict('/home/xiayan/testdir/MinkLoc3D-SI/weights/model_MinkFPNGeM_20220818_141758/epoch40_time.csv')
