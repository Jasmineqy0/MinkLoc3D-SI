# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by:  Qianyun Li (Technical University of Munich 2022)

import argparse
import os
import pathlib
import numpy as np
from decimal import Decimal
import csv
import random
import pickle
import pandas as pd
from sklearn.neighbors import KDTree
from position import pnt2utm

#####For training and test data split##### 

def get_test_range(northing_shift, easting_shift):

    z_center = -6.919999999999999

    # corner points of street in the middle
    l_ini = np.array([-39.516170501709,51.071258544922, z_center], dtype=np.float64)
    l_end = np.array([-117.093261718750,-129.892379760742, z_center], dtype=np.float64)
    w_ini = np.array([-138.423934936523,-26.235725402832, z_center], dtype=np.float64)
    w_end = np.array([-47.159458160400,-66.141090393066, z_center], dtype=np.float64)
    corner_points = [l_ini, l_end, w_ini, w_end]

    northing_vals, easting_vals = [], []
    for point in corner_points:
        northing, easting = pnt2utm(point)
        northing_vals.append(northing)
        easting_vals.append(easting)
    
    northing_min, northing_max = min(northing_vals), max(northing_vals)
    easting_min, easting_max = min(easting_vals), max(easting_vals)

    northing_range = [northing_min-northing_shift, northing_max+northing_shift]
    easting_range = [easting_min-easting_shift, easting_max+easting_shift]

    print('min for northing: ', northing_range[0], '\nmax for northing: ', northing_range[1])
    print('min for easting: ', easting_range[0], '\nmax for y: ', easting_range[1])
    return northing_range, easting_range

def check_in_test_set(northing, easting, northing_range, easting_range):
    in_test_set=False
    northing_min, northing_max = northing_range[0], northing_range[1]
    easting_min, easting_max = easting_range[0], easting_range[1]
    if(northing_min < northing and northing < northing_max and easting_min < easting and easting < easting_max):
        in_test_set=True
    return in_test_set

##########################################


#####Constructing testing pickles#####

def output_to_file(output, filename, protocol):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=protocol)
    print("\nDone ", filename)

def construct_query_and_database_sets(tag_dirs, tag_filenames, northing_range, easting_range, protocol, query_test_dist, pickle_dir, dataset_name):
    database_trees, test_trees = [], []
    for folder, filename in zip(tag_dirs, tag_filenames):
        data_database, data_test = [], []
        with open(os.path.join(folder,filename), 'r') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                northing, easting = Decimal(row[1]), Decimal(row[2])
                if check_in_test_set(northing, easting, northing_range, easting_range):
                    data_test.append((row[0], row[1], row[2]))
                data_database.append((row[0], row[1], row[2]))
                
        df_database = pd.DataFrame.from_records(data_database, columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame.from_records(data_test, columns=['file', 'northing', 'easting'])
#         print('df_database in {}\n'.format(filename), df_database, '\n\n df_test in {}\n'.format(filename), df_test)
        
        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets, database_sets = [], []
    cnt = 0
    for folder, filename in zip(tag_dirs, tag_filenames):
        database, test = {}, {}
        with open(os.path.join(folder,filename), 'r') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                file, northing, easting = row[0], Decimal(row[1]), Decimal(row[2])
                if check_in_test_set(northing, easting, northing_range, easting_range):
                    cnt += 1
                    test[len(test.keys())] = {'query': file, 'northing': northing, 'easting': easting}
                database[len(database.keys())] = {'query': file, 'northing': northing, 'easting': easting}
        database_sets.append(database)
        test_sets.append(test)
    print('\ncnt:', cnt)
    
    cnt=0
    for i in np.arange(len(database_sets)):
        tree = database_trees[i]
        for j in np.arange(len(test_sets)):
            if (i == j):
                continue
            for key in np.arange(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                # DISTANCES: CORRECT<5
                index = tree.query_radius(coor, r=query_test_dist)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()
            
                print('\n', test_sets[j][key]['query'])
                tmp = test_sets[j][key][i]
                if tmp:
                    cnt +=1
                    print('\n', tmp, database_sets[i][tmp[0]]['query'])
    print('\ncnt:', cnt)

    output_to_file(database_sets, os.path.join(pickle_dir, f'{dataset_name}_evaluation_database.pickle'), protocol)
    output_to_file(test_sets, os.path.join(pickle_dir, f'{dataset_name}_evaluation_query.pickle'), protocol)

def generate_test_pickles(tag_dirs, tag_filenames,  northing_range, easting_range, protocol, pickle_dir, query_test_dist, dataset_name):
    pickle_dir = pickle_dir + f'python{protocol}'
    if not os.path.exists(pickle_dir):
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    construct_query_and_database_sets(tag_dirs=tag_dirs,
                                      tag_filenames=tag_filenames,
                                      northing_range=northing_range,
                                      easting_range=easting_range,
                                      protocol=protocol,
                                      pickle_dir=pickle_dir,
                                      query_test_dist=query_test_dist,
                                      dataset_name=dataset_name)

#####################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TUM Data Testing sets Parameters')
    parser.add_argument('--range', type=int, default=None, help='range of each frame')
    parser.add_argument('--frame_dist', default=5, type=int, help='distance between frames')

    args = parser.parse_args()

    frame_dist = args.frame_dist
    frame_range = args.range

    ##### constant dirs & patterns & other params#####
    northing_shift, easting_shift = 52, 52

    dataset_name = f'frame_{frame_dist}m'
    dataset_name = dataset_name + f'_range_{frame_range}m' if frame_range is not None else dataset_name
    dataset_dir = f'/home/xiayan/testdir/datasets/{dataset_name}/'
    inter_2018_dir = f'/home/xiayan/testdir/datasets/tum_prep/intermediate/frame_{frame_dist}m_range_{frame_range}m_2018'
    inter_2016_dir = f'/home/xiayan/testdir/datasets/tum_prep/intermediate/frame_{frame_dist}m_range_{frame_range}m_2016'
    pickle_dir = f'/home/xiayan/testdir/datasets/{dataset_name}/pickles/'

    tag_dirs = [inter_2018_dir, inter_2016_dir]
    tag_filenames = ['2018-tags.csv', '2016-tags.csv']
    northing_range, easting_range = get_test_range(northing_shift, easting_shift)

    query_train_dist = {"pos":5, 'neg':12.5}
    query_test_dist = 5
    ##################################################


    generate_test_pickles(tag_dirs=tag_dirs,
                          tag_filenames=tag_filenames,
                          northing_range=northing_range,
                          easting_range=easting_range,
                          protocol=5,
                          pickle_dir=pickle_dir,
                          query_test_dist=query_test_dist,
                          dataset_name=dataset_name)

    generate_test_pickles(tag_dirs=tag_dirs,
                          tag_filenames=tag_filenames,
                          northing_range=northing_range,
                          easting_range=easting_range,
                          protocol=4,
                          pickle_dir=pickle_dir,
                          query_test_dist=query_test_dist,
                          dataset_name=dataset_name)
