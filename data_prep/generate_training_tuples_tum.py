# Modified PointNetVLAD code: https://github.com/mikacuy/pointnetvlad
# Modified by:  Qianyun Li (Technical University of Munich 2022)

import os
import pathlib
import numpy as np
from decimal import Decimal
import csv
import random
import argparse
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

    northing_range = [5335969.471942793, 5336110.305563628]
    easting_range =  [690875.4143936138, 691050.9016333856]

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

#####Initialize pandas DataFrame#####
def get_dataframes(tag_dirs, tag_filenames, northing_range, easting_range):
    data_train, data_test = [], []

    for idx, folder in enumerate(tag_dirs):
        filename = tag_filenames[idx]
        with open(os.path.join(folder,filename), 'r') as f:
            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                northing, easting = Decimal(row[1]), Decimal(row[2])
                if(check_in_test_set(northing, easting, northing_range, easting_range)):
                    data_test.append((row[0], row[1], row[2]))
                else:
                    data_train.append((row[0], row[1], row[2]))
        
    df_train = pd.DataFrame.from_records(data_train, columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame.from_records(data_test, columns=['file', 'northing', 'easting'])

    # print(df_train)
    # print(df_test)
    print("Number of training submaps: "+str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: "+str(len(df_test['file'])))

    return df_train, df_test
#######################################


#####Constructing training pickles#####
def output_to_file(output, filename, protocol):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=protocol)
    print("\nDone ", filename)

def construct_query_dict(df_centroids, filename, protocol, query_dist):
    tree = KDTree(df_centroids[['northing','easting']])
    ind_nn = tree.query_radius(df_centroids[['northing','easting']],r=query_dist['pos']) #5
    ind_r = tree.query_radius(df_centroids[['northing','easting']], r=query_dist['neg']) #12.5
    queries={}
    for i in np.arange(ind_nn.size):
        query=df_centroids.iloc[i]["file"]
        positives=np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives=np.setdiff1d(df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i]={"query":query,"positives":positives,"negatives":negatives}
          
    output_to_file(queries, filename, protocol)

def generate_training_pickles(df_train, df_test, query_train_dist,  protocol, pickle_dir, dataset_name):
    protocol = 5
    pickle_dir = pickle_dir + f'protocol{protocol}'

    if not os.path.exists(pickle_dir):
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
        
    construct_query_dict(df_centroids=df_train,
                         filename=os.path.join(pickle_dir, f"tum_{dataset_name}_training_queries_baseline.pickle"),
                         protocol=protocol,
                         query_dist=query_train_dist)
    
    construct_query_dict(df_centroids=df_test,
                         filename=os.path.join(pickle_dir, f"tum_{dataset_name}_test_queries_baseline.pickle"),
                         protocol=protocol,
                         query_dist=query_train_dist)

#####################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TUM Data Training Pickle Parameters')
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
    ##################################################

    df_train, df_test = get_dataframes(tag_dirs, tag_filenames, northing_range, easting_range)

    generate_training_pickles(df_train = df_train,
                              df_test = df_test,
                              query_train_dist = query_train_dist,
                              protocol = 5,
                              pickle_dir = pickle_dir,
                              dataset_name = dataset_name)

    generate_training_pickles(df_train = df_train,
                              df_test = df_test,
                              query_train_dist = query_train_dist,
                              protocol = 4,
                              pickle_dir = pickle_dir,
                              dataset_name = dataset_name)


