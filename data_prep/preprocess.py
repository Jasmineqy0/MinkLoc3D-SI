from genericpath import exists
import os
import open3d as o3d
import numpy as np
from glob import glob
from pypcd import pypcd
import pyproj
from pathlib import Path
import pickle
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
import subprocess
from typing import Union
import warnings
import argparse

def ecef2lla(pnt_ecef):
    M = np.array([[-0.20053982643488838, -0.72975436304732511,  0.653637780140389870, 4177139.442822214700], 
                  [0.979685550579095120, -0.14937937302342402,  0.133798448801411370, 855052.7445900742900],
                  [0.000000000000000000,  0.667191406216027790, 0.744886318488585660, 4728408.463954962800],
                  [0.000000000000000000,  0.000000000000000000, 0.000000000000000000, 1.000000000000000000]])
    x = np.append(pnt_ecef.T, np.array([1], ndmin=2), axis=0)
    
    pnt  = M @ x
    
    transformer = pyproj.Transformer.from_crs(
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    )
    
    lon, lat, alt = transformer.transform(pnt[0],pnt[1],pnt[2],radians=False)
    return lon[0], lat[0], alt[0]

def lla2utm(lon, lat):
    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=32, preserve_units=False)
    east, north = proj(lon, lat)
    return east, north

def pnt2utm(pnt):
    lon, lat, _ = ecef2lla(pnt.reshape(1,-1))
    east, north = lla2utm(lon, lat)
    return north, east

# Code From:https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1000

def save_frames_list(frames_list, path):
    with open(path, 'wb') as f:
        pickle.dump(frames_list, f)

def load_frames_list(path):
    with open(path, 'rb') as f:
        frames_list = pickle.load(f)
    return frames_list

def reshape_pcd(pcd):
    np_x = (np.array(pcd.pc_data['x'], dtype=np.float64)).astype(np.float64)
    np_y = (np.array(pcd.pc_data['y'], dtype=np.float64)).astype(np.float64)
    np_z = (np.array(pcd.pc_data['z'], dtype=np.float64)).astype(np.float64)
    np_vpx = (np.array(pcd.pc_data['vp_x'], dtype=np.float64)).astype(np.float64)
    np_vpy = (np.array(pcd.pc_data['vp_y'], dtype=np.float64)).astype(np.float64)
    np_vpz = (np.array(pcd.pc_data['vp_z'], dtype=np.float64)).astype(np.float64)
    points = np.transpose(np.vstack((np_x, np_y, np_z, np_vpx, np_vpy, np_vpz)))
    points = points[~np.isnan(points).any(axis=1)] # remove nan points
    assert points.shape[1] == 6

    return points

def collect_frames_list(input_dir, output_dir, idx_file_stt, idx_file_end, frame_dist):
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    is_start = True
    dist_cum = 0
    frames = []
    for i in range(idx_file_stt, idx_file_end+1):
        input_file = os.path.join(input_dir, f'{i:05d}.pcd')

        pcd = pypcd.PointCloud.from_path(input_file)
        points = reshape_pcd(pcd)
        vp_xyz = np.array([points[0][3:]], dtype=np.float64)
        lon, lat, _ = ecef2lla(vp_xyz)

        if is_start:
            print("new ini from {}".format(i))
            frames.append(input_file)
            # new_pcd = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(points[:, :3]))
            # assert np.asarray(new_pcd.points).shape[-1] == 3
            # o3d.io.write_point_cloud(output_file, new_pcd, write_ascii=True)

            pnt_prev = [lon, lat]
            is_start = False
        else:
            dist_cum += haversine(pnt_prev[0], pnt_prev[1], lon, lat)
            pnt_prev = [lon, lat]
            print(i, dist_cum, points.shape[0])

            if dist_cum > frame_dist:
                is_start = True
                dist_cum = 0 

    return frames

def collect_frames(frames_list, inter_frames_dir, year_prefix, frame_range):
    for frame in tqdm(frames_list):
        file_name = Path(frame).stem
        output_file = os.path.join(inter_frames_dir, f'{year_prefix}_{file_name}.pcd')

        pcd = pypcd.PointCloud.from_path(frame)
        points = reshape_pcd(pcd)
        if frame_range is not None:
            points = points[(points[:, 1] - points[:, 4]) < frame_range]
        new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
        assert np.asarray(new_pcd.points).shape[-1] == 3
        o3d.io.write_point_cloud(output_file, new_pcd, write_ascii=True)

    print(f'Finished for year {year_prefix}!')


def downsample_pcd(pcd, sample_size, ini_voxel_size):
    voxel_size = ini_voxel_size
    while (np.asarray(pcd.points).shape[0] > sample_size):
        voxel_size += 0.01
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    assert np.asarray(pcd.points).shape[0] <= sample_size, f'number of points after downsampling should be smaller than or equal to{sample_size}'

    return pcd

def upsample_pcd(points, sample_size):
    pnt_num = points.shape[0]
    idx_rand = np.arange(pnt_num)
    idx_left = np.random.randint(0, pnt_num, sample_size-pnt_num)
    idx_rand = np.concatenate([idx_rand, idx_left])
    sample = points[idx_rand]
    assert sample.shape[0] == sample_size

    return sample

def normalize_pcd(points, center):
    # normalize in range [-1, 1]
    sum = np.sum(np.linalg.norm(points - center, axis=1))
    d = sum / points.shape[0]
    s = 0.15/d #adjusted by visualizing intermediate result
    points = s * (points - center)

    more_n1 = np.where(np.all(points>=-1,axis=1), 1, 0)
    less_p1 = np.where(np.all(points<=1, axis=1), 1, 0)
    idx_norm = np.where(more_n1 + less_p1 == 2)[0]
    points = points[idx_norm] 
    assert np.all(points >= -1) and np.all(points <= 1) and np.any(points < 0)
    return points

def preprocess2bin(input_dir, output_dir, name_prefix, sample_size, is_normalize, seed):
    tags = []
    np.random.seed(seed)
    for file in tqdm(glob(os.path.join(input_dir, '*[0-9].pcd'))):
        file_name = file.split("/")[-1]
        file_name_vol_pcd = file_name.split('.')[0] + '_vox.pcd'
        file_name_bin = file_name.split('.')[0] + '.bin'

        pcd = o3d.io.read_point_cloud(file, remove_nan_points=True)
        
        # voxel grid downsampling
        downsampled_pcd = downsample_pcd(pcd, sample_size, ini_voxel_size=0.05)
        voxel_file = os.path.join(input_dir, file_name_vol_pcd)
        o3d.io.write_point_cloud(voxel_file, downsampled_pcd, write_ascii=True)
        points = np.asarray(downsampled_pcd.points)
        assert points.shape[0] <= sample_size, f'number of points after downsampling should be smaller than or equal to {sample_size}'

        # get the position
        center = np.mean(points, axis=0, keepdims=True)
        lon, lat, _ = ecef2lla(center)
        east, north = lla2utm(lon, lat)
        tags.append("{},{},{}\n".format(file_name_bin, north, east))
        
        # normalize
        if is_normalize:
            points = normalize_pcd(points, center)
        
        # upsample to sample size
        sample = upsample_pcd(points, sample_size)
        
        assert not np.any(np.all(sample == 0, axis=1)), f'false for {file_name}'
        sample.tofile(os.path.join(output_dir, file_name_bin))

    with open(os.path.join(input_dir,'{}-tags.csv'.format(name_prefix)), 'w') as f:
        f.write('file,northing,easting\n')
        f.writelines(tags)

    print(f'{name_prefix} Finished')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TUM Data Preprocessing Parameters')
    parser.add_argument('--get_frames', default=False, action="store_true", help='get frames')
    parser.add_argument('--remove_ground', default=False, action="store_true", help='remove ground')
    parser.add_argument('--preprocess_frames', default=False, action="store_true", help='preprocess frames')
    parser.add_argument('--range', type=int, default=None, help='range of each frame')
    parser.add_argument('--seed', type=int, default=10, help='seed')
    parser.add_argument('--not_normalize', default=False, action="store_true", help='normalized point cloud')
    parser.add_argument('--sample_size', default=4096, type=int, help='number of points in point cloud')
    parser.add_argument('--frame_dist', default=5, type=int, help='distance between frames')

    args = parser.parse_args()

    get_frames = args.get_frames
    frame_dist = args.frame_dist
    remove_ground = True if get_frames else args.remove_ground
    preprocess_frames = args.preprocess_frames
    frame_range = args.range
    seed = args.seed
    is_normalize = not args.not_normalize
    sample_size = args.sample_size

    ##### constant params#####
    warnings.filterwarnings('ignore', ".*can't understand line.*", )

    data_2018_dir = f'/home/xiayan/testdir/datasets/tum_prep/original/original_2018/laserscanner1'
    data_2016_dir = f'/home/xiayan/testdir/datasets/tum_prep/original/original_2016/laserscanner1'
    dataset_name = f'frame_{frame_dist}m'
    dataset_name = dataset_name + f'_range_{frame_range}m' if frame_range is not None else dataset_name + '_range_full'
    inter_2018_dir = f'/home/xiayan/testdir/datasets/tum_prep/intermediate/{dataset_name}_2018'
    inter_2016_dir = f'/home/xiayan/testdir/datasets/tum_prep/intermediate/{dataset_name}_2016'
    dataset_dir = f'/home/xiayan/testdir/datasets/{dataset_name}/'
    # pickle_dir_pattern = '{cwd}/models/datasets/frame_datasets/{dataset_name}/pickles/{version}'

    Path(inter_2018_dir).mkdir(exist_ok=True, parents=True)
    Path(inter_2016_dir).mkdir(exist_ok=True, parents=True) 
    Path(dataset_dir).mkdir(exist_ok=True, parents=True)

    idx_stt_2018, idx_end_2018 = 6, 10590
    idx_stt_2016, idx_end_2016 = 11, 8888

    saved_frames_list = {'2018':'frames_2018_list.bin', '2016':'frames_2016_list.bin'}
    ##################################################

    if get_frames:
        if not os.path.exists(saved_frames_list['2018']):
            print('Collecting frame list in 2018:')
            frames_2018_list = collect_frames_list( input_dir=data_2018_dir,
                                                    output_dir=inter_2018_dir,
                                                    idx_file_stt=idx_stt_2018,
                                                    idx_file_end=idx_end_2018,
                                                    frame_dist=frame_dist)
            save_frames_list(frames_2018_list, saved_frames_list['2018'])
        else:
            frames_2018_list = load_frames_list(saved_frames_list['2018'])

        if not os.path.exists(saved_frames_list['2016']):
            print('Collecting frame list in 2016:')
            frames_2016_list = collect_frames_list( input_dir=data_2016_dir,
                                                    output_dir=inter_2016_dir,
                                                    idx_file_stt=idx_stt_2016,
                                                    idx_file_end=idx_end_2016,
                                                    frame_dist=frame_dist)
            save_frames_list(frames_2016_list, saved_frames_list['2016'])
        else:
            frames_2016_list = load_frames_list(saved_frames_list['2016'])

        print(f'Number of frames collected in 2018: {len(frames_2018_list)}')
        print(f'Number of frames collected in 2016: {len(frames_2016_list)}')
        print(f'Total:  { len(frames_2018_list) + len(frames_2016_list) } ')

        print('Start to collect frames:')
        collect_frames( frames_list=frames_2018_list,
                        inter_frames_dir=inter_2018_dir,
                        year_prefix='2018',
                        frame_range=frame_range)
        collect_frames( frames_list=frames_2016_list,
                        inter_frames_dir=inter_2016_dir,
                        year_prefix='2016',
                        frame_range=frame_range)


    #### execute script to remove ground ####
    if remove_ground:
        print('Start to remove ground:')
        subprocess.run(f'bash remove_ground.sh  {inter_2018_dir} {inter_2016_dir}',  shell=True, check=True)

    # nohup ./remove_ground.sh > remove_ground.log 2>&1 &

    #### After removing ground, preprocess ####

    if preprocess_frames:
        print('Start to preprocess:')
        preprocess2bin(input_dir=inter_2018_dir,
                       output_dir=dataset_dir,
                       name_prefix='2018',
                       sample_size=sample_size,
                       is_normalize=is_normalize,
                       seed=10)

        preprocess2bin(input_dir=inter_2016_dir,
                       output_dir=dataset_dir,
                       name_prefix='2016',
                       sample_size=sample_size,
                       is_normalize=is_normalize,
                       seed=10)