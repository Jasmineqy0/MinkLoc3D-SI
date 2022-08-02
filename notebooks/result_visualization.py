from cmath import exp
import pandas as pd
import numpy as np
import json
from glob import glob
import os
import plotly.express as px
from tqdm import tqdm
import re

tf_logdir = '../tf_logs/'

def get_finished_configs(tf_logdir):
    """
    Get the list of finished configs, return a list.
    """
    if not os.path.exists(tf_logdir):
        raise ValueError('The directory {} does not exist.'.format(tf_logdir))

    configs, dirs = [], []
    for dir in os.listdir(tf_logdir):
        if os.path.exists(os.path.join(tf_logdir, dir, 'config.json')):
            dirs.append(dir)
            with open(os.path.join(tf_logdir, dir, 'config.json')) as f:
                config = json.load(f)
                configs.append(config)
    return configs, dirs

def get_columns(configs, dirs): 

    # data-related columns
    datasets = []
    epochs = []
    batch_size = []
    batch_size_limit = []

    # model-related columns
    version = []
    backbone = []
    pooling = []
    pointnet = []
    pointnet_pnt2s = []
    self_attention = []
    self_attention_num_layers = []
    pointnet_cross_attention = []
    pointnet_cross_attention_attention_types = []
    multi_cross_attention = []
    multi_cross_attention_attention_types = []

    # result columns
    loss = []
    average_recall = []
    average_1p_recall = []

    for config in configs:
        datasets.append(config['params']['dataset_name'])
        epochs.append(config['params']['epochs'])
        batch_size.append(config['params']['batch_size'])
        batch_size_limit.append(config['params']['batch_size_limit'])
        backbone.append(config['params']['model_params']['backbone'])
        pooling.append(config['params']['model_params']['pooling'])

        pointnet.append(True if 'pointnet' in config['params']['model_params']['combine_params'] else False)
        pointnet_pnt2s.append(config['params']['model_params']['combine_params']['pointnet']['pnt2s'] if \
                             'pointnet' in config['params']['model_params']['combine_params'] else None)
        self_attention.append(True if 'self_attention' in config['params']['model_params']['combine_params'] else False)
        self_attention_num_layer = 1 if 'self_attention' in config['params']['model_params']['combine_params'] else None
        self_attention_num_layer = config['params']['model_params']['combine_params']['self_attention']['num_layers'] if \
                                    'self_attention' in config['params']['model_params']['combine_params'] and \
                                    'num_layers' in config['params']['model_params']['combine_params']['self_attention'] else self_attention_num_layer
        self_attention_num_layers.append(self_attention_num_layer)
        pointnet_cross_attention.append(True if 'pointnet_cross_attention' in config['params']['model_params']['combine_params'] else False)
        pointnet_cross_attention_attention_types.append(config['params']['model_params']['combine_params']['pointnet_cross_attention']['attention_type'] if \
                                                        'pointnet_cross_attention' in config['params']['model_params']['combine_params'] else None)
        multi_cross_attention.append(True if 'multi_cross_attention' in config['params']['model_params']['combine_params'] else False)
        multi_cross_attention_attention_types.append(config['params']['model_params']['combine_params']['multi_cross_attention']['attention_type'] if \
                                                        'multi_cross_attention' in config['params']['model_params']['combine_params'] else None)
        version.append(backbone[-1]+pooling[-1])

        exp_loss = []
        for i in range(len(config['stats']['train'])):
            exp_loss.append(float(config['stats']['train'][i]['loss']))
        loss.append(exp_loss)
        
        exp_avg_recall, exp_1p_avg_recall = [], []
        for j in range(len(config['stats']['eval'])):
            epoch = re.match(r'(epoch[0-9]+)', next(iter(config['stats']['eval'][j]))).group(1)
            exp_1p_avg_recall.append(float(config['stats']['eval'][j][epoch][config['params']['dataset_name'].lower()]['ave_one_percent_recall']))
            exp_avg_recall_str = config['stats']['eval'][j][epoch][config['params']['dataset_name'].lower()]['ave_recall']
            exp_avg_recall.append(float(re.match(r'\[\s*([0-9]+.[0-9]+).*', exp_avg_recall_str).group(1)))
        average_recall.append(exp_avg_recall)
        average_1p_recall.append(exp_1p_avg_recall)
        assert len(average_recall) == len(average_1p_recall), 'The length of average_recall and average_1p_recall is not equal.'
    
    model_names = []
    for v, p, s, pc, m in zip(version, pointnet, self_attention, pointnet_cross_attention, multi_cross_attention):
        model_name = v
        model_name = model_name + ' + pointnet' if p else model_name
        model_name = model_name + ' + self_attention' if s else model_name
        model_name = model_name + ' + pointnet_cross_attention' if pc else model_name
        model_name = model_name + ' + multi_cross_attention' if m else model_name
        if v == 'MinkFPNGeM' and not p and not s and not pc and not m:
            model_name = model_name + ' Baseline'
        model_names.append(model_name)

    dirs = [d.replace('-', '_')[:-2] if int(d.split('-')[0][-3:]) < 731 else d.replace('-', '_') for d in dirs]
    dirs = [f'model_{v}_{d}' for v,d in zip(version, dirs)]

    df = pd.DataFrame(
        {   'dataset_name': datasets,
            'model_name': model_names,
            'dir': dirs,
            'epochs': epochs,
            'batch_size': batch_size,
            'batch_size_limit': batch_size_limit,
            'version': version,
            'pointnet.pnt2s': pointnet_pnt2s,
            'self_attention.num_layers': self_attention_num_layers,
            'pointnet_cross_attention.attention_types': pointnet_cross_attention_attention_types,
            'loss': loss,
            'multi_cross_attention': multi_cross_attention,
        })

    metrics = {
        'average_recall': np.array(average_recall),
        'average_1p_recall': np.array(average_1p_recall)
    }
    return df, metrics


def get_highest_by_criterion(criterion, df, metrics):
    all_metrics = ['average_recall', 'average_1p_recall']
    assert criterion in all_metrics, 'criterion must be one of {}'.format(all_metrics)
    
    criterion1 = next(iter (set(all_metrics) - set([criterion])))
    max_vals = [np.max(l) for l in metrics[criterion]]
    max_idx =  [np.argmax(l) for l in metrics[criterion]]
    assert len(max_vals) == len(max_idx), 'The length of max_vals and max_idx is not equal.'
    df['best_epoch'] = np.asarray(max_idx) + 1
    df.loc[df['dataset_name'] != 'TUM', 'best_epoch'] =  df.loc[df['epochs'] != 'TUM', 'epochs']
    df[criterion] = np.asarray(max_vals, dtype="object")
    df[criterion1] = np.asarray([l[i] for i, l in zip(max_idx, metrics[criterion1])],  dtype="object")
    assert criterion in df.columns and criterion1 in df.columns and 'best_epoch' in df.columns
    
    return df

def get_results(tf_logdir):
    """
    Get the table of results, return a dataframe.
    """
    if not os.path.exists(tf_logdir):
        raise ValueError('The directory {} does not exist.'.format(tf_logdir))

    configs, dirs = get_finished_configs(tf_logdir)

    df = get_columns(configs, dirs)

    return df 
    

df, metrics = get_results(tf_logdir)
criterion = 'average_recall'
df_criterion = get_highest_by_criterion('average_recall', df, metrics)
df_criterion.sort_values(by=criterion, ascending=False)

assert False