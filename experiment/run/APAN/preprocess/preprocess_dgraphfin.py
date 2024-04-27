import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from torch._C import dtype
from tqdm import tqdm
import csv
import os


def preprocess():

  u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []
  feat_l = []
  np_file = np.load('./data/dgraphfin.npz')
  node_feat = np_file['x']
  y = np_file['y']
  edge_type = np_file['edge_type']
  edge_timestamp = np_file['edge_timestamp']
  edge_index = np_file['edge_index']
  for idx, interactions in enumerate(edge_index):
    u = interactions[0]
    i = interactions[1]

    ts = edge_timestamp[idx]
    label = y[u]
    edge_feature = edge_type[idx]
    one_hot = np.zeros(11)
    one_hot[(edge_feature - 1)] = 1
    edge_label_oh = one_hot

    u_list.append(u)
    i_list.append(i)
    ts_list.append(ts)
    label_list.append(label)
    idx_list.append(idx)

    feat_l.append(edge_label_oh)

    df = pd.DataFrame({'u': u_list,
                        'i': i_list,
                        'ts': ts_list,
                        'label': label_list,
                        'feat': feat_l})

    df = df.sort_values('ts')
    # ind = [str(i) for i in range(len(edge_timestamp))]
    # ind_add_one = [j for j in range(len(edge_timestamp))]
    # df.index = ind
    # df['idx'] = ind_add_one
    # edge_feat = df['feat_l'].tolist()
    # df = df.drop(['feat_l'], axis=1)

    # print(len(df.u.unique()) / df.u.max())
    # print(len(df.i.unique()) / df.i.max())

    df["u"] = df["u"].astype("category")
    df['u'] = df['u'].cat.codes
    df["i"] = df["i"].astype("category")
    df['i'] = df['i'].cat.codes

    # feat = np.array(feat_l, dtype='float32')
    # empty = np.zeros(feat.shape[1], dtype='float32')[np.newaxis, :]
    # feat = np.vstack([empty, feat])
    df.to_csv('./data/dgraphfin_raw.csv')



preprocess()