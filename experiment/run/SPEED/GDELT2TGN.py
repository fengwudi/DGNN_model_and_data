import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import argparse


def preprocess():
    edges_name = './DATA/GDELT/edges.csv'
    node_name = './DATA/GDELT/node_features.pt'
    edge_name = './DATA/GDELT/edge_features.pt'

    df = pd.read_csv(edges_name)
    edge_feat = torch.load(edge_name)
    node_feat = torch.load(node_name)

    edges = {
        'u': df['src'],
        'i': df['dst'],
        'ts': df['time'],
        'label': 0
    }

    df = pd.DataFrame(edges)
    df['idx'] = df.index

    print(len(df.u.unique()) / df.u.max())
    print(len(df.i.unique()) / df.i.max())

    df["u"] = df["u"].astype("category")
    df['u'] = df['u'].cat.codes
    df["i"] = df["i"].astype("category")
    df['i'] = df['i'].cat.codes

    return df, np.array(edge_feat), node_feat


def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path("data/").mkdir(parents=True, exist_ok=True)
    # PATH = './Datasets/GDELT/edge.csv'
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat, node_feat = preprocess()
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())

    # rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, node_feat)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()


if __name__ == '__main__':
    run('gdelt', bipartite=True)
