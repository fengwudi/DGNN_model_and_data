import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = int(e[3]) - 789652009
            label = 0

            feat = float(e[2])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)

    df = pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list,
                       'feat_l': feat_l})

    print(len(df.u.unique()) / df.u.max())
    print(len(df.i.unique()) / df.i.max())

    df["u"] = df["u"].astype("category")
    df['u'] = df['u'].cat.codes
    df["i"] = df["i"].astype("category")
    df['i'] = df['i'].cat.codes

    df = df.sort_values('ts')
    ind = [str(i) for i in range(len(idx_list))]
    ind_add_one = [j for j in range(len(idx_list))]
    df.index = ind
    df['idx'] = ind_add_one
    edge_feat = df['feat_l'].tolist()
    df = df.drop(['feat_l'], axis=1)
    edge_feat = np.array(edge_feat)
    edge_feat = edge_feat.reshape(-1, 1)
    return df, edge_feat


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
    PATH = './Datasets/ml-25m/ratings.csv'
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(1)[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

if __name__ == '__main__':
    run('ml25m', bipartite=True)
