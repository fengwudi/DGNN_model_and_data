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
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            raw_ts = int(e[4])
            if 1511481600 < raw_ts < 1512345600:
                u = int(e[0])
                i = int(e[1])
                ts = int(e[4]) - 1511481613

                label = int(e[2])

                feat = e[3]

                one_hot = np.zeros(4)
                if feat == 'pv':
                    one_hot[0] = 1
                elif feat == 'buy':
                    one_hot[1] = 1
                elif feat == 'cart':
                    one_hot[2] = 1
                elif feat == 'fav':
                    one_hot[3] = 1

                feat = one_hot

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
    print('unique_u:{}, max_u:{}, unique_i:{}, max_i:{}, num_label:{}'
          .format(len(df.u.unique()), df.u.max(), len(df.i.unique()), df.i.max(), len(df.label.unique())))
    df["u"] = df["u"].astype("category")
    df['u'] = df['u'].cat.codes
    df["i"] = df["i"].astype("category")
    df['i'] = df['i'].cat.codes
    df["label"] = df["label"].astype("category")
    df['label'] = df['label'].cat.codes

    print('u,i,label reindex finished')
    print('unique_u:{}, max_u:{}, unique_i:{}, max_i:{}, total_uniq_nodes:{}, num_label:{}'
          .format(len(df.u.unique()), df.u.max(), len(df.i.unique()),
                  df.i.max(), (len(df.u.unique())+len(df.i.unique())), len(df.label.unique())))

    df = df.sort_values('ts')

    print('ts rearrange finished')

    ts_min, ts_max = df.ts.min(), df.ts.max()
    print(ts_min, ts_max)

    ind = [str(i) for i in range(len(idx_list))]
    ind_add_one = [j for j in range(len(idx_list))]
    df.index = ind
    df['idx'] = ind_add_one
    edge_feat = df['feat_l'].tolist()
    df = df.drop(['feat_l'], axis=1)

    print('returning df and edge_feat')
    return df, np.array(edge_feat)


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
    PATH = './Datasets/UserBehavior.csv'
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
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
    run('taobao', bipartite=True)
