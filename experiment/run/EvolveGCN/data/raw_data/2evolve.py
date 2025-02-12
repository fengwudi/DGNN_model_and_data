import numpy as np
import pandas as pd
import torch
import argparse


def changewiki():
    
    # df = pd.read_csv('wikipedia.csv', index_col=False)
    # edges = {
    #     'source': df['user_id'],
    #     'target': df['item_id'],
    #     'label': df['state_label'],
    #     'timestamp': df['timestamp'],
    #     'edge_feat': df['comma_separated_list_of_features']
    # }
    # df_edges = pd.DataFrame(edges)
    # df_edges = df_edges.sort_values(['timestamp', 'source'])
    
    # df_edges.to_csv('../wikipedia.csv', index=False, header=None)
    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open('./wikipedia.csv') as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    df_edges = pd.DataFrame({'source': u_list, 'target': i_list, 'label': label_list, 'timestamp': ts_list})
    df_edges = df_edges.sort_values(['timestamp', 'source'])
    df_edges.to_csv('../wikipedia.csv', index=False, header=None)
    # print(feat_l)
    # feat_a = np.array(feat_l)
    # np.save('../wikipedia.npy', feat_a)
    
def changereddit():
    
    # df = pd.read_csv('reddit.csv', index_col=False)
    # edges = {
    #     'source': df['user_id'],
    #     'target': df['item_id'],
    #     'label': df['state_label'],
    #     'timestamp': df['timestamp'],
    #     'edge_feat': df['comma_separated_list_of_features']
    # }
    # df_edges = pd.DataFrame(edges)
    # df_edges = df_edges.sort_values(['timestamp', 'source'])
    
    # df_edges.to_csv('../reddit.csv', index=False, header=None)
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open('./reddit.csv') as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    df_edges = pd.DataFrame({'source': u_list, 'target': i_list, 'label': label_list, 'timestamp': ts_list})
    df_edges = df_edges.sort_values(['timestamp', 'source'])
    df_edges.to_csv('../reddit.csv', index=False, header=None)
    
def showwiki():
    df = pd.read_csv('../wikipedia.csv')
    print(df)


def changeFlight():
    
    df = pd.read_csv('ml_Flights.csv', index_col=False)
    # print(df.iloc[0])
    edges = {
        'source': df['u'],
        'target': df['i'],
        'label': df['label'],
        'timestamp': df['ts']
    }
    df_edges = pd.DataFrame(edges)
    # df_edges = df_edges.sort_values(['timestamp', 'source'])
    
    df_edges.to_csv('../flight.csv', index=False, header=None)


def changemooc():
    
    # df = pd.read_csv('mooc.csv', index_col=False)
    # edges = {
    #     'source': df['user_id'],
    #     'target': df['item_id'],
    #     'label': df['state_label'],
    #     'timestamp': df['timestamp']
    # }
    # df_edges = pd.DataFrame(edges)
    # df_edges = df_edges.sort_values(['timestamp', 'source'])
    
    # df_edges.to_csv('../mooc.csv', index=False, header=None)
    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open('./mooc.csv') as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = int(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    df_edges = pd.DataFrame({'source': u_list, 'target': i_list, 'label': label_list, 'timestamp': ts_list})
    df_edges = df_edges.sort_values(['timestamp', 'source'])
    df_edges.to_csv('../mooc.csv', index=False, header=None)


def changeml25m():
    df =pd.read_csv('ml25m.csv')
    edges = {
        'src': df['userId'],
        'dst': df['movieId'],
        'label': df['rating'],
        'time': df['timestamp'] - 789652009
    }
    df_edges = pd.DataFrame(edges)
    df_edges = df_edges.sort_values(['time', 'src'])
    
    df_edges.to_csv('../ml25m.csv', index=False, header=None)

# changeml25m()

def countTime():
    df =pd.read_csv('ml25m.csv')
    edges = {
        'src': df['userId'],
        'dst': df['movieId'],
        'label': df['rating'],
        'timestamp': df['timestamp'] - 789652009
    }
    # df = pd.read_csv('reddit.csv', index_col=False)
    # edges = {
    #     'src': df['user_id'],
    #     'dst': df['item_id'],
    #     'label': df['state_label'],
    #     'timestamp': df['timestamp'],
    #     'edge_feat': df['comma_separated_list_of_features']
    # }
    print(edges['timestamp'].max() - edges['timestamp'].min())
    # print(edges.dst.max() - edges.dst.min())

def changedgraphfin():
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    np_file = np.load('./dgraphfin.npz')
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
                       'label': label_list,
                       'ts': ts_list
                       })

    df = df.sort_values('ts')

    print(len(df.u.unique()) / df.u.max())
    print(len(df.i.unique()) / df.i.max())

    df["u"] = df["u"].astype("category")
    df['u'] = df['u'].cat.codes
    df["i"] = df["i"].astype("category")
    df['i'] = df['i'].cat.codes
    
    df.to_csv('../dgraphfin.csv', index=False, header=None)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')

args = parser.parse_args()
if args.data == 'wikipedia':
    changewiki()
elif args.data == 'reddit':
    changereddit()
elif args.data == 'mooc':
    changemooc()
elif args.data == 'Flights':
    changeFlight()
elif args.data == 'ml25m':
    changeml25m()
elif args.data == 'dgraphfin':
    changedgraphfin()