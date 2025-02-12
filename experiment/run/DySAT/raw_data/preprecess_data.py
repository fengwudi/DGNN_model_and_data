import os.path

import networkx as nx
import pandas
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix
import argparse

from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="wikipedia",type=str,choices=["wikipedia","reddit","Flights","mooc"])

    args = parser.parse_args()

    edge_file = "raw_data/ml_{}.csv".format(args.dataset)
    node_feat_file = "raw_data/ml_{}_node.npy".format(args.dataset)

    df = pandas.read_csv(edge_file)
    node_feats = np.load(node_feat_file)

    aggr_time = 100000 if args.dataset != "Flights" else 7
    print(f'Dataset {args.dataset}: num_nodes: {node_feats.shape[0]},'
          f' num_edge: {len(df)}, max_ts: {df["ts"].max()}, min_ts: {df["ts"].min()}')

    df["ts"] = df["ts"] // aggr_time
    print(f"snapshot num: {df['ts'].max()}, ts_sorted:{df['ts'].is_monotonic_increasing}")

    snapshots = defaultdict(lambda: nx.MultiGraph())
    for row in df.itertuples():
        u = row.u
        i = row.i
        ts = row.ts
        label = int(row.label)
        # print(ts)

        if ts not in snapshots.keys():
            if ts > 0:
                snapshots[int(ts)].add_nodes_from(snapshots[ts - 1].nodes(data=True))
                assert (len(snapshots[ts].edges()) == 0)
        snapshots[int(ts)].add_edge(u,i,label=label)




    used_nodes = []
    for id, snapshot in snapshots.items():
        print(f"snapshot {id}: #nodes:{snapshot.number_of_nodes()}, #edges:{snapshot.number_of_edges()}")
        for node in snapshot.nodes():
            if node not in used_nodes:
                used_nodes.append(node)

    # remap nodes in graph
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}
    for id, snapshot in snapshots.items():
        snapshots[id] = nx.relabel_nodes(snapshot,nodes_consistent_map)

    reversed_nodes_consistent_map = {node:idx for idx, node in nodes_consistent_map.items()}
    graphs = []
    for id, snapshot in snapshots.items():
        tmp_feature = []
        for node in snapshot.nodes():
            tmp_feature.append(node_feats[reversed_nodes_consistent_map[node]])
        snapshot.graph["feature"] = csr_matrix(tmp_feature)
        graphs.append(snapshot)



    save_path = "data/{}".format(args.dataset)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + "/graph.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(graphs, f)
    print("Processed Data Saved at {}".format(save_path))




