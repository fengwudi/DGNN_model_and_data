#!/bin/bash

wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt     
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/ext_full.npz            
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv                  
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/ec_edge_features_e0.pt
wget -P ./DATA/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/ec_edge_class.pt.pt

wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features_e0.pt
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges+uniq.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/ext_full.npz
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/learned_node_feats.pt

wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features_e0.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv  
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/ext_full.npz
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/learned_node_feats.pt
