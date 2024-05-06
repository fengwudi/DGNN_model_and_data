#!/bin/bash

wget -P ./DATA/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
wget -P ./DATA/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv
wget -P ./DATA/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv