#!/usr/bin/env bash
# Example run file, choose the configuration you want.
# Test for running a single experiment. --repeat means run how many different random seeds.

# Notes:
# complete heteroGNN: numNodeTypes * numEdgeTypes * numNodeTypes multiplicity.
# partial heteroGNN: only numNodeTypes + numEdgeTypes multiplicity.

# python3 main.py --cfg './roland_example.yaml' --repeat 1 --override_data_dir '/home/tianyudu/Data/roland_public_data' --override_remark 'roland_example'
nohup python main.py --cfg './wikipedia.yaml' --repeat 1 > res/wiki.log &
nohup python main.py --cfg './reddit.yaml' --repeat 1 > res/reddit.log &
nohup python main.py --cfg './mooc.yaml' --repeat 1 > res/mooc.log &
nohup python main.py --cfg './flights.yaml' --repeat 1 > res/flights.log &
nohup python main.py --cfg './ml25m.yaml' --repeat 1 > res/ml25m.log &
nohup python main.py --cfg './dgraphfin.yaml' --repeat 1 > res/dgraphfin.log &