#!bin/bash
export NCCL_IB_DISABLE=1
nohup python -m torch.distributed.run --nproc_per_node=4 train.py --data WIKI --pbar --group 4 --batchsize 4000 > wiki.log 2>&1 &