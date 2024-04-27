import argparse
import json
import os
import math
import pathlib
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from CHANGELOG import MODEL_VERSION
from tiger.data.data_loader import ChunkSampler, GraphCollator, load_jodie_data
from tiger.data.graph import Graph
from tiger.eval_utils import eval_edge_prediction, warmup
from tiger.model.feature_getter import NumericalFeature
from tiger.model.tiger import TIGER
from tiger.utils import BackgroundThreadGenerator
from tiger.model.restarters import SeqRestarter, StaticRestarter, WalkRestarter
from tiger.eval_utils import eval_edge_prediction, warmup

from init_utils import init_data, init_model, init_parser, process_args_presets, init_model_for_ddp
from train_utils import EarlyStopMonitor, hash_args, seed_all, get_logger, dist_setup, dist_cleanup, DummyLogger

from ddp_train_utils import train_one_epoch, evaluate, divide_nodes, infinite_loop, rewrite_state_dict, \
    reconstruct_graph_dl, divided_nodes_from_txt, BackupMem, sync_memory

import sys


def worker(rank, world_size, args):
    dist_setup(rank, world_size)

    run(rank, world_size, prefix=args.prefix, root=args.root, data=args.data,
        dim=args.dim, feature_as_buffer=not args.no_feat_buffer,
        gpu=args.gpu, seed=args.seed, num_workers=args.num_workers, subset=args.subset,
        hit_type=args.hit_type, restarter_type=args.restarter_type,
        hist_len=args.hist_len,
        n_neighbors=args.n_neighbors,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
        strategy=args.strategy, msg_src=args.msg_src, upd_src=args.upd_src,
        mem_update_type=args.upd_fn, msg_tsfm_type=args.tsfm_fn,
        lr=args.lr, n_epochs=args.n_epochs, bs=args.bs,
        mutual_coef=args.mutual_coef, patience=args.patience,
        restart_prob=args.restart_prob, part_exp=args.part_exp,
        recover_from=args.recover_from, recover_step=args.recover_step,
        force=args.force, warmup_steps=args.warmup,
        dyrep=args.dyrep, embedding_type=args.embedding_type, no_memory=args.no_memory,
        divide_method=args.divide_method, testing_mode=args.testing_mode,
        testing_on_cpu=args.testing_on_cpu, backup_memory_cpu=args.backup_memory_to_cpu,
        top_k=args.top_k, static_shared_nodes=args.static_shared_nodes, sync_mode=args.sync_mode,
        shuffle_parts=args.shuffle_parts, no_ind_val=args.no_ind_val, save_mode=args.save_mode
        )

    dist_cleanup()


def run(rank, world_size, *, prefix,
        root, data, dim, feature_as_buffer,
        gpu, seed, num_workers, subset,
        hit_type, restarter_type, hist_len,
        n_neighbors, n_layers, n_heads, dropout,
        strategy, msg_src, upd_src,
        mem_update_type, msg_tsfm_type,
        lr, n_epochs, bs, mutual_coef, patience,
        restart_prob,
        part_exp,
        recover_from, recover_step, force, warmup_steps,
        dyrep, embedding_type, no_memory, divide_method,
        testing_mode, testing_on_cpu, backup_memory_cpu,
        top_k, static_shared_nodes, sync_mode, shuffle_parts, no_ind_val,
        save_mode
        ):
    # Get hash
    args = {k: v for k, v in locals().items()
            if not k in {'gpu', 'force', 'rank', 'recover_from', 'recover_step'}}
    HASH = hash_args(**args, MODEL_VERSION=MODEL_VERSION)
    prefix = HASH if prefix == '' else f'{prefix}.{HASH}'
    device = torch.device(f'cuda:{rank}')

    restart_mode = restart_prob > 0

    # Sanity check
    if (not restart_mode) and (warmup_steps > 0):
        raise ValueError('Warmup is not needed without restart.')

    # Path
    MODEL_SAVE_PATH = f'./saved_models/{prefix}.pth'
    RESULT_SAVE_PATH = f"results/{prefix}.json"
    PICKLE_SAVE_PATH = "results/{}.pkl".format(prefix)

    if divide_method == 'pre':
        DIVIDED_NODES_PATHs = []
        for i in range(2**part_exp):
            DIVIDED_NODES_PATH = f'./partition/divided_nodes_seed/{data}/{seed}/{data}_{2**part_exp}parts_top{top_k}/output{i}.txt'
            DIVIDED_NODES_PATHs.append(DIVIDED_NODES_PATH)
        if top_k == 0:
            SHARED_NODES_PATH = None
        else:
            SHARED_NODES_PATH = f'./partition/divided_nodes_seed/{data}/{seed}/{data}_{2**part_exp}parts_top{top_k}/outputshared.txt'
    elif divide_method == 'pre_kl':
        DIVIDED_NODES_PATHs = []
        for i in range(2**part_exp):
            DIVIDED_NODES_PATH = f'./partition/divided_nodes_seed/{data}_kl/{seed}/{data}_{2**part_exp}parts/output{i}.txt'
            DIVIDED_NODES_PATHs.append(DIVIDED_NODES_PATH)
    elif divide_method == 'random':
        DIVIDED_NODES_PATHs = []
        for i in range(2**part_exp):
            DIVIDED_NODES_PATH = f'./partition/divided_nodes_seed/{data}_random/{seed}/{data}_{2**part_exp}parts/output{i}.txt'
            DIVIDED_NODES_PATHs.append(DIVIDED_NODES_PATH)

    pathlib.Path("graph_list/").mkdir(parents=True, exist_ok=True)

    pathlib.Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    ckpts_dir = pathlib.Path(f"./saved_checkpoints/{prefix}")
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("results/").mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda epoch: ckpts_dir / f'rank{rank}-{epoch}.pth'

    sns_dir = pathlib.Path(f"./sub_nodes/{prefix}")
    sns_dir.mkdir(parents=True, exist_ok=True)
    get_sub_nodes_path = lambda epoch: sns_dir / f'rank{rank}-{epoch}.pth'

    # init logger
    logger = get_logger(HASH) if rank == 0 else DummyLogger()
    if rank == 0:
        logger.info(f'[START {HASH}]')
        logger.info(f'Model version: {MODEL_VERSION}')
        logger.info(", ".join([f"{k}={v}" for k, v in args.items()]))

        if pathlib.Path(RESULT_SAVE_PATH).exists() and not force:
            logger.info('Duplicate task! Abort!')
            return False

    if world_size > 1:
        dist.barrier()  # for single process should also work

    save_mode = "save"

    try:
        # Init
        seed_all(seed)
        # ============= Load Data ===========
        (
            nfeats, efeats, full_data, train_data, val_data, test_data,
            inductive_val_data, inductive_test_data
        ) = load_jodie_data(data, train_seed=seed, root=root)


        n_parts=2**part_exp

        if world_size > 1:
            dist.barrier()  # for single process should also work

        # ============= Divide graph and Init max memory ===========
        if divide_method == "pre":
            divided_nodes, shared_nodes = divided_nodes_from_txt(DIVIDED_NODES_PATHs, SHARED_NODES_PATH)
        elif divide_method == "buildin_kl":
            divided_nodes, shared_nodes = divide_nodes(train_data, full_data.num_node, world_size, seed, full_data, k=part_exp)
        elif divide_method == "pre_kl" or divide_method == "random":
            divided_nodes, shared_nodes = divided_nodes_from_txt(DIVIDED_NODES_PATHs, None)
        n_shared_nodes = len(shared_nodes) if shared_nodes is not None else 0

        if world_size == len(divided_nodes):
            sub_nodes = divided_nodes[rank]

            init_data_time = time.time()
            full_sub_data, full_sub_graph, _, _, _ = reconstruct_graph_dl(full_data, sub_nodes, shared_nodes, strategy, seed,
                                                                       n_neighbors, n_layers, restarter_type, hist_len, bs,
                                                                       save_mode, data, divide_method, 2**part_exp, top_k,
                                                                       "full_sub_graph", rank, pin_memory=True)
            logger.info(f'[I/O TIME] Init full_sub_graphs use {(time.time()-init_data_time):.2f}')

            n_nodes = torch.tensor([full_sub_graph.num_node + n_shared_nodes + 2]).to(device)
            dist.all_reduce(n_nodes, op=dist.ReduceOp.MAX)
            ts_delta_mean, ts_delta_std, *_ = full_data.get_stats()
            n_edges = torch.tensor([len(full_sub_data)]).to(device)
            dist.all_reduce(n_edges, op=dist.ReduceOp.MAX)
        elif world_size > len(divided_nodes):
            raise ValueError("Divide graph to more sub-graphs or Use less GPUs, you are using {} GPU(s), "
                             "but only divide graph to {} parts".format(world_size, len(divided_nodes)))
        elif world_size < len(divided_nodes) and len(divided_nodes) % world_size == 0:
            parts_multi = int(len(divided_nodes) / world_size)
            n_nodes = 0
            n_edges = 0
            for parts in divided_nodes:
                init_data_time = time.time()
                full_sub_data, full_sub_graph, _, _, _ = reconstruct_graph_dl(full_data, parts, shared_nodes, strategy, seed,
                                                                           n_neighbors, n_layers, restarter_type, hist_len, bs,
                                                                           save_mode, data, divide_method, 2**part_exp, top_k,
                                                                           "full_sub_graph", rank, pin_memory=True)
                logger.info(f'[I/O TIME] Init full_sub_graphs use {(time.time()-init_data_time):.2f}')

                n_sub_nodes = full_sub_graph.num_node
                n_sub_edges = len(full_sub_data)
                if n_sub_nodes > n_nodes:
                    n_nodes = n_sub_nodes
                if n_sub_edges > n_edges:
                    n_edges = n_sub_edges
            n_nodes *= parts_multi
            n_edges *= parts_multi
            n_nodes = torch.tensor([n_nodes]).to(device)
            n_edges = torch.tensor([n_edges]).to(device)
        else:
            raise ValueError("Check world_size and graph parts! You are using {} GPU(s),"
                             " but the graph have been divided to {} parts".format(world_size, len(divided_nodes)))
        
        init_data_time = time.time()
        _, train_sub_graph, train_sub_dl, _, _ = reconstruct_graph_dl(train_data, sub_nodes, shared_nodes, strategy, seed,
                                                                   n_neighbors, n_layers, restarter_type, hist_len, bs,
                                                                   save_mode, data, divide_method, 2**part_exp, top_k,
                                                                   "train_sub_graph", rank, pin_memory=True)
        logger.info(f'[I/O TIME] Init train_sub_graphs use {(time.time()-init_data_time):.2f}')

        if rank == 0 and top_k == 0:
            init_data_time = time.time()
            full_graph = Graph.from_npy(save_mode, full_data, data, seed, divide_method, n_parts, "for_all", "full_graph", "for_all", strategy=strategy)
            logger.info(f'[I/O TIME] Init full_graph uses {(time.time()-init_data_time):.2f}')
            init_data_time = time.time()
            train_graph = Graph.from_npy(save_mode, train_data, data, seed, divide_method, n_parts, "for_all", "train_graph", "for_all", strategy=strategy)
            logger.info(f'[I/O TIME] Init train_graph uses {(time.time()-init_data_time):.2f}')

        if save_mode == "save":
            sys.exit(0)

    except Exception as e:
        if rank == 0:
            logger.error(traceback.format_exc())
            logger.error(e)
        raise



def get_args():
    parser = init_parser()
    # Exp Setting
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # DDP
    parser.add_argument('--gpu', type=str, default='0', help='Cuda index')
    parser.add_argument('--port', type=str, default='29500', help='port for DDP')
    # Data
    parser.add_argument('--subset', type=float, default=1.0, help='Only use a subset of training data')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers in train dataloader')
    # Training
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=200, help='Batch size')
    # MISC
    parser.add_argument('--force', action='store_true', help='Overwirte the existing task')
    parser.add_argument('--recover_from', type=str, default='', help='ckpt path')
    parser.add_argument('--recover_step', type=int, default=0, help='recover step')
    # Data Distribute
    parser.add_argument('--part_exp', type=int, default=0, help='Partition graph into 2^k parts')
    parser.add_argument('--divide_method', type=str, default='pre', 
                        choices=["pre", "buildin_kl", "pre_kl", "random"], help='methods used for dividing')
    parser.add_argument('--testing_mode', type=str, default='hybrid', 
                        choices=["from_begin", "from_val", "hybrid"], 
                        help='The memory used in testing, copy from the end of val or rerun the train and val')
    parser.add_argument('--testing_on_cpu', action='store_true', help='testing run on cpu')
    parser.add_argument('--backup_memory_to_cpu', action='store_true', help='backup memory to cpu')
    parser.add_argument('--top_k', type=int, default=0, help='Use top k shared nodes')
    parser.add_argument('--static_shared_nodes', action='store_true', help='shared nodes memory is static')
    parser.add_argument('--sync_mode', type=str, default='none', 
                        choices=["none", "average", "last"], help='methods of sync memories between gpus')
    parser.add_argument('--shuffle_parts', action='store_true', help='when ws != parts, shuffle every epoch or not')
    parser.add_argument('--no_ind_val', action='store_true', help='skip inductive validation process')
    parser.add_argument('--save_mode', type=str, default='none', 
                    choices=["none", "read", "save"], help='save mode')

    args = parser.parse_args()
    process_args_presets(args)
    return args


if __name__ == '__main__':
    args = get_args()
    gpus = args.gpu.split(',')

    # Sanity Check
    if len(gpus) <= 1 or '-1' in gpus:
        raise ValueError('Please use the single-gpu train file instead!')

    WORLD_SIZE = len(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    mp.spawn(worker, nprocs=WORLD_SIZE, args=(WORLD_SIZE, args), join=True, daemon=True)