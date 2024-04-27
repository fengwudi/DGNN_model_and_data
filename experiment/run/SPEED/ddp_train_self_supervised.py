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
        shuffle_parts=args.shuffle_parts, no_ind_val=args.no_ind_val
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
        top_k, static_shared_nodes, sync_mode, shuffle_parts, no_ind_val
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
    if testing_mode == "hybrid":
        RESULT_SAVE_PATH_from_val = f"results/from_val_{prefix}.json"
        PICKLE_SAVE_PATH_from_val = "results/from_val_{}.pkl".format(prefix)
        RESULT_SAVE_PATH_from_begin = f"results/from_begin_{prefix}.json"
        PICKLE_SAVE_PATH_from_begin = "results/from_begin_{}.pkl".format(prefix)

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

    # if rank == 0:  # only the first process logs and saves
    pathlib.Path("./saved_models/").mkdir(parents=True, exist_ok=True)
    ckpts_dir = pathlib.Path(f"./saved_checkpoints/{prefix}")
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("results/").mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda rank, epoch: ckpts_dir / f'rank{rank}-{epoch}.pth'

    sns_dir = pathlib.Path(f"./sub_nodes/{prefix}")
    sns_dir.mkdir(parents=True, exist_ok=True)
    get_sub_nodes_path = lambda rank, epoch: sns_dir / f'rank{rank}-{epoch}.pth'

    # init logger
    logger = get_logger(HASH) if rank == 0 or (rank == 1 and testing_mode == "hybrid") else DummyLogger()
    if rank == 0:
        logger.info(f'[START {HASH}]')
        logger.info(f'Model version: {MODEL_VERSION}')
        logger.info(", ".join([f"{k}={v}" for k, v in args.items()]))

        if (pathlib.Path(RESULT_SAVE_PATH).exists() or pathlib.Path(RESULT_SAVE_PATH_from_val).exists()) and not force:
            logger.info('Duplicate task! Abort!')
            return False

    if world_size > 1:
        dist.barrier()  # for single process should also work

    try:
        # Init
        seed_all(seed)
        # ============= Load Data ===========
        basic_data, graphs, dls = init_data(
            data, root, seed,
            rank=rank, world_size=world_size,
            num_workers=num_workers, bs=bs, warmup_steps=warmup_steps,
            subset=subset, strategy=strategy,
            n_layers=n_layers, n_neighbors=n_neighbors,
            restarter_type=restarter_type, hist_len=hist_len,
            part_exp=0
        )
        (
            nfeats, efeats, full_data, train_data, val_data, test_data,
            inductive_val_data, inductive_test_data
        ) = basic_data
        train_graph, full_graph = graphs
        (
            train_dl, offline_dl, val_dl, ind_val_dl,
            test_dl, ind_test_dl, val_warmup_dl, test_warmup_dl,
            test_train_dl
        ) = dls

        eval_dls = (val_warmup_dl, offline_dl, val_dl, ind_val_dl)

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

            full_sub_data, full_sub_graph, _, _, global_list_full = reconstruct_graph_dl(full_data, sub_nodes, shared_nodes, strategy, seed,
                                                                       n_neighbors, n_layers, restarter_type, hist_len,
                                                                       bs, pin_memory=True)
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

                full_sub_data, full_sub_graph, _, _, global_list_full= reconstruct_graph_dl(full_data, parts, shared_nodes, strategy, seed,
                                                                           n_neighbors, n_layers, restarter_type,
                                                                           hist_len, bs, pin_memory=True)

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
            ts_delta_mean, ts_delta_std, *_ = full_data.get_stats()
        else:
            raise ValueError("Check world_size and graph parts! You are using {} GPU(s),"
                             " but the graph have been divided to {} parts".format(world_size, len(divided_nodes)))

        # ============= Init Model ===========
        model = init_model_for_ddp(
            nfeats, efeats, train_graph, n_nodes.item(), n_edges.item(), ts_delta_mean, ts_delta_std, device,
            feature_as_buffer=feature_as_buffer, dim=dim,
            n_layers=n_layers, n_heads=n_heads, n_neighbors=n_neighbors,
            hit_type=hit_type, dropout=dropout,
            restarter_type=restarter_type, hist_len=hist_len,
            msg_src=msg_src, upd_src=upd_src,
            msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type,
            dyrep=dyrep, embedding_type=embedding_type, no_memory=no_memory, n_shared_nodes=n_shared_nodes,
            static_shared_nodes=static_shared_nodes
        )

        # recover training, this function will not be used for now.
        if rank == 0 and recover_from != '':
            model.load_state_dict(torch.load(recover_from, map_location=device))
            epoch_start = recover_step
        else:
            epoch_start = 0

        ddp_model = DDP(model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
        optimizer = optim.Adam(ddp_model.parameters(), lr=lr * np.sqrt(world_size))  # NB: larger lr

        val_aps = []
        ind_val_aps = []
        val_aucs = []
        ind_val_aucs = []
        val_recalls = []
        ind_val_recalls = []
        val_accs = []
        ind_val_accs = []
        val_f1s = []
        ind_val_f1s = []
        train_epoch_times = []
        total_epoch_times = []
        train_losses = []
        train_memorys = []

        early_stopper = EarlyStopMonitor(max_round=patience, epoch_start=epoch_start)
        signal = torch.tensor([1]).to(device)  # as a flag for multiprocessing
        done = False

        for epoch in range(epoch_start, n_epochs):
            dist.all_reduce(signal, op=dist.ReduceOp.MIN)
            if signal.item() == 0:  # finish training
                break

            if world_size != len(divided_nodes):
                parts_multi = int(len(divided_nodes) / world_size)
                # random shuffling
                if shuffle_parts:
                    np.random.seed(epoch+seed)
                    np.random.shuffle(divided_nodes)
                sub_nodes_lists = divided_nodes[rank * parts_multi: (rank * parts_multi + parts_multi)]
                sub_nodes = np.empty(0, dtype=np.int64)
                for nodes_list in sub_nodes_lists:
                    sub_nodes = np.concatenate([sub_nodes, nodes_list])

            train_sub_data, train_sub_graph, train_sub_dl, _, global_list_train = reconstruct_graph_dl(train_data, sub_nodes, shared_nodes,
                                                                                    strategy,
                                                                                    seed, n_neighbors, n_layers,
                                                                                    restarter_type, hist_len, bs,
                                                                                    pin_memory=True)

            offline_dl = None

            # Training
            model.reset()
            model.graph = train_sub_graph

            max_len = torch.tensor([len(train_sub_dl)]).to(device)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX)

            backup_mem = BackupMem()

            if rank == 0:
                logger.info('Start {} epoch'.format(epoch))

            m_contrast_loss, m_mutual_loss, m_loss, train_epoch_time, train_memory = \
                train_one_epoch(
                    ddp_model, model, optimizer, train_sub_dl, device, restart_prob,
                    mutual_coef, max_len.item(), world_size, backup_mem, backup_memory_cpu
                )

            train_epoch_times.append(train_epoch_time)
            train_memorys.append(train_memory)

            left_mem, right_mem, msg_store = backup_mem.data
            
            if backup_memory_cpu:
                model.left_memory.to('cpu'), model.right_memory.to('cpu'), model.msg_store.to('cpu'), model.upd_memory.to('cpu'), model.msg_memory.to('cpu')
                torch.cuda.empty_cache()
                left_mem.to(device), right_mem.to(device), msg_store.to(device)
                del backup_mem
                torch.cuda.empty_cache()

            if shared_nodes and not static_shared_nodes:
                sync_memory(sync_mode, left_mem, right_mem, n_shared_nodes, world_size, device)

            memory_state_train_end = (left_mem, right_mem, msg_store)
            model.load_memory_state(memory_state_train_end)

            # ============= Evaluate ===========
            full_data.eval = True
            val_data.eval = True
            inductive_val_data.eval = True
            val_sub_data, _ = val_data.get_subset_and_reindex_by_nodes(sub_nodes, shared_nodes)
            full_sub_data, _ = full_data.get_subset_and_reindex_by_nodes(sub_nodes, shared_nodes)

            full_sub_graph = Graph.from_data(full_sub_data, strategy=strategy, seed=seed)
            eval_sub_collator = GraphCollator(full_sub_graph, n_neighbors, n_layers,
                                              restarter=restarter_type, hist_len=hist_len)

            val_sub_dl = DataLoader(val_sub_data, batch_size=bs, collate_fn=eval_sub_collator)
            if no_ind_val:
                inductive_val_sub_data = None
                ind_val_sub_dl = None
            else:
                inductive_val_sub_data, _ = inductive_val_data.get_subset_and_reindex_by_nodes(sub_nodes, shared_nodes)
                ind_val_sub_dl = DataLoader(inductive_val_sub_data, batch_size=bs, collate_fn=eval_sub_collator)

            model.flush_msg()
            model.graph = full_sub_graph
            model.msg_store.clear()

            if warmup_steps:
                uptodate_nodes = warmup(model, val_warmup_dl, device)
            else:
                uptodate_nodes = set()

            val_ap, val_auc, ind_val_ap, ind_val_auc, eval_epoch_time, val_recall, val_acc, val_f1, ind_val_recall, ind_val_acc, ind_val_f1 = evaluate(
                model, eval_dls, val_sub_dl, ind_val_sub_dl, device, restart_prob, warmup_steps, uptodate_nodes, subset,
                backup_memory_cpu, world_size, n_shared_nodes, static_shared_nodes, sync_mode, no_ind_val)

            total_epoch_time = train_epoch_time + eval_epoch_time
            total_epoch_times.append(total_epoch_time)

            val_ap, val_auc, val_recall, val_acc, val_f1 = torch.tensor([val_ap]).to(device), torch.tensor([val_auc]).to(device), torch.tensor([val_recall]).to(device), torch.tensor([val_acc]).to(device), torch.tensor([val_f1]).to(device)

            dist.all_reduce(val_ap)
            dist.all_reduce(val_auc)
            dist.all_reduce(val_recall)
            dist.all_reduce(val_acc)
            dist.all_reduce(val_f1)

            val_ap, val_auc, val_recall, val_acc, val_f1 = val_ap.item() / world_size, val_auc.item() / world_size, val_recall.item() / world_size, val_acc.item() / world_size, val_f1.item() / world_size

            if not no_ind_val:
                ind_val_ap, ind_val_auc, ind_val_recall, ind_val_acc, ind_val_f1 = torch.tensor([ind_val_ap]).to(device), torch.tensor([ind_val_auc]).to(device), torch.tensor([ind_val_recall]).to(device), torch.tensor([ind_val_acc]).to(device), torch.tensor([ind_val_f1]).to(device)

                dist.all_reduce(ind_val_ap)
                dist.all_reduce(ind_val_auc)
                dist.all_reduce(ind_val_recall)
                dist.all_reduce(ind_val_acc)
                dist.all_reduce(ind_val_f1)

                ind_val_ap, ind_val_auc, ind_val_recall, ind_val_acc, ind_val_f1c = ind_val_ap.item() / world_size, ind_val_auc.item() / world_size, ind_val_recall.item() / world_size, ind_val_acc.item() / world_size, ind_val_f1c.item() / world_size

            # save
            model.flush_msg()
            if shared_nodes and not static_shared_nodes:
                sync_memory(sync_mode, model.left_memory, model.right_memory, n_shared_nodes, world_size, device)
            torch.save(model.state_dict(), get_checkpoint_path(rank, epoch))
            global_list_train = global_list_train.astype(int)
            torch.save(global_list_train, get_sub_nodes_path(rank, epoch))

            if rank == 0:
                logger.info('Epoch {:4d} total    took  {:.2f}s'.format(epoch, total_epoch_time))
                logger.info('Epoch {:4d} training took  {:.2f}s'.format(epoch, train_epoch_time))
                logger.info('Epoch {:4d} training took  memory {:.2f}MB'.format(epoch, train_memory))
                logger.info(f'Epoch mean    total loss: {m_loss:.4f}')
                logger.info(f'Epoch mean contrast loss: {m_contrast_loss:.4f}')
                logger.info(f'Epoch mean   mutual loss: {m_mutual_loss:.4f}')
                logger.info(f'Val     ap: {val_ap:.4f}, Val     auc: {val_auc:.4f}')
                if not no_ind_val:
                    logger.info(f'Val ind ap: {ind_val_ap:.4f}, Val ind auc: {ind_val_auc:.4f}')

            val_aps.append(val_ap)
            ind_val_aps.append(ind_val_ap)
            val_aucs.append(val_auc)
            ind_val_aucs.append(ind_val_auc)
            val_recalls.append(val_recall)
            ind_val_recalls.append(ind_val_recall)
            val_accs.append(val_acc)
            ind_val_accs.append(ind_val_acc)
            val_f1s.append(val_f1)
            ind_val_f1s.append(ind_val_f1)
            train_losses.append(m_loss)

            if early_stopper.early_stop_check(val_ap):
                logger.info('No improvement over {} epochs, stop training'.format(
                    early_stopper.max_round))
                done = True

            if epoch == n_epochs-1:
                done = True

            dist.barrier()  # wait

            if done:
                # if rank == 0:  # send signal to all processes
                dist.all_reduce(torch.tensor([0]).to(device), op=dist.ReduceOp.MIN)

                buffer_list = ['left_memory.vals', 'left_memory.update_ts', 'left_memory.active_mask',
                                'right_memory.vals', 'right_memory.update_ts', 'right_memory.active_mask',
                                'msg_memory.vals', 'msg_memory.update_ts', 'msg_memory.active_mask',
                                'upd_memory.vals', 'upd_memory.update_ts', 'upd_memory.active_mask']
                
                if rank == 0 or (testing_mode == "hybrid" and rank == 1):
                    for r in range(world_size):
                        best_model_path = get_checkpoint_path(r, early_stopper.best_epoch)
                        model_state_sep = torch.load(best_model_path)

                        best_model_sub_nodes = get_sub_nodes_path(r, early_stopper.best_epoch)
                        epoch_sub_nodes = torch.load(best_model_sub_nodes)
                        # rewrite state_dict
                        if r == 0:
                            model_state_rewritten = rewrite_state_dict(None, model_state_sep, full_graph.num_node, shared_nodes, epoch_sub_nodes, buffer_list, testing_mode, rank)
                        else:
                            model_state_rewritten = rewrite_state_dict(model_state_rewritten, model_state_sep, full_graph.num_node, shared_nodes, epoch_sub_nodes, buffer_list, testing_mode, rank)
                    torch.save(model_state_rewritten, get_checkpoint_path(rank, 'updated_buffer'))
                break

            # ============= Evaluate END ================

        if rank > 1:
            # only main process evals test data
            # other processes can safely exit now
            return True
        if testing_mode != "hybrid" and rank == 1:
            return True

        if rank == 0 or 1:
            best_epoch = early_stopper.best_epoch - epoch_start
            best_val_ap = val_aps[best_epoch]
            best_val_auc = val_aucs[best_epoch]
            best_ind_val_ap = ind_val_aps[best_epoch]
            best_ind_val_auc = ind_val_aucs[best_epoch]
            max_memory_allocated = torch.cuda.max_memory_allocated() / 1e6
            max_memory_reserved = torch.cuda.max_memory_reserved() / 1e6
        if rank == 0:
            # Testing
            logger.info(f'[ Train] Max Memory Allocated: {max_memory_allocated:.2f} MB Max Memory Reserved: {max_memory_reserved:.2f} MB on Main process')
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            logger.info(f'[ Val] Best     ap: {best_val_ap:.4f} Best     auc: {best_val_auc:.4f}')
            if not no_ind_val:
                logger.info(f'[ Val] Best ind ap: {best_ind_val_ap:.4f} Best ind auc: {best_ind_val_auc:.4f}')
            logger.info('Average training time: {:.2f} s'.format(np.mean(train_epoch_times)))
            logger.info('Average total time: {:.2f} s'.format(np.mean(total_epoch_times)))
            logger.info('Average memory usage: {:.2f} MB'.format(np.mean(train_memorys)))

        if rank == 0 or 1:
            del model
            torch.cuda.empty_cache()

            # ============= Init Testing Model ===========
            if testing_on_cpu: device = "cpu"

            model = init_model(
                nfeats, efeats, train_graph, full_graph, full_data, device,
                feature_as_buffer=feature_as_buffer, dim=dim,
                n_layers=n_layers, n_heads=n_heads, n_neighbors=n_neighbors,
                hit_type=hit_type, dropout=dropout,
                restarter_type=restarter_type, hist_len=hist_len,
                msg_src=msg_src, upd_src=upd_src,
                msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type,
                dyrep=dyrep, embedding_type=embedding_type, no_memory=no_memory,
                n_shared_nodes=0
            )
            # ============= Testing ===========
            model_state_path = get_checkpoint_path(rank, 'updated_buffer')
            model_state = torch.load(model_state_path)
            model.load_state_dict(model_state, strict=False)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # save to the model save folder

            model.eval()
            model.graph = full_graph
            if restart_mode:
                model.msg_store.clear()
                if warmup_steps:
                    uptodate_nodes = warmup(model, test_warmup_dl, device)
                else:
                    uptodate_nodes = set()
            
            start_test_time = time.time()

            if (testing_mode == "from_begin" or testing_mode == "hybrid") and rank == 0:
                model.reset()
                _, _, _, _, _ = eval_edge_prediction(model, test_train_dl, device, restart_mode, uptodate_nodes=uptodate_nodes)
                _, _, _, _, _ = eval_edge_prediction(model, val_dl, device, restart_mode, uptodate_nodes=uptodate_nodes)

            memory_state_val_end = model.save_memory_state()  # save states at t_valid_end

            test_ap, test_auc, test_recall, test_acc, test_f1 = eval_edge_prediction(
                model, test_dl, device, restart_mode,
                uptodate_nodes=uptodate_nodes.copy()
            )  # memory modified
            model.load_memory_state(memory_state_val_end)  # load states at t_valid_end
            ind_test_ap, ind_test_auc, ind_test_recall, ind_test_acc, ind_test_f1 = eval_edge_prediction(
                model, ind_test_dl, device, restart_mode,
                uptodate_nodes=uptodate_nodes.copy()
            )

            test_time = time.time() - start_test_time

            if testing_mode == "hybrid" and rank == 0:
                logger.info('Test_from_begin took  {:.2f}s'.format(test_time))
                logger.info(f'[Test_from_begin]     ap: {test_ap:.4f}     auc: {test_auc:.4f}     recall: {test_recall:.4f}     acc: {test_acc:.4f}     f1: {test_f1:.4f}')
                logger.info(f'[Test_from_begin] ind ap: {ind_test_ap:.4f} ind auc: {ind_test_auc:.4f}     recall: {ind_test_recall:.4f}     acc: {ind_test_acc:.4f}     acc: {ind_test_f1:.4f}')
            elif testing_mode == "hybrid" and rank == 1:
                logger.info('Test_from_val took  {:.2f}s'.format(test_time))
                logger.info(f'[Test_from_val]     ap: {test_ap:.4f}     auc: {test_auc:.4f}')
                logger.info(f'[Test_from_val] ind ap: {ind_test_ap:.4f} ind auc: {ind_test_auc:.4f}')
            elif testing_mode != "hybrid":
                logger.info('Test took  {:.2f}s'.format(test_time))
                logger.info(f'[Test_{testing_mode}]     ap: {test_ap:.4f}     auc: {test_auc:.4f}')
                logger.info(f'[Test_{testing_mode}] ind ap: {ind_test_ap:.4f} ind auc: {ind_test_auc:.4f}')

            # Save results for this run
            if rank == 0 and testing_mode == "hybrid":
                with open(PICKLE_SAVE_PATH, "rb") as f:
                    results = pickle.load(f)
                    results.update({
                        "test_ap_from_begin": test_ap,
                        "ind_test_ap_from_begin": ind_test_ap,
                        "test_auc_from_begin": test_auc,
                        "ind_test_auc_from_begin": ind_test_auc,
                        "test_time_from_begin": test_time})
                with open(PICKLE_SAVE_PATH, "wb") as f:
                    pickle.dump(results, f)
            elif rank == 1 and testing_mode == "hybrid":
                pickle.dump({
                    "rank": rank,
                    "testing_mode": testing_mode,
                    "val_aps": val_aps,
                    "val_aucs": val_aucs,
                    "ind_val_aps": ind_val_aps,
                    "ind_val_aucs": ind_val_aucs,
                    "test_ap_from_val": test_ap,
                    "ind_test_ap_from_val": ind_test_ap,
                    "test_auc_from_val": test_auc,
                    "ind_test_auc_from_val": ind_test_auc,
                    "epoch_times": train_epoch_times,
                    "train_losses": train_losses,
                    "total_epoch_times": total_epoch_times,
                    "test_time_from_val": test_time
                }, open(PICKLE_SAVE_PATH, "wb"))
            else:
                pickle.dump({
                    "testing_mode": testing_mode,
                    "val_aps": val_aps,
                    "val_aucs": val_aucs,
                    "ind_val_aps": ind_val_aps,
                    "ind_val_aucs": ind_val_aucs,
                    f"test_ap_{testing_mode}": test_ap,
                    f"ind_test_ap_{testing_mode}": ind_test_ap,
                    f"test_auc_{testing_mode}": test_auc,
                    f"ind_test_auc_{testing_mode}": ind_test_auc,
                    "epoch_times": train_epoch_times,
                    "train_losses": train_losses,
                    "total_epoch_times": total_epoch_times,
                    f"test_time_{testing_mode}": test_time
                }, open(PICKLE_SAVE_PATH, "wb"))
            
            if rank == 0 and testing_mode == "hybrid":
                with open(RESULT_SAVE_PATH, 'r') as f:
                    results = json.load(f)
                    results.update(
                        test_ap_from_begin=test_ap, test_auc_from_begin=test_auc,
                        ind_test_ap_from_begin=ind_test_ap, ind_test_auc_from_begin=ind_test_auc,
                        test_time_from_begin=test_time,
                        max_memory_allocated=max_memory_allocated, max_memory_reserved=max_memory_reserved,
                    )
                with open(RESULT_SAVE_PATH, 'w') as f:
                    json.dump(results, f)
            elif rank == 1 and testing_mode == "hybrid":
                results = args.copy()
                results.update(
                    HASH=HASH,
                    VERSION=MODEL_VERSION,
                    rank=rank, testing_mode=testing_mode,
                    average_training_time=np.mean(train_epoch_times),
                    average_total_time=np.mean(total_epoch_times),
                    val_ap=best_val_ap, ind_val_ap=best_ind_val_ap,
                    val_auc=best_val_auc, ind_val_auc=best_ind_val_auc,
                    test_ap_from_val=test_ap, test_auc_from_val=test_auc,
                    ind_test_ap_from_val=ind_test_ap, ind_test_auc_from_val=ind_test_auc,
                    test_time_from_val=test_time
                )
                json.dump(results, open(RESULT_SAVE_PATH, 'w'))
            else:
                results = args.copy()
                results.update(
                    HASH=HASH,
                    VERSION=MODEL_VERSION,
                    rank=rank, testing_mode=testing_mode,
                    val_ap=best_val_ap, ind_val_ap=best_ind_val_ap,
                    val_auc=best_val_auc, ind_val_auc=best_ind_val_auc,
                    test_ap=test_ap, test_auc=test_auc,
                    ind_test=ind_test_ap, ind_test_auc=ind_test_auc,
                    test_time=test_time
                )
                json.dump(results, open(RESULT_SAVE_PATH, 'w'))
            
            # remove all ckpts
            if rank == 0:
                shutil.rmtree(ckpts_dir)
                shutil.rmtree(sns_dir)
            return True

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
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=200, help='Batch size')
    # MISC
    parser.add_argument('--force', action='store_true', help='Overwirte the existing task')
    parser.add_argument('--recover_from', type=str, default='', help='ckpt path')
    parser.add_argument('--recover_step', type=int, default=0, help='recover step')
    # Data Distribute
    parser.add_argument('--part_exp', type=int, default=0, help='Partition graph into 2^k parts')
    parser.add_argument('--divide_method', type=str, default='pre', 
                        choices=["pre", "buildin_kl", "pre_kl", "random"], help='methods used for dividing, '
                                                                                'do not use the buildin_kl')
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

    args = parser.parse_args()
    process_args_presets(args)
    return args


if __name__ == '__main__':
    args = get_args()
    gpus = args.gpu.split(',')

    # Sanity Check
    #if len(gpus) <= 1 or '-1' in gpus:
    #    raise ValueError('Please use the single-gpu train file instead!')

    WORLD_SIZE = len(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    mp.spawn(worker, nprocs=WORLD_SIZE, args=(WORLD_SIZE, args), join=True, daemon=True)
