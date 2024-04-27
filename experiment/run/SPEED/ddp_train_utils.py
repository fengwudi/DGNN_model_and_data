import math
import sys

from tqdm import tqdm
import torch

from ddp_utils import reduce_value, is_main_process
import time
import numpy as np
import torch
from tiger.eval_utils import eval_edge_prediction, warmup
from tiger.utils import BackgroundThreadGenerator
from torch import distributed as dist

from tiger.data.data_loader import ChunkSampler, GraphCollator, load_jodie_data
from tiger.data.graph import Graph

from torch.utils.data import DataLoader


class BackupMem:
    def __init__(self) -> None:
        self.data = None


def train_one_epoch(ddp_model, model, optimizer, train_dl, device, restart_prob,
                    mutual_coef, max_len, world_size, backup_mem,
                    backup_memory_cpu):
    start_epoch_t0 = time.time()

    uptodate_nodes = set()
    
    ddp_model.train()

    m_contrast_loss = torch.zeros(1).to(device)
    m_mutual_loss = torch.zeros(1).to(device)
    m_loss = torch.zeros(1).to(device)
    train_memory = 0

    restarting = False

    # add flags for the current GPU and global signal
    flag_world = torch.zeros(1).to(device)
    flag_self = torch.zeros(1).to(device)

    # init a generator
    infinite_generator = infinite_loop(train_dl, max_len)

    i_batch = 0
    while True:
        dist.barrier()
        dist.all_reduce(flag_world, op=dist.ReduceOp.MIN)

        if flag_world.item() == 1:
            break
        else:
            flag_world = torch.clone(flag_self)

        (src_ids, dst_ids, neg_dst_ids, ts, eids, _, comp_graph), loop_start, loop_end = \
            next(infinite_generator)

        if loop_start:
            ddp_model.module.reset()

        src_ids = src_ids.long().to(device)
        dst_ids = dst_ids.long().to(device)
        neg_dst_ids = neg_dst_ids.long().to(device)
        ts = ts.float().to(device)
        eids = eids.long().to(device)
        comp_graph.to(device)
        optimizer.zero_grad()

        # Restart
        if np.random.rand() < restart_prob and i_batch:
            restarting = True
            uptodate_nodes = set()
            model.msg_store.clear()

        if restarting:  # in lazy mode
            involved_nodes = comp_graph.np_computation_graph_nodes
            restart_nodes = set(involved_nodes) - set(uptodate_nodes)
            r_nids = torch.tensor(list(restart_nodes)).long().to(device)
            model.restart(r_nids, torch.full((len(r_nids),), ts.min().item()).to(device))
            uptodate_nodes.update(restart_nodes)

        # compute losses
        contrast_loss, mutual_loss = ddp_model(
            src_ids, dst_ids, neg_dst_ids, ts, eids, comp_graph,
            contrast_only=(restart_prob == 0)
        )
        # backpropagation & optimize
        loss = contrast_loss + mutual_coef * mutual_loss
        loss.backward()
        optimizer.step()
        train_memory += torch.cuda.max_memory_allocated() / 1e6

        if loop_end:
            if backup_memory_cpu:
                torch.cuda.empty_cache()
                backup_mem.data = ddp_model.module.backup_to_cpu()
            else:
                backup_mem.data = ddp_model.module.save_memory_state()

            flag_self = torch.ones(1).to(device)

        m_contrast_loss = (m_contrast_loss * i_batch + contrast_loss.detach()) / (i_batch + 1)
        m_mutual_loss = (m_mutual_loss * i_batch + mutual_loss.detach()) / (i_batch + 1)
        m_loss = (m_loss * i_batch + loss.detach()) / (i_batch + 1)
        i_batch += 1

        # exchange info. between cards
        dist.all_reduce(m_loss)
        dist.all_reduce(m_contrast_loss)
        dist.all_reduce(m_mutual_loss)
        m_contrast_loss = (m_contrast_loss.item() / world_size)
        m_mutual_loss = (m_mutual_loss.item() / world_size)
        m_loss = (m_loss.item() / world_size)

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    train_memory /= i_batch
    train_epoch_time = time.time() - start_epoch_t0
    return m_contrast_loss, m_mutual_loss, m_loss, train_epoch_time, train_memory


@torch.no_grad()
def evaluate(model, eval_dls, val_dl, ind_val_dl, device, restart_prob, warmup_steps, uptodate_nodes, subset,
             backup_memory_cpu, world_size, n_shared_nodes, static_shared_nodes, sync_mode, no_ind_val):
    start_eval_t0 = time.time()
    model.eval()

    val_warmup_dl, offline_dl, _, _ = eval_dls

    restart_mode = restart_prob > 0

    if restart_mode:
        model.msg_store.clear()
        if warmup_steps:
            uptodate_nodes = warmup(model, val_warmup_dl, device)
        else:
            uptodate_nodes = set()
    elif subset < 1.0:
        _ = eval_edge_prediction(model, offline_dl, device, restart_mode=False)

    if not no_ind_val:
        if backup_memory_cpu:
            memory_state_train_end_cpu = BackupMem()
            memory_state_train_end_cpu.data = model.backup_to_cpu()
        else:
            memory_state_train_end = model.save_memory_state()

    # ============= Transductive Evaluate ================
    val_ap, val_auc, val_recall, val_acc, val_f1 = eval_edge_prediction(
        model, val_dl, device, restart_mode, uptodate_nodes=uptodate_nodes.copy()
    )  # memory modified
    # ============= Transductive Evaluate ================

    if no_ind_val:
        if n_shared_nodes != 0 and not static_shared_nodes:
            sync_memory(sync_mode, model.left_memory, model.right_memory, n_shared_nodes, world_size, device)

        ind_val_ap = None
        ind_val_auc = None
        ind_val_recall = None
        ind_val_acc = None
        ind_val_f1 = None
    else:
        if backup_memory_cpu:
            memory_state_valid_end = BackupMem()
            memory_state_valid_end.data = model.backup_to_cpu()
            ve_left_mem, ve_right_mem, ve_msg_store = memory_state_valid_end.data

            (te_left_mem, te_right_mem, te_msg_store) = memory_state_train_end_cpu.data
            memory_state_train_end = (te_left_mem.to(device), te_right_mem.to(device), te_msg_store.to(device))
        else:
            memory_state_valid_end = model.save_memory_state()  # save states at t_valid_end

        # ============= Inductive Evaluate ================
        model.load_memory_state(memory_state_train_end)  # load states at t_train_end
        ind_val_ap, ind_val_auc, ind_val_recall, ind_val_acc, ind_val_f1 = eval_edge_prediction(
            model, ind_val_dl, device, restart_mode, uptodate_nodes=uptodate_nodes.copy()
        )
        # ============= Transductive Evaluate ================

        del memory_state_train_end
        torch.cuda.empty_cache()

        if backup_memory_cpu:
            if n_shared_nodes != 0 and not static_shared_nodes:
                left_mem, right_mem = (ve_left_mem.to(device), ve_right_mem.to(device))
                sync_memory(sync_mode, left_mem, right_mem, n_shared_nodes, world_size, device)

                memory_state_valid_end_gpu = (left_mem, right_mem, ve_msg_store.to(device))
            else:
                memory_state_valid_end_gpu = (ve_left_mem.to(device), ve_right_mem.to(device), ve_msg_store.to(device))

            model.load_memory_state(memory_state_valid_end_gpu)
        else:
            if n_shared_nodes != 0 and not static_shared_nodes:
                left_mem, right_mem, _ = memory_state_valid_end
                sync_memory(sync_mode, left_mem, right_mem, n_shared_nodes, world_size, device)

            model.load_memory_state(memory_state_valid_end)

    eval_epoch_time = time.time() - start_eval_t0

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return val_ap, val_auc, ind_val_ap, ind_val_auc, eval_epoch_time, val_recall, val_acc, val_f1, ind_val_f1, ind_val_recall, ind_val_acc


# Divide nodes to lists
def divide_nodes(train_data, n_nodes, world_size, seed, full_data, k=1):
    # train_size = len(train_data)
    from tiger.partition.kl_partition import partition_data  # , distribute_node
    parts, cut = partition_data(full_data, n_nodes, k)

    shared_nodes = []
    shared_nodes = None if not shared_nodes else shared_nodes

    if shared_nodes is not None:
        for i, part in enumerate(parts):
            part = list(filter(lambda x: x not in shared_nodes, part))
            parts[i] = part

    return parts, shared_nodes


def divided_nodes_from_txt(nodes_paths, shared_nodes_path):
    divided_node_lists = []
    for nodes_path in nodes_paths:
        with open(nodes_path) as f:
            nodes_list = []
            for _, line in enumerate(f):
                e = line.strip().split(',')
                node = int(e[0])
                nodes_list.append(node)
            divided_node_lists.append(nodes_list)

    if shared_nodes_path:
        with open(shared_nodes_path) as f:
            shared_nodes_list = []
            for _, line in enumerate(f):
                e = line.strip().split(',')
                node = int(e[0])
                shared_nodes_list.append(node)
    else:
        shared_nodes_list = None
    return divided_node_lists, shared_nodes_list


def infinite_loop(data_loader, max_len):
    while True:
        it = BackgroundThreadGenerator(data_loader)
        if is_main_process():
            it = tqdm(it, total=len(data_loader), ncols=50, mininterval=10)
        for i, data in enumerate(it):
            yield data, (i == 0), (i == len(data_loader) - 1)


def rewrite_state_dict(model_state, model_state_sep, n_nodes, shared_nodes, sub_nodes, buffer_list, testing_mode, rank):
    if model_state is None:
        model_state_rewritten = model_state_sep
    else:
        model_state_rewritten = model_state
    for buffer in buffer_list:
        memory = model_state_sep[buffer]
        shape = list(memory.shape)
        shape[0] = n_nodes
        if model_state is None:
            new_memory = torch.zeros(shape, dtype=memory.dtype)
        else:
            new_memory = model_state_rewritten[buffer]

        if testing_mode == "from_val" or (testing_mode == "hybrid" and rank == 1):
            #print('sub_nodes len:',len(sub_nodes))
            #print('memory len:',len(memory))
            #print('new_memory len:',len(new_memory))
            for i in range(len(sub_nodes)):
                original_index = sub_nodes[i]
                if i<len(memory):
                    new_memory[original_index] = memory[i]
                else:
                    new_memory[original_index] = memory[len(memory)-1]
        elif testing_mode == "from_begin" or (testing_mode == "hybrid" and rank == 0):
            pass

        model_state_rewritten[buffer] = new_memory
    return model_state_rewritten


def reconstruct_graph_dl(data, sub_nodes, shared_nodes, strategy, seed, n_neighbors, n_layers, restarter_type, hist_len,
                         bs, pin_memory):
    sub_data, global_list = data.get_subset_and_reindex_by_nodes(sub_nodes, shared_nodes)
    sub_graph = Graph.from_data(sub_data, strategy=strategy, seed=seed)

    sub_collator = GraphCollator(sub_graph, n_neighbors, n_layers, restarter=restarter_type, hist_len=hist_len)
    sub_dl = DataLoader(sub_data, batch_size=bs, collate_fn=sub_collator, pin_memory=pin_memory)
    return sub_data, sub_graph, sub_dl, sub_collator, global_list


def sync_last(mem, n_shared_nodes):
    backup_ts = mem.update_ts[:n_shared_nodes + 1].clone()
    dist.all_reduce(mem.update_ts[:n_shared_nodes + 1], op=dist.ReduceOp.MAX)
    non_equal_mask = backup_ts != mem.update_ts[:n_shared_nodes + 1]  # [False(same), True(not_same)]
    mem.shared_memory.weight.data[non_equal_mask] = 0.
    dist.all_reduce(mem.shared_memory.weight)
    mem.update_ts[:n_shared_nodes + 1] = backup_ts
    mem.vals.data = torch.cat([mem.shared_memory.weight.data, mem.non_shared_vals.data], dim=0)


def sync_average(mem, world_size, device):
    dist.all_reduce(mem.shared_memory.weight)
    world_size_tensor = torch.tensor(world_size).to(device)
    mem.shared_memory.weight.data = torch.div(mem.shared_memory.weight, world_size_tensor)
    mem.vals.data = torch.cat([mem.shared_memory.weight.data, mem.non_shared_vals.data], dim=0)


def sync_memory(sync_mode, left_mem, right_mem, n_shared_nodes, world_size, device):
    if sync_mode == "average":
        sync_average(left_mem, world_size, device)
        sync_average(right_mem, world_size, device)
    elif sync_mode == "last":
        sync_last(left_mem, n_shared_nodes)
        sync_last(right_mem, n_shared_nodes)
    elif sync_mode == "none":
        pass

