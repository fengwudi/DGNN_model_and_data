import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--train_neg_samples', type=int, default=1)
parser.add_argument('--neg_sets', type=int, default=32, help='how many groups of negative samples')
parser.add_argument('--seed', type=int, default=0, help='random seed to use')
parser.add_argument('--gen_eval', action='store_true', help='whether to generate evaluation minibatches')
parser.add_argument('--eval_cap', type=int, default=0, help='maximum number of minibatches in eval and test set')
parser.add_argument('--minibatch_parallelism', type=int, default=1, help='how many nubmer of GPU per minibatches')
parser.add_argument('--edge_classification', action='store_true', help='whether to train with the edge classification task')
args = parser.parse_args()

if args.data in ['GDELT', 'LINK']:
    args.eval_cap = 5000
    args.neg_sets = 16

#if args.data in ['GDELT']:
    #args.edge_classification = True

import torch
import dgl
import datetime
import random
import math
import threading
import pickle
import numpy as np
from modules import *
from sampler import *
from utils import *
from get_config import *
from tqdm import tqdm
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

tot_rank = 1
sample_param, memory_param, gnn_param, train_param = get_config(args.model, args.data, tot_rank)
if args.train_neg_samples > 0:
    train_param['train_neg_samples'] = args.train_neg_samples

if args.edge_classification:
    args.neg_sets = 0
    train_param['train_neg_samples'] = 0
    train_param['eval_neg_samples'] = 0
    edge_cls = torch.load('../tgl-main/DATA/{}/ec_edge_class.pt'.format(args.data))

g, df = load_graph(args.data)
train_edge_end = df[df['ext_roll'].gt(0)].index[0]
val_edge_end = df[df['ext_roll'].gt(1)].index[0]
num_nodes = g['indptr'].shape[0] - 1

sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                            sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                            sample_param['strategy']=='recent', sample_param['prop_time'],
                            sample_param['history'], float(sample_param['duration']))
is_bipartite = True if args.data in ['WIKI', 'REDDIT', 'MOOC', 'LASTFM', 'Taobao', 'sTaobao', 'LINK'] else False
if is_bipartite:
    neg_link_sampler = NegLinkSampler(num_nodes, df=df)
else:
    neg_link_sampler = NegLinkSampler(num_nodes)

if not os.path.isdir('minibatches'):
    os.mkdir('minibatches')
if args.minibatch_parallelism == 1:
    path = 'minibatches/{}_{}_{}_{}/'.format(args.data, train_param['train_neg_samples'], train_param['eval_neg_samples'], args.neg_sets)
else:
    path = 'minibatches/{}_{}_{}_{}_{}/'.format(args.minibatch_parallelism, args.data, train_param['train_neg_samples'], train_param['eval_neg_samples'], args.neg_sets)
if not os.path.isdir(path):
    os.mkdir(path)

stats = {'num_nodes': g['indptr'].shape[0] - 1, 'num_edges': len(df)}
with open('minibatches/{}_stats.pkl'.format(args.data), 'wb') as f:
    pickle.dump(stats, f)

train_df = df[:train_edge_end]
val_df = df[train_edge_end:val_edge_end]
test_df = df[val_edge_end:]

mem_ts = torch.zeros(num_nodes)
mail_ts = torch.zeros(num_nodes)
mail_e = torch.zeros(num_nodes, dtype=torch.long) + len(df)

last_updated_mem = None
i = 0
pos_mfgs = list()
pos_mfgs_i = list()
srcs = list()
dsts = list()
tss = list()
eids = list()
for _, rows in tqdm(train_df.groupby(train_df.index // train_param['batch_size']), total=len(train_df) // train_param['batch_size']):
    # positive mfg
    root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
    ts = np.tile(rows.time.values, 2).astype(np.float32)
    sampler.sample(root_nodes, ts)
    ret = sampler.get_ret()
    mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
    mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
    mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
    mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
    mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
    if args.edge_classification:
        mfg.edge_cls = edge_cls[rows.index.values].float()
    # mfg.start_eidx = eid[0].item()
    # mfg.end_eidx = eid[-1].item() + 1
    pos_mfgs.append(mfg)
    pos_mfgs_i.append(i)
    
    srcs.append(torch.from_numpy(rows.src.values))
    dsts.append(torch.from_numpy(rows.dst.values))
    tss.append(torch.from_numpy(rows.time.values.astype(np.float32)))
    eids.append(torch.from_numpy(rows.index.values))

    # src = torch.from_numpy(rows.src.values)
    # dst = torch.from_numpy(rows.dst.values)
    # ts = torch.from_numpy(ts)
    # eid = torch.from_numpy(rows.index.values)
    # mailseid = torch.cat([eid, eid])
    # mailsts = ts
    # nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
    # update_mask = torch.zeros(nid.shape[0], dtype=torch.bool)
    # idx = unique_last_idx(nid)
    # nid = torch.cat([src, dst])
    # idx_map = torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(dst.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1)
    # idx = idx_map[idx]
    # update_mask[idx] = 1
    # mfg.node_memory_mask = update_mask

    # negative mfg
    if not args.edge_classification:
        for j in range(args.neg_sets):
            root_nodes = neg_link_sampler.sample(len(rows) * train_param['train_neg_samples'])
            ts = np.tile(rows.time.values, train_param['train_neg_samples']).astype(np.float32)
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
            mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
            mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
            with open('{}/train_neg_{}_{}.pkl'.format(path, i, j), 'wb') as f:
                pickle.dump(mfg, f)

    if len(pos_mfgs) == args.minibatch_parallelism:

        idx_map = list()
        length = 0
        for src in srcs:
            idx_map.append(torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(src.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1) + length)
            length += idx_map[-1].shape[0]
        idx_map = torch.cat(idx_map)

        src = torch.cat(srcs)
        dst = torch.cat(dsts)
        ts = torch.cat(tss)
        eid = torch.cat(eids)

        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        update_mask = torch.zeros(nid.shape[0], dtype=torch.bool)
        idx_raw = unique_last_idx(nid)
        idx = idx_map[idx_raw]
        update_mask[idx] = 1
        start = 0
        for mfg, mfg_i in zip(pos_mfgs, pos_mfgs_i):
            mfg.node_memory_mask = update_mask[start:start + mfg.num_dst_nodes()]
            start += mfg.num_dst_nodes()
            with open('{}/train_pos_{}.pkl'.format(path, mfg_i), 'wb') as f:
                pickle.dump(mfg, f)

        # update mem_ts, mail_ts, and mail_e
        mailseid = torch.cat([eid, eid])
        mailsts = torch.cat([ts, ts])
        nid = torch.cat([src, dst])

        idx_map = torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(dst.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1)
        idx = idx_map[idx_raw]

        nid = nid[idx].long()
        mailseid = mailseid[idx]
        mailsts = mailsts[idx]
        mem_ts[nid] = mail_ts[nid]
        mail_ts[nid] = mailsts
        mail_e[nid] = mailseid

        pos_mfgs = list()
        pos_mfgs_i = list()
        srcs = list()
        dsts = list()
        tss = list()
        eids = list()
    i += 1


if args.gen_eval:
    path = 'minibatches/{}_{}_eval/'.format(args.data, train_param['eval_neg_samples'])
    if not os.path.isdir(path):
        os.mkdir(path)

    # validation
    last_updated_mem = None
    i = 0
    for _, rows in tqdm(val_df.groupby(val_df.index // train_param['batch_size']), total=len(val_df) // train_param['batch_size']):
        # positive mfg
        root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        ts = np.tile(rows.time.values, 2).astype(np.float32)
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
        mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
        mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
        mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
        mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
        if args.edge_classification:
            mfg.edge_cls = edge_cls[rows.index.values].float()

        src = torch.from_numpy(rows.src.values)
        dst = torch.from_numpy(rows.dst.values)
        ts = torch.from_numpy(ts)
        eid = torch.from_numpy(rows.index.values)
        mailseid = torch.cat([eid, eid])
        mailsts = ts
        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        update_mask = torch.zeros(nid.shape[0], dtype=torch.bool)
        idx = unique_last_idx(nid)
        nid = torch.cat([src, dst])
        idx_map = torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(dst.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1)
        idx = idx_map[idx]

        update_mask[idx] = 1
        mfg.node_memory_mask = update_mask
        mfg.start_eidx = eid[0].item()
        mfg.end_eidx = eid[-1].item() + 1

        with open('{}/val_pos_{}.pkl'.format(path, i), 'wb') as f:
            pickle.dump(mfg, f)

        # negative mfg
        if not args.edge_classification:
            root_nodes = neg_link_sampler.sample(len(rows) * train_param['eval_neg_samples'])
            ts = np.tile(rows.time.values, train_param['eval_neg_samples']).astype(np.float32)
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
            mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
            mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
            with open('{}/val_neg_{}.pkl'.format(path, i), 'wb') as f:
                pickle.dump(mfg, f)
            
        # update mem_ts, mail_ts, and mail_e
        nid = nid[idx].long()
        mailseid = mailseid[idx]
        mailsts = mailsts[idx]
        mem_ts[nid] = mail_ts[nid]
        mail_ts[nid] = mailsts
        mail_e[nid] = mailseid
        
        i += 1

        if args.eval_cap > 0:
            if i > args.eval_cap:
                break

    # test
    last_updated_mem = None
    i = 0
    for _, rows in tqdm(test_df.groupby(test_df.index // train_param['batch_size']), total=len(test_df) // train_param['batch_size']):
        # positive mfg
        root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
        ts = np.tile(rows.time.values, 2).astype(np.float32)
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
        mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
        mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
        mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
        mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
        if args.edge_classification:
            mfg.edge_cls = edge_cls[rows.index.values].float()
        
        src = torch.from_numpy(rows.src.values)
        dst = torch.from_numpy(rows.dst.values)
        ts = torch.from_numpy(ts)
        eid = torch.from_numpy(rows.index.values)
        mailseid = torch.cat([eid, eid])
        mailsts = ts
        nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
        update_mask = torch.zeros(nid.shape[0], dtype=torch.bool)
        idx = unique_last_idx(nid)
        nid = torch.cat([src, dst])
        idx_map = torch.cat([torch.arange(src.shape[0]).unsqueeze(1), torch.arange(dst.shape[0]).unsqueeze(1) + src.shape[0]], dim=1).reshape(-1)
        idx = idx_map[idx]

        update_mask[idx] = 1
        mfg.node_memory_mask = update_mask
        mfg.start_eidx = eid[0].item()
        mfg.end_eidx = eid[-1].item() + 1

        with open('{}/test_pos_{}.pkl'.format(path, i), 'wb') as f:
            pickle.dump(mfg, f)

        # negative mfg
        if not args.edge_classification:
            root_nodes = neg_link_sampler.sample(len(rows) * train_param['eval_neg_samples'])
            ts = np.tile(rows.time.values, train_param['eval_neg_samples']).astype(np.float32)
            sampler.sample(root_nodes, ts)
            ret = sampler.get_ret()
            mfg = to_dgl_blocks(ret, sample_param['history'], 0, cuda=False)[0][0]
            mfg.srcdata['ID'] = mfg.srcdata['ID'].long()
            mfg.srcdata['mem_ts'] = mem_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_ts'] = mail_ts[mfg.srcdata['ID']]
            mfg.srcdata['mail_e'] = mail_e[mfg.srcdata['ID']]
            with open('{}/test_neg_{}.pkl'.format(path, i), 'wb') as f:
                pickle.dump(mfg, f)

        # update mem_ts, mail_ts, and mail_e
        nid = nid[idx].long()
        mailseid = mailseid[idx]
        mailsts = mailsts[idx]
        mem_ts[nid] = mail_ts[nid]
        mail_ts[nid] = mailsts
        mail_e[nid] = mailseid

        i += 1

        if args.eval_cap > 0:
            if i > args.eval_cap:
                break