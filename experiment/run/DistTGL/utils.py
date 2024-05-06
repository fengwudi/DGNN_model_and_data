import torch
import os
import yaml
import dgl
import time
import pandas as pd
import numpy as np
from sklearn import metrics

class WriteBuffer:
    # this class is to avoid the shallow copy for the input parameters in the .forward() function in torch.distributed.model
    def __init__(self, rank, memory_write_buffer, mail_write_buffer, write_1idx_buffer, write_status):
        self.rank = rank
        self.memory_write_buffer = memory_write_buffer
        self.mail_write_buffer = mail_write_buffer
        self.write_1idx_buffer = write_1idx_buffer
        self.write_status = write_status

def calc_f1_mic(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return metrics.f1_score(y_true, y_pred, average="micro")

def load_feat(d):
    node_feats = None
    if os.path.exists('../TGL/DATA/{}/node_features.pt'.format(d)):
        node_feats = torch.load('../TGL/DATA/{}/node_features.pt'.format(d))
    elif os.path.exists('../TGL/DATA/{}/learned_node_feats.pt'.format(d)):
        node_feats = torch.load('../TGL/DATA/{}/learned_node_feats.pt'.format(d))
    edge_feats = None
    if os.path.exists('../TGL/DATA/{}/edge_features_e0.pt'.format(d)):
        edge_feats = torch.load('../TGL/DATA/{}/edge_features_e0.pt'.format(d))
    elif os.path.exists('../TGL/DATA/{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('../TGL/DATA/{}/edge_features.pt'.format(d))
        edge_feats = torch.cat([edge_feats, torch.zeros((1, edge_feats.shape[1]), dtype=edge_feats.dtype)])
        torch.save(edge_feats, '../TGL/DATA/{}/edge_features_e0.pt'.format(d))
    if node_feats is not None:
        if node_feats.dtype == torch.bool:
            node_feats = node_feats.type(torch.int8)
    if edge_feats is not None:
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.int8)
    return node_feats, edge_feats

def load_graph(d):
    if os.path.isfile('../TGL/DATA/{}/edges+uniq.csv'.format(d)):
        df = pd.read_csv('../TGL/DATA/{}/edges+uniq.csv'.format(d))
    else:
        df = pd.read_csv('../TGL/DATA/{}/edges.csv'.format(d))
    g = np.load('../TGL/DATA/{}/ext_full.npz'.format(d))
    return g, df

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

def unique_last_idx(a):
    # find index of last appearance of an element in a array
    uni, inv = torch.unique(a, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
    return perm

def to_dgl_blocks(ret, hist, num_neg_samples, cuda=True, combine=True):
    mfgs = list()
    for r in ret:
        positive_nodes = r.dim_out() // (num_neg_samples + 2) * 2
        src_id = torch.from_numpy(r.nodes())
        src_uniq, src_uniq_idx = torch.unique(src_id, sorted=False, return_inverse=True)
        col = src_uniq_idx[torch.from_numpy(r.col()).long()]
        row = torch.from_numpy(r.row()).long()

        b = dgl.create_block((col, row), num_src_nodes=src_uniq.shape[0], num_dst_nodes=r.dim_out())
        b.srcdata['ID'] = src_uniq

        b.edata['dt'] = torch.from_numpy(r.dts())[b.num_dst_nodes():]
        b.edata['ID'] = torch.from_numpy(r.eid())

        b.positive_memory_idx_end = positive_nodes
        b.src_idx = src_uniq_idx

        if cuda:
            b.src_idx = b.src_idx.cuda()
            mfgs.append(b.to('cuda:0'))
        else:
            mfgs.append(b)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs

def combine_mfgs(pos_mfg, neg_mfg):
    col = torch.cat([pos_mfg.edges()[0], neg_mfg.edges()[0] + pos_mfg.num_src_nodes()])
    row = torch.cat([pos_mfg.edges()[1], neg_mfg.edges()[1] + pos_mfg.num_dst_nodes()])
    mfg = dgl.create_block((col, row), num_src_nodes=pos_mfg.num_src_nodes() + neg_mfg.num_src_nodes(), num_dst_nodes=pos_mfg.num_dst_nodes() + neg_mfg.num_dst_nodes())
    mfg.combined = True
    mfg.num_pos_src = pos_mfg.num_src_nodes()
    mfg.num_pos_dst = pos_mfg.num_dst_nodes()
    mfg.num_neg_src = neg_mfg.num_src_nodes()
    mfg.num_neg_dst = neg_mfg.num_dst_nodes()
    mfg.num_pos_idx = pos_mfg.src_idx.shape[0]
    mfg.num_neg_idx = neg_mfg.src_idx.shape[0]
    mfg.node_memory_mask = pos_mfg.node_memory_mask
    mfg.srcdata['ID'] = torch.cat([pos_mfg.srcdata['ID'], neg_mfg.srcdata['ID']])
    mfg.srcdata['mem_ts'] = torch.cat([pos_mfg.srcdata['mem_ts'], neg_mfg.srcdata['mem_ts']])
    mfg.srcdata['mail_ts'] = torch.cat([pos_mfg.srcdata['mail_ts'], neg_mfg.srcdata['mail_ts']])
    mfg.srcdata['mail_e'] = torch.cat([pos_mfg.srcdata['mail_e'], neg_mfg.srcdata['mail_e']])
    mfg.edata['dt'] = torch.cat([pos_mfg.edata['dt'], neg_mfg.edata['dt']])
    mfg.edata['ID'] = torch.cat([pos_mfg.edata['ID'], neg_mfg.edata['ID']])
    mfg.src_idx = torch.cat([pos_mfg.src_idx, neg_mfg.src_idx + pos_mfg.num_src_nodes()])
    return mfg

def node_to_dgl_blocks(root_nodes, ts, cuda=True):
    mfgs = list()
    b = dgl.create_block(([],[]), num_src_nodes=root_nodes.shape[0], num_dst_nodes=root_nodes.shape[0])
    b.srcdata['ID'] = torch.from_numpy(root_nodes)
    b.srcdata['ts'] = torch.from_numpy(ts)
    if cuda:
        mfgs.insert(0, [b.to('cuda:0')])
    else:
        mfgs.insert(0, [b])
    return mfgs

def mfg_to_cuda(mfg, cuda_device_id=None):
    if cuda_device_id is not None:
        mfg = mfg.to('cuda:{}'.format(cuda_device_id))
    else:
        mfg = mfg.to('cuda:{}'.format(torch.cuda.current_device()))
    mfg.src_idx_cuda = mfg.src_idx.cuda()
    if hasattr(mfg, 'edge_cls'):
        mfg.edge_cls_cuda = mfg.edge_cls.cuda()
    if hasattr(mfg, 'node_memory_mask'):
        mfg.node_memory_mask_cuda = mfg.node_memory_mask.cuda()
    return mfg

def prepare_input(mfg, node_feats, edge_feats):
    with torch.no_grad():
        if node_feats is not None:
            mfg.srcdata['rh'] = torch.index_select(node_feats, 0, mfg.srcdata['ID']).float()
        if edge_feats is not None:
            mfg.edata['f'] = torch.index_select(edge_feats, 0, mfg.edata['ID']).float()
            mfg.srcdata['mail_ef'] = torch.index_select(edge_feats, 0, mfg.srcdata['mail_e']).float()
    return

def get_ids(mfgs, node_feats, edge_feats):
    nids = list()
    eids = list()
    if node_feats is not None:
        for b in mfgs[0]:
            nids.append(b.srcdata['ID'].long())
    if edge_feats is not None:
        for mfg in mfgs:
            for b in mfg:
                eids.append(b.edata['ID'].long())
    return nids, eids

def get_pinned_buffers(sample_param, batch_size, node_feats, edge_feats):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if 'neighbor' in sample_param:
        for i in sample_param['neighbor']:
            limit *= i + 1
            if edge_feats is not None:
                for _ in range(sample_param['history']):
                    pinned_efeat_buffs.insert(0, torch.zeros((limit, edge_feats.shape[1]), pin_memory=True))
    if node_feats is not None:
        for _ in range(sample_param['history']):
            pinned_nfeat_buffs.insert(0, torch.zeros((limit, node_feats.shape[1]), pin_memory=True))
    return pinned_nfeat_buffs, pinned_efeat_buffs

def send_recv_mfg_GPU(mfgs, src, dst, nccl_group):
    local_rank = int(os.environ['LOCAL_RANK'])
    # print(local_rank, src, dst)
    if local_rank == src:
        block = mfgs[0][0]
        dim_block = torch.tensor([block.num_src_nodes(), block.num_dst_nodes(), block.num_edges(), block.srcdata['mem'].shape[1], block.srcdata['mem_input'].shape[1], block.edata['f'].shape[1], block.src_idx.shape[0]])
        torch.distributed.send(dim_block, dst)
        torch.distributed.send(block.edges()[0], dst, group=nccl_group)
        torch.distributed.send(block.edges()[1], dst, group=nccl_group)

        torch.distributed.send(block.srcdata['ID'], dst, group=nccl_group)
        # torch.distributed.send(block.srcdata['ts'], dst, group=nccl_group)
        torch.distributed.send(block.srcdata['mem'], dst, group=nccl_group)
        torch.distributed.send(block.srcdata['mem_ts'], dst, group=nccl_group)
        torch.distributed.send(block.srcdata['mem_input'], dst, group=nccl_group)
        torch.distributed.send(block.srcdata['mail_ts'], dst, group=nccl_group)

        torch.distributed.send(block.edata['ID'], dst, group=nccl_group)
        torch.distributed.send(block.edata['dt'], dst, group=nccl_group)
        torch.distributed.send(block.edata['f'], dst, group=nccl_group)

        torch.distributed.send(block.src_idx, dst, group=nccl_group)

    elif local_rank == dst:
        dim_block = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        torch.distributed.recv(dim_block, src)
        dim_block = dim_block.tolist()
        edges0 = torch.zeros(dim_block[2], dtype=torch.int64, device=torch.device('cuda:0'))
        edges1 = torch.zeros(dim_block[2], dtype=torch.int64, device=torch.device('cuda:0'))
        torch.distributed.recv(edges0, src, group=nccl_group)
        torch.distributed.recv(edges1, src, group=nccl_group)
        block = dgl.create_block((edges0, edges1), num_src_nodes=dim_block[0], num_dst_nodes=dim_block[1])

        src_id = torch.zeros(dim_block[0], dtype=torch.int32, device=torch.device('cuda:0'))
        # src_ts = torch.zeros(dim_block[0], dtype=torch.float32, device=torch.device('cuda:0'))
        src_mem = torch.zeros((dim_block[0], dim_block[3]), dtype=torch.float32, device=torch.device('cuda:0'))
        src_mem_ts = torch.zeros(dim_block[0], dtype=torch.float32, device=torch.device('cuda:0'))
        src_mem_input = torch.zeros((dim_block[0], dim_block[4]), dtype=torch.float32, device=torch.device('cuda:0'))
        src_mail_ts = torch.zeros((dim_block[0], 1), dtype=torch.float32, device=torch.device('cuda:0'))
        torch.distributed.recv(src_id, src, group=nccl_group)
        # torch.distributed.recv(src_ts, src, group=nccl_group)
        torch.distributed.recv(src_mem, src, group=nccl_group)
        torch.distributed.recv(src_mem_ts, src, group=nccl_group)
        torch.distributed.recv(src_mem_input, src, group=nccl_group)
        torch.distributed.recv(src_mail_ts, src, group=nccl_group)
        block.srcdata['ID'] = src_id
        # block.srcdata['ts'] = src_ts
        block.srcdata['mem'] = src_mem
        block.srcdata['mem_ts'] = src_mem_ts
        block.srcdata['mem_input'] = src_mem_input
        block.srcdata['mail_ts'] = src_mail_ts

        e_id = torch.zeros(dim_block[2], dtype=torch.int32, device=torch.device('cuda:0'))
        e_dt = torch.zeros(dim_block[2], dtype=torch.float32, device=torch.device('cuda:0'))
        e_f = torch.zeros((dim_block[2], dim_block[5]), dtype=torch.float32, device=torch.device('cuda:0'))
        torch.distributed.recv(e_id, src, group=nccl_group)
        torch.distributed.recv(e_dt, src, group=nccl_group)
        torch.distributed.recv(e_f, src, group=nccl_group)
        block.edata['ID'] = e_id
        block.edata['dt'] = e_dt
        block.edata['f'] = e_f

        src_idx = torch.zeros(dim_block[6], dtype=torch.int64, device=torch.device('cuda:0'))
        torch.distributed.recv(src_idx, src, group=nccl_group)
        block.src_idx = src_idx

    return [[block]]