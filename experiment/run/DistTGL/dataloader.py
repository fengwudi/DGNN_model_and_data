import os
import torch
import pickle
import dgl
import time
import numpy as np
from memorys import *
from utils import *
from threading import Thread

class DataLoader:
    def __init__(self, data, train_neg_samples, eval_neg_samples, sets, mode, minibatch_parallelism=1, mailbox=None, node_feats=None, edge_feats=None, edge_classification=False):
        self.edge_classification = edge_classification
        if edge_classification:
            train_neg_samples = 0
            eval_neg_samples = 0
            sets = 0
        self.minibatch_parallelism = minibatch_parallelism
        if mode == 'train' and minibatch_parallelism > 1:
            self.path = 'minibatches/{}_{}_{}_{}_{}/'.format(minibatch_parallelism, data, train_neg_samples, eval_neg_samples, sets)
        else:
            self.path = 'minibatches/{}_{}_{}_{}/'.format(data, train_neg_samples, eval_neg_samples, sets)
        if mode != 'train':
            if not os.path.isfile(self.path + 'val_pos_0.pkl'):
                self.path = 'minibatches/{}_{}_eval/'.format(data, eval_neg_samples)
        self.sets = sets
        self.mode = mode
        self.tot_length = len([fn for fn in os.listdir(self.path) if fn.startswith('{}_pos'.format(mode))]) // minibatch_parallelism
        self.idx = 0
        self.rng = np.random.default_rng()
        self.mailbox = mailbox
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.cuda_device_id = torch.cuda.current_device()

        self.rank = None
        self.memory_read_buffer = None
        self.mail_read_buffer = None
        self.read_1idx_buffer = None
        self.read_status = None

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # only usable in validation/test
        if self.idx < self.tot_length:
            if self.mode == 'train':
                chosen_set = '_' + str(self.rng.integers(self.sets))
            else:
                chosen_set = ''
            with open('{}{}_pos_{}.pkl'.format(self.path, self.mode, self.idx), 'rb') as f:
                pos_mfg = pickle.load(f)
            neg_mfg = None
            if not self.edge_classification:
                with open('{}{}_neg_{}{}.pkl'.format(self.path, self.mode, self.idx, chosen_set), 'rb') as f:
                    neg_mfg = pickle.load(f)
            self.idx += 1
            return pos_mfg, neg_mfg
        else:
            raise StopIteration

    def get(self, idx, num_neg=1, offset=0):
        idx = idx * self.minibatch_parallelism + offset
        with open('{}{}_pos_{}.pkl'.format(self.path, self.mode, idx), 'rb') as f:
            pos_mfg = pickle.load(f)
        neg_mfgs = list()
        for _ in range(num_neg):
            if self.mode == 'train':
                chosen_set = '_' + str(self.rng.integers(self.sets))
            else:
                chosen_set = ''
            if not self.edge_classification:
                with open('{}{}_neg_{}{}.pkl'.format(self.path, self.mode, idx, chosen_set), 'rb') as f:
                    neg_mfgs.append(pickle.load(f))
        return pos_mfg, neg_mfgs

    def _load_and_slice_mfgs(self, pos_idx, neg_idxs, zero_mails=False):
        # only usable in training set
        t_s = time.time()
        with open('{}{}_pos_{}.pkl'.format(self.path, self.mode, pos_idx), 'rb') as f:
            pos_mfg = pickle.load(f)
        if not self.edge_classification:
            for idx, neg_idx in enumerate(neg_idxs):
                with open('{}{}_neg_{}_{}.pkl'.format(self.path, self.mode, pos_idx, neg_idx), 'rb') as f:
                    neg_mfg = pickle.load(f)
                mfg = combine_mfgs(pos_mfg, neg_mfg)
                prepare_input(mfg, self.node_feats, self.edge_feats)
                self.prefetched_mfgs[idx] = mfg
                if zero_mails:
                    num_nodes = mfg.srcdata['ID'].shape[0]
                    mfg.srcdata['mem'] = torch.zeros((num_nodes, self.mailbox.node_memory.shape[1]), device=mfg.srcdata['ID'].device)
                    mfg.srcdata['mail'] = torch.zeros((num_nodes, self.mailbox.mailbox.shape[1]), device=mfg.srcdata['ID'].device)
                else:
                    self.read_1idx_buffer[self.rank][idx][0] = mfg.srcdata['ID'].shape[0]
                    self.read_1idx_buffer[self.rank][idx][1:1 + mfg.srcdata['ID'].shape[0]].copy_(mfg.srcdata['ID'])
        else:
            mfg = pos_mfg
            mfg.combined = False
            prepare_input(mfg, self.node_feats, self.edge_feats)
            self.prefetched_mfgs[0] = mfg
            if zero_mails:
                num_nodes = mfg.srcdata['ID'].shape[0]
                mfg.srcdata['mem'] = torch.zeros((num_nodes, self.mailbox.node_memory.shape[1]), device=mfg.srcdata['ID'].device)
                mfg.srcdata['mail'] = torch.zeros((num_nodes, self.mailbox.mailbox.shape[1]), device=mfg.srcdata['ID'].device)
            else:
                self.read_1idx_buffer[self.rank][0][0] = mfg.srcdata['ID'].shape[0]
                self.read_1idx_buffer[self.rank][0][1:1 + mfg.srcdata['ID'].shape[0]].copy_(mfg.srcdata['ID'])
        if not zero_mails:
            self.read_status[self.rank] = 1
            # print('here2 read_status:', self.read_status.data_ptr(), self.read_status)
        return

    def init_prefetch(self, idx, num_neg=1, offset=0, prefetch_interval=1, rank=None, memory_read_buffer=None, mail_read_buffer=None, read_1idx_buffer=None, read_status=None):
        # initilize and prefetch the first minibatches with all zero node memory and mails
        # only works in training

        self.rank = rank
        self.memory_read_buffer = memory_read_buffer
        self.mail_read_buffer = mail_read_buffer
        self.read_1idx_buffer = read_1idx_buffer
        self.read_status = read_status
        # print('here1 read_status:', read_status.data_ptr(), 'self.read_status:', read_status.data_ptr())

        self.prefetch_interval = prefetch_interval
        self.prefetch_offset = offset
        self.prefetch_num_neg = num_neg
        self.next_prefetch_idx = idx + prefetch_interval

        self.fetched_mfgs = list()
        self.thread_idx = 0
        self.prefetched_mfgs = [None] * num_neg if not self.edge_classification else [None]
        self.prefetch_threads = None
        idx = 0 if idx < 0 else idx
        pos_idx = idx * self.minibatch_parallelism + offset
        neg_idxs = list()
        for i in range(num_neg):
            if not self.edge_classification:
                neg_idx = self.rng.integers(self.sets)
            else:
                neg_idx = None
            neg_idxs.append(neg_idx)
        self.prefetch_thread = Thread(target=self._load_and_slice_mfgs, args=(pos_idx, neg_idxs, True))
        self.prefetch_thread.start()
        return

    def prefetch_next(self):
        # put current prefetched to fetched and start next prefetch
        t_s = time.time()
        self.prefetch_thread.join()
        if not 'mem' in self.prefetched_mfgs[0].srcdata:
            # print('here3 read_status:', self.read_status.data_ptr())
            while self.read_status[self.rank] == 1:
                pass
            if not self.edge_classification:
                for idx, mfg in enumerate(self.prefetched_mfgs):
                    mfg.srcdata['mem'] = self.memory_read_buffer[self.rank][idx][:mfg.srcdata['ID'].shape[0]]
                    mfg.srcdata['mail'] = self.mail_read_buffer[self.rank][idx][:mfg.srcdata['ID'].shape[0]]
            else:
                self.prefetched_mfgs[0].srcdata['mem'] = self.memory_read_buffer[self.rank][0][:self.prefetched_mfgs[0].srcdata['ID'].shape[0]]
                self.prefetched_mfgs[0].srcdata['mail'] = self.mail_read_buffer[self.rank][0][:self.prefetched_mfgs[0].srcdata['ID'].shape[0]]
        self.thread_idx = 0
        self.fetched_mfgs = self.prefetched_mfgs

        # print('\twait for previous thread time {}ms'.format((time.time() - t_s) * 1000))

        if self.next_prefetch_idx != -1:
            if self.next_prefetch_idx >= self.tot_length:
                self.next_prefetch_idx = self.tot_length - 1

            self.prefetched_mfgs = [None] * self.prefetch_num_neg if not self.edge_classification else [None]
            self.prefetch_threads = None

            pos_idx = self.next_prefetch_idx * self.minibatch_parallelism + self.prefetch_offset
            neg_idxs = list()
            for i in range(self.prefetch_num_neg):
                if not self.edge_classification:
                    neg_idx = self.rng.integers(self.sets)
                else:
                    neg_idx = None
                neg_idxs.append(neg_idx)
            # print('launch prefetch thread to prefetch {}'.format(pos_idx))
            self.prefetch_thread = Thread(target=self._load_and_slice_mfgs, args=(pos_idx, neg_idxs))
            self.prefetch_thread.start()

            if self.next_prefetch_idx != self.tot_length - 1:
                self.next_prefetch_idx += self.prefetch_interval
            else:
                self.next_prefetch_idx = -1

    def get_fetched(self):
        ret = self.fetched_mfgs[self.thread_idx]
        self.thread_idx += 1
        return ret
