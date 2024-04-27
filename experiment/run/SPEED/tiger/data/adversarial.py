
from collections import defaultdict
import math
from typing import Dict, Optional, Set, Tuple
import numpy as np


class AdversarialEdgeSampler:
    """
    Adversarial Random Edge Sampling as Negative Edges
    """

    def __init__(self, full_srcs, full_dsts, full_ts, test_srcs, test_ts, neg_type, seed=None):
        """
        'src_list', 'dst_list', 'ts_list' are related to the full data! All possible edges in train, validation, test
        """
        if not (neg_type == 'hist' or neg_type == 'ind'):
            raise ValueError("Undefined Negative Edge Sampling Strategy!")

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.neg_type = neg_type
        self.full_srcs = full_srcs
        self.full_dsts = full_dsts
        self.full_ts = full_ts
        self.full_srcs_distinct = np.unique(full_srcs)
        self.full_dst_distinct = np.unique(full_dsts)
        self.full_ts_distinct = np.unique(full_ts)
        self.test_srcs = test_srcs
        self.test_ts = test_ts
        self.ts_init = min(self.full_ts_distinct)
        self.ts_end = max(self.full_ts_distinct)
        self.ts_hist_end = self.full_ts[-len(test_srcs)-1]
        self.train_edge_dict = self.get_edges_within(self.ts_init, self.ts_hist_end)

    def get_edges_within(self, t0: float, t1:float, subset: Optional[Set]=None
                        ) -> Dict[int, Set[int]]:
        """
        return edges of within the given time interval
        """
        a = np.searchsorted(self.full_ts, t0, side='left')
        b = np.searchsorted(self.full_ts, t1, side='right')
        srcs = self.full_srcs[a:b]
        dsts = self.full_dsts[a:b]
        edge_dict = defaultdict(set)
        for src, dst in zip(srcs, dsts):
            if (subset is not None) and (src not in subset):
                continue
            edge_dict[src].add(dst)
        return edge_dict

    def get_difference_edge_list(self, first_e_set, second_e_set):
        """
        return edges in the first_e_set that are not in the second_e_set
        """
        difference_e_set = set(first_e_set) - set(second_e_set)
        src_l, dst_l = [], []
        for e in difference_e_set:
            src_l.append(e[0])
            dst_l.append(e[1])
        return np.array(src_l), np.array(dst_l)

    def sample(self, srcs, t0, t1):
        if self.neg_type == 'hist':
            neg_srcs, neg_dsts = self.sample_hist(srcs, t0, t1)
        elif self.neg_type == 'ind':
            neg_srcs, neg_dsts = self.sample_ind(srcs, t0, t1)
        else:
            raise ValueError("Undefined Negative Edge Sampling Strategy!")
        return neg_srcs, neg_dsts
    
    def sample_hist(self, srcs, t0, t1):
        hist_edge_dict = self.get_edges_within(self.ts_init, t0, srcs)
        current_edge_dict = self.get_edges_within(t0, t1, srcs)
        neg_dsts = []
        for src in srcs:
            diff_edges = hist_edge_dict[src] - current_edge_dict[src]
            if len(diff_edges):
                neg_dst = self.rng.choice(list(diff_edges))
            else:
                neg_idx = self.rng.randint(0, len(self.full_dst_distinct))
                neg_dst = self.full_dst_distinct[neg_idx]
            neg_dsts.append(neg_dst)
        
        return srcs, np.array(neg_dsts)
    
    def sample_ind(self, srcs, t0, t1):
        hist_edge_dict = self.get_edges_within(self.ts_init, t0, srcs)
        current_edge_dict = self.get_edges_within(t0, t1, srcs)
        neg_dsts = []
        for src in srcs:
            diff_edges = hist_edge_dict[src] - self.train_edge_dict[src] - current_edge_dict[src]
            if len(diff_edges):
                neg_dst = self.rng.choice(list(diff_edges))
            else:
                neg_idx = self.rng.randint(0, len(self.full_dst_distinct))
                neg_dst = self.full_dst_distinct[neg_idx]
            neg_dsts.append(neg_dst)
        
        return srcs, np.array(neg_dsts)

    def reset_random_state(self):
        self.rng = np.random.RandomState(self.seed)

    def pre_sample_neg_dsts(self, n_total: int, bs: int=200) -> np.ndarray:
        self.reset_random_state()
        assert len(self.test_srcs) == n_total
        n_iters = math.ceil(n_total / bs)
        neg_dsts = []
        for i in range(n_iters):
            srcs = self.test_srcs[i*bs:(i+1)*bs]
            ts = self.test_ts[i*bs:(i+1)*bs]
            _, negs = self.sample(srcs, ts[0], ts[-1])
            neg_dsts.append(negs)
        all_neg_dsts = np.concatenate(neg_dsts)
        assert len(all_neg_dsts) == n_total
        return all_neg_dsts