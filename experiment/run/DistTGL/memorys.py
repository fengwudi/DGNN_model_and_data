import torch
import dgl
import time
from layers import TimeEncode

class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, _node_memory=None, _mailbox=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
        self.mailbox = torch.zeros((num_nodes, 2 * memory_param['dim_out']), dtype=torch.float32) if _mailbox is None else _mailbox
        self.device = torch.device('cpu')
        
    def reset(self):
        self.node_memory.fill_(0)
        self.mailbox.fill_(0)

    def prep_input_mails(self, mfg):
        if hasattr(mfg, 'stream'):
            with torch.cuda.stream(mfg.stream):
                idx = mfg.srcdata['ID'].cpu()
                mfg.srcdata['mem'] = torch.index_select(self.node_memory, 0, idx).to(mfg.srcdata['ID'].device)
                mfg.srcdata['mail'] = torch.index_select(self.mailbox, 0, idx).to(mfg.srcdata['ID'].device)
        else:
            idx = mfg.srcdata['ID'].cpu()
            mfg.srcdata['mem'] = torch.index_select(self.node_memory, 0, idx).to(mfg.srcdata['ID'].device)
            mfg.srcdata['mail'] = torch.index_select(self.mailbox, 0, idx).to(mfg.srcdata['ID'].device)

    def prep_zero_mails(self, mfg):
        num_nodes = mfg.srcdata['ID'].shape[0]
        mfg.srcdata['mem'] = torch.zeros((num_nodes, self.node_memory.shape[1]), device=mfg.srcdata['ID'].device)
        mfg.srcdata['mail'] = torch.zeros((num_nodes, self.mailbox.shape[1]), device=mfg.srcdata['ID'].device)

    def update_memory_and_mailbox(self, mfg, memory):
        with torch.no_grad():
            if mfg.combined:
                mask = mfg.node_memory_mask.to(self.device)
                nid = mfg.srcdata['ID'][:mfg.num_pos_src].to(self.device)[mfg.src_idx[:mask.shape[0]]]
                memory = memory.to(self.device)[mfg.src_idx[:mask.shape[0]]]
            else:
                mask = mfg.node_memory_mask.to(self.device)
                nid = mfg.srcdata['ID'].to(self.device)[mfg.src_idx[:mask.shape[0]]]
                memory = memory.to(self.device)[mfg.src_idx[:mask.shape[0]]]
            masked_nid = nid[mask]
            masked_memory = memory[mask]
            self.node_memory[masked_nid] = masked_memory

            num_root = mask.shape[0] // 2
            mail_src = masked_memory
            mail_dst = torch.cat([memory[num_root:][mask[:num_root]], memory[:num_root][mask[num_root:]]])
            mail = torch.cat([mail_src, mail_dst], dim=1)
            self.mailbox[masked_nid] = mail

class SmartMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat, num_node, no_learn_node=False):
        super(SmartMemoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time

        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.addc = False

        # self.no_learn_node = no_learn_node
        if dim_node_feat > 0:
            if dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        # if not no_learn_node:
        #     if dim_node_feat == 0:
        #         self.constant = torch.nn.Embedding(num_node, dim_hid, sparse=True)

        #     self.choice = torch.nn.Embedding(num_node, 1, sparse=True)
        #     torch.nn.init.constant_(self.choice.weight, 0)

    def forward(self, b, write_buffer=None):
        # torch.cuda.synchronize()
        # t_s=time.time()
        if self.dim_time > 0:
            time_feat = self.time_enc(b.srcdata['mail_ts'].squeeze() - b.srcdata['mem_ts'])
            if 'mail_ef' in b.srcdata:
                b.srcdata['t_mem_input'] = torch.cat([b.srcdata['mail'], b.srcdata['mail_ef'], time_feat], dim=1)
            else:
                b.srcdata['t_mem_input'] = torch.cat([b.srcdata['mail'], time_feat], dim=1)
        updated_memory = self.updater(b.srcdata['t_mem_input'], b.srcdata['mem'])
        new_memory = updated_memory
        # torch.cuda.synchronize()
        # print('t_gru:{:.4f}'.format(time.time() - t_s))

        with torch.no_grad():
            if write_buffer is not None:
                rank = write_buffer.rank
                memory_write_buffer = write_buffer.memory_write_buffer
                mail_write_buffer = write_buffer.mail_write_buffer
                write_1idx_buffer = write_buffer.write_1idx_buffer
                write_status = write_buffer.write_status
                if b.combined:
                    memory = updated_memory[:b.num_pos_src]
                else:
                    memory = updated_memory
                if b.combined:
                    mask = b.node_memory_mask_cuda
                    nid = b.srcdata['ID'][:b.num_pos_src][b.src_idx[:mask.shape[0]]]
                    memory = memory[b.src_idx[:mask.shape[0]]]
                else:
                    mask = b.node_memory_mask_cuda
                    nid = b.srcdata['ID'][b.src_idx[:mask.shape[0]]]
                    memory = memory[b.src_idx[:mask.shape[0]]]

                masked_nid = nid[mask]
                masked_memory = memory[mask]
                write_1idx_buffer[rank][0] = masked_nid.shape[0]
                write_1idx_buffer[rank][1:1 + masked_nid.shape[0]].copy_(masked_nid)
                memory_write_buffer[rank][:masked_nid.shape[0]].copy_(masked_memory)

                num_root = mask.shape[0] // 2
                mail_src = masked_memory
                mail_dst = torch.cat([memory[num_root:][mask[:num_root]], memory[:num_root][mask[num_root:]]])
                mail = torch.cat([mail_src, mail_dst], dim=1)
                mail_write_buffer[rank][:masked_nid.shape[0]].copy_(mail)

                write_status[rank] = 1
                # import pdb; pdb.set_trace()
                # pass
            else:
                if b.combined:
                    self.last_updated_memory = updated_memory[:b.num_pos_src].detach().clone()
                else:
                    self.last_updated_memory = updated_memory.detach().clone()

        # if not self.no_learn_node:
        #     if self.dim_node_feat > 0:
        #         if self.dim_node_feat == self.dim_hid:
        #             constant_memory = b.srcdata['rh']
        #         else:
        #             constant_memory = self.node_feat_map(b.srcdata['rh'])
        #     else:
        #         constant_memory = self.constant(b.srcdata['ID'])
        #     thres = torch.sigmoid(self.choice(b.srcdata['ID']))
        #     b.srcdata['h'] = new_memory * thres + constant_memory * (1 - thres)
        # else:
        if self.dim_node_feat > 0:
            if self.dim_node_feat == self.dim_hid:
                constant_memory = b.srcdata['rh']
            else:
                constant_memory = self.node_feat_map(b.srcdata['rh'])
            b.srcdata['h'] = new_memory + constant_memory
        else:
            b.srcdata['h'] = new_memory