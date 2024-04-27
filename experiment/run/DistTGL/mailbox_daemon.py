import os
import time
from multiprocessing import Process

def start_mailbox_daemon(omp_num_threads, group_id, dims_mailbox, edge_classification=False):
    os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(omp_num_threads)

    import torch
    import dgl
    import random
    import numpy as np
    from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

    num_nodes, mem_dim, max_read_nodes, max_write_nodes, tot_group_rank, mb_tot_group_rank = dims_mailbox
    minibatch_parallelism = tot_group_rank // mb_tot_group_rank

    node_memory = get_shared_mem_array(str(group_id) + 'node_memory', torch.Size([num_nodes, mem_dim]), dtype=torch.float32)
    mails = get_shared_mem_array(str(group_id) + 'mails', torch.Size([num_nodes, 2 * mem_dim]), dtype=torch.float32)

    memory_read_buffer = get_shared_mem_array('{}_memory_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, mem_dim]), dtype=torch.float32)
    mail_read_buffer = get_shared_mem_array('{}_mail_read_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes, 2 * mem_dim]), dtype=torch.float32)
    read_1idx_buffer = get_shared_mem_array('{}_read_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, mb_tot_group_rank, max_read_nodes + 1]), dtype=torch.int64)
    read_status = get_shared_mem_array('{}_read_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
    memory_write_buffer = get_shared_mem_array('{}_memory_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, mem_dim]), dtype=torch.float32)
    mail_write_buffer = get_shared_mem_array('{}_mail_write_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes, 2 * mem_dim]), dtype=torch.float32)
    write_1idx_buffer = get_shared_mem_array('{}_write_1idx_buffer'.format(group_id), torch.Size([tot_group_rank, max_write_nodes + 1]), dtype=torch.int64)
    write_status = get_shared_mem_array('{}_write_status'.format(group_id), torch.Size([tot_group_rank]), dtype=torch.int)
    reset_status = get_shared_mem_array('{}_reset_status'.format(group_id), torch.Size([1]), dtype=torch.int)

    # print('deamon read_status:', read_status.data_ptr(), 'write_status:', write_status.data_ptr())

    def write_rank(rank):
        nonlocal node_memory, mails, memory_write_buffer, mail_write_buffer, write_1idx_buffer
        # print('write for rank {}'.format(rank))
        with torch.no_grad():
            write_1idx = write_1idx_buffer[rank]
            write_idx = write_1idx[1:1 + write_1idx[0]]
            node_memory[write_idx] = memory_write_buffer[rank][:write_idx.shape[0]]
            mails[write_idx] = mail_write_buffer[rank][:write_idx.shape[0]]

    def read_rank(rank):
        nonlocal node_memory, mails, memory_read_buffer, mail_read_buffer, read_1idx_buffer
        # print('read for rank {}'.format(rank))
        with torch.no_grad():
            for i in range(mb_tot_group_rank):
                read_1idx = read_1idx_buffer[rank][i]
                read_idx = read_1idx[1:1 + read_1idx[0]]
                torch.index_select(node_memory, 0, read_idx, out=memory_read_buffer[rank][i][:read_idx.shape[0]])
                torch.index_select(mails, 0, read_idx, out=mail_read_buffer[rank][i][:read_idx.shape[0]])
                if edge_classification:
                    break

    with torch.no_grad():
        while True:
            node_memory.zero_()
            mails.zero_()
            # read_status.zero_()
            # write_status.zero_()
            curr_rank = 0
            reset_flag = False
            while True:
                # write to node memory
                # print('curr_rank:{}'.format(curr_rank))
                for rank in range(curr_rank, curr_rank + minibatch_parallelism):
                    # print('\twait for write request from {}...'.format(rank), end='')
                    while write_status[rank] == 0:
                        if reset_status[0] == 1:
                            reset_flag = True
                            reset_status[0] = 2
                            break
                        # print('\tread status:{} write status:{}'.format(read_status, write_status))
                        # time.sleep(0.01)
                        pass
                    # print('done!')
                    if not reset_flag:
                        write_rank(rank)
                        # print('\tset write_status[{}] to 0'.format(rank))
                        write_status[rank] = 0
                
                if not reset_flag:
                    # serving next rank
                    curr_rank += minibatch_parallelism
                    curr_rank = 0 if curr_rank >= tot_group_rank else curr_rank
                    # print('\t advance curr_rank to {}'.format(curr_rank))
                    # print('\twait for read request from {}...'.format(curr_rank), end='')
                    while not (read_status[curr_rank] == 1):
                        if reset_status[0] == 1:
                            reset_flag = True
                            reset_status[0] = 2
                            break
                        pass
                    if not reset_flag:
                        for rank in range(curr_rank, curr_rank + minibatch_parallelism):
                            while not (read_status[rank] == 1):
                                if reset_status[0] == 1:
                                    reset_flag = True
                                    reset_status[0] = 2
                                    break
                                pass
                            if not reset_flag:
                                # serve read request
                                # print('done!')
                                read_rank(rank)
                                # print('reset read status to 0')
                                read_status[rank] = 0
                if reset_flag:
                    # print('\tbreak inner loop!')
                    break

            