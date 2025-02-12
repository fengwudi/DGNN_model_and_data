import math

def get_config(model, data, tot_rank, minibatch_parallelism=1):
    if model == 'tgn':
        sample_param = {
            'layer': 1, 
            'neighbor': [20],
            'strategy': 'uniform',
            'prop_time': False,
            'history': 1,
            'duration': 0,
            'num_thread': 32
        }
        memory_param = {
            'type': 'node',
            'dim_time': 100,
            'deliver_to': 'self',
            'mail_combine': 'last',
            'memory_update': 'smart',
            'mailbox_size': 1,
            'combine_node_feature': True,
            'dim_out': 100
        }
        gnn_param = {
            'arch': 'transformer_attention',
            'layer': 1,
            'att_head': 2,
            'dim_time': 100,
            'dim_out': 100
        }
    elif model == 'tgat':
        sample_param = {
            'layer': 1, 
            'neighbor': [10],
            'strategy': 'uniform',
            'prop_time': False,
            'history': 1,
            'duration': 0,
            'num_thread': 32
        }
        memory_param = {
            'type': 'none',
            'dim_out': 0
        }
        gnn_param = {
            'arch': 'transformer_attention',
            'layer': 1,
            'att_head': 2,
            'dim_time': 100,
            'dim_out': 100
        }
    elif model == 'jodie':
        sample_param = {
            'no_sample': True,
            'history': 1
        }
        memory_param = {
            'type': 'node',
            'dim_time': 100,
            'deliver_to': 'self',
            'mail_combine': 'last',
            'memory_update': 'smart',
            'mailbox_size': 1,
            'combine_node_feature': True,
            'dim_out': 100
        }
        gnn_param = {
            'arch': 'identity',
            'time_transform': 'JODIE'
        }
    elif model == 'apan':
        sample_param = {
            'layer': 1, 
            'neighbor': [10],
            'strategy': 'recent',
            'prop_time': False,
            'history': 1,
            'duration': 0,
            'num_thread': 32,
            'no_neg': True
        }
        memory_param = {
            'type': 'node',
            'dim_time': 100,
            'deliver_to': 'neighbors',
            'mail_combine': 'last',
            'memory_update': 'smart',
            'mailbox_size': 2,
            'combine_node_feature': False,
            'dim_out': 100
        }
        gnn_param = {
            'arch': 'identity'
        }
    elif model == 'dysat':
        sample_param = {
            'layer': 1, 
            'neighbor': [10],
            'strategy': 'uniform',
            'prop_time': True,
            'history': 3,
            'duration': 100000,
            'num_thread': 32
        }
        memory_param = {
            'type': 'none',
            'dim_out': 0
        }
        gnn_param = {
            'arch': 'transformer_attention',
            'layer': 2,
            'att_head': 2,
            'dim_time': 0,
            'dim_out': 100,
            'combine': 'rnn'
        }
    # if data in ['GDELT', 'LINK', 'MAG', 'TaoBao', 'DGraphFin', 'ML25M']:
    #     epoch = 10
    # else:
    #     epoch = 100
    train_param = {
        #'epoch': math.ceil(epoch / tot_rank * minibatch_parallelism),
        'epoch': 50,
        'batch_size': 1000,
        'lr': 0.0001,
        'dropout': 0.1,
        'att_dropout': 0.1
    }

    train_param['train_neg_samples'] = 1
    train_param['eval_neg_samples'] = 1

    #if data in ['GDELT', 'LINK']:
        #train_param['batch_size'] = 3200
        # train_param['lr'] = 0.00001

    return sample_param, memory_param, gnn_param, train_param