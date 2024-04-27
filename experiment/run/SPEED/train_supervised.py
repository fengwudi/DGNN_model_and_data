import argparse
import json
import pathlib
import time
import traceback

import numpy as np
import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from CHANGELOG import MODEL_VERSION, CHECK_VERSION
from tiger.data.data_loader import GraphCollator, load_jodie_data_for_node_task
from tiger.data.graph import Graph
from tiger.eval_utils import eval_node_classification
from tiger.model.basic_modules import MLP
from tiger.model.feature_getter import NumericalFeature
from tiger.model.tiger import TIGER
from tiger.utils import BackgroundThreadGenerator

from init_utils import init_data, init_model
from train_utils import EarlyStopMonitor, get_logger, hash_args, seed_all



def run(*, prefix, gpu, seed, lr, n_epochs,
        patience, force, ckpt_path, root, bs,
        use_valid, dropout,
        # above are new parameters
        data, dim, feature_as_buffer, num_workers,
        hit_type, restarter_type, hist_len,
        n_neighbors, n_layers, n_heads, 
        strategy, msg_src, upd_src,
        mem_update_type, msg_tsfm_type, no_memory,
        dyrep, embedding_type, 
        part_exp, divide_method, top_k, static_shared_nodes, sync_mode, shuffle_parts, world_size,
        **kwargs
        ):
    args = {k: v for k, v in locals().items() 
            if not k in {'gpu', 'force', 'kwargs'}}
    HASH = hash_args(**args, MODEL_VERSION=MODEL_VERSION)
    prefix += '.' + HASH
    if not prefix:
        raise ValueError('Prefix should be given explicitly.')
    if gpu == "-1":
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')

    RESULT_SAVE_PATH = f"results/{prefix}.json"
    PICKLE_SAVE_PATH = "results/{}.pkl".format(prefix)

    ckpts_dir = pathlib.Path(f"./saved_checkpoints/{prefix}")
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    get_checkpoint_path = lambda epoch: ckpts_dir / f'{epoch}.pth'

    logger = get_logger(prefix)
    logger.info(f'[START {prefix}]')
    logger.info(f'Model version: {MODEL_VERSION}')
    logger.info(", ".join([f"{k}={v}" for k, v in args.items()]))

    if pathlib.Path(RESULT_SAVE_PATH).exists() and not force:
        logger.info('Duplicate task! Abort!')
        return False

    try:
        seed_all(seed)
        # ============= Load Data ===========
        (
            nfeats, efeats, full_data, train_data, val_data, test_data, 
        ) = load_jodie_data_for_node_task(data, train_seed=seed, root=root, 
                                          use_validation=use_valid)

        train_graph = Graph.from_data(train_data, strategy=strategy, seed=seed)
        full_graph = Graph.from_data(full_data, strategy=strategy, seed=seed)

        train_collator = GraphCollator(train_graph, n_neighbors, n_layers,
                                       restarter=restarter_type, hist_len=hist_len)
        eval_collator = GraphCollator(full_graph, n_neighbors, n_layers,
                                      restarter=restarter_type, hist_len=hist_len)

        train_dl = DataLoader(train_data, batch_size=bs, collate_fn=train_collator, pin_memory=True, num_workers=num_workers)
        val_dl = DataLoader(val_data, batch_size=bs, collate_fn=eval_collator)
        test_dl = DataLoader(test_data, batch_size=bs, collate_fn=eval_collator)

        # ============= Init Model ===========
        encoder = init_model(
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
        
        # load model ckpt
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
        encoder.eval()

        decoder = MLP(encoder.nfeat_dim, dropout=dropout).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(decoder.parameters(), lr=lr)

        val_aucs = []
        if use_valid:
            early_stopper = EarlyStopMonitor(max_round=patience)
        for epoch in range(n_epochs):
            start_epoch_t0 = time.time()
            logger.info('Start {} epoch'.format(epoch))

            m_loss = []
            it = BackgroundThreadGenerator(train_dl)
            it = tqdm.tqdm(it, total=len(train_dl), ncols=50)

            encoder.reset()
            decoder.train()
            for i_batch, (src_ids, dst_ids, neg_dst_ids, ts, eids, labels, comp_graph) in enumerate(it):
                bs = len(src_ids)
                src_ids = src_ids.long().to(device)
                dst_ids = dst_ids.long().to(device)
                neg_dst_ids = neg_dst_ids.long().to(device)
                ts = ts.float().to(device)
                eids = eids.long().to(device)
                labels = labels.float().to(device)
                comp_graph.to(device)
                with torch.no_grad():
                    _, h, *_ = encoder.contrast_learning(src_ids, dst_ids, neg_dst_ids,
                                                       ts, eids, comp_graph)
                optimizer.zero_grad()
                pred_y = decoder(h[:bs])  # only positive nodes
                loss = loss_fn(pred_y, labels)
                loss.backward()
                optimizer.step()
                m_loss.append(loss.item())

            epoch_time = time.time() - start_epoch_t0

            val_auc = eval_node_classification(encoder, decoder, val_dl, device)
            val_aucs.append(val_auc)

            logger.info('Epoch {:4d} training took  {:.2f}s'.format(epoch, epoch_time))
            logger.info(f'Epoch mean loss: {np.mean(m_loss):.4f}')
            logger.info(f'Epoch validation auc: {val_auc:.4f}')

            if use_valid:
                if early_stopper.early_stop_check(val_auc):
                    logger.info('No improvement over {} epochs, stop training'.format(
                        early_stopper.max_round))
                    break
                else:
                    torch.save(decoder.state_dict(), get_checkpoint_path(epoch))
        
        if use_valid:
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model_state = torch.load(best_model_path)
            decoder.load_state_dict(model_state)

            best_val_auc = val_aucs[early_stopper.best_epoch]
            logger.info(f'[ Val] auc: {best_val_auc:.4f}')

            test_auc = eval_node_classification(encoder, decoder, test_dl, device)
        else:
            logger.info('No validation set. Use the last epoch result.')
            test_auc = val_aucs[-1]

        logger.info(f'[Test] auc: {test_auc:.4f}')

        results = args.copy()
        results.update(
            prefix=prefix,
            VERSION=MODEL_VERSION,
            test_auc=test_auc,
        )
        json.dump(results, open(RESULT_SAVE_PATH, 'w'))

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(e)
        raise

    
def get_args():
    parser = argparse.ArgumentParser()
    # Exp Setting
    parser.add_argument('--code', type=str, default='', help='Name of the saved result and model')
    parser.add_argument('--json', type=str, default='', help='Path to model result (json file)')
    parser.add_argument('--ckpt', type=str, default='', help='Path to model check point')
    parser.add_argument('--root', type=str, default='.', help='Dataset root')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='Cuda index')

    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=100, help='Batch size')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--use_valid', action='store_true', help='Use validation set')

    parser.add_argument('--force', action='store_true', help='Overwirte the existing task')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    if args.code:
        with open(f'./results/{args.code}.json') as fh:
            saved_results = json.load(fh)
        ckpt_path = f'./saved_models/{args.code}.pth'
    else:
        with open(args.json) as fh:
            saved_results = json.load(fh)
        ckpt_path = args.ckpt_path
    
    if not CHECK_VERSION(saved_results['VERSION'], MODEL_VERSION):
        raise ValueError('version not match: {} != {}'.format(
            saved_results['VERSION'], MODEL_VERSION))
    
    prefix = saved_results['HASH'] if saved_results['prefix'] == '' else saved_results['prefix']
    prefix += '-node'
    kwargs = {k: v for k, v in saved_results.items() if k not in 
                {'prefix', 'seed', 'lr', 'n_epochs', 'bs', 'patience', 'root', 'dropout'}
             }

    run(
        prefix=prefix, gpu=args.gpu, seed=args.seed,
        lr=args.lr, dropout=args.dropout, bs=args.bs, n_epochs=args.n_epochs,
        patience=args.patience, force=args.force,
        use_valid=args.use_valid, root=args.root, ckpt_path=ckpt_path,
        **kwargs
    )
