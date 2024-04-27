# nohup python train.py --data WIKI --config ./config/JODIE.yml --gpu 3 > res/jodie_wiki.log &
# nohup python train.py --data REDDIT --config ./config/JODIE.yml --gpu 3 > res/jodie_reddit.log &
# nohup python train.py --data Flight --config ./config/JODIE.yml --rand_edge_features 172 --gpu 3 > res/jodie_flight.log &

# wait

# nohup python train.py --data WIKI --config ./config/TGAT.yml --gpu 3 > res/tgat_wiki.log &
# nohup python train.py --data REDDIT --config ./config/TGAT.yml --gpu 3 > res/tgat_reddit.log &
# nohup python train.py --data Flight --config ./config/TGAT.yml --rand_edge_features 172 --gpu 3 > res/tgat_flight.log &

# wait

# nohup python train.py --data WIKI --config ./config/TGN.yml --gpu 3 > res/tgn_wiki.log &
# nohup python train.py --data REDDIT --config ./config/TGN.yml --gpu 3 > res/tgn_reddit.log &
# nohup python train.py --data Flight --config ./config/TGN.yml --rand_edge_features 172 --gpu 3 > res/tgn_flight.log &

# wait
# nohup python train.py --data WIKI --config ./config/APAN.yml --gpu 3 > res/apan_wiki.log &
# nohup python train.py --data REDDIT --config ./config/APAN.yml --gpu 3 > res/apan_reddit.log &
# nohup python train.py --data Flight --config ./config/APAN.yml --rand_edge_features 172 --gpu 3 > res/apan_flight.log &

# wait

# nohup python train.py --data WIKI --config ./config/DySAT.yml --gpu 3 > res/dysat_wiki.log &
# nohup python train.py --data REDDIT --config ./config/DySAT.yml --gpu 3 > res/dysat_reddit.log &
nohup python train.py --data Flight --config ./config/DySAT.yml --rand_edge_features 172 --gpu 1 > res/dysat_flight.log &

# wait


# nohup python train.py --data ML25M --config ./config/JODIE.yml --gpu 3 > res/jodie_ml25m.log &
# nohup python train.py --data ML25M --config ./config/TGAT.yml --gpu 2 > res/tgat_ml25m.log &
# nohup python train.py --data ML25M --config ./config/TGN.yml --gpu 3 > res/tgn_ml25m.log &
nohup python train.py --data ML25M --config ./config/APAN.yml --gpu 3 > res/apan_ml25m.log &
nohup python train.py --data ML25M --config ./config/DySAT.yml --gpu 3 > res/dysat_ml25m1.log &

# wait


nohup python train.py --data MOOC --config ./config/JODIE.yml --gpu 1 --rand_edge_features 172 > res/jodie_mooc.log &
nohup python train.py --data MOOC --config ./config/TGAT.yml --gpu 2 --rand_edge_features 172 > res/tgat_mooc.log &
nohup python train.py --data MOOC --config ./config/TGN.yml --gpu 2 --rand_edge_features 172 > res/tgn_mooc.log &
nohup python train.py --data MOOC --config ./config/APAN.yml --gpu 2 --rand_edge_features 172 > res/apan_mooc.log &
nohup python train.py --data MOOC --config ./config/DySAT.yml --gpu 2 --rand_edge_features 172 > res/dysat_mooc.log &

nohup python train.py --data DGraphFin --config ./config/JODIE.yml --gpu 0 > res/jodie_dgraphfin.log &
nohup python train.py --data DGraphFin --config ./config/TGAT.yml --gpu 1 > res/tgat_dgraphfin.log &
nohup python train.py --data DGraphFin --config ./config/TGN.yml --gpu 2 > res/tgn_dgraphfin.log &
nohup python train.py --data DGraphFin --config ./config/APAN.yml --gpu 3 > res/apan_dgraphfin.log &
nohup python train.py --data DGraphFin --config ./config/DySAT.yml --gpu 3 > res/dysat_dgraphfin.log &



#多GPU

# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data ML25M --config ./config/dist/APAN.yml --num_gpus 1 > res/apan_ml25m_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12348 --nproc_per_node=3 train_dist.py --data ML25M --config ./config/dist/APAN.yml --num_gpus 2 > res/apan_ml25m_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12349 --nproc_per_node=5 train_dist.py --data ML25M --config ./config/dist/APAN.yml --num_gpus 4 > res/apan_ml25m_4gpu.log &
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data DGraphFin --config ./config/dist/APAN.yml --num_gpus 1 > res/apan_dgraphfin_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12350 --nproc_per_node=3 train_dist.py --data DGraphFin --config ./config/dist/APAN.yml --num_gpus 2 > res/apan_dgraphfin_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12351 --nproc_per_node=5 train_dist.py --data DGraphFin --config ./config/dist/APAN.yml --num_gpus 4 > res/apan_dgraphfin_4gpu.log &

# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data ML25M --config ./config/dist/JODIE.yml --num_gpus 1 > res/jodie_ml25m_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12352 --nproc_per_node=3 train_dist.py --data ML25M --config ./config/dist/JODIE.yml --num_gpus 2 > res/jodie_ml25m_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12353 --nproc_per_node=5 train_dist.py --data ML25M --config ./config/dist/JODIE.yml --num_gpus 4 > res/jodie_ml25m_4gpu.log &
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data DGraphFin --config ./config/dist/JODIE.yml --num_gpus 1 > res/jodie_dgraphfin_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12354 --nproc_per_node=3 train_dist.py --data DGraphFin --config ./config/dist/JODIE.yml --num_gpus 2 > res/jodie_dgraphfin_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12355 --nproc_per_node=5 train_dist.py --data DGraphFin --config ./config/dist/JODIE.yml --num_gpus 4 > res/jodie_dgraphfin_4gpu.log &

# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data ML25M --config ./config/dist/TGAT.yml --num_gpus 1 > res/tgat_ml25m_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12356 --nproc_per_node=3 train_dist.py --data ML25M --config ./config/dist/TGAT.yml --num_gpus 2 > res/tgat_ml25m_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12357 --nproc_per_node=5 train_dist.py --data ML25M --config ./config/dist/TGAT.yml --num_gpus 4 > res/tgat_ml25m_4gpu.log &
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data DGraphFin --config ./config/dist/TGAT.yml --num_gpus 1 > res/tgat_dgraphfin_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12358 --nproc_per_node=3 train_dist.py --data DGraphFin --config ./config/dist/TGAT.yml --num_gpus 2 > res/tgat_dgraphfin_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12359 --nproc_per_node=5 train_dist.py --data DGraphFin --config ./config/dist/TGAT.yml --num_gpus 4 > res/tgat_dgraphfin_4gpu.log &

# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data ML25M --config ./config/dist/TGN.yml --num_gpus 1 > res/tgn_ml25m_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12360 --nproc_per_node=3 train_dist.py --data ML25M --config ./config/dist/TGN.yml --num_gpus 2 > res/tgn_ml25m_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12361 --nproc_per_node=5 train_dist.py --data ML25M --config ./config/dist/TGN.yml --num_gpus 4 > res/tgn_ml25m_4gpu.log &
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data DGraphFin --config ./config/dist/TGN.yml --num_gpus 1 > res/tgn_dgraphfin_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12362 --nproc_per_node=3 train_dist.py --data DGraphFin --config ./config/dist/TGN.yml --num_gpus 2 > res/tgn_dgraphfin_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12363 --nproc_per_node=5 train_dist.py --data DGraphFin --config ./config/dist/TGN.yml --num_gpus 4 > res/tgn_dgraphfin_4gpu.log &

nohup python -m torch.distributed.launch --master_port=12359 --nproc_per_node=2 train_dist.py --data ML25M --config ./config/dist/DySAT.yml --num_gpus 1 > res/dysat_ml25m_1gpu.log &
nohup python -m torch.distributed.launch --master_port=12360 --nproc_per_node=3 train_dist.py --data ML25M --config ./config/dist/DySAT.yml --num_gpus 2 > res/dysat_ml25m_2gpu.log &
nohup python -m torch.distributed.launch --master_port=12361 --nproc_per_node=5 train_dist.py --data ML25M --config ./config/dist/DySAT.yml --num_gpus 4 > res/dysat_ml25m_4gpu.log &
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data DGraphFin --config ./config/dist/DySAT.yml --num_gpus 1 > res/dysat_dgraphfin_1gpu.log &
# nohup python -m torch.distributed.launch --master_port=12362 --nproc_per_node=3 train_dist.py --data DGraphFin --config ./config/dist/DySAT.yml --num_gpus 2 > res/dysat_dgraphfin_2gpu.log &
# nohup python -m torch.distributed.launch --master_port=12363 --nproc_per_node=5 train_dist.py --data DGraphFin --config ./config/dist/DySAT.yml --num_gpus 4 > res/dysat_dgraphfin_4gpu.log &