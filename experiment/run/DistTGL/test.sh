nohup python -m torch.distributed.run --master_port=29501 --nproc_per_node=2 train.py --data WIKI --pbar --group 2 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_wiki.log &
nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data REDDIT --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_reddit.log &
nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data Flight --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_flight.log &
nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data ML25M --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_ml25m1.log &
nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data MOOC --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_mooc.log &


nohup python -m torch.distributed.run --master_port=29500 --nproc_per_node=2 train.py --data ML25M --pbar --group 2 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_ml25m_2gpu.log &
nohup python -m torch.distributed.run --master_port=29000 --nproc_per_node=4 train.py --data ML25M --pbar --group 4 --seed 0 --minibatch_parallelism 1 --model tgn > res/tgn_ml25m_4gpu1.log &






torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 train.py --data ML25M --pbar --group 2  --seed 0 --minibatch_parallelism 1 --model tgn

# wait

# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data WIKI --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgat > res/tgat_wiki.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data REDDIT --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgat > res/tgat_reddit.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data Flight --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgat > res/tgat_flight.log &

# wait

# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data WIKI --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model jodie > res/jodie_wiki.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data REDDIT --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model jodie > res/jodie_reddit.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data Flight --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model jodie > res/jodie_flight.log &

# wait

# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data WIKI --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model apan > res/apan_wiki.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data REDDIT --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model apan > res/apan_reddit.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data Flight --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model apan > res/apan_flight.log &

# wait

# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data WIKI --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model dysat > res/dysat_wiki.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data REDDIT --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model dysat > res/dysat_reddit.log &
# nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data Flight --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model dysat > res/dysat_flight.log &

# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=0 --rdzv_backend=c10d --rdzv_endpoint=10.214.211.185:29400 train.py --data WIKI --pbar --group 8 --seed 0 --minibatch_parallelism 1 --model tgn
# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=10.214.211.185:29400 train.py --data WIKI --pbar --group 8 --seed 0 --minibatch_parallelism 1 --model tgn

# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint=10.214.211.185:12345 train.py --data WIKI --data WIKI --pbar --group 8 --seed 0 --minibatch_parallelism 1 --model tgn
# torchrun --nnodes=2 --nproc_per_node=4 --rdzv_endpoint=10.214.211.185:12345 train.py --data WIKI --data WIKI --pbar --group 8 --seed 0 --minibatch_parallelism 1 --model tgn


nohup torchrun --nnodes 2  --node_rank 0 --nproc_per_node 4 --master_addr  10.214.211.184  --master_port 29500 train.py --data DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/tgn_dgraphfin_8gpu1.log &
nohup torchrun --nnodes 2  --node_rank 1 --nproc_per_node 4 --master_addr  10.214.211.184  --master_port 29500 train.py --data DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/tgn_dgraphfin_8gpu1.log &

# python -m torch.distributed.launch --use_env --master_port=12355 --nproc_per_node=4 --nnodes=2 --node_rank=0  --master_addr=10.214.211.185  train.py --data WIKI --group 8 --seed 0 --profile --minibatch_parallelism 1
# python -m torch.distributed.launch --use_env --master_port=12355 --nproc_per_node=4 --nnodes=2 --node_rank=1  --master_addr=10.214.211.185  train.py --data WIKI --group 8 --seed 0 --profile --minibatch_parallelism 1

nohup torchrun --nnodes 2  --node_rank 0 --nproc_per_node 4 --master_addr  10.214.211.184  --master_port 29500 train.py --data ML25M --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/tgn_ml25m_8gpu1.log &
nohup torchrun --nnodes 2  --node_rank 1 --nproc_per_node 4 --master_addr  10.214.211.184  --master_port 29500 train.py --data ML25M --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/tgn_ml25m_8gpu1.log &



nohup python -m torch.distributed.run --nproc_per_node=1 train.py --data DGraphFin --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn > res/new/tgn_dgraphfin1.log &
nohup python -m torch.distributed.run --nproc_per_node=2 train.py --data DGraphFin --pbar --group 2 --seed 0 --minibatch_parallelism 1 --model tgn > res/new/tgn_dgraphfin_2gpu2.log &
nohup python -m torch.distributed.run --nproc_per_node=4 train.py --data DGraphFin --pbar --group 4 --seed 0 --minibatch_parallelism 1 --model tgn > res/new/tgn_dgraphfin_4gpu2.log &

nohup torchrun --nnodes 2  --node_rank 0 --nproc_per_node 4 --master_addr  10.214.211.185  --master_port 29500 train.py --data DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/new/tgn_dgraphfin_8gpu.log &
nohup torchrun --nnodes 2  --node_rank 1 --nproc_per_node 4 --master_addr  10.214.211.185  --master_port 29500 train.py --data DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn > res/new/tgn_dgraphfin_8gpu.log &