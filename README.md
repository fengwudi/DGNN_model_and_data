# DGNN Model/Framework and Datasets

## 1 The Code Link of DGNN Model and Framwork
You can find them at [paper_code](https://github.com/fengwudi/DGNN_model_and_data/tree/main/paper_code)

## 2 Experiment of Our Paper

### 2.1 Environment
You can install the required conda environment for the experiment through [environment](https://github.com/fengwudi/DGNN_model_and_data/tree/main/experiment/environment).

 We run TGN/JODIE/DyREP/EvolveGCN/TGAT/APAN/CAW use `apan.yml`.

 Run Roland use `roland.yml`.

 Run DySAT/TGL use `tgl.yml`.

 Run DistTGL/SPEED use `speed.yml`.

 Run DyGFormer/DyGLib/TGLite use `tglite.yml`
.
### 2.2 Datasets

We give the experiment datasets, please notice that you show process data at each model and framework


|Datasets |  Link |
| ----------- | ----------- |
Wikipedia | http://snap.stanford.edu/jodie/wikipedia.csv
Reddit  | http://snap.stanford.edu/jodie/reddit.csv
MOOC | http://snap.stanford.edu/jodie/mooc.csv
Flights | https://zenodo.org/records/7213796#.Y1cO6y8r30o
DGraphFin | https://dgraph.xinye.com/dataset
ML25M | https://grouplens.org/datasets/movielens/25m/

For ML25M，we use `ratings.csv`.

Please download them at `experiment/datasets/`

Modify `ratings.csv` to `ml25m.csv` use 
```shell
mv experiment/datasets/ratings.csv experiment/datasets/ml25m.csv
```

### 2.3 Run

If you have prepared the environment and dataset, you can start the experiment.

Our experiment is at 9 models and 3 frameworks.

For JODIE and DyREP, we use the version implemented by TGN.

For DySAT, we use the version implemented by pytorch.

| Model | Code |
| ----------- | ----------- |
APAN |https://github.com/WangXuhongCN/APAN  
EvolveGCN(-H/-O) |https://github.com/IBM/EvolveGCN  
DySAT  |https://github.com/FeiGSSS/DySAT_pytorch
TGAT  |https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs 
TGN(TGN/DyREP/JODIE) |https://github.com/twitter-research/tgn 
CAW  |http://snap.stanford.edu/caw/  
ROLAND |https://github.com/snap-stanford/roland 
DyGFormer |https://github.com/yule-BUAA/DyGLib

| Framework | Code |
| ----------- | ----------- |
TGL|https://github.com/amazon-science/tgl
DistTGL|https://github.com/amazon-science/disttgl
SPEED |https://github.com/chenxi1228/SPEED
DyGLib |https://github.com/yule-BUAA/DyGLib
TGLite |https://github.com/ADAPT-uiuc/tglite

#### 2.3.1 Preprocess Data in TGN Format

As many model use TGN's format dataset, we first preprocess dataset of TGN.

```shell
cd experiment/run/TGN/
cp ../../datasets/{wikipedia.csv,reddit.csv,mooc.csv,Flights.csv,ml25m.csv,dgraphfin.npz} data/
python utils/preprocess_data.py --data wikipedia/reddit/mooc --bipartite
python utils/preprocess_data.py --data dgraphfin/Flights/ml25m
```

#### 2.3.2 DGNN Model

The detailed steps for the model are listed below.

##### (1) APAN

```shell
conda activate apan
cd experiment/run/APAN/
cp ../../datasets/{wikipedia.csv,reddit.csv,mooc.csv,Flights.csv,ml25m.csv,dgraphfin.npz} data/
mv ./data/wikipedia.csv ./data/wikipedia_raw.csv
mv ./data/reddit.csv ./data/reddit_raw.csv
mv ./data/mooc.csv ./data/mooc_raw.csv
mv ./data/Flights.csv ./data/flights_raw.csv
mv ./data/ml25m.csv ./data/ml25m_raw.csv

# preprocess datasets
(1) python preprocess/preprocess_csv.py --data wikipedia/reddit/mooc/flights
    python preprocess/preprocess_ml25m.py
    python preprocess/preprocess_dgraphfin.py

(2) python preprocess/BuildDglGraph.py --data wikipedia/reddit/mooc/flights/ml25m/dgraphfin

# run link prediction
python train.py -d wikipedia/reddit/mooc/flights/ml25m/dgraphfin

# run hyperparameter batch size 100/500/2000
python train.py -d wikipedia/reddit/flights --bs 100/500/2000

# run hyperparameter GNN layer 2l
python train.py -d wikipedia/reddit/flights --n_layer 2

# run node classification
python train.py -d wikipedia/reddit/mooc --tasks NC --bs 100
```


##### (2) CAW
```shell
conda activate apan
cd experiment/run/CAW/
ln -s ../TGN/data/ ./processed

# run link prediction
python main.py -d wikipedia/reddit/mooc/Flights --bs 1000 --n_degree 20 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --n_layer 1
python main.py -d ml25m/dgraphfin --bs 1000 --n_degree 20 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --n_layer 1 --walk_n_head 1

# run hyperparameter batch size 100/500/2000
python main.py -d wikipedia/reddit/Flights --bs 100/500/2000 --n_degree 20 --mode t --bias 1e-5 --pos_enc lp --walk_pool sum --n_layer 1
```

##### (3) DySAT
```shell
conda activate pytorch1.10
cd experiment/run/DySAT/
cp ../TGN/data/{ml_wikipedia.npy,ml_wikipedia.csv,ml_wikipedia_node.npy,ml_reddit.npy,ml_reddit.csv,ml_reddit_node.npy,ml_mooc.npy,ml_mooc.csv,ml_mooc_node.npy,ml_Flights.npy,ml_Flights.csv,ml_Flights_node.npy,ml_ml25m.npy,ml_ml25m.csv,ml_ml25m_node.npy,ml_dgraphfin.npy,ml_dgraphfin.csv,ml_dgraphfin_node.npy} raw_data/
cd raw_data/
python raw_data/preprecess_data.py --dataset wikipedia/reddit/mooc/Flights/ml25m/dgraphfin

# run link prediction
python -u train.py --dataset wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --time_steps -1 --featureless False --epochs 50 --early_stop 3
```
##### (4) EvolveGCN
```shell
conda activate apan
cd experiment/run/EvolveGCN/data/raw_data
cp ../../../../datasets/wikipedia.csv .
cp ../../../../datasets/reddit.csv .
cp ../../../../datasets/mooc.csv .
cp ../../../TGN/data/ml_Flights.csv .
cp ../../../../datasets/ml25m.csv .
cp ../../../../datasets/dgraphfin.npz .
python 2evolve.py --data wikipedia/reddit/mooc/Flights/ml25m/dgraphfin
cd ../..

# run link prediction
python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_o.yaml 

python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_o.yaml 

python run_exp.py --config_file ./experiments/parameters_mooc_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_mooc_linkpred_egcn_o.yaml 

python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_o.yaml 

python run_exp.py --config_file ./experiments/parameters_dgraphfin_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_dgraphfin_linkpred_egcn_o.yaml 

python run_exp.py --config_file ./experiments/parameters_ml25m_linkpred_egcn_h.yaml 
python run_exp.py --config_file ./experiments/parameters_ml25m_linkpred_egcn_o.yaml 

# run node classification
python run_exp.py --config_file ./experiments/parameters_wikipedia_nodecls_egcn_h.yaml
python run_exp.py --config_file ./experiments/parameters_wikipedia_nodecls_egcn_o.yaml

python run_exp.py --config_file ./experiments/parameters_reddit_nodecls_egcn_h.yaml
python run_exp.py --config_file ./experiments/parameters_reddit_nodecls_egcn_o.yaml

python run_exp.py --config_file ./experiments/parameters_mooc_nodecls_egcn_h.yaml
python run_exp.py --config_file ./experiments/parameters_mooc_nodecls_egcn_o.yaml
```
##### (5) Roland
```shell
conda activate roland
cd experiment/run/Roland
python setup.py develop
ln -s ../TGN/data/ ./roland_public_data
cd run/

# run link prediction
python main.py --cfg './(wikipedia/reddit/mooc/flights/ml25m/dgraphfin).yaml' --repeat 1
```

##### (6) TGAT
```shell
conda activate apan
cd experiment/run/TGAT
ln -s ../TGN/data/ ./processed

# run link prediction
python -u learn_edge.py -d wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --n_layer 1 --n_head 2 --prefix wiki/reddit/mooc/Flights/ml25m/dgraphfin

# run hyperparameter batch size 100/500/2000
python -u learn_edge.py -d wikipedia/reddit/Flights --bs 100/500/2000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --n_layer 1 --n_head 2 --prefix wiki/reddit/Flights

# run hyperparameter GNN layer 2l
python -u learn_edge.py -d wikipedia/reddit/Flights --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --n_layer 2 --n_head 2 --prefix wiki/reddit/Flights

# run node classification
python -u learn_node.py -d wikipedia/reddit/mooc --n_layer 1 --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod--n_head 2 --prefix wiki/reddit/mooc
```

##### (7) TGN/DyREP/JODIE
```shell
conda activate apan
cd experiment/run/TGN/

# run link prediction
python train_self_supervised.py -d wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --use_memory --prefix tgn-attn # TGN
python train_self_supervised.py -d wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn # JODIE
python train_self_supervised.py -d wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn # DyREP

# run hyperparameter batch size 100/500/2000
python train_self_supervised.py -d wikipedia/reddit/Flights --use_memory --prefix tgn-attn --bs 100/500/2000  # TGN
python train_self_supervised.py -d wikipedia/reddit/Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 100/500/2000 # JODIE
python train_self_supervised.py -d wikipedia/reddit/Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 100/500/2000 # DyREP

# run hyperparameter GNN layer 2l
python train_self_supervised.py -d wikipedia/reddit/Flights --use_memory --prefix tgn-attn --n_layer 2  # TGN
python train_self_supervised.py -d wikipedia/reddit/Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_layer 2 # DyREP

# run node classification
python train_supervised.py -d wikipedia/reddit/mooc --use_memory --prefix tgn-attn  # TGN
python train_supervised.py -d wikipedia/reddit/mooc --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn # JODIE
python train_supervised.py -d wikipedia/reddit/mooc --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn # DyREP
```

##### (8) DyGFormer
```shell
conda activate tglite
cd experiment/run/DyGLib/
ln -s ../TGN/data/ ./processed_data

python train_link_prediction.py --dataset_name wikipedia --model_name DyGFormer --batch_size 1000 # wikipedia
python train_link_prediction.py --dataset_name reddit --model_name DyGFormer --batch_size 1000 # reddit
python train_link_prediction.py --dataset_name mooc --model_name DyGFormer --batch_size 1000 # mooc
python train_link_prediction.py --dataset_name Flights --model_name DyGFormer --batch_size 1000 # Flights
python train_link_prediction.py --dataset_name dgraphfin --model_name DyGFormer --batch_size 1000 # dgraphfin
python train_link_prediction.py --dataset_name ml25m --model_name DyGFormer --batch_size 1000 # ml25m
```

#### 2.3.3 DGNN Framework

##### (1) TGL
```shell
conda activate pytorch1.10
cd experiment/run/TGL/
python setup.py build_ext --inplace
bash down.sh
cp ../../datasets/{dgraphfin.npz,ml25m.csv,Flights.csv} DATA/
cd DATA/
mkdir DGraphFin Flights ML25M
python dgraphfin2tgl.py
python flights2tgl.py
python ml25m2tgl.py
cd ..
python gen_graph.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin

# run link prediction
(you should add --rand_edge_features 172 if data is MOOC)
python train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --config ./config/APAN.yml # TGL-APAN
python train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --config ./config/DySAT.yml # TGL-DySAT you should change yaml file `duration`:100000 for WIKI/REDDIT/MOOC ; 7 for Flights ; 50 for DGraphFin ; 32000000 for ML25M
python train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --config ./config/JODIE.yml # TGL-JODIE
python train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --config ./config/TGAT.yml # TGL-TGAT
python train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --config ./config/TGN.yml # TGL-TGN

# run multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 train_dist.py --data ML25M/DGraphFin --config ./config/dist/(APAN.yml/DySAT.yml/JODIE.yml/TGAT.yml/TGN.yml) --num_gpus 1 # 1GPU
python -m torch.distributed.launch --nproc_per_node=3 train_dist.py --data ML25M/DGraphFin --config ./config/dist/(APAN.yml/DySAT.yml/JODIE.yml/TGAT.yml/TGN.yml) --num_gpus 2 # 2GPU
python -m torch.distributed.launch --nproc_per_node=5 train_dist.py --data ML25M/DGraphFin --config ./config/dist/(APAN.yml/DySAT.yml/JODIE.yml/TGAT.yml/TGN.yml) --num_gpus 4 # 4GPU
```

##### (2) DistTGL
```shell
conda activate speed
cd experiment/run/DistTGL/
python setup.py build_ext --inplace

# Please generate TGL dataset first
python gen_minibatch.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --gen_eval --minibatch_parallelism 1 --model tgn

# run mult-GPU
python -m torch.distributed.run --nproc_per_node=1 train.py --data WIKI/REDDIT/MOOC/Flights/ML25M/DGraphFin --pbar --group 1 --seed 0 --minibatch_parallelism 1 --model tgn # 1GPU

python -m torch.distributed.run --nproc_per_node=2 train.py --data ML25M/DGraphFin --pbar --group 2 --seed 0 --minibatch_parallelism 1 --model tgn # 2GPU
python -m torch.distributed.run --nproc_per_node=4 train.py --data ML25M/DGraphFin --pbar --group 4 --seed 0 --minibatch_parallelism 1 --model tgn # 4GPU

torchrun --nnodes 2  --node_rank 0 --nproc_per_node 4 --master_addr  <CurrentServerIP>  --master_port <Port> train.py --data ML25M/DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn # 8GPU
torchrun --nnodes 2  --node_rank 1 --nproc_per_node 4 --master_addr  <RemoveServerIP>  --master_port <Port> train.py --data ML25M/DGraphFin --pbar --group 8 --seed 0 --profile --minibatch_parallelism 1 --model tgn # 8GPU
```

##### (3) SPEED
```shell
conda activate speed
cd experiment/run/SPEED/
ln -s ../TGN/data/ ./data
python partition/transform.py --data wikipedia/reddit/mooc/Flights/ml25m/dgraphfin
cd partition
java -jar dist/partition.jar wikipedia/reddit/mooc/Flights/ml25m/dgraphfin 1/2/4 0.01 -degree_compute decay -algorithm hashing -lambda 1 -beta 0.1  -seed 0 -threads 8 -output output
cd ..

# run mult-GPU
python ddp_train_self_supervised.py --gpu 0 --data wikipedia/reddit/mooc/Flights/ml25m/dgraphfin --part_exp 0 --[tgat/tgn/dyrep/jodie] --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val # 1GPU
python ddp_train_self_supervised.py --gpu 0,1 --data ml25m/dgraphfin --part_exp 1 --[tgat/tgn/dyrep/jodie] --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val # 2GPU
python ddp_train_self_supervised.py --gpu 0,1,2,3 --data ml25m/dgraphfin --part_exp 2 --[tgat/tgn/dyrep/jodie] --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val # 4GPU
```

#### (4) DyGLib
```shell
conda activate tglite
cd experiment/run/DyGLib/
ln -s ../TGN/data/ ./processed_data

# JODIE
python train_link_prediction.py --dataset_name wikipedia --model_name JODIE --batch_size 1000
python train_link_prediction.py --dataset_name reddit --model_name JODIE --batch_size 1000
python train_link_prediction.py --dataset_name mooc --model_name JODIE --batch_size 1000
python train_link_prediction.py --dataset_name Flights --model_name JODIE --batch_size 1000
python train_link_prediction.py --dataset_name dgraphfin --model_name JODIE --batch_size 1000
python train_link_prediction.py --dataset_name ml25m --model_name JODIE --batch_size 1000

# DyRep
python train_link_prediction.py --dataset_name wikipedia --model_name DyRep --batch_size 1000
python train_link_prediction.py --dataset_name reddit --model_name DyRep --batch_size 1000
python train_link_prediction.py --dataset_name mooc --model_name DyRep --batch_size 1000
python train_link_prediction.py --dataset_name Flights --model_name DyRep --batch_size 1000
python train_link_prediction.py --dataset_name dgraphfin --model_name DyRep --batch_size 1000
python train_link_prediction.py --dataset_name ml25m --model_name DyRep --batch_size 1000

# TGAT
python train_link_prediction.py --dataset_name wikipedia --model_name TGAT --batch_size 1000
python train_link_prediction.py --dataset_name reddit --model_name TGAT --batch_size 1000
python train_link_prediction.py --dataset_name mooc --model_name TGAT --batch_size 1000
python train_link_prediction.py --dataset_name Flights --model_name TGAT --batch_size 1000
python train_link_prediction.py --dataset_name dgraphfin --model_name TGAT --batch_size 1000
python train_link_prediction.py --dataset_name ml25m --model_name TGAT --batch_size 1000

# TGN
python train_link_prediction.py --dataset_name wikipedia --model_name TGN --batch_size 1000
python train_link_prediction.py --dataset_name reddit --model_name TGN --batch_size 1000
python train_link_prediction.py --dataset_name mooc --model_name TGN --batch_size 1000
python train_link_prediction.py --dataset_name Flights --model_name TGN --batch_size 1000
python train_link_prediction.py --dataset_name dgraphfin --model_name TGN --batch_size 1000
python train_link_prediction.py --dataset_name ml25m --model_name TGN --batch_size 1000

# CAWN
python train_link_prediction.py --dataset_name wikipedia --model_name CAWN --batch_size 1000
python train_link_prediction.py --dataset_name reddit --model_name CAWN --batch_size 1000
python train_link_prediction.py --dataset_name mooc --model_name CAWN --batch_size 1000
python train_link_prediction.py --dataset_name Flights --model_name CAWN --batch_size 1000
python train_link_prediction.py --dataset_name dgraphfin --model_name CAWN --batch_size 1000
python train_link_prediction.py --dataset_name ml25m --model_name CAWN --batch_size 1000
```

#### (5) TGLite
```shell
conda activate tglite
cd experiment/run/TGLite/
ln -s ../TGL/DATA/ ./data

#apan
python examples/apan/train.py -d WIKI --data-path ./examples --bsize 1000 --opt-all 
python examples/apan/train.py -d REDDIT --data-path ./examples --bsize 1000 --opt-all 
python examples/apan/train.py -d MOOC --data-path ./examples --bsize 1000 --opt-all 
python examples/apan/train.py -d Flights --data-path ./examples --bsize 1000 --opt-all 
python examples/apan/train.py -d DGraphFin --data-path ./examples --bsize 1000 --opt-all
python examples/apan/train.py -d ML25M --data-path ./examples --bsize 1000 --opt-all

#jodie
python examples/jodie/train.py -d WIKI --data-path ./examples --bsize 1000
python examples/jodie/train.py -d REDDIT --data-path ./examples --bsize 1000
python examples/jodie/train.py -d MOOC --data-path ./examples --bsize 1000
python examples/jodie/train.py -d Flights --data-path ./examples --bsize 1000
python examples/jodie/train.py -d DGraphFin --data-path ./examples --bsize 1000
python examples/jodie/train.py -d ML25M --data-path ./examples --bsize 1000

#tgat
python examples/tgat/train.py -d WIKI --data-path ./examples --bsize 1000 --opt-all 
python examples/tgat/train.py -d REDDIT --data-path ./examples --bsize 1000 --opt-all
python examples/tgat/train.py -d MOOC --data-path ./examples --bsize 1000 --opt-all
python examples/tgat/train.py -d Flights --data-path ./examples --bsize 1000 --opt-all
python examples/tgat/train.py -d DGraphFin --data-path ./examples --bsize 1000 --opt-all
python examples/tgat/train.py -d ML25M --data-path ./examples --bsize 1000 --opt-all

#tgn
python examples/tgn/train.py -d WIKI --data-path ./examples --bsize 1000 --opt-all
python examples/tgn/train.py -d REDDIT --data-path ./examples --bsize 1000 --opt-all
python examples/tgn/train.py -d MOOC --data-path ./examples --bsize 1000 --opt-all
python examples/tgn/train.py -d Flights --data-path ./examples --bsize 1000 --opt-all 
python examples/tgn/train.py -d DGraphFin --data-path ./examples --bsize 1000 --opt-all
python examples/tgn/train.py -d ML25M --data-path ./examples --bsize 1000 --opt-all
```