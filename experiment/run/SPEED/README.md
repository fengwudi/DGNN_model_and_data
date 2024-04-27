# SPEED
Codes for the paper "SPEED: Streaming Partition and Parallel Acceleration for Temporal Interaction Graph Embedding"

# Data

For the datasets Wikipedia, Reddit, MOOC and LastFM, please download data from the [project homepage of JODIE](https://snap.stanford.edu/jodie/) and pre-process them with the script provided by [TGN](https://github.com/twitter-research/tgn).

For ML25m, please download data from the [grouplens](https://grouplens.org/datasets/movielens/25m/) and put the file ratings.csv into the folder [Datasets](Datasets) then, pre-process it with the [ML25m2TGN.py](ML25m2TGN.py). Or you can download the dataset which is pre-processed by us from [Kaggle](https://www.kaggle.com/datasets/chenxi1228/ml25m-tgn-style), and put it in the folder [data](data).

For DGraphfin, please download data from the [DGraph](https://dgraph.xinye.com/dataset) and put it into the folder [Datasets](Datasets) then, pre-process it with the [DGraphFin2TGN.py](DGraphFin2TGN.py). Or you can download the dataset which is pre-processed by us from [Kaggle](https://www.kaggle.com/datasets/chenxi1228/dgraphfin-tgn-style), and put it in the folder [data](data).

For Taobao, please download data from the [Tianchi](https://tianchi.aliyun.com/dataset/649) and put the file into the folder [Datasets](Datasets) then, pre-process it with the [Taobao2TGN.py](Taobao2TGN.py). Or you can download the dataset which is pre-processed by us from [Kaggle](https://www.kaggle.com/datasets/chenxi1228/taobao-tgn-style), and put it in the folder [data](data).

# How to use

## Streaming Edge Partitioning Component (SEP)

### Pre-processing for partition

If you would like to using our SEP, please first do one more pre-processing step, which transform the csv data file to txt format.

```
python partition/transform.py --data [DATA]
```

We also provide the processed files for datasets Wikipedia, Reddit, MOOC and LastFM in this repo, so for these four datasets, you could directly conduct the SEP.

Then move forward to the partition folder (by `cd partition`).

### Usage: Parameters:

`graphfile`: the name of the file that stores the graph to be partitioned.

`nparts`: the number of parts that the graph will be partitioned into. Maximum value 256.

`topk`: topk nodes are selected as hubs.

Options:
`-degree_compute string` -> options to compute nodes' centrality (normal decay)

`-algorithm string` -> specifies the algorithm to be used (hdrf hashing mymethod). Default mymethod.

`-lambda double` -> specifies the lambda parameter for hdrf and mymethod. Default 1.

`-beta double` -> specifies the beta parameter for hdrf. Default 0.1.

`-seed int` -> seed for repeated experiments. Default 0.

`-threads integer` -> specifies the number of threads used by the application. Default all available processors.

`-output string` -> specifies the prefix for the name of the files where the output will be stored (files: prefix.info, prefix.edges and prefix.vertices).


You can run the partition for temporal interaction graphs by:

```
java -jar dist/partition.jar [DATA] [2/4/8...] [0.01/0.05/0.10...] -degree_compute [normal/decay] -algorithm [mymethod/hdrf/hashing] -lambda [1/2...] -beta [0.1/0.2...]  -seed [0/1/2/...] -threads [1/2/4...] -output output
```

## Parallel Acceleration Component (PAC)

Some parameters need to be transfered from SEP:

|       SEP       |     PAC    |
|:---------------:|:----------:|
|       topk      |  top_k/100 |
|      nparts     | 2^part_exp |
|     topk*100    |    top_k   |
| $\sqrt{nparts}$ |  part_exp  |


### Regular training
Please use the main branch to proceed a regular training. You can also use it to train big datasets, however, by applying the codes in the branch "big_datasets" would save you some time.

```
python ddp_train_self_supervised.py --gpu 0,1,2,3 --data [DATA] --part_exp [1/2/3...] --[jodie/tgn/tgat/dyrep/tige] --prefix [add_your_prefered_prefix] --top_k [0/1/5/10/-1] --seed [0/1/2...] --sync_mode [last/none/average] --divide_method pre
```

You could also choose to add argument `--shuffle_parts` to enable the random shuffling, when you have more graph partitions than your number of GPUs.


### Training big datasets
If you would like to train the three big datasets, you may need to pass two arguments to avoid OOM problem:

`--backup_memory_to_cpu` `--testing_on_cpu`.

And you may need to specify a vector dimension for nodes and edges, e.g.,`--dim 100`.

```
python ddp_train_self_supervised.py --gpu 0,1,2,3 --data [DATA] --part_exp [1/2/3...] --[jodie/tgn/tgat/dyrep/tige] --prefix [add_your_prefered_prefix] --top_k [0/1/5/10/-1] --seed [0/1/2...] --sync_mode [last/none/average] --divide_method pre --backup_memory_to_cpu --testing_on_cpu --no_ind_val --dim 100
```

### I/O Accelerating and reusing sub-graphs
If you would like to train big datasets and want to accelerate the I/O process to save some time, please use the codes in the branch "IO-Accelerating".

In this branch, the whole training process is being accelerated by optimising the I/O process and omitting unnecessary inductive validation and only using `testing_from_begin` setting.

You should (optionally) first save the graphs and sub-graphs by running `save_graphs.py` for different (parameters): `--data`, `--part_exp`, `--gpu`, `--divide_method`, `--top_k` and `--seed` (the saved graphs will be the same if these parameters are the same), you may also would like to set a different `--prefix`:

```
python save_graphs.py --gpu 0,1,2,3 --data [DATA] --part_exp [1/2/3...] --prefix [add_your_prefered_prefix] --top_k [0/1/5/10/-1] --seed [0/1/2...] --divide_method pre --save_mode save
```

Then you can reuse the saved graphs by setting `--save_mode read` for training for different backbone models or other parameters. 

```
python ddp_train_self_supervised.py --gpu 0,1,2,3 --data [DATA] --part_exp [1/2/3...] --[jodie/tgn/tgat/dyrep/tige] --prefix [add_your_prefered_prefix] --top_k [0/1/5/10/-1] --seed [0/1/2...] --sync_mode [last/none/average] --divide_method pre --backup_memory_to_cpu --testing_on_cpu --no_ind_val --dim 100 --save_mode read
```

Note that this branch only support for the situation, that your number of GPUs equals to 2^part_exp, for now.


Node Classification
```
python train_supervised.py --code [CODE]
```
Here, [CODE] is the HASH code of a trained model with `train_self_supervised.py`.


