# DistTGL: Distributed Memory-based Temporal Graph Neural Network Training

## Overview

This repo is the open-sourced code for our work *DistTGL: Distributed Memory-based Temporal Graph Neural Network Training*.

## Requirements
- python >= 3.8.13
- pytorch >= 1.11.0
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.8.2
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

## Dataset
Download the dataset using the `down.sh` script. Note that we do not release the Flights dataset due to license restriction. You can download the Flight dataset directly from [this](https://github.com/fpour/DGB) link.

## Mini-batch Preparation

DistTGL pre-compute mini-batches before training. To ensure fast mini-batch loading from disk, please store the mini-batches in a fast SSD. In the paper, we use RAID0 array of two NVMe SSDs.

We first compile the sampler from [TGL](https://github.com/amazon-science/tgl) by
> python setup.py build_ext --inplace

Then generate the mini-batches using
> python gen_minibatch.py --data \<DatasetName> --gen_eval --minibatch_parallelism \<NumberofMinibatchParallelism>

where `<NumberofMinibatchParallelism>` is the `i` in `(i x j x k)` in the paper.

## Run

On each machine, execute
> torchrun --nnodes=\<NumberofMachines> --nproc_per_node=\<NumberofGPUPerMachine> --rdzv_id=\<JobID> --rdzv_backend=c10d --rdzv_endpoint=\<HostNodeIPAddress>:\<HostNodePort> train5.py --data \<DatsetName> --group \<NumberofGroupParallelism>

where `<NumberofGroupParallelism>` is the `k` in `(i x j x k)` in the paper.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
