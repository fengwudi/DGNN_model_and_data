java -jar dist/partition.jar Flights 1 0.1 -degree_compute decay -algorithm hashing -lambda 1 -beta 0.1  -seed 0 -threads 8 -output output


# nohup python ddp_train_self_supervised.py --port 12348 --gpu 0 --data wikipedia --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgn_wiki.log &
# nohup python ddp_train_self_supervised.py --port 12349 --gpu 1 --data reddit --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgn_reddit.log &
# nohup python ddp_train_self_supervised.py --port 12350 --gpu 2 --data Flights --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgn_flight.log &


# nohup python ddp_train_self_supervised.py --port 12352 --gpu 3 --data wikipedia --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgat_wiki.log &
# nohup python ddp_train_self_supervised.py --port 12356 --gpu 3 --data wikipedia --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/jodie_wiki.log &
# nohup python ddp_train_self_supervised.py --port 12360 --gpu 3 --data wikipedia --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/dyrep_wiki.log &

# wait

# nohup python ddp_train_self_supervised.py --port 12353 --gpu 0 --data reddit --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgat_reddit.log &
# nohup python ddp_train_self_supervised.py --port 12357 --gpu 1 --data reddit --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/jodie_reddit.log &
# nohup python ddp_train_self_supervised.py --port 12361 --gpu 2 --data reddit --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/dyrep_reddit.log &


# wait

# nohup python ddp_train_self_supervised.py --port 12362 --gpu 0 --data Flights --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/dyrep_flight.log &
# nohup python ddp_train_self_supervised.py --port 12354 --gpu 1 --data Flights --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgat_flight.log &
nohup python ddp_train_self_supervised.py --port 12358 --gpu 1 --data Flights --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/jodie_flight.log &

# wait

# echo "MOOC"

# nohup python ddp_train_self_supervised.py --port 20000 --gpu 0 --data mooc --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgn_mooc.log &
# nohup python ddp_train_self_supervised.py --port 20001 --gpu 1 --data mooc --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/dyrep_mooc.log &
# nohup python ddp_train_self_supervised.py --port 20002 --gpu 2 --data mooc --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgat_mooc.log &
# nohup python ddp_train_self_supervised.py --port 20003 --gpu 3 --data mooc --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/jodie_mooc.log &

wait

echo "ML25M 1GPU:"
# nohup python ddp_train_self_supervised.py --port 12351 --gpu 0 --data ml25m --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgn_ml25m.log & 
nohup python ddp_train_self_supervised.py --port 12355 --gpu 1 --data ml25m --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/tgat_ml25m.log & 
nohup python ddp_train_self_supervised.py --port 12359 --gpu 2 --data ml25m --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/jodie_ml25m.log & 
nohup python ddp_train_self_supervised.py --port 12363 --gpu 3 --data ml25m --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val > res/final/dyrep_ml25m.log & 


wait

echo "ML25M 2GPU:"
# nohup python ddp_train_self_supervised.py --port 12450 --gpu 0,1 --data ml25m --part_exp 1 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgn_ml25m_2gpu.log &
# nohup python ddp_train_self_supervised.py --port 12451 --gpu 0,1 --data ml25m --part_exp 1 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgat_ml25m_2gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12452 --gpu 2,3 --data ml25m --part_exp 1 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/jodie_ml25m_2gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12453 --gpu 2,3 --data ml25m --part_exp 1 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/dyrep_ml25m_2gpu.log & 

wait

echo "ML25M 4GPU:"
# nohup python ddp_train_self_supervised.py --port 12454 --gpu 0,1,2,3 --data ml25m --part_exp 2 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgn_ml25m_4gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12455 --gpu 0,1,2,3 --data ml25m --part_exp 2 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgat_ml25m_4gpu.log &
# nohup python ddp_train_self_supervised.py --port 12456 --gpu 0,1,2,3 --data ml25m --part_exp 2 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/jodie_ml25m_4gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12457 --gpu 0,1,2,3 --data ml25m --part_exp 2 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/dyrep_ml25m_4gpu.log & 

wait

# echo "DGraphFin 1GPU:"
# nohup python ddp_train_self_supervised.py --port 12358 --gpu 0 --data dgraphfin --part_exp 0 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgn_dgraphfin.log & 
# nohup python ddp_train_self_supervised.py --port 12359 --gpu 1 --data dgraphfin --part_exp 0 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgat_dgraphfin.log &
# nohup python ddp_train_self_supervised.py --port 12360 --gpu 2 --data dgraphfin --part_exp 0 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/jodie_dgraphfin.log &
# nohup python ddp_train_self_supervised.py --port 12361 --gpu 3 --data dgraphfin --part_exp 0 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/dyrep_dgraphfin.log & 


wait

echo "DGraphFin 2GPU:"
# nohup python ddp_train_self_supervised.py --port 12362 --gpu 1,2 --data dgraphfin --part_exp 1 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgn_dgraphfin_2gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12363 --gpu 1,2 --data dgraphfin --part_exp 1 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgat_dgraphfin_2gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12364 --gpu 1,2 --data dgraphfin --part_exp 1 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/jodie_dgraphfin_2gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12365 --gpu 2,3 --data dgraphfin --part_exp 1 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/dyrep_dgraphfin_2gpu.log & 

wait

echo "DGraphFin 4GPU:"
# nohup python ddp_train_self_supervised.py --port 12366 --gpu 0,1,2,3 --data dgraphfin --part_exp 2 --tgn --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgn_dgraphfin_4gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12367 --gpu 0,1,2,3 --data dgraphfin --part_exp 2 --tgat --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/tgat_dgraphfin_4gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12368 --gpu 0,1,2,3 --data dgraphfin --part_exp 2 --jodie --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/jodie_dgraphfin_4gpu.log & 
# nohup python ddp_train_self_supervised.py --port 12369 --gpu 0,1,2,3 --data dgraphfin --part_exp 2 --dyrep --top_k 10 --seed 0 --sync_mode last --n_epochs 50 --divide_method pre --bs 1000 --no_ind_val  --backup_memory_to_cpu --testing_on_cpu > res/final/dyrep_dgraphfin_4gpu.log & 