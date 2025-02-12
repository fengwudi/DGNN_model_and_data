# nohup python main.py -d wikipedia --bs 1000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 3 > res/wiki-1000.log &
# nohup python main.py -d reddit --bs 1000 --n_degree 20 --walk_n_head 2  --seed 2024 --n_layer 1 --gpu 3 > res/reddit-1000.log &
nohup python main.py -d mooc --bs 1000 --n_degree 20 --walk_n_head 2  --n_layer 1 --gpu 0 > res/mooc-1000.log &
# nohup python main.py -d Flights --bs 1000 --n_degree 20 --walk_n_head 2   --seed 2024 --n_layer 1 --gpu 2 > res/flight-1000.log &



# nohup python main.py -d wikipedia --bs 100 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 0 > res/wiki-100.log &
# nohup python main.py -d reddit --bs 100 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 2 > res/reddit-100.log &
# nohup python main.py -d Flights --bs 100 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 1 > res/flight-100.log &

# nohup python main.py -d wikipedia --bs 500 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 0 > res/wiki-500.log & 
# nohup python main.py -d reddit --bs 500 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 1 > res/reddit-500.log &
# nohup python main.py -d Flights --bs 500 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 0 > res/flight-500.log &

# nohup python main.py -d wikipedia --bs 2000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 0 > res/wiki-2000.log & 
# nohup python main.py -d reddit --bs 2000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 1 > res/reddit-2000.log &
# nohup python main.py -d Flights --bs 2000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 3 > res/flight-2000.log &

# nohup python main.py -d ml25m --pos_dim 2 --bs 1000 --n_degree 20 --walk_n_head 2  --seed 2024 --n_layer 1 --gpu 3 > res/ml25m.log &
# nohup python main.py -d dgraphfin --bs 1000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 1 --gpu 0 > res/dgraphfin.log &


# nohup python main.py -d wikipedia --bs 1000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 2 --gpu 0 > res/wiki_2l.log &
# nohup python main.py -d reddit --bs 1000 --n_degree 20 --walk_n_head 2 --seed 2024 --n_layer 2 --gpu 1 > res/reddit_2l.log &
# nohup python main.py -d Flights --bs 1000 --n_degree 20  --walk_n_head 2 --seed 2024 --n_layer 2 --gpu 0 > res/flight_2l.log &



# fligh-100 & reddit-2000      flight-2000 & reddit-1000    fligh-500 & dgraphfin