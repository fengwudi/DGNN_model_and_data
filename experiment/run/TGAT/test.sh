# nohup python -u learn_edge.py -d wikipedia --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_layer 1 --n_head 2 --prefix wiki > res/wiki-1000.log &

# nohup python -u learn_edge.py -d reddit --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0  --n_layer 1 --n_head 2 --prefix reddit > res/reddit-1000.log &

# nohup python -u learn_edge.py -d Flights --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_layer 1  --n_head 2 --prefix flight > res/flights-1000.log &

# nohup python -u learn_edge.py -d mooc --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_layer 1  --n_head 2 --prefix mooc > res/mooc-1000.log &


# nohup python -u learn_node.py -d wikipedia --n_layer 1 --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix wiki > res/wiki_node.log &
# nohup python -u learn_node.py -d reddit --n_layer 1 --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 2 --n_head 2 --prefix reddit > res/reddit_node.log &
nohup python -u learn_node.py -d mooc --n_layer 1 --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 3 --n_head 2 --prefix mooc > res/mooc_node.log &


# nohup python -u learn_edge.py -d wikipedia --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 3 --n_layer 1 --n_head 2 --prefix wiki100 > res/wiki-100.log &
# nohup python -u learn_edge.py -d wikipedia --bs 500 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1  --n_layer 1 --n_head 2 --prefix wiki500 > res/wiki-500.log &
# nohup python -u learn_edge.py -d wikipedia --bs 2000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 2 --n_layer 1  --n_head 2 --prefix wiki2000 > res/wiki-2000.log &


# nohup python -u learn_edge.py -d reddit --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 3 --n_layer 1 --n_head 2 --prefix reddit100 > res/reddit-100.log &
# nohup python -u learn_edge.py -d reddit --bs 500 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 0  --n_layer 1 --n_head 2 --prefix reddit500 > res/reddit-500.log &
# nohup python -u learn_edge.py -d reddit --bs 2000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_layer 1  --n_head 2 --prefix reddit2000 > res/reddit-2000.log &

# nohup python -u learn_edge.py -d Flights --bs 100 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_layer 1 --n_head 2 --prefix flight100 > res/flight-100.log &
# nohup python -u learn_edge.py -d Flights --bs 500 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 3  --n_layer 1 --n_head 2 --prefix flight500 > res/flight-500.log &
# nohup python -u learn_edge.py -d Flights --bs 2000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 2 --n_layer 1  --n_head 2 --prefix flight2000 > res/flight-2000.log &



# nohup python -u learn_edge.py -d wikipedia --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 2 --prefix wiki-2l > res/wiki-2l.log &
# nohup python -u learn_edge.py -d reddit --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 2 --prefix reddit-2l > res/reddit-2l.log &
nohup python -u learn_edge.py -d Flights --bs 1000 --uniform  --n_degree 20 --agg_method attn --attn_mode prod --gpu 3 --prefix flight-2l > res/flights-2l.log &


nohup python -u learn_edge.py -d ml25m --bs 1000 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 1 --n_layer 1 --n_head 1 --prefix ml25m > res/ml25m.log &

nohup python -u learn_edge.py -d dgraphfin --bs 1000 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_layer 1 --n_head 1 --prefix dgraphfin > res/dgraphfin.log &
