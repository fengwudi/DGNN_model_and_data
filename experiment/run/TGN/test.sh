
nohup python train_self_supervised.py --use_memory --prefix tgn-attn --gpu 2 --uniform > res/tgn_wiki.log &
nohup python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --gpu 3 --uniform > res/tgn_reddit.log &
nohup python train_self_supervised.py -d mooc --use_memory --prefix tgn-attn-mooc --gpu 0 --uniform > res/tgn_mooc.log &
nohup python train_self_supervised.py -d Flights --use_memory --prefix tgn-attn-flight --gpu 3 --uniform > res/tgn_flights.log &

nohup python train_self_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --gpu 2 --uniform > res/jodie_wiki.log &
nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn-reddit --gpu 3 --uniform > res/jodie_reddit.log &
nohup python train_self_supervised.py -d mooc --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn-mooc --gpu 0 --uniform > res/jodie_mooc.log &
nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn-flight --gpu 2 --uniform > res/jodie_flights.log &

nohup python train_self_supervised.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --gpu 0 --uniform > res/dyrep_wiki.log &
nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn-reddit --gpu 3 --uniform > res/dyrep_reddit.log &
nohup python train_self_supervised.py -d mooc --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn-mooc --gpu 2 --uniform > res/dyrep_mooc.log &
nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn-flight --gpu 0 --uniform > res/dyrep_flights.log &


echo "tgn wiki ing:"
nohup python train_supervised.py -d wikipedia --use_memory --prefix tgn-attn --gpu 1 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/wiki_tgn_node.log &
wait
echo "tgn reddit ing:"
nohup python train_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --gpu 2 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/reddit_tgn_node.log &
wait
echo "tgn mooc ing:"
nohup python train_supervised.py -d mooc --use_memory --prefix tgn-attn-mooc --gpu 3 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/mooc_tgn_node.log &
wait

echo "jodie wiki ing:"
nohup python train_supervised.py -d wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --gpu 1 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/wiki_jodie_node.log &
wait
echo "jodie reddit ing:"
nohup python train_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --gpu 2 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/reddit_jodie_node.log &
wait
echo "jodie mooc ing:"
nohup python train_supervised.py -d mooc --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --gpu 3 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/mooc_jodie_node.log &
wait

echo "dyrep wiki ing:"
nohup python train_supervised.py -d wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --gpu 1 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/wiki_dyrep_node.log &
wait
echo "dyrep reddit ing:"
nohup python train_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --gpu 2 --n_epoch 50 --n_degree 20 --bs 1000 --uniform > res/reddit_dyrep_node.log &
wait
echo "dyrep mooc ing:"
nohup python train_supervised.py -d mooc --use_memory --memory_update


nohup python train_self_supervised.py -d wikipedia --n_layer 2 --gpu 0 --use_memory --prefix tgn-2l-attn --uniform > res/tgn_wiki_2l.log &
nohup python train_self_supervised.py -d wikipedia --n_layer 2 --gpu 1 --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-2l_rnn --uniform > res/dyrep_wiki_2l.log &
nohup python train_self_supervised.py -d reddit --n_layer 2 --gpu 2 --use_memory --prefix tgn-2l-attn --uniform > res/tgn_reddit_2l.log &
nohup python train_self_supervised.py -d reddit --n_layer 2 --gpu 3 --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-2l_rnn --uniform > res/dyrep_reddit_2l.log &
nohup python train_self_supervised.py -d Flights --n_layer 2 --gpu 2 --use_memory --prefix tgn-2l-attn --uniform > res/tgn_flight_2l.log &
nohup python train_self_supervised.py -d Flights --n_layer 2 --gpu 3 --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep-2l_rnn --uniform > res/dyrep_flight_2l.log &


# echo "tgn batch size:"

# nohup python train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --bs 100 --gpu 0 > res/batch_size/wiki-100.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --bs 500 --gpu 1 > res/batch_size/wiki-500.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --bs 1000 --gpu 2 > res/batch_size/wiki-1000.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --prefix tgn-attn --bs 2000 --gpu 3 > res/batch_size/wiki-2000.log &

# nohup python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn --bs 100 --gpu 0 > res/batch_size/reddit-100.log &
# nohup python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn --bs 500 --gpu 1 > res/batch_size/reddit-500.log &
# nohup python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn --bs 1000 --gpu 2 > res/batch_size/reddit-1000.log &
# nohup python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn --bs 2000 --gpu 3 > res/batch_size/reddit-2000.log &

# nohup python train_self_supervised.py -d Flights --use_memory --prefix tgn-attn --bs 100 --gpu 0 > res/batch_size/tgn_flight-100.log &
# nohup python train_self_supervised.py -d Flights --use_memory --prefix tgn-attn --bs 500 --gpu 1 > res/batch_size/tgn_flight-500.log &
# nohup python train_self_supervised.py -d Flights --use_memory --prefix tgn-attn --bs 1000 --gpu 2 > res/batch_size/tgn_flight-1000.log &
# nohup python train_self_supervised.py -d Flights --use_memory --prefix tgn-attn --bs 2000 --gpu 3 > res/batch_size/tgn_flight-2000.log &

# echo "jodie batch size:"

# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 100 --gpu 0 > res/batch_size/jodie_wiki-100.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 500 --gpu 1 > res/batch_size/jodie_wiki-500.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 1000 --gpu 2 > res/batch_size/jodie_wiki-1000.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 2000 --gpu 3 > res/batch_size/jodie_wiki-2000.log &

# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 100 --gpu 0 > res/batch_size/jodie_reddit-100.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 500 --gpu 1 > res/batch_size/jodie_reddit-500.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 1000 --gpu 2 > res/batch_size/jodie_reddit-1000.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 2000 --gpu 3 > res/batch_size/jodie_reddit-2000.log &

# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 100 --gpu 0 > res/batch_size/jodie_flight-100.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 500 --gpu 1 > res/batch_size/jodie_flight-500.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 1000 --gpu 2 > res/batch_size/jodie_flight-1000.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --bs 2000 --gpu 3 > res/batch_size/jodie_flight-2000.log &


# echo "dyrep batch size:"

# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 100 --gpu 0  > res/batch_size/dyrep_wiki-100.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 500 --gpu 1  > res/batch_size/dyrep_wiki-500.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 1000 --gpu 2  > res/batch_size/dyrep_wiki-1000.log &
# nohup python train_self_supervised.py -d wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 2000 --gpu 3  > res/batch_size/dyrep_wiki-2000.log &

# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 100 --gpu 0  > res/batch_size/dyrep_reddit-100.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 500 --gpu 1  > res/batch_size/dyrep_reddit-500.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 1000 --gpu 2  > res/batch_size/dyrep_reddit-1000.log &
# nohup python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 2000 --gpu 3  > res/batch_size/dyrep_reddit-2000.log &

# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 100 --gpu 0  > res/batch_size/dyrep_flight-100.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 500 --gpu 1  > res/batch_size/dyrep_flight-500.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 1000 --gpu 2  > res/batch_size/dyrep_flight-1000.log &
# nohup python train_self_supervised.py -d Flights --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --bs 2000 --gpu 3  > res/batch_size/dyrep_flight-2000.log &