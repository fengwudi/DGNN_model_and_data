nohup python -u train.py --dataset Flights --time_steps -1 --epochs 50 --early_stop 3 --GPU_ID 1 --featureless False > logs/flights.log &



nohup python -u train.py --dataset wikipedia --time_steps -1 --epochs 50 --early_stop 3 --GPU_ID 0 --featureless False > logs/wikipedia.log &
nohup python -u train.py --dataset reddit --time_steps -1 --epochs 50 --early_stop 3 --GPU_ID 2 --featureless False > logs/reddit.log &
nohup python -u train.py --dataset mooc --time_steps -1 --epochs 50 --early_stop 3 --GPU_ID 3 --featureless False > logs/mooc.log &