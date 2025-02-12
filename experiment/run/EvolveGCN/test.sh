# python test.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_h.yaml

# python test.py --config_file ./experiments/parameters_reddit_linkpred_egcn_h.yaml


nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_h.yaml > res/wiki_h.log &
# nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_nodecls_egcn_h.yaml > res/wiki_h_node.log &
# nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_o.yaml > res/wiki_o.log &
# nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_nodecls_egcn_o.yaml > res/wiki_o_node.log &


# nohup python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_h.yaml > res/flight_h.log &
# nohup python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_o.yaml > res/flight_o.log &
# nohup python run_exp.py --config_file ./experiments/parameters_mooc_nodecls_egcn_h.yaml > res/mooc_h_node.log &
# nohup python run_exp.py --config_file ./experiments/parameters_mooc_nodecls_egcn_o.yaml > res/mooc_o_node.log &




# nohup python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_h.yaml > res/reddit_h.log &
# nohup python run_exp.py --config_file ./experiments/parameters_reddit_nodecls_egcn_h.yaml > res/reddit_h_node.log &
# nohup python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_o.yaml > res/reddit_o.log &
# nohup python run_exp.py --config_file ./experiments/parameters_reddit_nodecls_egcn_o.yaml > res/reddit_o_node.log &






# nohup python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_h.yaml > res/reddit_h.log &
# nohup python run_exp.py --config_file ./experiments/parameters_reddit_linkpred_egcn_o.yaml > res/reddit_o.log &

# nohup python run_exp.py --config_file ./experiments/parameters_mooc_linkpred_egcn_o.yaml > res/mooc_o.log &
nohup python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_o.yaml > res/flight_o.log &


nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_h.yaml > res/new/wiki_h.log &
nohup python run_exp.py --config_file ./experiments/parameters_wikipedia_linkpred_egcn_o.yaml > res/new/wiki_o.log &
nohup python run_exp.py --config_file ./experiments/parameters_mooc_linkpred_egcn_h.yaml > res/new/mooc_h.log &
nohup python run_exp.py --config_file ./experiments/parameters_flight_linkpred_egcn_h.yaml > res/new/flight_h.log &