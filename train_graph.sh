# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mma
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mmac
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mmc
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mma --gnn-type gat
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mma --gnn-type gin
# CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mma --gnn-type gcn
CUDA_VISIBLE_DEVICES=1 python train_graph.py --model mma --gnn-type graphsage


# python train_graph.py --task 0
# python train_graph.py --task 1
# python train_graph.py --task 2
# python train_graph.py --task 3
# python train_graph.py --task 4
# python train_graph.py --task 5