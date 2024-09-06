import argparse
import collections
import json
import logging
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsoluteError, KendallRankCorrCoef, PearsonCorrCoef, RelativeSquaredError

# from dataset import ADataset, HDataset, collate_fn
from model_single import GraphGNN, MMGraph, SiameseFC
from dataset_single import load_dataset, load_dataset_saap
from network import MMPeptide, SEQPeptide, VoxPeptide, MMFPeptide
from train import train, train_reg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import MLCE, SuperLoss, unbiased_curriculum_loss
from utils import load_pretrain_model, set_seed


import torch
import numpy as np
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef, KendallRankCorrCoef

def eval_model(test_loader, model):
    # Move metrics computation to the appropriate device
    metric_mae = MeanAbsoluteError().cuda()
    metric_rse = RelativeSquaredError().cuda()
    metric_pcc = PearsonCorrCoef().cuda()
    metric_krcc = KendallRankCorrCoef().cuda()

    # Prepare to collect predictions and ground truths
    preds = []
    ys = []

    # Disable gradient calculations for evaluation
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data_graph_1, seq = data[0]
            data_graph_2, seq2 = data[1]
            gt_1, gt_2 = data[2]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            out_1 = model(data_graph_1)
            out_2 = model(data_graph_2)
            preds.append(out_1-out_2)  # Detach and send predictions to CPU
            ys.append(gt_1-gt_2)

    # Concatenate all batched predictions and ground truths
    preds = torch.cat(preds, dim=0).squeeze(-1)
    gt_list_valid = torch.cat(ys, dim=0).cuda().squeeze(-1)

    # Calculate metrics
    mae = round(metric_mae(preds, gt_list_valid.float()).item(), 3)
    rse = round(metric_rse(preds, gt_list_valid.float()).item(), 3)
    pcc = round(metric_pcc(preds, gt_list_valid.float()).item(), 3)
    kcc = round(metric_krcc(preds, gt_list_valid.float()).item(), 3)

    return mae, rse, pcc, kcc


def eval_model_one(test_loader, model):
    metric_mae = MeanAbsoluteError().cuda()
    metric_rse = RelativeSquaredError().cuda()
    metric_pcc = PearsonCorrCoef().cuda()
    metric_krcc = KendallRankCorrCoef().cuda()
    preds = []
    ys = []
    with torch.no_grad():
        for data in test_loader:
            data_graph_1, seq = data[0]
            data_graph_2, seq2 = data[1]
            gt_1, gt_2 = data[2]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            out_1 = model(data_graph_1)
            out_2 = model(data_graph_2)
            preds.append(out_1[0]-out_2[0])
            ys.append(torch.tensor(gt_1-gt_2))
    preds = torch.cat(preds, dim=0)

    gt_list_valid = torch.cat(ys).cuda()

    mae = round(metric_mae(preds, gt_list_valid).item(), 3)
    rse = round(metric_rse(preds, gt_list_valid).item(), 3)
    pcc = round(metric_pcc(preds, gt_list_valid.float()).item(), 3)
    kcc = round(metric_krcc(preds, gt_list_valid.float()).item(), 3)
    return mae, rse, pcc, kcc


def main(args):
    set_seed(args.seed)

    if args.loss == "mlce":
        criterion = MLCE()
    elif args.loss == "rse":
        criterion = nn.MSELoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()
    else:
        raise NotImplementedError

    weight_dir = "./run/" + args.task + "-" + args.gnn_type + '-' + args.loss + str(args.batch_size) + str(
        args.lr) + str(args.epochs)
    print('saving_dir: ', weight_dir)
    os.makedirs(weight_dir, exist_ok=True)

    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    logging.info('Loading Training Dataset')
    data_list = load_dataset(args.task)

    logging.info('Loading Test Dataset')
    test_set = load_dataset_saap(args.task)
    test_loader = DataLoader(test_set, batch_size=1, follow_batch=['x_s'], shuffle=False)

    random.shuffle(data_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    five_fold_index = [([j for j in range(len(data_list)) if (j + i) % 5 != 0],
                        [j for j in range(len(data_list)) if (j + i) % 5 == 0]) for i in range(5)]
    best_perform_list = [[] for i in range(5)]
    best_perform_list_test = [[] for i in range(5)]

    for i, (train_index, valid_index) in enumerate(five_fold_index):
        print(len(train_index))
        print(len(valid_index))
        # model = GraphGNN(num_layer=args.num_layer, input_dim=40, emb_dim=args.emb_dim, out_dim=1, JK="last",
        #                  drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        model = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
                        drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        # model = SiameseFC(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
        #                 drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        model.to(device)

        train_direct_dataset, valid_direct_dataset = [data_list[i] for i in train_index], [data_list[j] for j in
                                                                                           valid_index]

        print(len(train_direct_dataset) + len(valid_direct_dataset))
        train_loader = DataLoader(train_direct_dataset, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=True)
        valid_loader = DataLoader(valid_direct_dataset, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=False, drop_last=True)

        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=True, weight_decay=5e-5)
        print(weight_dir)
        weights_path = f"{weight_dir}/model_{i + 1}.pth"
        # early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {i + 1}')

        # Create a SummaryWriter instance (it will create a log directory)
        writer = SummaryWriter(weight_dir + '/fold_' + str(i))

        # Close the writer when you're done
        writer.close()

        best_metric_val = 0
        best_metric_test = 0
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            # print(epoch)
            yt = []
            for data in train_loader:
                data_graph_1, seq = data[0]
                data_graph_2, seq = data[1]
                gt1, gt2 = data[2]
                data_graph_1 = data_graph_1.to(device)
                data_graph_2 = data_graph_2.to(device)
                out = model(data_graph_1).squeeze(1)
                loss = F.mse_loss(out, torch.tensor(np.asarray(gt1)).cuda().float())
                yt.append(torch.tensor(np.asarray(gt1)))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                out = model(data_graph_2).squeeze(1)
                loss = F.mse_loss(out, torch.tensor(np.asarray(gt2)).cuda().float())
                yt.append(torch.tensor(np.asarray(gt2)))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if epoch == 1:
                yt = torch.cat(yt).int()
                # print(torch.sum(yt, 0))

            model.eval()
            val_mae, val_rse, val_pcc, val_kcc = eval_model(valid_loader, model)

            print(
                'Eval result:' + str(epoch) + ' ' + str(val_mae) + ' ' + str(val_rse) + ' ' + str(val_pcc) + ' ' + str(
                    val_kcc))

            overall_value = (val_pcc + val_kcc) / (val_mae * val_rse)
            if overall_value > best_metric_val:
                best_metric_val = overall_value
                best_perform_list[i] = np.asarray([val_mae, val_rse, val_pcc, val_kcc])

            mae, rse, pcc, kcc = eval_model_one(test_loader, model)
            overall_value = (abs(pcc + kcc)) / (mae * rse)
            if overall_value > best_metric_test:
                best_metric_test = overall_value
                best_perform_list_test[i] = np.asarray([mae, rse, pcc, kcc])
                torch.save(model.state_dict(), f"{weight_dir}/model_{i + 1}.pth")
            print('Test result:' + str(epoch) + ' ' + str(mae) + ' ' + str(rse) + ' ' + str(pcc) + ' ' + str(kcc))

            # Log the validation metrics to TensorBoard
            writer.add_scalars('MAE', {'Validation': val_mae, 'Test': mae}, epoch)
            writer.add_scalars('RSE', {'Validation': val_rse, 'Test': rse}, epoch)
            writer.add_scalars('PCC', {'Validation': val_pcc, 'Test': pcc}, epoch)
            writer.add_scalars('KCC', {'Validation': val_kcc, 'Test': kcc}, epoch)

            # if epoch == 1:
            #     print(torch.sum(ys, 0))
            # print('used time', time.time() - start_time)

    logging.info(f'Cross Validation Finished!')
    best_perform_list = np.asarray(best_perform_list)
    best_perform_list_test = np.asarray(best_perform_list_test)
    perform = open(weight_dir + '/result.txt', 'w')
    print(best_perform_list)
    print(best_perform_list_test)
    print('best_perform_list', np.mean(best_perform_list, 0))
    # print('best_perform_list', np.std(best_perform_list, 0))
    print('best_perform_list_test', np.mean(best_perform_list_test, 0))
    # print('best_perform_list_test', np.std(best_perform_list_test, 0))
    perform.write('IID\n')
    for row in best_perform_list:
        perform.write(','.join([str(round(i, 4)) for i in row]) + '\n')
    perform.write(','.join([str(round(i, 4)) for i in np.mean(best_perform_list, 0)]) + '\n')
    perform.write(','.join([str(round(i, 4)) for i in np.std(best_perform_list, 0)]) + '\n')

    perform.write('OOD\n')
    for row in best_perform_list_test:
        perform.write(','.join([str(round(i, 4)) for i in row]) + '\n')
    perform.write(','.join([str(round(i, 4)) for i in np.mean(best_perform_list_test, 0)]) + '\n')
    perform.write(','.join([str(round(i, 4)) for i in np.std(best_perform_list_test, 0)]) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='resnet26')
    # model setting
    parser.add_argument('--model', type=str, default='mm',
                        help='model resnet26, bi-gru')
    parser.add_argument('--fusion', type=str, default='1',
                        help="Seed for splitting dataset (default 1)")

    # task & dataset setting
    parser.add_argument('--task', type=str, default='all',
                        help='task: anti toxin anti-all mechanism anti-binary anti-regression mic')
    parser.add_argument('--classes', type=int, default=6,
                        help='model')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=1,
                        help='number of GNN message passing layers (default: 1)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=90,
                        help='embedding dimensions (default: 200)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="attention",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gat",
                        help='gnn type (gin, gcn, gat, graphsage)')
    # training setting
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=0,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=25,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--loss', type=str, default='mlce',
                        help='loss function (mlce, sl, mix)')
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                        help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth

    args = parser.parse_args()

    main(args)
