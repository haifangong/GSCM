import argparse
import json
import logging
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsoluteError, KendallRankCorrCoef, PearsonCorrCoef, RelativeSquaredError

# from dataset import ADataset, HDataset, collate_fn
from model import MMGraphCat, MMGraphCatAtt, MMGraph, MMSingleGraph # no qkv and residual, no residual, qkv+residual
from dataset_graph import load_dataset, load_dataset_saap
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import MLCE, SuperLoss, unbiased_curriculum_loss
from utils import load_pretrain_model, set_seed


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
            data_graph_1 = data[0]
            data_graph_2 = data[1]
            gt = data[2]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            out1 = model(data_graph_1).squeeze(1)
            out2 = model(data_graph_2).squeeze(1)
            preds.append(out1-out2)  # Detach and send predictions to CPU
            ys.append(gt)

    # Concatenate all batched predictions and ground truths
    preds = torch.cat(preds, dim=0).squeeze(-1)
    gt_list_valid = torch.cat(ys, dim=0).cuda()

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
            data_graph_1 = data[0]
            data_graph_2 = data[1]
            gt = data[2]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            out1 = model(data_graph_1).squeeze(1)
            out2 = model(data_graph_2).squeeze(1)
            preds.append(out1[0]-out2[0])
            ys.append(torch.tensor(gt))
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
        loss_function = MLCE()
    elif args.loss == "mse":
        loss_function = nn.MSELoss()
    elif args.loss == "smoothl1":
        loss_function = nn.SmoothL1Loss()
    elif args.loss == "ce":
        loss_function = nn.CrossEntropyLoss()
    elif args.loss == "bce":
        loss_function = nn.BCELoss()
    elif args.loss == "ucl":
        loss_function = unbiased_curriculum_loss
    elif args.loss == "superloss":
        loss_function = SuperLoss()
    else:
        raise NotImplementedError
    if 'abla_data' in args.abla:
        weight_dir = "./run/abla_data-" + args.model + "-" + str(args.similarity) + "-" + str(args.balance_num) + '-' + str(args.batch_size) + str(args.lr) + str(args.epochs)
    elif 'abla_feature' in args.abla:
        weight_dir = "./run/abla_feature-" + args.model + '-' + args.feature + '-' + args.structure_feature + '-' + str(args.batch_size) + str(args.lr) + str(args.epochs)
    else:
        weight_dir = "./run/" + args.task + "-" + args.model + "-" + args.gnn_type + '-' + args.loss + str(args.batch_size) + str(args.lr) + str(args.epochs)
    args.weight_dir = weight_dir
    print('saving_dir: ', weight_dir)
    os.makedirs(weight_dir, exist_ok=True)

    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    data_list = load_dataset(args)
    logging.info('Load Training Dataset Number: ' + str(len(data_list)))

    test_set = load_dataset_saap(args)
    logging.info('Load Test Dataset Number: ' + str(len(test_set)))

    test_loader = DataLoader(test_set, batch_size=1, follow_batch=['x_s'], shuffle=False)

    random.shuffle(data_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    five_fold_index = [([j for j in range(len(data_list)) if (j + i) % 5 != 0],
                        [j for j in range(len(data_list)) if (j + i) % 5 == 0]) for i in range(5)]
    best_perform_list = [[] for i in range(5)]
    best_perform_list_test = [[] for i in range(5)]

    for i, (train_index, valid_index) in enumerate(five_fold_index):
        logging.info(f'Running Cross Validation {i + 1}')
        if args.model == 'mma':
            model = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
                            drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        elif args.model == 'mmac':
            model = MMGraphCatAtt(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
                            drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        elif args.model == 'mmc':
            model = MMGraphCat(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
                            drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        elif args.model == 'mms':
            model = MMSingleGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, JK="last",
                            drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)
        # model.load_state_dict(torch.load(args.pretrain))
        model.to(device)

        train_direct_dataset, valid_direct_dataset = [data_list[i] for i in train_index], [data_list[j] for j in
                                                                                           valid_index]

        print(len(train_direct_dataset) + len(valid_direct_dataset))
        train_loader = DataLoader(train_direct_dataset, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_direct_dataset, batch_size=args.batch_size, follow_batch=['x_s'], shuffle=False, drop_last=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Create a SummaryWriter instance (it will create a log directory)
        writer = SummaryWriter(weight_dir + '/fold_' + str(i))

        best_metric_val = 0
        best_metric_test = 0

        for epoch in range(1, args.epochs + 1):
            for data in train_loader:
                data_graph_1 = data[0]
                data_graph_2 = data[1]
                gt1, gt2 = data_graph_1.gt, data_graph_2.gt
                data_graph_1 = data_graph_1.to(device)
                data_graph_2 = data_graph_2.to(device)
                out = model(data_graph_1).squeeze(1)
                loss = F.mse_loss(out, torch.tensor(np.asarray(gt1)).cuda().float())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                out = model(data_graph_2).squeeze(1)
                loss = F.mse_loss(out, torch.tensor(np.asarray(gt2)).cuda().float())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            val_mae, val_rse, val_pcc, val_kcc = eval_model(valid_loader, model)
            logging.info('Eval result:' + str(epoch) + ' ' + str(val_mae) + ' ' + str(val_rse) + ' ' + str(val_pcc) + ' ' + str(val_kcc))

            overall_value = (val_pcc + val_kcc) / (val_mae + val_rse)
            if overall_value > best_metric_val:
                best_metric_val = overall_value
                best_perform_list[i] = np.asarray([val_mae, val_rse, val_pcc, val_kcc])
            mae, rse, pcc, kcc = eval_model(test_loader, model)
            # mae, rse, pcc, kcc = eval_model_one(test_loader, model)
            overall_value = (pcc + kcc) / (mae + rse)
            if overall_value > best_metric_test:
                best_metric_test = overall_value
                best_perform_list_test[i] = np.asarray([mae, rse, pcc, kcc])
                torch.save(model.state_dict(), f"{weight_dir}/model_{i + 1}.pth")
            logging.info('Test result:' + str(epoch) + ' ' + str(mae) + ' ' + str(rse) + ' ' + str(pcc) + ' ' + str(kcc))

            # Log the validation metrics to TensorBoard
            writer.add_scalars('MAE', {'Validation': val_mae, 'Test': mae}, epoch)
            writer.add_scalars('RSE', {'Validation': val_rse, 'Test': rse}, epoch)
            writer.add_scalars('PCC', {'Validation': val_pcc, 'Test': pcc}, epoch)
            writer.add_scalars('KCC', {'Validation': val_kcc, 'Test': kcc}, epoch)

        # Close the writer when you're done
        writer.close()

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
    # model structure setting
    parser.add_argument('--model', type=str, default='mms',
                        help='mma, mmac, mmc, mms')
    parser.add_argument('--fusion', type=str, default='1',
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--num-layer', type=int, dest='num_layer', default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=90,
                        help='embedding dimensions (default: 200)')
    parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="attention",
                        help='graph level pooling (sum, mean, max, attention)')
    parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gatv2",
                        help='gnn type (gin, gcn, gat, graphsage)')

    # validation setting
    parser.add_argument('--seed', type=int, default=1,
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")

    # task setting
    parser.add_argument('--task', type=str, default='all',
                        help='task: anti toxin anti-all mechanism anti-binary anti-regression mic')
    parser.add_argument('--abla', type=str, default='',
                        help='abla_data, abla_feature')
    
    # dataset setting
    parser.add_argument('--similarity', type=float, default=0.3,
                        help="Similarity for construct training set.")
    parser.add_argument('--balance_num', type=int, default=1000,
                        help="Set up the max number of sample in dataset constructure.")

    # feature setting
    parser.add_argument('--feature', type=str, default='gsh',
                        help="g: global feature, s: sequence feature, h: helixfold structure feature")
    parser.add_argument('--structure_feature', type=str, default='sep',
                        help="s: sequence embedding, e: energy function, p: position embedding")
    parser.add_argument('--classes', type=int, default=6,
                        help='model')

        
    # training setting
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=8192,
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss function')
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='./run/all-gatv2-mlce2560.002110/model_1.pth',
                        help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth

    args = parser.parse_args()

    main(args)
