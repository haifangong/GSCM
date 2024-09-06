import argparse
import json
import logging
import os
import time

from dataset import ADataset, collate_fn
from network import MMPeptide, SEQPeptide, VoxPeptide, MMFPeptide, SMPeptide
from train import train, train_reg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from loss import MLCE, SuperLoss, LogCoshLoss
from utils import load_pretrain_model, set_seed
from torchmetrics import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef


def main():
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

    # training setting
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--warm-steps', type=int, dest='warm_steps', default=0,
                        help='number of warm start steps for learning rate (default: 10)')
    parser.add_argument('--patience', type=int, default=25,
                        help='patience for early stopping (default: 10)')
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='',
                        help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth

    # args for losses
    parser.add_argument('--loss', type=str, default='ce',
                        help='loss function (mlce, sl, mix)')

    parser.add_argument('--bias-curri', dest='bias_curri', action='store_true', default=False,
                        help='directly use loss as the training data (biased) or not (unbiased)')
    parser.add_argument('--anti-curri', dest='anti_curri', action='store_true', default=False,
                        help='easy to hard (curri), hard to easy (anti)')
    parser.add_argument('--std-coff', dest='std_coff', type=float, default=1,
                        help='the hyper-parameter of std')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.loss == "mlce":
        criterion = MLCE()
    elif args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "sl":
        criterion = SuperLoss()
    elif args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "logcosh":
        criterion = LogCoshLoss()
    elif args.loss == "bce":
        criterion = nn.BCELoss()
    else:
        criterion = 0
        pass
    weight_dir = "./run/" + args.task + "m-" + args.model + '-' + args.loss + str(args.batch_size) + str(args.lr) + str(args.epochs)
    print('saving_dir: ', weight_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(weight_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)

    with open(os.path.join(weight_dir, "config.json"), "w") as f:
        f.write(json.dumps(vars(args)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info('Loading Test Dataset')

    best_perform_list = [[] for i in range(5)]

    for i in range(1):
        start_loading_time = time.time()
        train_set = ADataset(mode='train', fold=i, task=args.task)
        valid_set = ADataset(mode='valid', fold=i, task=args.task)
        test_set = ADataset(mode='test', fold=i, task=args.task)
        print('loading_time: ', time.time()-start_loading_time)
        train_set.num_classes = 1
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        if args.model == 'seq':
            model = SEQPeptide(classes=train_set.num_classes, q_encoder='mlp')
            args.classes = train_set.num_classes
        elif args.model == 'voxel':
            model = VoxPeptide(classes=train_set.num_classes)
            args.classes = train_set.num_classes
        elif args.model == 'mm':
            # model = MMPeptide(classes=train_set.num_classes, q_encoder='tf', ) # attention='hamburger'
            model = SMPeptide(classes=train_set.num_classes, q_encoder='mlp', max_length=30) # attention='hamburger'
            # model = MMPeptide(classes=train_set.num_classes, q_encoder='mlp', ) # attention='hamburger'
            args.classes = train_set.num_classes
        elif args.model == 'mmf':
            model = MMFPeptide(classes=train_set.num_classes, q_encoder='mlp', max_length=30) # attention='hamburger'
            args.classes = train_set.num_classes
        if len(args.pretrain) != 0:
            print('loading pretrain model')
            # model = load_pretrain_model(model, torch.load(args.pretrain))
            model_state = model.state_dict()
            pretrained_state = torch.load(args.pretrain)
            pretrained_state = {k: v for k, v in pretrained_state.items() if
                                k in model_state and v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            # model.load_state_dict(torch.load(args.pretrain), strict=False)
        model.to(device)
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=True, weight_decay=5e-5)
        print(weight_dir)
        weights_path = f"{weight_dir}/model_{i + 1}.pth"
        # early_stopping = EarlyStopping(patience=args.patience, path=weights_path)
        logging.info(f'Running Cross Validation {i + 1}')
        metric_mae = MeanAbsoluteError().to(device)
        metric_mse = MeanSquaredError().to(device)
        metric_pcc = PearsonCorrCoef().to(device)
        best_metric = 10000
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            if True:
                train_loss, mae, mse, pcc = train_reg(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, pcc: {pcc:.3f}')

                if mae < best_metric:
                    logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, mae: {mae:.3f}, mse: {mse:.3f}, pcc: {pcc:.3f}')
                    best_metric = mae
                    torch.save(model.state_dict(), weights_path)
                    preds = []
                    gt_list_valid = []
                    with torch.no_grad():
                        for data in test_loader:
                            voxel, seq = data[0]
                            voxel2, seq2 = data[1]
                            gt = data[2]
                            out = model(((voxel.to(device), seq.to(device)), (voxel2.to(device), seq2.to(device))))
                            gt_list_valid.append(gt.cuda())
                            preds.append(out)

                    # calculate metrics
                    preds = torch.cat(preds, dim=0)
                    gt_list_valid = torch.cat(gt_list_valid).int()

                    mae = metric_mae(preds, gt_list_valid).item()
                    mse = metric_mse(preds, gt_list_valid).item()
                    pcc = metric_pcc(preds.squeeze(-1), gt_list_valid.float().squeeze(-1)).item()
                    logging.info(f'Epoch: {epoch:03d}, Test metrics, mae: {mae:.3f}, mse: {mse:.3f}, pcc: {pcc:.3f}')
                    best_perform_list[i] = np.asarray([mae, mse, pcc])

            else:
                train_loss, macro_ap, macro_f1, macro_acc, macro_auc = train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer)
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, macro_ap: {macro_ap:.3f}, macro_f1: {macro_f1:.3f}, macro_acc: {macro_acc:.3f}, macro_auc: {macro_auc:.3f}')
                avg_metric = macro_ap + macro_f1 + macro_acc + macro_auc
                if avg_metric > best_metric:
                    logging.info(
                        f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, macro_ap: {macro_ap:.3f}, macro_f1: {macro_f1:.3f}, macro_acc: {macro_acc:.3f}, macro_auc: {macro_auc:.3f}')
                    best_metric = avg_metric
                    best_perform_list[i] = np.asarray([macro_ap, macro_f1, macro_acc, macro_auc])
                    torch.save(model.state_dict(), weights_path)
            print('used time', time.time()-start_time)

    logging.info(f'Cross Validation Finished!')
    best_perform_list = np.asarray(best_perform_list)
    perform = open(weight_dir+'/result.txt', 'w')
    print(best_perform_list)
    print(np.mean(best_perform_list, 0))
    print(np.std(best_perform_list, 0))
    perform.write(','.join([str(i) for i in np.mean(best_perform_list, 0)])+'\n')
    perform.write(','.join([str(i) for i in np.std(best_perform_list, 0)])+'\n')


if __name__ == "__main__":
    main()
