import torch
from torchmetrics import F1Score, Accuracy, AveragePrecision, AUROC
from torchmetrics import MeanAbsoluteError, RelativeSquaredError, PearsonCorrCoef

from loss import unbiased_curriculum_loss


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    train_loss = 0
    metric_macro_acc = Accuracy(num_classes=args.classes, task='multilabel', num_labels=6, average='macro').to(device)
    metric_macro_f1 = F1Score(average='macro', task='multilabel', num_labels=6, num_classes=args.classes).to(device)
    metric_macro_ap = AveragePrecision(num_classes=args.classes, num_labels=6, task='multilabel', threshold=.0).to(
        device)
    metric_auc = AUROC(num_classes=args.classes, task='multilabel', num_labels=6, threshold=.0).to(device)

    for data in train_loader:
        voxel, seq, second_struct = data[0]
        voxel2, seq2, second_struct2 = data[1]
        gt = data[2]
        # print(seq_lengths)
        out = model((voxel.to(device), seq.to(device), second_struct.to(device)), (voxel2.to(device), seq2.to(device), second_struct2.to(device)))
        # print(out[0])
        # print(gt[0])
        loss = criterion(out, gt.to(device).float())
        # loss_0 = criterion(out[0], gt.to(device).float())
        # loss_1 = criterion(out[1], gt.to(device).float())
        # loss_2 = criterion(out[2], gt.to(device).float())
        # loss = loss_0 + loss_1 + loss_2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
            voxel, seq, second_struct, gt, seq_lengths = data
            gt_list_valid.append(gt.cuda())
            out = model((voxel.to(device), seq.to(device), second_struct.to(device)), seq_lengths)
            preds.append(out)

    # calculate metrics
    preds = torch.cat(preds, dim=0)
    gt_list_valid = torch.cat(gt_list_valid).int().squeeze(1)

    macro_ap = metric_macro_ap(preds, gt_list_valid).item()
    # class_ap = [round(i.item(), 5) for i in metric_class_ap(preds, gt_list_valid)]
    macro_auc = metric_auc(preds, gt_list_valid).item()
    macro_f1 = metric_macro_f1(preds, gt_list_valid).item()
    macro_acc = metric_macro_acc(preds, gt_list_valid).item()
    return train_loss, macro_ap, macro_f1, macro_acc, macro_auc


def train_reg(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    model.train()
    train_loss = 0
    metric_mae = MeanAbsoluteError().to(device)
    metric_mse = RelativeSquaredError().to(device)
    metric_pcc = PearsonCorrCoef().to(device)
    for data in train_loader:
        voxel, seq = data[0]
        voxel2, seq2 = data[1]
        gt = data[2]
        out = model(((voxel.to(device), seq.to(device)), (voxel2.to(device), seq2.to(device))))
        if args.loss == 'curri':
            loss = unbiased_curriculum_loss(out, gt.to(device).float(), args, epoch, args.epochs)
        else:
            loss = criterion(out, gt.to(device).float())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    model.eval()
    preds = []
    gt_list_valid = []
    with torch.no_grad():
        for data in valid_loader:
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
    return train_loss, mae, mse, pcc
