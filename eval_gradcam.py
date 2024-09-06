import numpy as np
import torch
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchcam.methods import CAM, GradCAM
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torchmetrics.functional import pearson_corrcoef, extended_edit_distance
from torchmetrics import F1Score
from tqdm import tqdm
from model import resnet26, gru, Bottleneck, ResNet3D_OLD
from dataset import ADataset, PDataset, GDataset, MDataset

toxin_f = open('pseudo_pretrain.csv', 'w')


def eval_toxin():

    # model = resnet26(1).cuda()
    f = open('gen_toxin2.csv', 'w')
    model1 = gru(input_dim=22, classes=1).cuda()
    # print(model1)
    # cam = GradCAM(model1, 'rnn', )

    # def grad_cam_for_ques(model, output, index=None):
    #     # print(q_feat.shape)
    #     if index == None:
    #         index = np.argmax(output.cpu().data.numpy())
    #     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #     one_hot[0][index] = 1
    #     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * output)
    #     model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     q_grads = model.gradients[0].cpu().data.numpy()
    #
    #     weights = np.mean(q_grads, axis=1)[0, :]
    #     print("weights.shape: {}".format(weights.shape))
    #     # cam = np.zeros(q_feat.shape[0], dtype=np.float32)
    #     #
    #     # for i, w in enumerate(weights):
    #     #     cam += w * q_feat[:, i]
    #     #
    #     # # cam = np.maximum(cam, 0)
    #     # cam = cam - np.min(cam)
    #     # cam = cam / np.max(cam)
    #     return
    model1.load_state_dict(torch.load('./runs/toxin_binary_gru/model_1.pkl'))
    model2 = gru(input_dim=22, classes=1).cuda()
    model2.load_state_dict(torch.load('./runs/toxin_binary_gru/model_2.pkl'))
    model3 = gru(input_dim=22, classes=1).cuda()
    model3.load_state_dict(torch.load('./runs/toxin_binary_gru/model_3.pkl'))
    model4 = gru(input_dim=22, classes=1).cuda()
    model4.load_state_dict(torch.load('./runs/toxin_binary_gru/model_4.pkl'))
    model5 = gru(input_dim=22, classes=1).cuda()
    model5.load_state_dict(torch.load('./runs/toxin_binary_gru/model_5.pkl'))

    test_set = GDataset(mode='all', fold=0)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for data in tqdm(valid_loader):
        result = 0
        voxel, seq, gt, index = data
        pred_result1 = model1(seq.cuda())
        # grad_cam_for_ques(model1, )
        # result = cam(class_idx=0, scores=pred_result1)
        # print(len(result))
        # print(result[0].shape)
        pred_result2 = model2(seq.cuda())
        pred_result3 = model3(seq.cuda())
        pred_result4 = model4(seq.cuda())
        pred_result5 = model5(seq.cuda())
        result += 0 if pred_result1.item() > 0 else 1
        result += 0 if pred_result2.item() > 0 else 1
        result += 0 if pred_result3.item() > 0 else 1
        result += 0 if pred_result4.item() > 0 else 1
        result += 0 if pred_result5.item() > 0 else 1

        f.write(str(index.item())+','+str(int(result/5)) + '\n')


def eval_anti():
    f = open('gen_anti2.csv', 'w')

    model1 = resnet26(num_classes=6).cuda()
    model1.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_1.pkl'))
    model2 = resnet26(num_classes=6).cuda()
    model2.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_2.pkl'))
    model3 = resnet26(num_classes=6).cuda()
    model3.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_3.pkl'))
    model4 = resnet26(num_classes=6).cuda()
    model4.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_4.pkl'))
    model5 = resnet26(num_classes=6).cuda()
    model5.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_5.pkl'))

    test_set = GDataset(mode='all', fold=0)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    # index = 0
    for data in tqdm(valid_loader):
        voxel, seq, gt, index = data
        pred_result1 = model1((voxel.cuda(), seq.cuda()))

        pred_result2 = model2((voxel.cuda(), seq.cuda()))
        pred_result3 = model3((voxel.cuda(), seq.cuda()))
        pred_result4 = model4((voxel.cuda(), seq.cuda()))
        pred_result5 = model5((voxel.cuda(), seq.cuda()))
        result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5)/5
        final_result = [str(i.item()) for i in result[0]]
        # index += 1
        # f.write(str(index)+','.join(final_result) + '\n')
        f.write(str(index.item())+','+','.join(final_result) + '\n')


def eval_mechanism():

    # model = resnet26(1).cuda()
    f = open('gen_mechanism2.csv', 'w')
    model1 = ResNet3D_OLD(Bottleneck, [1, 2, 4, 1], num_classes=4).cuda()
    print(model1)
    #
    model1.load_state_dict(torch.load('./runs/mlce-vq1-mech/model_1.pkl'))
    # model2 = resnet26(num_classes=4).cuda()
    # model2.load_state_dict(torch.load('./runs/mlce-vq1-mech/model_2.pkl'))
    # model3 = resnet26(num_classes=4).cuda()
    # model3.load_state_dict(torch.load('./runs/mlce-vq1-mech/model_3.pkl'))
    # model4 = resnet26(num_classes=4).cuda()
    # model4.load_state_dict(torch.load('./runs/mlce-vq1-mech/model_4.pkl'))
    # model5 = resnet26(num_classes=4).cuda()
    # model5.load_state_dict(torch.load('./runs/mlce-vq1-mech/model_5.pkl'))

    # gradcam

    # cam = GradCAM(model1, 'seq_fc.0', )
    # cam = GradCAM(model1, 'layer1.0.conv1', )
    cam = GradCAM(model1, 'conv1', )

    test_set = GDataset(mode='all', fold=0)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    # model2.eval()
    # model3.eval()
    # model4.eval()
    # model5.eval()
    count = 0
    for data in tqdm(valid_loader):
        count += 1
        result = 0
        voxel, seq, gt, index = data
        pred_result1 = model1((voxel.cuda(), seq.cuda()))
        # grayscale_cam = cam(input_tensor=(voxel.cuda(), seq.cuda()))
        # grayscale_cam = grayscale_cam[0, :]
        out = model1((voxel.cuda(), seq.cuda()))
        result = cam(class_idx=0, scores=out)

        result_image = sitk.GetImageFromArray(voxel.cpu().numpy())
        # print(result[0].cpu().numpy())
        # print(result)
        # print(np.sum(result[0].unsqueeze(0).cpu().numpy()))
        cam_data = np.repeat(result[0].unsqueeze(0).cpu().numpy()*255, 3, axis=1)
        print(cam_data.shape)
        cam_image = sitk.GetImageFromArray(cam_data)
        sitk.WriteImage(result_image, './gradcam/voxel'+str(count)+'.nii.gz')
        sitk.WriteImage(cam_image, './gradcam/cam'+str(count)+'.nii.gz')

        print(voxel.cpu().shape)
        print(result[0].shape)
        # visualization = show_cam_on_image(voxel.cpu(), result[0].cpu(), use_rgb=True)

        # print(grayscale_cam)
        # pred_result2 = model2((voxel.cuda(), seq.cuda()))
        # pred_result3 = model3((voxel.cuda(), seq.cuda()))
        # pred_result4 = model4((voxel.cuda(), seq.cuda()))
        # pred_result5 = model5((voxel.cuda(), seq.cuda()))
        # result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5)/5
        # final_result = [str(i.item()) for i in result[0]]
        #
        # f.write(str(index.item())+','+','.join(final_result) + '\n')


def eval_mutate():
    f = open('mutate.csv', 'w')

    model1 = resnet26(num_classes=6).cuda()
    model1.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_1.pkl'))
    model2 = resnet26(num_classes=6).cuda()
    model2.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_2.pkl'))
    model3 = resnet26(num_classes=6).cuda()
    model3.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_3.pkl'))
    model4 = resnet26(num_classes=6).cuda()
    model4.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_4.pkl'))
    model5 = resnet26(num_classes=6).cuda()
    model5.load_state_dict(torch.load('./runs/mlce-vq1-anti/model_5.pkl'))

    test_set = MDataset(mode='all', fold=0)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for data in tqdm(valid_loader):
        wide, mutate = data
        pred_w1 = model1((wide[0].cuda(), wide[1].cuda()))
        pred_m1 = model1((mutate[0].cuda(), mutate[1].cuda()))
        pred_w2 = model2((wide[0].cuda(), wide[1].cuda()))
        pred_m2 = model2((mutate[0].cuda(), mutate[1].cuda()))
        pred_w3 = model3((wide[0].cuda(), wide[1].cuda()))
        pred_m3 = model3((mutate[0].cuda(), mutate[1].cuda()))
        pred_w4 = model4((wide[0].cuda(), wide[1].cuda()))
        pred_m4 = model4((mutate[0].cuda(), mutate[1].cuda()))
        pred_w5 = model5((wide[0].cuda(), wide[1].cuda()))
        pred_m5 = model5((mutate[0].cuda(), mutate[1].cuda()))
        result_w = torch.mean(pred_w1 + pred_w2 + pred_w3 + pred_w4 + pred_w5)/5
        result_m = torch.mean(pred_m1 + pred_m2 + pred_m3 + pred_m4 + pred_m5)/5
        print(result_w.item())
        print(result_m.item())
        # final_result = [str(i.item()) for i in result[0]]
        # # index += 1
        # # f.write(str(index)+','.join(final_result) + '\n')
        # f.write(str(index.item())+','+','.join(final_result) + '\n')
eval_toxin()
eval_anti()
# eval_mechanism()
# eval_mutate()
