import argparse
import os

import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsoluteError, KendallRankCorrCoef, PearsonCorrCoef, RelativeSquaredError
from tqdm import tqdm
import warnings

# from dataset import ADataset, HDataset, collate_fn
from model import MMGraph
from mutation.dataset_graph import construct_graph

parser = argparse.ArgumentParser(description='resnet26')

parser.add_argument('--num-layer', type=int, dest='num_layer', default=1,
                    help='number of GNN message passing layers (default: 2)')
parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=80,
                    help='embedding dimensions (default: 200)')
parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="attention",
                    help='graph level pooling (sum, mean, max, attention)')
parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="graphsage",
                    help='gnn type (gin, gcn, gat, graphsage)')
args = parser.parse_args()


def load_dataset_v2(task='anti', mode='train'):
    p = PDBParser()

    data_list = []
    all_data = pd.read_csv('metadata/p123-sequences.csv', encoding="unicode_escape").values[:200]
    seq_list, labels = all_data[:, 0], all_data[:, 1]

    seq = 'KWMIKWPSNWFTML'
    gt_2 = 123

    pdb_path = "./pdb/pdb_cp/" + seq + ".pdb"

    structure = p.get_structure(0, pdb_path)
    voxel_temp, seq_emb_temp, _ = construct_graph(structure, seq, gt_2, pdb_path)

    # build mutation dataset
    for idx in tqdm(range(len(seq_list)-1)):
        seq = seq_list[idx+1].upper().strip()
        gt = 0
        pdb_path = "./pdb/pdb_cp/" + seq + ".pdb"
        if not os.path.exists(pdb_path):
            print(seq)
        # Suppress specific PDBConstructionWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = p.get_structure(idx, pdb_path)

        voxel1, seq_emb1, _ = construct_graph(structure, seq, gt, pdb_path)
        data_list.append([(voxel_temp, seq_emb_temp), (voxel1, seq_emb1), seq])

    return data_list


def gen_group_pair(sub_data, temp_seq, pdb_path):
    p = PDBParser()
    data_list = []
    seq_list, labels = sub_data[:, 0], sub_data[:, 1]
    temp_index = np.where(seq_list == temp_seq)[0]
    temp_value = int(labels[temp_index])

    structure = p.get_structure(0, pdb_path + temp_seq + ".pdb")
    voxel_temp, seq_temp, _ = construct_graph(structure, temp_seq, temp_value, pdb_path + temp_seq + ".pdb")

    seq_list = np.delete(seq_list, temp_index)
    labels = np.delete(labels, temp_index)
    for idx in range(len(seq_list)):
        seq = seq_list[idx].upper().strip()
        label = int(labels[idx])
        structure = p.get_structure(idx, pdb_path + seq + ".pdb")
        voxel, seq_emb, _ = construct_graph(structure, seq, label, pdb_path + temp_seq + ".pdb")
        data_list.append([(voxel_temp, seq_temp), (voxel, seq_emb), temp_value-label])
    return data_list


def load_dataset_qlx(csv_path='./metadata/data_qlx.csv', pdb_path='./pdb/pdb_cp/'):
    all_data = pd.read_csv(csv_path, encoding="unicode_escape").values
    temp_1 = 'KWMIKWPSNWFTML'
    data_list = gen_group_pair(all_data, temp_1, pdb_path)
    print('Total length:', len(data_list))
    return data_list


def eval_model(args):
    model = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-graphsage-mlce2560.002116/model_1.pth'))
    model1 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model1.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-graphsage-mlce2560.002116/model_2.pth'))
    model2 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model2.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-graphsage-mlce2560.002116/model_3.pth'))
    model3 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model3.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-graphsage-mlce2560.002116/model_4.pth'))
    model4 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model4.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-graphsage-mlce2560.002116/model_5.pth'))

    # test_set = load_dataset('anti', mode='test')
    test_set = load_dataset_v2('anti', mode='test')
    test_loader = DataLoader(test_set, batch_size=1, follow_batch=['x_s'], shuffle=False)

    metric_mae = MeanAbsoluteError().cuda()
    metric_rse = RelativeSquaredError().cuda()
    metric_pcc = PearsonCorrCoef()
    metric_krcc = KendallRankCorrCoef()
    seq_list = []
    preds = []
    ys = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data_graph_1, seq = data[0]
            data_graph_2, seq2 = data[1]
            seq_list.append(seq2[0])
            gt = data[2]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            out0 = model((data_graph_1, data_graph_2))
            out1 = model1((data_graph_1, data_graph_2))
            out2 = model2((data_graph_1, data_graph_2))
            out3 = model3((data_graph_1, data_graph_2))
            out4 = model4((data_graph_1, data_graph_2))
            # print(out)
            out = (out0+out1+out2+out3+out4)/5
            # out = out0
            loss = F.mse_loss(out.squeeze(1), torch.tensor(np.asarray(gt)).cuda().float())
            preds.append(out[0].detach().cpu())
            ys.append(torch.tensor(np.asarray(gt)))
    preds = torch.cat(preds, dim=0)

    gt_list_valid = torch.cat(ys).int()

    mae = round(metric_mae(preds, gt_list_valid).item(), 3)
    rse = round(metric_rse(preds, gt_list_valid).item(), 3)
    pcc = round(metric_pcc(preds.squeeze(-1), gt_list_valid.float().squeeze(-1)).item(), 3)
    kcc = round(metric_krcc(preds.squeeze(-1), gt_list_valid.float().squeeze(-1)).item(), 3)
    print()
    preds_final = [float(i.item()) for i in preds]
    gt_final = [i.item() for i in gt_list_valid]
    f = open('ranking_p123.csv', 'w')
    for i in range(len(preds_final)):
        seq = seq_list[i]
        f.write(str(seq)+','+str(preds_final[i])+','+str(gt_final[i]) +'\n')

    return mae, rse, pcc, kcc

# eval_model(args)
import numpy as np
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def calculate_property(seq):
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    hydrophobic_moment_list = analysed_seq.protein_scale(window=9, param_dict=ProtParamData.kd)
    hydrophobic_moment = np.mean(hydrophobic_moment_list)
    aliphatic_index = ((aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / len(seq))

    charged_amino_acids = ['D', 'E', 'R', 'K', 'H']
    total_charged = sum(aa_counts[aa] for aa in charged_amino_acids)
    charge_density = total_charged / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()

    return analysed_seq.gravy(), hydrophobic_moment, aliphatic_index, analysed_seq.aromaticity(), analysed_seq.instability_index(), alpha_helix, beta_helix, turn_helix, analysed_seq.charge_at_pH(7), analysed_seq.isoelectric_point(), charge_density, np.mean(analysed_seq.flexibility())


# print(eval_model(args))
def eval_5_models(data_graph_1, data_graph_2, model0, model1, model2, model3, model4):
    out0 = model0((data_graph_1, data_graph_2))
    out1 = model1((data_graph_1, data_graph_2))
    out2 = model2((data_graph_1, data_graph_2))
    out3 = model3((data_graph_1, data_graph_2))
    out4 = model4((data_graph_1, data_graph_2))
    out = (out0 + out1 + out2 + out3 + out4) / 5
    return round(out.item(),3)


def calculate_peptide_charge(peptide):
    positive_residues = ['R', 'K', 'H']
    negative_residues = ['D', 'E']
    charge = sum(1 for aa in peptide if aa in positive_residues) - sum(1 for aa in peptide if aa in negative_residues)
    return charge



# Assuming `MMGraph` and `load_dataset_v2` are defined elsewhere
# and `args` is an argparse.Namespace or similar containing configuration
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_gradient_map(grad, title):
    # Ensure the gradient tensor is on CPU and detached from the current graph
    grad_np = grad.cpu().detach().numpy().T

    # Apply exponential scaling
    grad_exp = np.exp(grad_np)

    fig, ax = plt.subplots(figsize=(20, 10))
    cax = ax.matshow(grad_exp, cmap='coolwarm')
    fig.colorbar(cax)

    plt.title(title)
    plt.xlabel('Node Index')
    plt.ylabel('Feature ID')

    ax.set_xticks(np.arange(grad_exp.shape[1]))
    ax.set_yticks(np.arange(grad_exp.shape[0]))
    ax.set_yticklabels(np.arange(1, grad_exp.shape[0] + 1))

    plt.show()
def eval_multi_anti(args):
    model0 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1,
                     gnn_type=args.gnn_type).cuda()
    model0.load_state_dict(
        torch.load('/home/duadua/Desktop/AMP/mutation/run/all-graphsage-mlce2560.002116/model_1.pth'))

    test_set = load_dataset_v2('anti', mode='test')
    test_loader = DataLoader(test_set, batch_size=1, follow_batch=['x_s'], shuffle=False)

    model0.eval()

    for data in tqdm(test_loader):
        data_graph_1, seq = data[0]
        data_graph_2, seq2 = data[1]
        data_graph_1 = data_graph_1.cuda()
        data_graph_2 = data_graph_2.cuda()

        data_graph_1.x_s.requires_grad_()
        data_graph_2.x_s.requires_grad_()

        out = model0((data_graph_1, data_graph_2))
        out.backward(torch.ones_like(out))

        grad_data_graph_1 = data_graph_1.x_s.grad
        grad_data_graph_2 = data_graph_2.x_s.grad

        # print("Gradients for data_graph_1: ", grad_data_graph_1)
        # print("Gradients for data_graph_2: ", grad_data_graph_2)
        # # Assuming `grad_data_graph_1` and `grad_data_graph_2` are the gradient tensors from your previous function
        plot_gradient_map(grad_data_graph_1, "Gradients for Data Graph 1")
        plot_gradient_map(grad_data_graph_2, "Gradients for Data Graph 2")

eval_multi_anti(args)
