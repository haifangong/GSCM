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
from dataset_graph import construct_graph

parser = argparse.ArgumentParser(description='resnet26')

parser.add_argument('--num-layer', type=int, dest='num_layer', default=2,
                    help='number of GNN message passing layers (default: 2)')
parser.add_argument('--emb-dim', type=int, dest='emb_dim', default=90,
                    help='embedding dimensions (default: 200)')
parser.add_argument('--dropout-ratio', type=float, dest='dropout_ratio', default=0.5,
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--graph-pooling', type=str, dest='graph_pooling', default="attention",
                    help='graph level pooling (sum, mean, max, attention)')
parser.add_argument('--gnn-type', type=str, dest='gnn_type', default="gatv2",
                    help='gnn type (gin, gcn, gat, gatv2)')
args = parser.parse_args()


def load_dataset_v2(task='anti', mode='train'):
    p = PDBParser()

    data_list = []
    all_data = pd.read_csv('/home/ubuntu/gonghaifan/search_all/scans/3_positive3.csv', encoding="unicode_escape").values
    seq_list, labels = all_data[:, 0], all_data[:, 1]

    seq = 'KWKIKWPVRWFTKL' #  KWMIKWPSNWFTML
    gt_2 = 123

    pdb_path = "./pdb/pdb_gen/" + seq + ".pdb"

    structure = p.get_structure(0, pdb_path)
    voxel_temp, seq_emb_temp, _ = construct_graph(structure, seq, gt_2, pdb_path)

    # build mutation dataset
    for idx in tqdm(range(len(seq_list)-1)):
        seq = seq_list[idx+1].upper().strip()
        gt = 0
        pdb_path = "/home/ubuntu/gonghaifan/helixfold-single/pdb_relaxed/" + seq + ".pdb"
        if not os.path.exists(pdb_path):
            continue
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
    model.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/anti-gatv2-mlce2560.002110/model_1.pth'))
    model1 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model1.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/anti-gatv2-mlce2560.002110/model_2.pth'))
    model2 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model2.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/anti-gatv2-mlce2560.002110/model_3.pth'))
    model3 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model3.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/anti-gatv2-mlce2560.002110/model_4.pth'))
    model4 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model4.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/anti-gatv2-mlce2560.002110/model_5.pth'))

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

import time

# print(eval_model(args))
def eval_5_models(data_graph_1, data_graph_2, model0, model1, model2, model3, model4):
    start_time = time.time()
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


def eval_multi_anti(args):
    model0 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model0.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/all-gatv2-mlce2560.002110/model_1.pth'))
    model1 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model1.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/all-gatv2-mlce2560.002110/model_2.pth'))
    model2 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model2.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/all-gatv2-mlce2560.002110/model_3.pth'))
    model3 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model3.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/all-gatv2-mlce2560.002110/model_4.pth'))
    model4 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model4.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/all-gatv2-mlce2560.002110/model_5.pth'))

    model10 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model10.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/0-gatv2-mlce2560.002110/model_1.pth'))
    model11 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model11.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/0-gatv2-mlce2560.002110/model_2.pth'))
    model12 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model12.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/0-gatv2-mlce2560.002110/model_3.pth'))
    model13 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model13.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/0-gatv2-mlce2560.002110/model_4.pth'))
    model14 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model14.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/0-gatv2-mlce2560.002110/model_5.pth'))

    model20 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model20.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/1-gatv2-mlce2560.002110/model_1.pth'))
    model21 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model21.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/1-gatv2-mlce2560.002110/model_2.pth'))
    model22 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model22.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/1-gatv2-mlce2560.002110/model_3.pth'))
    model23 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model23.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/1-gatv2-mlce2560.002110/model_4.pth'))
    model24 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model24.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/1-gatv2-mlce2560.002110/model_5.pth'))

    model30 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model30.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/2-gatv2-mlce2560.002110/model_1.pth'))
    model31 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model31.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/2-gatv2-mlce2560.002110/model_2.pth'))
    model32 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model32.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/2-gatv2-mlce2560.002110/model_3.pth'))
    model33 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model33.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/2-gatv2-mlce2560.002110/model_4.pth'))
    model34 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model34.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/2-gatv2-mlce2560.002110/model_5.pth'))

    model40 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model40.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/3-gatv2-mlce2560.002110/model_1.pth'))
    model41 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model41.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/3-gatv2-mlce2560.002110/model_2.pth'))
    model42 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model42.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/3-gatv2-mlce2560.002110/model_3.pth'))
    model43 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model43.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/3-gatv2-mlce2560.002110/model_4.pth'))
    model44 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model44.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/3-gatv2-mlce2560.002110/model_5.pth'))

    model50 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model50.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/4-gatv2-mlce2560.002110/model_1.pth'))
    model51 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model51.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/4-gatv2-mlce2560.002110/model_2.pth'))
    model52 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model52.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/4-gatv2-mlce2560.002110/model_3.pth'))
    model53 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model53.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/4-gatv2-mlce2560.002110/model_4.pth'))
    model54 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model54.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/4-gatv2-mlce2560.002110/model_5.pth'))

    model60 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model60.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/5-gatv2-mlce2560.002110/model_1.pth'))
    model61 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model61.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/5-gatv2-mlce2560.002110/model_2.pth'))
    model62 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model62.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/5-gatv2-mlce2560.002110/model_3.pth'))
    model63 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model63.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/5-gatv2-mlce2560.002110/model_4.pth'))
    model64 = MMGraph(num_layer=args.num_layer, input_dim=43, emb_dim=args.emb_dim, out_dim=1, gnn_type=args.gnn_type).cuda()
    model64.load_state_dict(torch.load('/home/ubuntu/gonghaifan/Mutation/run/5-gatv2-mlce2560.002110/model_5.pth'))

    # test_set = load_dataset('anti', mode='test')
    test_set = load_dataset_v2('anti', mode='test')
    test_loader = DataLoader(test_set, batch_size=1, follow_batch=['x_s'], shuffle=False)

    f = open('ranking_k20_balanced_iid.csv', 'w')
    f.write('seq, rank, E, all, 1,2,3,4,5,6, 1_6 avg, hydrophobic, hydrophobic_moment, aliphatic_index, aromaticity, instability_index, alpha_helix, beta_helix, turn_helix, charge, isoelectric_point, charge_density, flexibility\n')

    with torch.no_grad():
        for data in tqdm(test_loader):
            data_graph_1, seq = data[0]
            data_graph_2, seq2 = data[1]
            seq_str = data[2][0]
            data_graph_1 = data_graph_1.cuda()
            data_graph_2 = data_graph_2.cuda()
            all_results = eval_5_models(data_graph_1, data_graph_2, model0.eval(), model1.eval(), model2.eval(), model3.eval(), model4.eval())
            cate_1 = eval_5_models(data_graph_1, data_graph_2, model10.eval(), model11.eval(), model12.eval(), model13.eval(), model14.eval())
            cate_2 = eval_5_models(data_graph_1, data_graph_2, model20.eval(), model21.eval(), model22.eval(), model23.eval(), model24.eval())
            cate_3 = eval_5_models(data_graph_1, data_graph_2, model30.eval(), model31.eval(), model32.eval(), model33.eval(), model34.eval())
            cate_4 = eval_5_models(data_graph_1, data_graph_2, model40.eval(), model41.eval(), model42.eval(), model43.eval(), model44.eval())
            cate_5 = eval_5_models(data_graph_1, data_graph_2, model50.eval(), model51.eval(), model52.eval(), model53.eval(), model54.eval())
            cate_6 = eval_5_models(data_graph_1, data_graph_2, model60.eval(), model61.eval(), model62.eval(), model63.eval(), model64.eval())
            avg = round((cate_1+cate_2+cate_3+cate_4+cate_5+cate_6)/6,3)
            rank = round((cate_1+cate_2+cate_3+cate_4+cate_5+cate_6+all_results)/7, 3)
            charge = calculate_peptide_charge(seq_str)
            f.write(str(seq_str) + ',' + str(rank) + ',' + str(charge) + ',' + str(all_results) + ',' +
                    str(cate_1) + ',' + str(cate_2) + ',' + str(cate_3) + ',' + str(cate_4) + ',' + str(cate_5) + ',' + str(cate_6) + ',' + str(avg) + ',')
            properties = calculate_property(seq_str)
            for p in properties:
                f.write(str(round(p, 3))+',')
            f.write('\n')


eval_multi_anti(args)
