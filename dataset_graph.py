import collections
import json
import math
import os
import random
from collections import defaultdict

import mdtraj as md
import networkx as nx
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from Bio.PDB import is_aa
from Bio.PDB.Polypeptide import three_to_one
from Bio.SeqUtils import ProtParamData
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from dataset import parse_ll37_testset, calculate_similarity


def gmean(labels):
    # Assuming 'labels' is a NumPy array of shape (n, m)
    # Calculate the geometric mean across axis 1

    # Take the product of elements along axis=1
    product = np.prod(labels, axis=1)

    # nth root of product, where n is the number of elements along axis=1
    geometric_means = product ** (1.0 / labels.shape[1])
    return geometric_means


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}


def gen_group_pair(args, sub_data, seq_temp, pdb_path, task):
    p = PDBParser(QUIET=True)
    data_list = []
    seq_list, labels = sub_data[:, 1], np.concatenate((sub_data[:, 4:9], sub_data[:, 10:11]), axis=1)
    if task == 'all':
        labels = gmean(labels)
    else:
        labels = labels[:, int(task):int(task) + 1]

    temp_index = int(np.where(seq_list == seq_temp)[0][0])
    temp_value = math.log2(int(labels[temp_index]))

    structure = p.get_structure(0, pdb_path + seq_temp + ".pdb")
    voxel_temp = construct_graph(args, structure, seq_temp, temp_value, pdb_path + seq_temp + ".pdb")

    seq_list = np.delete(seq_list, temp_index)
    labels = np.delete(labels, temp_index)
    for idx in range(len(seq_list)):
        seq = seq_list[idx].upper().strip()
        if 'X' in seq or 'B' in seq or 'J' in seq or 'Q' in seq or 'Z' in seq or 'U' in seq:
            continue
        if calculate_similarity(seq_temp, seq) < 0.5:
            continue
        label = math.log2(int(labels[idx]))
        structure = p.get_structure(idx, pdb_path + seq + ".pdb")
        voxel = construct_graph(args, structure, seq, label, pdb_path + seq + ".pdb")
        data_list.append([(voxel_temp), (voxel), temp_value - label])
    return data_list


def load_dataset_saap(args, task='all', csv_path='./metadata/data_saap.csv', pdb_path='./pdb/pdb_dbassp/'):
    data_list = []
    all_data = pd.read_csv(csv_path, encoding="unicode_escape").values
    group_1, group_2, group_3 = all_data[:32, :], all_data[33:70, :], all_data[71:, :]
    temp_1 = 'KRIVKLIKKWLR'
    temp_2 = 'GFKRLVQRLKDFLRNLV'
    temp_3_1 = 'LKRVWKAVFKLLKRYWRQLKPVR'
    temp_3_2 = 'LKRLYKRLAKLIKRLYRYLKKPVR'

    data_list.extend(gen_group_pair(args, group_1, temp_1, pdb_path, task))
    data_list.extend(gen_group_pair(args, group_2, temp_2, pdb_path, task))
    data_list.extend(gen_group_pair(args, group_3, temp_3_1, pdb_path, task))
    data_list.extend(gen_group_pair(args, group_3, temp_3_2, pdb_path, task))
    print('Total length:', len(data_list))
    return data_list


def train_valid_split(data_set, fold, mode):
    train_set = [data_set[i] for i in range(len(data_set)) if (i + fold) % 5 != 0]
    valid_set = [data_set[i] for i in range(len(data_set)) if (i + fold) % 5 == 0]
    return train_set if mode == "train" else valid_set


def plot_number_frequency(numbers, save_name=None):
    # Count the occurrence of each number
    counter = collections.Counter(numbers)

    # Separate the numbers and their counts for plotting
    num = list(counter.keys())
    counts = list(counter.values())

    # Sort the numbers and their counts
    sorted_pairs = sorted(zip(num, counts))
    num, counts = zip(*sorted_pairs)

    # Create a line plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size
    plt.plot(num, counts, color='skyblue', linewidth=1)

    # Customize the axes and title
    plt.xlabel('Number', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Number Frequency', fontsize=16)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.grid(axis='both', linestyle='--', alpha=0.7)  # Add grid lines

    # Show the plot
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')  # Save the figure with tight layout
    else:
        plt.show()


import mdtraj as md

def calculate_secondary_structure_percentages(seq: str) -> dict:
    analysed_seq = ProteinAnalysis(seq)
    alpha_helix, beta_sheet, turn = analysed_seq.secondary_structure_fraction()
    
    # 转换为百分比
    alpha_helix_percentage = alpha_helix * 100
    beta_sheet_percentage = beta_sheet * 100
    turn_percentage = turn * 100

    # 计算其他类型的百分比
    other_percentage = 100 - (alpha_helix_percentage + beta_sheet_percentage)

    return {
        "alpha_helix": alpha_helix_percentage,
        "beta_sheet": beta_sheet_percentage,
        "turn": turn_percentage,
        "other": other_percentage
    }

def read_secondary_structure(pdb_id: str, chain_id: str = "A") -> dict:
    traj = md.load(pdb_id)
    chain = next(chain for chain in traj.topology.chains if chain.chain_id == chain_id)
    chain_traj = traj.atom_slice(traj.topology.select(f"chainid {chain.index}"))
    secondary_structure = md.compute_dssp(chain_traj)[0]
    print(secondary_structure)
    # 计算各种类型的数量
    alpha_helix_count = sum(1 for ss in secondary_structure if ss == 'H')
    beta_sheet_count = sum(1 for ss in secondary_structure if ss in ['B', 'E'])
    other_count = sum(1 for ss in secondary_structure if ss not in ['H', 'B', 'E', 'NA'])
    total_residues = sum(1 for ss in secondary_structure if ss != 'NA')
    
    if total_residues == 0:
        raise ValueError("No residues found for the specified chain.")
    
    # 计算百分比
    alpha_helix_percentage = alpha_helix_count / total_residues
    beta_sheet_percentage = beta_sheet_count / total_residues
    other_percentage = other_count / total_residues
    
    return {
        "alpha_helix": alpha_helix_percentage,
        "beta_sheet": beta_sheet_percentage,
        "other": other_percentage
    }

# Define a function to calculate and pad flexibility with zeros
def calculate_and_pad_flexibility_with_zeros(seq):
    try:
        # Create a ProteinAnalysis object
        analysed_seq = ProteinAnalysis(seq)
        # Calculate the flexibility of the sequence
        flexibilities = analysed_seq.flexibility()
        # Initialize the padded flexibility list with zeros
        padded_flexibilities = [0.0] * len(seq)
        # Insert the actual flexibilities into the middle of the list
        # The first 4 and the last 4 residues will remain 0.0
        for i in range(len(flexibilities)):
            padded_flexibilities[i + 4] = flexibilities[i]
        return padded_flexibilities
    except Exception as e:
        # Handle any error that occurs during the analysis
        print(f"An error occurred: {e}")
        # Optionally, return a list of zeros if an error occurs
        return [0.0] * len(seq)


def calculate_property(seq):
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    aliphatic_index = ((aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / len(seq))

    positive_charged_amino_acids = ['R', 'K', 'H']
    negative_charged_amino_acids = ['D', 'E']

    total_positive_charged = sum(aa_counts.get(aa, 0) for aa in positive_charged_amino_acids)
    total_negative_charged = sum(aa_counts.get(aa, 0) for aa in negative_charged_amino_acids)

    total_charge = total_positive_charged - total_negative_charged
    charge_density = total_charge / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()

    return list(
        [round(analysed_seq.gravy(), 3) * 10, round(aliphatic_index, 3) * 10, round(analysed_seq.aromaticity(), 3) * 10,
         round(analysed_seq.instability_index(), 3), round(alpha_helix * 10, 3), round(beta_helix * 10, 3),
         round(turn_helix * 10, 3), round(analysed_seq.charge_at_pH(7), 3), round(analysed_seq.isoelectric_point(), 3),
         round(charge_density, 3) * 10])

def calculate_property_pdb(pdb_file):
    # Load the structure
    traj = md.load(pdb_file)
    topology = traj.topology

    # Extract sequence from the structure
    sequence = ''.join([residue.code for residue in topology.residues if residue.is_protein])
    analysed_seq = ProteinAnalysis(sequence)

    # Count amino acids directly from the structure
    aa_counts = {residue.code: 0 for residue in topology.residues if residue.is_protein}
    for residue in topology.residues:
        if residue.is_protein:
            aa_counts[residue.code] += 1

    # Calculate aliphatic index based on the structure
    aliphatic_index = ((aa_counts.get('A', 0) + 2.9 * aa_counts.get('V', 0) + 3.9 * (aa_counts.get('I', 0) + aa_counts.get('L', 0))) / len(sequence))

    # Positive and negative charged amino acids
    positive_charged_amino_acids = ['R', 'K', 'H']
    negative_charged_amino_acids = ['D', 'E']

    total_positive_charged = sum(aa_counts.get(aa, 0) for aa in positive_charged_amino_acids)
    total_negative_charged = sum(aa_counts.get(aa, 0) for aa in negative_charged_amino_acids)

    total_charge = total_positive_charged - total_negative_charged
    charge_density = total_charge / len(sequence)

    # Secondary structure (using DSSP)
    dssp = md.compute_dssp(traj)
    total_residues = len(sequence)
    alpha_helix = np.sum(dssp == 'H') / total_residues
    beta_sheet = np.sum(dssp == 'E') / total_residues
    turn_helix = np.sum(dssp == 'T') / total_residues

    # Calculate properties using ProteinAnalysis for those not readily available from structure alone
    gravy = analysed_seq.gravy()
    aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    charge_at_pH7 = analysed_seq.charge_at_pH(7)
    isoelectric_point = analysed_seq.isoelectric_point()

    return [
        round(gravy, 3) * 10,
        round(aliphatic_index, 3) * 10,
        round(aromaticity, 3) * 10,
        round(instability_index, 3),
        round(alpha_helix * 10, 3),
        round(beta_sheet * 10, 3),
        round(turn_helix * 10, 3),
        round(charge_at_pH7, 3),
        round(isoelectric_point, 3),
        round(charge_density, 3) * 10
    ]


def read_scoring_functions(pdb):
    scoring = False
    profile = []
    for line in open(pdb):
        if line.startswith("#END_POSE_ENERGIES_TABLE"):
            scoring = False
        if scoring:
            data = [float(v) for v in line.split()[1:]]
            profile.append(data)
        if line.startswith("pose"):
            scoring = True
    return profile


def load_aa_features(feature_path):
    aa_features = {}
    for line in open(feature_path):
        line = line.strip().split()
        aa, features = line[0], [float(feature) for feature in line[1:]]
        aa_features[aa] = features
    return aa_features


def construct_graph(args, structure, seq, gt, pdb_path):
    # Initialize an empty graph
    G = nx.Graph()

    # Check if chain 'A' is present in the structure, return None if not
    if 'A' not in structure[0]:
        return None
    chain = structure[0]['A']

    # Add nodes to the graph for each amino acid in the chain with their 3D positions
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            alpha_c_atom = res["CA"]
            x, y, z = alpha_c_atom.get_coord()  # Get 3D coordinates
            G.add_node(res.id[1], name=resname, position=(x, y, z))

    # Add edges between nodes if the distance between alpha carbons is <= 5Å
    for i, m in enumerate(G.nodes):
        for n in list(G.nodes)[i + 1:]:
            distance = chain[m]["CA"] - chain[n]["CA"]
            if distance <= 5:
                G.add_edge(m, n, weight=5 / distance)

    # Load scoring functions and features
    scoring = read_scoring_functions(pdb_path)
    aa_features = load_aa_features('metadata/features.txt')

    # Encode structure feature
    for node_id, node_data in G.nodes(data=True):
        res = chain[node_id]
        aa_feature = aa_features[three_to_one(res.get_resname())]
        energy_feature = scoring[node_id - 1]
        position_feature = list(node_data['position'])  # Convert position tuple to list
        if 'h' in args.feature:
            node_data['x'] = aa_feature + energy_feature + position_feature
            if 's' not in args.structure_feature:
                node_data['x'][:20]=[0 for i in range(20)]
            if 'e' not in args.structure_feature:
                node_data['x'][20:40]=[0 for i in range(20)]
            if 'p' not in args.structure_feature:
                node_data['x'][40:43]=[0 for i in range(3)]
        else:
            global_properties = [0] * 43
        node_data['x'] = aa_feature + energy_feature + position_feature

    # Convert graph to data format used in ML models
    G = nx.convert_node_labels_to_integers(G)
    data_wt = from_networkx(G)
    data_graph = PairData(data_wt.edge_index, data_wt.x)

    # Encode global feature
    if 'g' in args.feature:
        global_properties = calculate_property(seq)
    else:
        global_properties = [0] * 10

    # Encode the sequence and pad it to a fixed length
    if 's' in args.feature:
        seq_emb = [AMAs[char] for char in seq] + [0] * (30 - len(seq))
    else:
        seq_emb = [0] * 30

    data_graph.gt = gt
    data_graph.seq = seq_emb
    data_graph.global_f = global_properties
    return data_graph

def extract_sequences_from_csv(csv_data):
    with open(csv_data, 'r') as f:
        return [line.strip().split(',')[1] for line in f.readlines()[1:]]
# Usage example
# structure, seq, gt, pdb_path need to be defined before calling the function
# data_graph, seq_emb, gt = construct_graph(structure, seq, gt, pdb_path)
class PairData(Data):
    def __init__(self, edge_index_s, x_s):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s

    def __inc__(self, key, value, *args):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'wide_nodes':
            return self.x_s.num_nodes
        else:
            return super().__inc__(key, value, *args)


def load_dataset(args, gen_dict=True):
    task = args.task
    test_sequences_big = extract_sequences_from_csv('metadata/data_simi.csv')
    print(len(test_sequences_big))
    # test_sequences_big = parse_ll37_testset(path='metadata/LL37_v0.csv')
    # test_sequences = parse_ll37_testset(path='metadata/LL37.csv')
    p = PDBParser(QUIET=True)

    # all_data = pd.read_csv('metadata/data_0119.csv', encoding="unicode_escape").values[:-35]  # -35 for ood
    all_data = pd.read_csv('metadata/data_0511.csv', encoding="unicode_escape").values[:-94]

    idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], np.concatenate((all_data[:, 4:9], all_data[:, 10:11]), axis=1)
    labels[labels > 256] = 256
    labels[labels < 2] = 2
    if task == 'all':
        labels = gmean(labels)
    else:
        labels = labels[:, int(task):int(task) + 1]

    filter_idx_list = []
    seq_new_list = []
    label_list = []
    # print(test_sequences)
    # missing_pdb = open('missing_pdb.csv', 'w')
    for idx in range(len(idx_list)):
        seq = seq_list[idx].upper().strip()
        if 'X' in seq or 'B' in seq or 'J' in seq or 'Q' in seq or 'Z' in seq or 'U' in seq or len(seq) > 30 or len(seq) < 6:
            continue
        # print(seq)
        if seq in test_sequences_big:
            continue

        filter_idx_list.append(idx)
        seq_new_list.append(seq)
        label_list.append(math.log2(int(labels[idx])))

    print('sequences number:', len(idx_list))
    print('sequences number filtered:', len(filter_idx_list))


    # 第一部分：筛选序列对
    pair_info_list = []
    diff_count = defaultdict(int)  # 用于计数每个差值的样本数

    # 第一部分：筛选序列对
    for idx_temp in tqdm(range(len(seq_new_list))):
        gt_temp = label_list[idx_temp]
        # if gt_temp > 7 or gt_temp < 5:
        #     continue
        seq_temp = seq_new_list[idx_temp]

        for idx_target in range(len(seq_new_list)):
            seq_target = seq_new_list[idx_target]
            if abs(len(seq_temp) - len(seq_target)) > 5:
                continue
            if calculate_similarity(seq_temp, seq_target) < args.similarity:
                continue
            if seq_temp == seq_target:
                continue

            gt_target = label_list[idx_target]
            diff = abs(gt_temp - gt_target) * 100

            # 检查当前差值的样本数是否达到上限
            if diff_count[diff] < args.balance_num:
                pair_info_list.append((idx_temp, seq_temp, gt_temp, idx_target, seq_target, gt_target))
                diff_count[diff] += 1
        

    # 定义保存成对数据和标签差异的列表

    # 构造一个字典，key是序列，value是基于PDB文件构建的数据
    pdb_data_dict = {}

    # 读取所有PDB文件并构建字典
    for seq in tqdm(seq_new_list):
        if os.path.exists("./pdb/pdb_dbassp/" + seq + ".pdb"):
            pdb_path = "./pdb/pdb_dbassp/" + seq + ".pdb"
        elif os.path.exists("./pdb/pdb_gen/" + seq + ".pdb"):
            pdb_path = "./pdb/pdb_gen/" + seq + ".pdb"
        else:
            raise FileNotFoundError

        structure = p.get_structure(seq, pdb_path)
        gt = label_list[seq_new_list.index(seq)]
        data = construct_graph(args, structure, seq, gt, pdb_path)
        pdb_data_dict[seq] = data

    # 使用先前生成的pair_info_list
    paired_list = []
    diff_label_list = []

    for (idx_temp, seq_temp, gt_temp, idx_target, seq_target, gt_target) in tqdm(pair_info_list):
        data_temp = pdb_data_dict.get(seq_temp)
        data_target = pdb_data_dict.get(seq_target)

        if data_temp is not None and data_target is not None:
            paired_list.append((data_temp, data_target, gt_temp - gt_target))
            diff_label_list.append(gt_temp - gt_target)

    plot_number_frequency(diff_label_list, save_name=args.weight_dir+'/data_distribution.jpg')
    return paired_list

if __name__ == "__main__":
    # pdb_id = "./pdb/pdb_real/2mlt.pdb"
    # percentages = read_secondary_structure(pdb_id, chain_id="A")
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2%}")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2%}")
    # print(f"Other Types Percentage: {percentages['other']:.2%}")

    # pdb_file = "./pdb/pdb_dbassp/GIGAVLKVLTTGLPALISWIKRKRQQ.pdb"
    # percentages = read_secondary_structure(pdb_file, chain_id="A")
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2%}")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2%}")
    # print(f"Other Types Percentage: {percentages['other']:.2%}")

    # seq = "GIGAVLKVLTTGLPALISWIKRKRQQ"
    # percentages = calculate_secondary_structure_percentages(seq)
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2f}%")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2f}%")
    # # print(f"Turn Percentage: {percentages['turn']:.2f}%")
    # print(f"Other Types Percentage: {percentages['other']:.2f}%")

    # pdb_id = "./pdb/pdb_real/2k6o.pdb"
    # percentages = read_secondary_structure(pdb_id, chain_id="A")
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2%}")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2%}")
    # print(f"Other Types Percentage: {percentages['other']:.2%}")

    # pdb_file = "./pdb/pdb_dbassp/LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES.pdb"
    # percentages = read_secondary_structure(pdb_file, chain_id="A")
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2%}")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2%}")
    # print(f"Other Types Percentage: {percentages['other']:.2%}")

    # seq = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    # percentages = calculate_secondary_structure_percentages(seq)
    # print(f"Alpha Helix Percentage: {percentages['alpha_helix']:.2f}%")
    # print(f"Beta Sheet Percentage: {percentages['beta_sheet']:.2f}%")
    # # print(f"Turn Percentage: {percentages['turn']:.2f}%")
    # print(f"Other Types Percentage: {percentages['other']:.2f}%")

    import argparse
    parser = argparse.ArgumentParser(description='resnet26')
    # model setting
    parser.add_argument('--model', type=str, default='mm',
                        help='model resnet26, bi-gru')
    parser.add_argument('--fusion', type=str, default='1',
                        help="Seed for splitting dataset (default 1)")
    parser.add_argument('--similarity', type=float, default=0.3,
                        help="Similarity for construct training set.")

    # task & dataset setting
    parser.add_argument('--task', type=str, default='all',
                        help='task: anti toxin anti-all mechanism anti-binary anti-regression mic')
    parser.add_argument('--classes', type=int, default=6,
                        help='model')
    parser.add_argument('--split', type=int, default=5,
                        help="Split k fold in cross validation (default: 5)")
    parser.add_argument('--seed', type=int, default=1,
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
    # training setting
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=110,
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
    parser.add_argument('--pretrain', type=str, dest='pretrain', default='./run/all-gatv2-mlce2560.002110/model_1.pth',
                        help='path of the pretrain model')  # /home/duadua/Desktop/fetal/3dpretrain/runs/e50.pth

    args = parser.parse_args()

    data_list = load_dataset(args)
    # test_set = load_dataset_saap()
    # gt_list = []
    # for i in test_set:
    #     gt_list.append(i[-1])
    # print(gt_list)
    # plot_number_frequency(gt_list, save_name='updown.jpg')
