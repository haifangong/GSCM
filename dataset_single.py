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


def gen_group_pair(sub_data, temp_seq, pdb_path, task):
    p = PDBParser(QUIET=True)
    data_list = []
    seq_list, labels = sub_data[:, 1], np.concatenate((sub_data[:, 4:9], sub_data[:, 10:11]), axis=1)
    if task == 'all':
        labels = gmean(labels)
    else:
        labels = labels[:, int(task):int(task) + 1]

    temp_index = int(np.where(seq_list == temp_seq)[0][0])
    temp_value = math.log2(int(labels[temp_index]))

    structure = p.get_structure(0, pdb_path + temp_seq + ".pdb")
    voxel_temp, seq_temp, _ = construct_graph(structure, temp_seq, temp_value, pdb_path + temp_seq + ".pdb")

    seq_list = np.delete(seq_list, temp_index)
    labels = np.delete(labels, temp_index)
    for idx in range(len(seq_list)):
        seq = seq_list[idx].upper().strip()
        if 'X' in seq:
            continue
        if calculate_similarity(temp_seq, seq) < 0.5:
            continue
        label = math.log2(int(labels[idx]))
        structure = p.get_structure(idx, pdb_path + seq + ".pdb")
        voxel, seq_emb, _ = construct_graph(structure, seq, label, pdb_path + seq + ".pdb")
        data_list.append([(voxel_temp, seq_temp), (voxel, seq_emb), (temp_value, label)])
    return data_list


def load_dataset_saap(task='all', csv_path='./metadata/data_saap.csv', pdb_path='./pdb/pdb_dbassp/'):
    data_list = []
    all_data = pd.read_csv(csv_path, encoding="unicode_escape").values
    group_1, group_2, group_3 = all_data[:32, :], all_data[33:70, :], all_data[71:, :]
    temp_1 = 'KRIVKLIKKWLR'
    temp_2 = 'GFKRLVQRLKDFLRNLV'
    temp_3_1 = 'LKRVWKAVFKLLKRYWRQLKPVR'
    temp_3_2 = 'LKRLYKRLAKLIKRLYRYLKKPVR'

    data_list.extend(gen_group_pair(group_1, temp_1, pdb_path, task))
    data_list.extend(gen_group_pair(group_2, temp_2, pdb_path, task))
    data_list.extend(gen_group_pair(group_3, temp_3_1, pdb_path, task))
    data_list.extend(gen_group_pair(group_3, temp_3_2, pdb_path, task))
    print('Total length:', len(data_list))
    return data_list


def train_valid_split(data_set, fold, mode):
    train_set = [data_set[i] for i in range(len(data_set)) if (i + fold) % 5 != 0]
    valid_set = [data_set[i] for i in range(len(data_set)) if (i + fold) % 5 == 0]
    return train_set if mode == "train" else valid_set


def plot_number_frequency(numbers, save_name=None):
    # count the occurrence of each number
    counter = collections.Counter(numbers)

    # separate the numbers and their counts for plotting
    num = list(counter.keys())
    counts = list(counter.values())

    # create a bar plot
    plt.bar(num, counts, color='skyblue', edgecolor='black')

    # customize the axes and title
    plt.xlabel('Number', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Number Frequency', fontsize=16)

    # show the plot
    # plt.show()
    # if save_name:
    #     plt.savefig(save_name)


def read_secondary_structure(pdb_id: str, chain_id: str = "A") -> str:
    pdb_file = f"{pdb_id}"
    traj = md.load(pdb_file)
    secondary_structure = md.compute_dssp(traj)[0]
    return "".join(secondary_structure)


def secondary_structure_to_fixed_tensor(secondary_structure: str, max_length: int = 30) -> torch.Tensor:
    sse_to_index = {"H": 1, "E": 2, "C": 3}
    num_sse = len(sse_to_index)
    fixed_tensor = torch.zeros(max_length)
    for i, sse in enumerate(secondary_structure):
        if i >= max_length:  #
            break
        fixed_tensor[i] = sse_to_index[sse]
    # fixed_tensor = torch.zeros(max_length, num_sse)
    # for i, sse in enumerate(secondary_structure):
    #     if i >= max_length:
    #         break
    #     index = sse_to_index[sse]
    #     fixed_tensor[i, index] = 1

    return fixed_tensor


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
    # print(seq)
    analysed_seq = ProteinAnalysis(seq)
    aa_counts = analysed_seq.count_amino_acids()
    aliphatic_index = ((aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / len(seq))

    charged_amino_acids = ['D', 'E', 'R', 'K', 'H']
    total_charged = sum(aa_counts[aa] for aa in charged_amino_acids)
    charge_density = total_charged / len(seq)
    alpha_helix, beta_helix, turn_helix = analysed_seq.secondary_structure_fraction()

    return list(
        [round(analysed_seq.gravy(), 3) * 10, round(aliphatic_index, 3) * 10, round(analysed_seq.aromaticity(), 3) * 10,
         round(analysed_seq.instability_index(), 3), round(alpha_helix * 10, 3), round(beta_helix * 10, 3),
         round(turn_helix * 10, 3), round(analysed_seq.charge_at_pH(7), 3), round(analysed_seq.isoelectric_point(), 3),
         round(charge_density, 3) * 10])


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


def construct_graph(structure, seq, gt, pdb_path):
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

    # Assign features to nodes in the graph, including the 3D position
    for node_id, node_data in G.nodes(data=True):
        res = chain[node_id]
        aa_feature = aa_features[three_to_one(res.get_resname())]
        energy_feature = scoring[node_id - 1]
        position_feature = list(node_data['position'])  # Convert position tuple to list
        node_data['x'] = aa_feature + energy_feature + position_feature
        # node_data['x'][20:40]=[0 for i in range(20)]
        # node_data['x'][40:43]=[0 for i in range(3)]

    # Convert graph to data format used in ML models
    G = nx.convert_node_labels_to_integers(G)
    data_wt = from_networkx(G)
    data_graph = PairData(data_wt.edge_index, data_wt.x)
    data_graph.gt = gt
    global_properties = calculate_property(seq)  # 12
    # print(global_properties)
    # print(len(global_properties))
    # return
    # Encode the sequence and pad it to a fixed length
    seq_emb = [AMAs[char] for char in seq] + [0] * (30 - len(seq))

    # print(seq_emb)
    data_graph.seq = seq_emb
    data_graph.global_f = global_properties
    return data_graph, seq_emb, gt


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


def load_dataset(task='all', gen_dict=False):
    test_sequences_big = parse_ll37_testset(path='metadata/LL37_v0.csv')
    test_sequences = parse_ll37_testset(path='metadata/LL37.csv')
    p = PDBParser(QUIET=True)

    all_data = pd.read_csv('metadata/data_0119.csv', encoding="unicode_escape").values[:-35]  # -35 for ood
    idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], np.concatenate((all_data[:, 4:9], all_data[:, 10:11]),
                                                                                axis=1)
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
        if 'X' in seq or 'O' in seq or 'U' in seq or len(seq) > 30 or len(seq) < 6:
            continue
        if seq in test_sequences_big:
            continue

        filter_idx_list.append(idx)
        seq_new_list.append(seq)
        label_list.append(math.log2(int(labels[idx])))

    print('class count:', np.sum(label_list, axis=0))
    print('sequences number:', len(idx_list))
    print('sequences number filtered:', len(filter_idx_list))
    # filter_idx_list = filter_idx_list[:1000]
    print(len(seq_new_list))
    threshold_simi = 0.3
    bin_count_origin = 50
    bin_count_diff = 1000

    if gen_dict:
        paired_dicts = {}  # key: index of group, value: number of group
        idx = 0
        for idx in tqdm(range(len(filter_idx_list) - 1)):
            current_idx = filter_idx_list[idx]
            temp_seq = seq_new_list[idx].upper().strip()

            # temp gt denotes the seq before mutation
            temp_gt = label_list[idx]
            if temp_gt > 7 or temp_gt < 5:
                continue

            for i in range(idx + 1, len(filter_idx_list) - 1):
                curr_seq = seq_new_list[i].upper().strip()
                curr_gt = label_list[i]

                if calculate_similarity(temp_seq, curr_seq) > threshold_simi:
                    if current_idx not in paired_dicts.keys():
                        paired_dicts[current_idx] = []
                        paired_dicts[current_idx].append((idx, temp_seq, temp_gt))
                        paired_dicts[current_idx].append((idx, curr_seq, curr_gt))
                    else:
                        paired_dicts[current_idx].append((idx, curr_seq, curr_gt))

        with open('./metadata/pair_dicts/' + task + '_dict.json', 'w') as f:
            json.dump(paired_dicts, f)
    else:
        paired_dicts = json.load(open('./metadata/pair_dicts/' + task + '_dict.json', 'r'))

    plot_number_frequency(label_list, save_name='updown.jpg')

    # label_distribution = [0 for i in range(257)]
    label_distribution = [0 for i in range(17)]

    # Calculate the initial label distribution
    for item in label_list:
        label_distribution[int(item)] += 1

    # Create a dictionary to hold the thresholds for each label that needs to be reduced
    labels_to_reduce = {}
    for label, count in enumerate(label_distribution):
        if count > bin_count_origin:
            labels_to_reduce[label] = bin_count_origin / count
    print('labels_to_reduce', labels_to_reduce)
    # remove unpaired sequences
    ori_gt_list = []
    clean_dict = {}
    length_list = []
    for key in paired_dicts.keys():
        if len(paired_dicts[key]) == 1:
            continue
        else:
            parsed_list = []
            for item in paired_dicts[key]:
                gt = item[-1]
                if gt in labels_to_reduce.keys():
                    if random.random() > float(labels_to_reduce[int(gt)]):
                        continue

                if os.path.exists("./pdb/pdb_dbassp/" + seq + ".pdb"):
                    pdb_path = "./pdb/pdb_dbassp/" + seq + ".pdb"
                elif os.path.exists("./pdb/pdb_gen/" + seq + ".pdb"):
                    pdb_path = "./pdb/pdb_gen/" + seq + ".pdb"
                else:
                    raise FileNotFoundError

                structure = p.get_structure(idx, pdb_path)
                seq = item[1]
                ori_gt_list.append(gt)
                voxel1, seq_emb1, _ = construct_graph(structure, seq, gt, pdb_path)
                parsed_list.append((item[0], item[-1], voxel1, seq_emb1))
            clean_dict[key] = parsed_list
            length_list.append(len(paired_dicts[key]))

    # plot_number_frequency(ori_gt_list, save_name='topdown.jpg')
    # plot_number_frequency(length_list)
    # print(clean_dict)
    print('items without combination:', len(clean_dict.keys()))

    filter_data = []
    for key in clean_dict.keys():
        clean_dict_list = clean_dict[key]
        for i in range(len(clean_dict_list)):
            for j in range(i + 1, len(clean_dict_list)):
                filter_data.append((clean_dict_list[i], clean_dict_list[j]))

    paired_list = filter_data

    # Preprocessing step to determine the indices to keep for balancing
    def get_balanced_indices(gt1_list, gt2_list, max_samples_per_bin=10):
        differences = np.abs(np.array(gt1_list) - np.array(gt2_list))
        indices_per_diff = defaultdict(list)
        for idx, diff in enumerate(differences):
            indices_per_diff[diff].append(idx)

        balanced_indices = set()
        for diff, indices in indices_per_diff.items():
            if len(indices) > max_samples_per_bin:
                balanced_indices.update(np.random.choice(indices, max_samples_per_bin, replace=False))
            else:
                balanced_indices.update(indices)

        return balanced_indices

    print('paired_list', len(paired_list))
    # Assuming paired_list is available and contains your data
    gt1_list = [int(item[0][1] * 100) for item in paired_list]
    gt2_list = [int(item[1][1] * 100) for item in paired_list]

    # Get the indices of the samples to keep
    balanced_indices = get_balanced_indices(gt1_list, gt2_list, max_samples_per_bin=bin_count_diff)

    # DataLoader loop
    data_list = []
    gt_list = []
    for idx, item in enumerate(tqdm(paired_list)):
        # Skip if the current index is not in the balanced set
        if idx not in balanced_indices:
            continue

        item_before, item_after = item
        idx1, gt1, voxel1, seq_emb1 = item_before
        idx2, gt2, voxel2, seq_emb2 = item_after
        # print(gt1)
        # if gt1 > 7:
        #     continue
        # if gt1 < 5:
        #     continue
        new_gt = gt1 - gt2
        data_list.append([(voxel1, seq_emb1), (voxel2, seq_emb2), (gt1, gt2)])
        # data_list.append([(voxel2, seq_emb2), (voxel1, seq_emb1), -new_gt])  # Note the negation for reverse order

        gt_list.append(new_gt)

    # Now `data_list` has been populated with balanced data according to the absolute differences of gt1 and gt2
    print('filter_out count:', len(paired_list) - len(gt_list))
    plot_number_frequency(gt_list, save_name='updown.jpg')
    gt_list = np.asarray(gt_list)
    gt_class_wise_count = np.sum(gt_list, 0)
    print('class wise counts', gt_class_wise_count)
    return data_list


if __name__ == "__main__":
    data_list = load_dataset('all')
    # test_set = load_dataset_saap()
    # gt_list = []
    # for i in test_set:
    #     gt_list.append(i[-1])
    # print(gt_list)
    # plot_number_frequency(gt_list, save_name='updown.jpg')
