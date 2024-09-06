import json
import math
import os
import torch
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_one
from torch.utils.data import random_split, Dataset
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
# from torchvision import transforms
import torch.nn.utils.rnn as rnn_utils

import collections
import matplotlib.pyplot as plt

def gmean(labels):
    # Assuming 'labels' is a NumPy array of shape (n, m)
    # Calculate the geometric mean across axis 1

    # Take the product of elements along axis=1
    product = np.prod(labels, axis=1)

    # nth root of product, where n is the number of elements along axis=1
    geometric_means = product ** (1.0 / labels.shape[1])
    return geometric_means

def plot_number_frequency(numbers, save_name=None):
    # count the occurrence of each number
    counter = collections.Counter(numbers)

    # separate the numbers and their counts for plotting
    num = list(counter.keys())
    counts = list(counter.values())

    # use a style template
    plt.style.use('ggplot')

    # create a bar plot
    plt.bar(num, counts, color='skyblue', edgecolor='black')

    # customize the axes and title
    plt.xlabel('Number', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Number Frequency', fontsize=16)

    # show the plot
    # plt.show()
    if save_name:
        plt.savefig(save_name)


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


def pdb_parser(structure):
    """

    """
    voxel = np.zeros((4, 64, 64, 64), dtype=np.int8)
    id = ''
    seq_str = ''
    for i in structure[0]:
        id = i.id
    chain = structure[0][id]
    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            resname = res.get_resname()
            amino = three_to_one(resname)
            seq_str += str(amino)
            ATOM_WATER = AMINO_ACID_WATER[amino]
            ATOM_CHARGE = AMINO_ACID_CHARGE[amino]
            ATOM_CATEGORY = AMAs[amino] * 20

            for i in res:
                if i.id not in ATOMS.keys():
                    continue
                x, y, z = i.get_coord()
                if abs(x) > 32:
                    x = clamp(x, -31, 31)
                if abs(y) > 32:
                    y = clamp(x, -31, 31)
                if abs(z) > 32:
                    z = clamp(x, -31, 31)
                x_i, y_i, z_i = int(x) + 32, int(y) + 32, int(z) + 32
                ATOM_WEIGHT = ATOMS[i.id]
                ATOM_R = ATOMS_R[i.id]

                if ATOM_R <= 1.5:
                    voxel[0, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WEIGHT
                    voxel[1, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_WATER
                    voxel[2, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CHARGE
                    voxel[3, x_i - 1:x_i + 1, y_i - 1:y_i + 1, z_i - 1:z_i + 1] = ATOM_CATEGORY
                else:
                    voxel[0, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WEIGHT
                    voxel[1, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_WATER
                    voxel[2, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CHARGE
                    voxel[3, x_i - ATOM_R: x_i + ATOM_R, x_i - ATOM_R: x_i + ATOM_R,
                    x_i - ATOM_R: x_i + ATOM_R] = ATOM_CATEGORY
    return voxel


ATOMS = {'H': 1, 'C': 12, 'N': 14, 'O': 16, 'S': 30}
ATOMS_R = {'H': 1, 'C': 1.5, 'N': 1.5, 'O': 1.5, 'S': 2}
AMINO_ACID_WATER = {'A': 255, 'V': 255, 'P': 255, 'F': 255, 'W': 255, 'I': 255, 'L': 255, 'G': 155, 'M': 155,
                    'Y': 55, 'S': 55, 'T': 55, 'C': 55, 'N': 55, 'Q': 55, 'D': 55, 'E': 55, 'K': 55, 'R': 55, 'H': 55}
AMINO_ACID_CHARGE = {'D': 55, 'E': 55, 'A': 155, 'V': 155, 'P': 155, 'F': 155, 'W': 155, 'I': 155, 'L': 155, 'G': 155,
                     'M': 155, 'Y': 155, 'S': 155, 'T': 155, 'C': 155, 'N': 155, 'Q': 155, 'K': 255, 'R': 255, 'H': 255}
# AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
#         'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19}
AMAs = {'G': 20, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'P': 5, 'F': 6, 'Y': 7, 'W': 8, 'S': 9, 'T': 10, 'C': 11,
        'M': 12, 'N': 13, 'Q': 14, 'D': 15, 'E': 16, 'K': 17, 'R': 18, 'H': 19, 'X': 21}


def train_valid_split(data_set, fold, mode):
    train_set, valid_set = {}, {}
    i = 0
    for key in data_set.keys():
        if (i + fold) % 5 != 0:
            train_set[key] = data_set[key]
        else:
            valid_set[key] = data_set[key]
        i += 1
    if mode == "train":
        return train_set
    elif mode == "valid":
        return valid_set


def parse_ll37_testset(path='metadata/LL37_v0.csv'):
    files = open(path, 'r').readlines()
    seqs = []
    for file in files:
        seqs.append(str(file.strip().split(',')[-1].strip().upper()))
    return seqs


def train_valid_split_random(data_set, fold, mode):
    train_set, valid_set = [], []
    for i in range(len(data_set)):
        if (i + fold) % 10 != 0:
            train_set.append(data_set[i])
        else:
            valid_set.append(data_set[i])
    if mode == "train":
        return train_set
    elif mode == "valid":
        return valid_set


def collate_fn(batch):
    vox = torch.stack([(v[0]) for v in batch])
    xs = [(v[1]) for v in batch]
    # print(len(xs))
    # print(xs[0].shape)
    ys = torch.stack([v[2] for v in batch])
    # 获得每个样本的序列长度
    seq_lengths = torch.LongTensor([v for v in map(len, xs)])
    # 每个样本都padding到当前batch的最大长度
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    # 把xs和ys按照序列长度从大到小排序
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    # print(seq_lengths)
    xs = xs[perm_idx]
    ys = ys[perm_idx]
    vox = vox[perm_idx]
    return vox, xs, ys, seq_lengths

    # data_tuple.sort(key=lambda x: len(x[1]), reverse=True)
    # seq = [sq[1] for sq in data_tuple]
    # vox = [sq[0] for sq in data_tuple]
    # label = [sq[2] for sq in data_tuple]
    # data_length = [len(sq) for sq in seq]
    # seq = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0.0)
    # return torch.tensor(np.asarray(vox)), torch.tensor(np.asarray(seq)), torch.tensor(np.asarray(label)), data_length


import torch
from Bio.PDB import PDBParser, DSSP
import mdtraj as md


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
        if i >= max_length:
            break
        fixed_tensor[i] = sse_to_index[sse]
    # fixed_tensor = torch.zeros(max_length, num_sse)
    # for i, sse in enumerate(secondary_structure):
    #     if i >= max_length:
    #         break
    #     index = sse_to_index[sse]
    #     fixed_tensor[i, index] = 1

    return fixed_tensor


def calculate_similarity(seq_gen, seq_tem):
    """
    seq1: generate
    seq2: template
    """
    from Bio import pairwise2
    from Bio.pairwise2 import format_alignment
    from Bio.Seq import Seq

    # Define the sequences to be compared
    # seq1 = Seq(seq_gen)
    # seq2 = Seq(seq_tem)
    seq1 = seq_gen
    seq2 = seq_tem

    # Define the scoring system
    match = 1
    mismatch = -1
    gap_opening = -0.5
    gap_extension = -0.1

    # Perform the alignment using the Needleman-Wunsch algorithm
    alignments = pairwise2.align.globalms(seq1, seq2, match, mismatch, gap_opening, gap_extension)
    return alignments[0].score / len(seq1)

from Bio.Align import PairwiseAligner
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

    # alignments = pairwise2.align.localxx(seq_gen, seq_tem)
    # print("Best local alignment(s):")
    # for alignment in alignments:
    #     print(format_alignment(*alignment))
    # # Normalize the score by the length of the generated sequence
aligner = PairwiseAligner()

def calculate_similarity(seq_gen, seq_tem):
    # Define the scoring system
    aligner.mode = 'global'  # Global alignment
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    # Perform the alignment
    score = aligner.score(seq_gen, seq_tem)
    return score / len(seq_gen)

print(calculate_similarity('IGKKFKRIVQRIKKFLRNL', 'RLGNFFRKVKEKIGGGLKKVGQKIKDFLGNLVPRTAS'))

class ADataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, mode='train', fold=0, task='all', max_length=30, min_length=10):
        test_sequences_big = parse_ll37_testset(path='metadata/LL37_v0.csv')
        test_sequences = parse_ll37_testset(path='metadata/LL37.csv')
        p = PDBParser(QUIET=True)

        all_data = pd.read_csv('metadata/data_0119.csv', encoding="unicode_escape").values[:-35]  # -35 for ood
        idx_list, seq_list, labels = all_data[:, 0], all_data[:, 1], np.concatenate(
            (all_data[:, 4:9], all_data[:, 10:11]),
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
        bin_count_diff = 200

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
        for key in tqdm(paired_dicts.keys()):
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
                    voxel1 = pdb_parser(structure)
                    seq_emb1 = [AMAs[char] for char in seq] + [0] * (30 - len(seq))
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
            indices_per_diff = collections.defaultdict(list)
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
        self.data_list = []
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
            self.data_list.append([(voxel1, seq_emb1), (voxel2, seq_emb2), new_gt])
            # data_list.append([(voxel2, seq_emb2), (voxel1, seq_emb1), -new_gt])  # Note the negation for reverse order

            gt_list.append(new_gt)

        # Now `data_list` has been populated with balanced data according to the absolute differences of gt1 and gt2
        print('filter_out count:', len(paired_list) - len(gt_list))
        plot_number_frequency(gt_list, save_name='updown.jpg')
        gt_list = np.asarray(gt_list)
        gt_class_wise_count = np.sum(gt_list, 0)
        print('class wise counts', gt_class_wise_count)

    def __getitem__(self, idx):
        sample1, sample2, gt = self.data_list[idx]
        voxel1, seq_emb1 = sample1
        voxel2, seq_emb2 = sample2
        # print(sample1)
        # voxel = self.im_aug(torch.Tensor(voxel).float())
        return (torch.Tensor(voxel1).float(), torch.Tensor(seq_emb1)), (
        torch.Tensor(voxel2).float(), torch.Tensor(seq_emb2)), torch.Tensor([gt])

    def __len__(self):
        return len(self.data_list)

    def encode_features(self, p, seq, count=0, max_length=40):
        if os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen"):
            voxel, seq_pdb = pdb_parser(p, count, "./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
            secondary_structure = read_secondary_structure("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
            secondary_structure_map = secondary_structure_to_fixed_tensor(secondary_structure)
        elif os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen"):
            voxel, seq_pdb = pdb_parser(p, count, "./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen")
            secondary_structure = read_secondary_structure("./pdb_gen/pdb_dbassp/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
            secondary_structure_map = secondary_structure_to_fixed_tensor(secondary_structure)
        else:
            return 100
            assert os.path.exists("./pdb_gen/pdb_dbassp/" + seq + "_real.pdb_gen")
        seq_code = [AMAs[char] for char in seq]
        # seq_emb = F.one_hot(torch.tensor(seq_code).to(torch.int64), num_classes=21).to(torch.float16)
        # print(seq)
        # print(len(seq_code))
        seq_emb = [seq_code + [0] * (max_length - len(seq_code))]
        return voxel, seq_emb, secondary_structure


class GDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, temp_seq, path='gendata/m1_37.csv', mode='dl', ):
        self.data_list = []
        self.mode = mode
        f = open(path, 'r')
        lines = f.readlines()[1:]
        all_data = pd.read_csv(path, encoding="unicode_escape").values
        idx_list, seq_list = all_data[:, 0], all_data[:, 1]
        filter_data = []
        for idx in range(1, len(seq_list) + 1):
            filter_data.append((idx, seq_list[idx - 1]))
        p = PDBParser()

        temp_seq_code = [AMAs[char] for char in temp_seq]
        temp_seq_emb = [temp_seq_code + [0] * (30 - len(temp_seq_code))]

        temp_voxel, seq_pdb = pdb_parser(p, idx, "./gendata/gen_v1_v3/" + temp_seq + "_relaxed_rank_1_model_3.pdb_gen")

        count = 0
        for item in tqdm(filter_data):
            index, seq = item
            idx = str(index).zfill(3)
            assert os.path.exists("./gendata/gen_v1_v3/" + seq + "_relaxed_rank_1_model_3.pdb_gen")

            seq_code = [AMAs[char] for char in seq]
            seq_emb = [seq_code + [0] * (30 - len(seq_code))]

            voxel, seq_pdb = pdb_parser(p, idx, "./gendata/gen_v1_v3/" + seq + "_relaxed_rank_1_model_3.pdb_gen")
            # print(lines[count])
            self.data_list.append([(temp_voxel, temp_seq_emb), (voxel, seq_emb), seq])

            count += 1

        print('used counts:', len(self.data_list))

    def __getitem__(self, idx):
        sample1, sample2, gt = self.data_list[idx]
        voxel1, seq_emb1 = sample1
        voxel2, seq_emb2 = sample2
        # print(sample1)
        # voxel = self.im_aug(torch.Tensor(voxel).float())
        return (torch.Tensor(voxel1).float(), torch.Tensor(seq_emb1)), (
        torch.Tensor(voxel2).float(), torch.Tensor(seq_emb2)), gt

    def __len__(self):
        return len(self.data_list)

# a = ADataset()
