import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GDataset
from network import MMPeptide, SMPeptide

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import mdtraj as md


def calculate_charge(sequence, pH=7.2):
    protein_analysis = ProteinAnalysis(sequence)
    return protein_analysis.charge_at_pH(pH)


def estimate_alpha_helix(pdb_file, sequence):
    traj = md.load(pdb_file)

    # Calculate secondary structure using DSSP
    secondary_structure = md.compute_dssp(traj)

    # Count the number of alpha-helices
    alpha_helices_count = (secondary_structure[0] == 'H').sum()

    # Print the result
    # print("Number of alpha-helices:", alpha_helices_count/len(sequence))
    return alpha_helices_count/len(sequence)


def load_saap148_emb(feature_fire='./148feature.txt'):
    f = open(feature_fire, 'r').readlines()
    f = f[0]
    feature = [float(i) for i in f.strip().split(',')]
    ft = torch.tensor(feature)
    return ft


# saap148_emb = load_saap148_emb()


def eval_charge_and_alpha(f_input='v2_filter_anti_mic.csv', f_output='v2_filter_anti_mic_toxin_prior.csv'):
    f = open(f_output, 'w')
    f.write('ID,SEQ,AMP,POSITIVE,NEGATIVE,AVG PN,DR,MDR,1,2,3,4,5,6,SIMI,MIC,TOXIN,Charge,AH \n')
    test_set = GDataset(path=f_input, mode="prior")
    for data in tqdm(test_set):
        pdb, seq, exist_info, index = data
        charge = calculate_charge(seq)
        alpha_helix = estimate_alpha_helix(pdb, seq)
        final_result = [str(charge), str(alpha_helix)]
        f.write(exist_info.replace('\n', '') + ',' + ','.join(final_result) + '\n')

def eval_toxin(f_input='v2_filter_anti_mic.csv', f_output='v2_filter_anti_mic_toxin.csv'):
    f = open(f_output, 'w')
    f.write('ID,SEQ,AMP,POSITIVE,NEGATIVE,AVG PN,DR,MDR,1,2,3,4,5,6,SIMI,MIC,TOXIN \n')
    model1 = MMPeptide(classes=1, attention='hamburger').cuda()
    model1.load_state_dict(torch.load('./run/toxin-mm-mlce1280.00249/model_1.pth'))
    model2 = MMPeptide(classes=1, attention='hamburger').cuda()
    model2.load_state_dict(torch.load('./run/toxin-mm-mlce1280.00249/model_2.pth'))
    model3 = MMPeptide(classes=1, attention='hamburger').cuda()
    model3.load_state_dict(torch.load('./run/toxin-mm-mlce1280.00249/model_3.pth'))
    model4 = MMPeptide(classes=1, attention='hamburger').cuda()
    model4.load_state_dict(torch.load('./run/toxin-mm-mlce1280.00249/model_4.pth'))
    model5 = MMPeptide(classes=1, attention='hamburger').cuda()
    model5.load_state_dict(torch.load('./run/toxin-mm-mlce1280.00249/model_5.pth'))

    test_set = GDataset(path=f_input)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for data in tqdm(valid_loader):
        voxel, seq, exist_info, index = data
        pred_result1 = model1((voxel.cuda(), seq.cuda()))[0]
        pred_result2 = model2((voxel.cuda(), seq.cuda()))[0]
        pred_result3 = model3((voxel.cuda(), seq.cuda()))[0]
        pred_result4 = model4((voxel.cuda(), seq.cuda()))[0]
        pred_result5 = model5((voxel.cuda(), seq.cuda()))[0]
        result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5) / 5
        final_result = [str(i.item()) for i in result[0]]
        f.write(exist_info[0].replace('\n', '') + ',' + ','.join(final_result) + '\n')


def eval_anti(f_input='gendata/v2_filter_r3.csv', f_output='v2_filter_anti.csv'):
    print('eval anti')
    f = open(f_output, 'w')
    f.write('ID,SEQ,AMP,POSITIVE,NEGATIVE,AVG PN,DR,MDR,1,2,3,4,5,6,SIMI,MIC,TOXIN \n')

    model1 = MMPeptide(classes=6).cuda()
    model1.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_1.pth'))
    model2 = MMPeptide(classes=6).cuda()
    model2.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_2.pth'))
    model3 = MMPeptide(classes=6).cuda()
    model3.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_3.pth'))
    model4 = MMPeptide(classes=6).cuda()
    model4.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_4.pth'))
    model5 = MMPeptide(classes=6).cuda()
    model5.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_5.pth'))

    test_set = GDataset(path=f_input)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('load anti dataset', len(valid_loader))
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for data in tqdm(valid_loader):
        voxel, seq, exist_info, index = data
        pred_result1, f1 = model1((voxel.cuda(), seq.cuda()))
        pred_result2, f2 = model2((voxel.cuda(), seq.cuda()))
        pred_result3, f3 = model3((voxel.cuda(), seq.cuda()))
        pred_result4, f4 = model4((voxel.cuda(), seq.cuda()))
        pred_result5, f5 = model5((voxel.cuda(), seq.cuda()))
        result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5) / 5
        average_feature = (f1 + f2 + f3 + f4 + f5)/5
        similarity = torch.nn.functional.cosine_similarity(average_feature, saap148_emb.cuda())
        final_result = [str(i.item()) for i in result[0]]
        f.write(exist_info[0].replace('\n', '') + ',' + ','.join(final_result) + ',' + str(similarity.item()) + '\n')


def eval_anti_mic(f_input='gendata/v2_filter_r3.csv', f_output='v2_filter_anti.csv'):
    print('eval anti mic')
    f = open(f_output, 'w')
    f.write('ID,SEQ,AMP,POSITIVE,NEGATIVE,AVG PN,DR,MDR,1,2,3,4,5,6,SIMI,MIC,TOXIN \n')
    model1 = MMPeptide(classes=1).cuda()
    model1.load_state_dict(torch.load('./run/mic-mm-mse1280.00249/model_1.pth'))
    model2 = MMPeptide(classes=1).cuda()
    model2.load_state_dict(torch.load('./run/mic-mm-mse1280.00249/model_2.pth'))
    model3 = MMPeptide(classes=1).cuda()
    model3.load_state_dict(torch.load('./run/mic-mm-mse1280.00249/model_3.pth'))
    model4 = MMPeptide(classes=1).cuda()
    model4.load_state_dict(torch.load('./run/mic-mm-mse1280.00249/model_4.pth'))
    model5 = MMPeptide(classes=1).cuda()
    model5.load_state_dict(torch.load('./run/mic-mm-mse1280.00249/model_5.pth'))

    test_set = GDataset(path=f_input)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for data in tqdm(valid_loader):
        voxel, seq, exist_info, index = data
        pred_result1, f1 = model1((voxel.cuda(), seq.cuda()))
        pred_result2, f2 = model2((voxel.cuda(), seq.cuda()))
        pred_result3, f3 = model3((voxel.cuda(), seq.cuda()))
        pred_result4, f4 = model4((voxel.cuda(), seq.cuda()))
        pred_result5, f5 = model5((voxel.cuda(), seq.cuda()))
        result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5) / 5
        # average_feature = (f1 + f2 + f3 + f4 + f5)/5
        # similarity = torch.nn.functional.cosine_similarity(average_feature, saap148_emb.cuda())
        final_result = [str(i.item()) for i in result[0]]
        f.write(exist_info[0].replace('\n', '') + ',' + ','.join(final_result) + '\n')


def eval_mechanism():
    # model = resnet26(1).cuda()
    f = open('gen_mechanism3.csv', 'w')
    model1 = ResNet3D_OLD(Bottleneck, [1, 2, 4, 1], num_classes=4).cuda()
    print(model1)
    #
    model1.load_state_dict(torch.load('./run/mlce-vq1-mech/model_1.pth'))
    model2 = resnet26(num_classes=4).cuda()
    model2.load_state_dict(torch.load('./run/mlce-vq1-mech/model_2.pth'))
    model3 = resnet26(num_classes=4).cuda()
    model3.load_state_dict(torch.load('./run/mlce-vq1-mech/model_3.pth'))
    model4 = resnet26(num_classes=4).cuda()
    model4.load_state_dict(torch.load('./run/mlce-vq1-mech/model_4.pth'))
    model5 = resnet26(num_classes=4).cuda()
    model5.load_state_dict(torch.load('./run/mlce-vq1-mech/model_5.pth'))

    test_set = GDataset(mode='all', fold=0)
    valid_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    count = 0
    for data in tqdm(valid_loader):
        count += 1
        voxel, seq, gt, index = data
        pred_result1 = model1((voxel.cuda(), seq.cuda()))
        pred_result2 = model2((voxel.cuda(), seq.cuda()))
        pred_result3 = model3((voxel.cuda(), seq.cuda()))
        pred_result4 = model4((voxel.cuda(), seq.cuda()))
        pred_result5 = model5((voxel.cuda(), seq.cuda()))
        result = (pred_result1 + pred_result2 + pred_result3 + pred_result4 + pred_result5) / 5
        final_result = [str(i.item()) for i in result[0]]
        f.write(str(index.item()) + ',' + ','.join(final_result) + '\n')


def eval_siamese_mutate_on_private():
    """
    KWMIKWFYDWFTML
    KWMIKWPSNWFTML
    """
    model1 = SMPeptide(classes=1, q_encoder='tf', ).cuda()  # attention='hamburger'
    # model1.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-mm-mse1280.02140 ood/model_1.pth'))
    model1.load_state_dict(torch.load('/home/duadua/Desktop/AMP/mutation/run/anti-mm-mse1280.02160/model_1.pth'))
    test_set = GDataset(path='./gendata/m1_40.csv', temp_seq="KWMIKWPSNWFTML")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model1.eval()
    f_out = open('prediction_private.csv', 'w')
    for data in tqdm(test_loader):
        voxel, seq = data[0]
        voxel2, seq2 = data[1]
        out = model1(((voxel.cuda(), seq.cuda()), (voxel2.cuda(), seq2.cuda())))
        # print(seq_lengths)
        f_out.write(data[2][0]+','+str(out.item())+'\n')


eval_siamese_mutate_on_private()


def eval_mutate():
    # f = open('mutate.csv', 'w')

    model1 = MMPeptide(classes=6).cuda()
    model1.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_1.pth'))
    model2 = MMPeptide(classes=6).cuda()
    model2.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_2.pth'))
    model3 = MMPeptide(classes=6).cuda()
    model3.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_3.pth'))
    model4 = MMPeptide(classes=6).cuda()
    model4.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_4.pth'))
    model5 = MMPeptide(classes=6).cuda()
    model5.load_state_dict(torch.load('./run/anti-mm-mlce1280.00250/model_5.pth'))

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
        result_w = (pred_w1 + pred_w2 + pred_w3 + pred_w4 + pred_w5) / 5
        result_m = (pred_m1 + pred_m2 + pred_m3 + pred_m4 + pred_m5) / 5
        print(result_w)
        print(result_m)
        # final_result = [str(i.item()) for i in result[0]]
        # # index += 1
        # # f.write(str(index)+','.join(final_result) + '\n')
        # f.write(str(index.item())+','+','.join(final_result) + '\n')

# eval_anti(f_input='gendata/v3_top20k.csv', f_output='v3_filter_anti.csv')
# eval_anti_mic(f_input='v3_filter_anti.csv', f_output='v3_filter_anti_mic.csv')
# eval_toxin(f_input='v3_filter_anti_mic.csv', f_output='v3_filter_anti_mic_toxin.csv')
# eval_charge_and_alpha(f_input='v3_filter_anti_mic.csv', f_output='v3_filter_anti_mic_toxin_prior.csv')
# eval_mechanism()
# eval_mutate()
