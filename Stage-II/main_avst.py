import os
import sys
sys.stdout.flush()
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile

from torch.utils.data import DataLoader
import time
from net_grd_avst.net_avst_rec_topk_scales import AVQA_Fusion_Net
from net_grd_avst.dataloader_avst_reco import AVQA_dataset
import ast
import json
import numpy as np
import random
from pathlib import Path
import inspect

print("\n--------------- Audio-Visual Spatial-Temporal Model --------------- \n")

def batch_organize(out_match_posi,out_match_nega):

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):

        flag_inter, audio, visual_posi, visual_nega, question, ques_len, target = sample['flag'].to('cuda'), sample['audio'].to('cuda'), \
            sample['visual_posi'].to('cuda'), sample['visual_nega'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['label'].to('cuda')

        optimizer.zero_grad()

        out_qa, out_match_posi,out_match_nega, loss_rec_v, loss_rec_a = model(flag_inter, audio, visual_posi, visual_nega, question, ques_len)  
        out_match,match_label=batch_organize(out_match_posi,out_match_nega)  
        out_match,match_label = out_match.type(torch.FloatTensor).cuda(), match_label.type(torch.LongTensor).cuda()

        loss_match=criterion(out_match,match_label)
        loss_qa = criterion(out_qa, target)
        loss = loss_qa
        
        loss = loss + 0.1*(loss_rec_a + loss_rec_v)

        loss.backward()
        optimizer.step()

        pred_index, predicted = torch.max(out_qa, 1)
        correct_qa += (predicted == target).sum().item()
        total_qa += out_qa.size(0)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), flush=True)

    return correct_qa


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):

            flag_inter, audio, visual_posi, visual_nega, question, ques_len, target = sample['flag'].to('cuda'), sample['audio'].to('cuda'), \
            sample['visual_posi'].to('cuda'), sample['visual_nega'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['label'].to('cuda')

            preds_qa, out_match_posi,out_match_nega, loss_rec_v, loss_rec_a = model(flag_inter, audio, visual_posi, visual_nega, question, ques_len)


            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))

    return 100 * correct_qa / total_qa


def test(model, test_loader, test_json_file):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open(test_json_file, 'r'))

    # useing index of question
    questionid_to_samples = {}
    for sample in samples:
        ques_id = sample['question_id']
        if ques_id not in questionid_to_samples.keys():
            questionid_to_samples[ques_id] = sample
        else:
            print("question_id_duplicated:", ques_id)

    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            
            flag_inter, audio, visual_posi, visual_nega, question, ques_len, target, video_id, question_id = sample['flag'].to('cuda'), sample['audio'].to('cuda'), \
                sample['visual_posi'].to('cuda'), sample['visual_nega'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['label'].to('cuda'), \
                sample['video_id'], sample['question_id']

            preds_qa, out_match_posi,out_match_nega, loss_rec_v, loss_rec_a = model(flag_inter, audio, visual_posi, visual_nega, question, ques_len)

            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            question_id = question_id.numpy().tolist()

            for index, ques_id in enumerate(question_id):
                x = questionid_to_samples[ques_id]
                type =ast.literal_eval(x['type'])

                if type[0] == 'Audio':
                    if type[1] == 'Counting':
                        A_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        A_cmp.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Visual':
                    if type[1] == 'Counting':
                        V_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        V_loc.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Audio-Visual':
                    if type[1] == 'Existential':
                        AV_ext.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Counting':
                        AV_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        AV_loc.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        AV_cmp.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Temporal':
                        AV_temp.append((predicted[index] == target[index]).sum().item())
    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))
    
    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))
    
    return 100 * correct / total


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

    parser.add_argument("--audio_dir", type=str, default='/home/lizhangbin/data/vggish', help="audio dir")
    # parser.add_argument(
    #     "--video_dir", type=str, default='/home/guangyao_li/dataset/avqa/avqa-frames-1fps', help="video dir")
    parser.add_argument("--video_res14x14_dir", type=str, default='/home/lizhangbin/data/res18_14_14', help="res14x14 dir")
    
    parser.add_argument("--flag_file_train", type=str, default="/home/lizhangbin/work_main/PMMR/data/missing_flag_videoId/missing_0.3/train_0.3.json", help="is flag file")
    parser.add_argument("--flag_file_val", type=str, default="/home/lizhangbin/work_main/PMMR/data/missing_flag_videoId/missing_0.3/val_0.3.json", help="is flag file")
    parser.add_argument("--flag_file_test", type=str, default="/home/lizhangbin/work_main/PMMR/data/missing_flag_videoId/missing_0.3/text_0.3.json", help="is flag file")

    parser.add_argument("--flag_file_test_missing_audio", type=str, default="/home/lizhangbin/work_main/PMMR/data/missing_flag_videoId/missing_certain_audio.json", help="is flag file")  # 设置audio全部缺失
    parser.add_argument("--flag_file_test_missing_visual", type=str, default="/home/lizhangbin/work_main/PMMR/data/missing_flag_videoId/missing_certain_visual.json", help="is flag file")  # 设置visual全部缺失


    parser.add_argument(
        "--label_train", type=str, default="/home/lizhangbin/work_main/PMMR/data/avqa-train.json", help="train csv file")
    parser.add_argument(
        "--label_val", type=str, default="/home/lizhangbin/work_main/PMMR/data/avqa-val.json", help="val csv file")
    parser.add_argument(
        "--label_test", type=str, default="/home/lizhangbin/work_main/PMMR/data/avqa-test.json", help="test csv file")
    
    parser.add_argument(
        '--scales', type=int, default=4, metavar='N', help='input batch size for training (default: 16)') 
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)') 
    parser.add_argument(
        '--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    # * save path
    parser.add_argument("--path_phase01", type=str, default='/data/lizhangbin/model_checkpoint__dir/', help="")

    parser.add_argument("--model_save_dir", type=str, default='/data/lizhangbin/model_checkpoint__dir', help="")
    parser.add_argument("--checkpoint_file", type=str, default='phase02_new_models', help="model name")
    parser.add_argument("--save_model_flag", type=str, default='False', help="flag as save model")


    args = parser.parse_args()
    print(format("main.py path", '<25'), Path(__file__).resolve())  
    print(format('network.py path', '<25'), Path(inspect.getabsfile(AVQA_Fusion_Net)).resolve())
    print(format('dataloader.py', '<25'), Path(inspect.getabsfile(AVQA_dataset)).resolve())
    
    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net(
            scales=args.scales,
        )
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train, flag_file=args.flag_file_train, audio_dir=args.audio_dir, video_dir=args.video_res14x14_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # val_dataset = AVQA_dataset(label=args.label_val, flag_file=args.flag_file_val, audio_dir=args.audio_dir, video_dir=args.video_res14x14_dir)
        # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = AVQA_dataset(label=args.label_test, flag_file=args.flag_file_test, audio_dir=args.audio_dir, video_dir=args.video_res14x14_dir)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset_missing_audio = AVQA_dataset(label=args.label_test, flag_file=args.flag_file_test_missing_audio, audio_dir=args.audio_dir, video_dir=args.video_res14x14_dir)
        test_loader_missing_audio = DataLoader(test_dataset_missing_audio, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset_missing_visual = AVQA_dataset(label=args.label_test, flag_file=args.flag_file_test_missing_visual, audio_dir=args.audio_dir, video_dir=args.video_res14x14_dir)
        test_loader_missing_visual = DataLoader(test_dataset_missing_visual, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


        # ===================================== load pretrained model ===============================================
        
        AVRecoNet_dict = torch.load(args.path_phase01)

        model_dict = model.state_dict()

        pretrained_reco_dict = { str(k).split('.')[0] + '.reco_func.' + str(k).split('.', 1)[1]: v for k, v in AVRecoNet_dict.items()}
        model_dict.update(pretrained_reco_dict)

        avst_dict = torch.load(
            "/home/lizhangbin/data/models/avst_update.pt")

        tmp = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias','module.fc_gl.weight','module.fc_gl.bias','module.fc1.weight', 'module.fc1.bias','module.fc2.weight', 'module.fc2.bias','module.fc3.weight', 'module.fc3.bias','module.fc4.weight', 'module.fc4.bias']
        tmp2 = ['module.fc_a1.weight', 'module.fc_a1.bias','module.fc_a2.weight','module.fc_a2.bias']
        pretrained_dict1 = {k: v for k, v in avst_dict.items() if k in tmp}
        pretrained_dict2 = {str(k).split('.')[0]+'.'+str(k).split('.')[1]+'_pure.'+str(k).split('.')[-1]: v for k, v in avst_dict.items() if k in tmp2}
        
        model_dict.update(pretrained_dict1)
        model_dict.update(pretrained_dict2)

        model.load_state_dict(model_dict, strict=False)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):

            correct_qa = train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            
            scheduler.step()

            F = eval(model, val_loader, epoch)


            if F >= best_F:

                save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
                save_model_path = Path(save_model_folder, f"model_{epoch}.pt")

                if not save_model_folder.exists():
                    save_model_folder.mkdir()

                if args.save_model_flag == 'True':
                    torch.save(model.state_dict(), str(save_model_path))
                    print(">>>save model path:", save_model_path)
                else:
                    print(">>>not save model.")


    else:
            #########################################################################################################
            # test. 
            #########################################################################################################
            print("test: all data>>>")
            print("*" * 30)
            print("below is normal result >>> ")
            test_acc = test(model, test_loader, args.label_test)
            
            if epoch > 0:
                print()
                print("*" * 30)
                print("below is missing audio modality test result >>> ")
                test_acc_audio = test(model, test_loader_missing_audio, args.label_test)
                
                print()
                print("*" * 30)
                print("below is missing visual modality test result >>> ")
                test_acc_visual = test(model, test_loader_missing_visual, args.label_test)
                print()
                print()


if __name__ == '__main__':
    main()