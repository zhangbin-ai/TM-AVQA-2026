import os
import sys
sys.stdout.flush()
import sys
sys.path.append("your path/work_main/TM_AVQA")

import time
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader_reco import AVQA_dataset
from network/network_reco import AVQA_Reconstruct

import ast
import inspect
import json
import os
from pathlib import Path
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def todevice(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    else:
        return tensor


def train(args, model, train_loader, optimizer, epoch, reserve_modality):
    model.train()

    running_loss = 0.0
    num_batch = 0

    for batch_idx, sample in enumerate(train_loader):
        
        video_id, question_id, flag, ques_len, visual, audio, text = [todevice(x, device) for x in sample.values()]

        optimizer.zero_grad()
        loss_mse_batch, _, reco_contra_item = model(reserve_modality=reserve_modality, topk=args.topk, flag=flag, ques_len=ques_len, visual=visual, audio=audio, text=text)

        loss_mse = loss_mse_batch.mean()
        loss_total = loss_mse + reco_contra_item*0.1
        loss_total.backward()
        optimizer.step()

        running_loss += loss_mse.item()
        num_batch += 1
        if batch_idx % args.log_interval == 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tContrastive_Loss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_mse.mean().item(), reco_contra_item.item()*0.1), flush=True)
    avg_loss = running_loss / num_batch

    return avg_loss

    
def main():
    parser = argparse.ArgumentParser(description='MUSIC-AVQA')

    parser.add_argument("--audio_dir", type=str, default='your path/data/vggish', help="audio dir")
    parser.add_argument("--video_dir", type=str, default='your path/data/res18_14_14', help="video clip feat dir")
    
    ############################################################################ dataset
    parser.add_argument("--reserve_modality_01", type=str, default='audio', help="reserve whi")
    parser.add_argument("--reserve_modality_02", type=str, default='visual', help="reserve whi")
    parser.add_argument("--mode", type=str, default='train', help="with mode to use")

    # * train setting
    parser.add_argument("--label_train", type=str, default="your path/work_main/PMMR/data/avqa-train.json", help="train csv file")
    parser.add_argument("--label_val", type=str, default="your path/work_main/PMMR/data/avqa-val.json", help="val csv file")
    parser.add_argument("--label_test", type=str, default="your path/work_main/PMMR/data/avqa-test.json", help="test csv file")
    parser.add_argument("--flag_file", type=str, default="your path/work_main/mywork-main/data/missing_flag/missing_0.3/train_0.3.json", help="is flag file")

    parser.add_argument('--scales', type=int, default=4, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--topk', type=int, default=8, metavar='N', help='Local pesudo lables are 4')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument('--epochs_02', type=int, default=10, metavar='N', help='number of epochs to train (default: 60)')
    parser.add_argument("--model", type=str, default='AVModel', help="with model to use")
    
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')
    parser.add_argument('--lr_02', type=float, default=1e-4, metavar='LR', help='learning rate (default: 3e-4)')

    parser.add_argument('--steplr_step', type=int, default=10, help='after x steps it goes down')
    parser.add_argument('--steplr_gamma', type=float, default=0.1, help='after x steps it goes down rate')
    parser.add_argument('--encoder_dim', type=int, default=512, help='encoder dimension')
    parser.add_argument('--encoder_depth', type=int, default=1, help='block layer number')
    parser.add_argument('--encoder_num_heads', type=int, default=4, help='head number')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='hidden dimension')

    # * save path
    parser.add_argument("--model_save_dir", type=str, default='your path/model_checkpoint__dir', help="")
    parser.add_argument("--checkpoint_file", type=str, default='new_reco_mem_0p1', help="model name")
    parser.add_argument("--save_model_flag", type=str, default='True', help="flag as save model")
    args = parser.parse_args()
    print(format("main.py path", '<25'), Path(__file__).resolve())
    print(format('network.py path', '<25'), Path(inspect.getabsfile(AVQA_Reconstruct)).resolve())
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

    if args.model == 'AVModel':
        print("loading AVModel Model ...")

        model = AVQA_Reconstruct(
            encoder_dim=args.encoder_dim,
            scale=args.scales,
            num_mem_token=10
        )
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')
    

    if args.mode == 'train':
        
        train_dataset = AVQA_dataset(
            label_file=args.label_train,
            flag_file=args.flag_file,
            audio_dir=args.audio_dir,
            video_dir=args.video_dir,
            mode=args.mode,
            reserve_modality=args.reserve_modality_01
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_step, gamma=args.steplr_gamma)

        # the first missing branch
        step_2_chechpoint_path = ""

        for epoch in range(1, args.epochs + 1):

            print(f"The first reserved modality is {args.reserve_modality_01}: the {epoch}-th learning rate is {optimizer.param_groups[0]['lr']}")

            start = time.time()

            loss_epoch = train(args, model, train_loader, optimizer, epoch=epoch, reserve_modality=args.reserve_modality_01)

            start02 = time.time()
            print(start02 - start, flush=True)
            print('Accuracy training avg loss is: %.3f' % (loss_epoch), flush=True)

            scheduler.step()

            if epoch > 0:
                save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
                save_model_path = Path(save_model_folder, f"model_reserve_01_{args.reserve_modality_01}_{epoch}.pt")

                if epoch == 10:
                    step_2_chechpoint_path = save_model_path
                if not save_model_folder.exists():
                    save_model_folder.mkdir()
                if args.save_model_flag == 'True':
                    torch.save(model.state_dict(), str(save_model_path))
                    print(">>>save model path:", save_model_path)
                else:
                    print(">>>not save model.")

        # the second missing branch
        model_dict = model.state_dict()
        pretrained_dict = torch.load(step_2_chechpoint_path)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        train_dataset_02 = AVQA_dataset(
            label_file=args.label_train,
            flag_file=args.flag_file,
            audio_dir=args.audio_dir,
            video_dir=args.video_dir,
            mode=args.mode,
            reserve_modality=args.reserve_modality_02
        )
        
        train_loader_02 = DataLoader(train_dataset_02, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        optimizer_02 = optim.Adam(model.parameters(), lr=args.lr_02)
        scheduler_02 = optim.lr_scheduler.StepLR(optimizer_02, step_size=args.steplr_step, gamma=args.steplr_gamma)


        for epoch in range(1, args.epochs_02 + 1):
            print(f"The second reserved modality is {args.reserve_modality_02}: the {epoch}-th learning rate is {optimizer_02.param_groups[0]['lr']}")
            loss_epoch = train(args, model, train_loader_02, optimizer_02, epoch=epoch, reserve_modality=args.reserve_modality_02)
            print('Accuracy training avg loss is: %.2f' % (loss_epoch), flush=True)

            scheduler_02.step(epoch)
            if epoch > 0:
                save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
                save_model_path = Path(save_model_folder, f"model_reserve_02_{args.reserve_modality_02}_{epoch}.pt")
                
                if args.save_model_flag == 'True':
                    torch.save(model.state_dict(), str(save_model_path))
                    print(">>>save model path:", save_model_path)
                else:
                    print(">>>not save model.")


if __name__ == '__main__':
    main()