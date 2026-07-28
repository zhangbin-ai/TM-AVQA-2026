import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random


def ids_to_multinomial(id, categories):
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]
    
class AVQA_dataset(Dataset):
    def __init__(self, label, flag_file, audio_dir, video_dir):

        samples = json.load(open('your path/data/avqa-train.json', 'r'))

        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]
            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])
        

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14
        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_dir

        flags_a_v = json.load(open(flag_file, 'r'))
        video_flags_dict = {}
        quesId_flags_dict = {}
        for node in flags_a_v:
            video_flags_dict[node["video_id"]] = self.get_flag(node["is_visual_missing"], node["is_audio_missing"])
        for node in flags_a_v:
            quesId_flags_dict[node["question_id"]] = self.get_flag(node["is_visual_missing"], node["is_audio_missing"])
        
        self.video_flags_dict = video_flags_dict
        self.quesId_flags_dict = quesId_flags_dict

        video_list = []
        for sample in self.samples:
            video_name = sample['video_id']
            if video_name not in video_list:
                video_list.append(video_name)

        self.video_list = video_list
        self.video_len = 10 * len(video_list)


    def __len__(self):
        return len(self.samples)


    def get_flag(self, is_visual_missing, is_audio_missing):
        if is_audio_missing is True and is_visual_missing is False:
            return 10
        elif is_audio_missing is False and is_visual_missing is True:
            return 20
        elif is_audio_missing is False and is_visual_missing is False:
            return 30
        else:
            print('dataloader flags error!!!')
    
    
    def __getitem__(self, idx):

        sample = self.samples[idx]

        name = sample['video_id']
        question_id = sample['question_id']

        audio_path = os.path.join(self.audio_dir, name + '.npy')
        visual_path = os.path.join(self.video_res14x14_dir, name + '.npy')

        falg_item = self.quesId_flags_dict[question_id]

        audio = np.zeros((10, 128)).astype(np.float32) if falg_item == 10 else np.load(audio_path)[::6, :]
        visual_posi = np.zeros((10, 512, 14, 14)).astype(np.float32) if falg_item == 20 else np.load(visual_path)

        video_idx=self.video_list.index(name)
        
        T,C,H,W = torch.from_numpy(visual_posi).size()
        visual_nega=torch.zeros(T,C,H,W)

        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        ques_len = len(question)
        ques_len = torch.from_numpy(np.array(ques_len))

        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        flag = torch.from_numpy(np.array(self.quesId_flags_dict[question_id]))
        
        sample = {
            'video_id': name,
            'question_id': question_id,
            'flag': flag,
            'visual_posi': torch.from_numpy(visual_posi).permute(0, 2, 3, 1),
            'visual_nega': visual_nega.permute(0, 2, 3, 1),
            'audio': torch.from_numpy(audio),
            'ques_len': ques_len,
            'question': ques, 
            'label': label
        }
        return sample