import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
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
    
    def __init__(self, label_file, flag_file, audio_dir, video_dir, mode='train', reserve_modality='audio'):
        
        train_samples = json.load(open('your path/data/avqa-train.json', 'r'))
        
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in train_samples:
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
        self.ques_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}
        self.ans_vocab = ans_vocab

        self.max_len = 14
        self.reserve_modality = reserve_modality


        full_samples = json.load(open(label_file, 'r'))

        flags_json = json.load(open(flag_file, 'r'))
        flags_dict = {}
        question_id_delete_list = []
        if self.reserve_modality == "audio":
            for node in flags_json:

                flags_dict[node["question_id"]] = 0 if node["is_visual_missing"] is True else 1
                if node["is_audio_missing"] is True:
                    question_id_delete_list.append(node["question_id"])
        
        elif self.reserve_modality == "visual":
            for node in flags_json:
                flags_dict[node["question_id"]] = 0 if node["is_audio_missing"] is True else 1
                if node["is_visual_missing"] is True:
                    question_id_delete_list.append(node["question_id"])
        else: 
            print("dataloader error!!! question_id")

        self.flags = flags_dict


        self.samples = [node for node in full_samples if node["question_id"] not in question_id_delete_list]
        
        self.audio_dir = audio_dir
        self.video_res14x14_dir = video_dir

    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']

        audio_path = os.path.join(self.audio_dir, name + '.npy')
        visual_path = os.path.join(self.video_res14x14_dir, name + '.npy')

        if self.reserve_modality == 'audio':
            audio = np.load(audio_path)[::6, :]
            visual = np.zeros((10, 512, 14, 14)).astype(np.float32) if self.flags[question_id] == 1 else np.load(visual_path)

        elif self.reserve_modality == 'visual':
            visual = np.load(visual_path)
            audio = np.zeros((10, 128)).astype(np.float32) if self.flags[question_id] == 1 else np.load(audio_path)[::6, :]
        else:
            print("dataloader error!!!")

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
        idxs = [self.ques_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)
        flag = torch.from_numpy(np.array(self.flags[question_id]))

        answer = sample['anser']
        label = ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()

        sample = {
            'video_id': name,
            'question_id': question_id,
            'flag': flag,
            'ques_len': ques_len,
            'visual': torch.from_numpy(visual).permute(0, 2, 3, 1),
            'audio': torch.from_numpy(audio),
            'text': ques
        }
        return sample