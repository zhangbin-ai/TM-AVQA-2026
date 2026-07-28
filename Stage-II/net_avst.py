import torch
import sys

# import torchvision
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce

from network.reconstruct_net_topk_lateloss_STR_TCF_0910 import AVQA_Reconstruct






def batch_organize(out_match_posi, out_match_nega):

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels




class AVQA_Fusion_Net(nn.Module):

    def __init__(self, scales):
        super(AVQA_Fusion_Net, self).__init__()

        # for baselines, the backbone network in Stage-II not need to revised.
        self.reco_func = AVQA_Reconstruct(encoder_dim=512, scales=scales, num_mem_token=10)

        
    def forward(self, flag_inter, audio, visual_posi, visual_nega, question, ques_len):

        '''
            input question shape:    [B, T]
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, H, W, C]
            input visual_nega shape: [B, T, H, W, C]
        '''

        """
            Recostructed Audio and Viusal Feature
            # Add integration mode to provide reconstruction data for the AVQA model
            if inter_pattern is True:
            The flag [10, 20, 30] is different from the flag in the reconstruction mode
            10: Indicates audio missing
            20: indicates visual missing
            30: indicates that neither audio nor visual is lacking
        """
        ############################################################################
        # The core code 
        ############################################################################

        audio_x1 = audio[flag_inter != 10]
        visual_posi_x1 = visual_posi[flag_inter != 10]
        question_x1 = question[flag_inter != 10]
        ques_len_x1 = ques_len[flag_inter != 10]
        
        flags_modified = torch.where(flag_inter == 20, torch.tensor(0, device=flag_inter.device), flag_inter)
        flags_modified = torch.where(flags_modified == 30, torch.tensor(1, device=flag_inter.device), flags_modified)
        flag_reco_01 = flags_modified[flags_modified != 10]
        
        if audio_x1.shape[0] != 0:
            mse_loss_audio, reco_visual, item_x = self.reco_func(reserve_modality="audio", flag=flag_reco_01, ques_len=ques_len_x1, visual=visual_posi_x1, audio=audio_x1, text=question_x1)
            reco_visual = reco_visual.view(-1, 10, 7, 7, 512)

            visual_posi_clone = visual_posi.clone()
            visual_posi_clone = reduce(
                visual_posi_clone, "b t (h h2) (w w2) d -> b t h w d", "mean", h2=2, w2=2 
            )
            visual_posi = reduce(
                visual_posi, "b t (h h2) (w w2) d -> b t h w d", "mean", h2=2, w2=2
            )
            visual_posi_clone[flag_inter != 10] = reco_visual
            visual_posi_clone[flag_inter == 30] = visual_posi[flag_inter == 30]
            visual_posi_clone = visual_posi_clone.permute(0, 1, 4, 2, 3)

            pool = nn.AdaptiveAvgPool2d((1, 1))
            visual_posi_clone = pool(visual_posi_clone).repeat(1, 1, 1, 7, 7)

        else:
            mse_loss_audio = torch.zeros(1).to('cuda')
            visual_posi_clone = visual_posi
            visual_posi_clone = visual_posi_clone.permute(0, 1, 4, 2, 3)

        audio_x2 = audio[flag_inter != 20]
        visual_posi_x2 = visual_posi[flag_inter != 20]
        question_x2 = question[flag_inter != 20]
        ques_len_x2 = ques_len[flag_inter != 20]

        flags_modified = torch.where(flag_inter == 10, torch.tensor(0, device=flag_inter.device), flag_inter)
        flags_modified = torch.where(flags_modified == 30, torch.tensor(1, device=flag_inter.device), flags_modified)
        flag_reco_02 = flags_modified[flags_modified != 20]

        if audio_x2.shape[0] != 0:
            mse_loss_visual, reco_audio, item_x = self.reco_func(reserve_modality="visual", flag=flag_reco_02, ques_len=ques_len_x2, visual=visual_posi_x2, audio=audio_x2, text=question_x2)
            reco_audio = reco_audio.view(-1, 10, 128)

            audio_clone = audio.clone()
            audio_clone[flag_inter != 20] = reco_audio
            audio_clone[flag_inter == 30] = audio[flag_inter == 30]

        else:
            mse_loss_visual = torch.zeros(1).to("cuda")
            audio_clone = audio


        visual = visual_posi_clone
        audio = audio_clone

        ###################################################################
        # Please add AVQA backbone for question answering. 
        ###################################################################

        return out_qa, mse_loss_visual, mse_loss_audio
