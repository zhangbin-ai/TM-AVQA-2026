import torch
import torch.nn as nn
import torch.nn.functional as F
from han_layer import AVHanLayer


class MemoryUnity(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(MemoryUnity, self).__init__()

        self.han_layer01 = AVHanLayer(d_model, nhead, dim_feedforward, dropout)
        self.han_layer02 = AVHanLayer(d_model, nhead, dim_feedforward, dropout)
        self.gate_weights = nn.Parameter(torch.zeros(2))


    def forward(self, src_v, src_a, src_t, reserve_modality="audio"):
        
        if reserve_modality == "audio":
            mem_v_for_t = self.han_layer01(src_v, src_t)
            mem_v_for_a = self.han_layer02(src_v, src_a)

            return self.gate_weights[0] * mem_v_for_t + self.gate_weights[1] * mem_v_for_a
        
        elif reserve_modality == "visual":
            mem_a_for_t = self.han_layer01(src_a, src_t)
            mem_a_for_v = self.han_layer02(src_a, src_v)

            return self.gate_weights[0] * mem_a_for_t + self.gate_weights[1] * mem_a_for_v

        else:
            print("memory error!!!")