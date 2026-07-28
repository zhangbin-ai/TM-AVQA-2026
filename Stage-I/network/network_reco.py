import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("your path/network")

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, repeat, reduce
from han_layer import AVHanLayer, SelfAttention
from contrastive_enhance import contrastive_loss
from utli.transformer_block import Transformer
from utli.pos_embedding import AbsolutePositionalEncoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class QstEncoder(nn.Module):
    
    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)
        
    def forward(self, question, qsn_lengths):

        qst_vec = self.word2vec(question)
        qst_vec = qst_vec.to(torch.float32).to(device)
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)
        self.lstm.flatten_parameters()
        output, (hidden, cell) = self.lstm(qst_vec)
        qst_feature = torch.cat((hidden, cell), 2)
        qst_feature = qst_feature.transpose(0, 1)
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)
        
        return output, qst_feature


class EncoderCompress(nn.Module):
    
    def __init__(self, N_layer, channels=512):
        super(EncoderCompress, self).__init__()
        self.N_layer = N_layer
        self.channels = channels

        compression_layers = []
        for i in range(N_layer):
            compression_layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ))
            compression_layers.append(nn.ReLU(inplace=True))

        compression_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.compression = nn.Sequential(*compression_layers)
    
    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.compression(x)

        _, _, H_compressed, W_compressed = x.shape
        if H_compressed != 1 or W_compressed != 1:
            raise ValueError(f"Error in compression: {x.shape}")
        
        x = x.view(B, T, C)
        return x


class Temp_multiscale_clue(nn.Module):
    def __init__(self, scales, in_channels):

        super(Temp_multiscale_clue, self).__init__()
        configs = [
            {"kernel_size": 1, "stride": 1, "padding": 0},
            {"kernel_size": 3, "stride": 1, "padding": 0},
            {"kernel_size": 3, "stride": 2, "padding": 0},
            {"kernel_size": 3, "stride": 4, "padding": 0},
            {"kernel_size": 10, "stride": 10, "padding": 0},
        ]
        
        conv_configs = configs[: scales]

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=config["kernel_size"],
                    stride=config["stride"],
                    padding=config["padding"]
                ),
                nn.ReLU(inplace=True)
            )
            for config in conv_configs
        ])

    def forward(self, x):
        """
        前向传播
        :param x: 输入特征，形状为 [B, T, C]
        :return: 输出特征，形状为 [B, T_out, C]
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)

        features = []
        for conv in self.conv_layers:
            temp_x = conv(x)
            features.append(temp_x)

        return features



class Daul_Reco(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):

        super(Daul_Reco, self).__init__()
        
        self.han_layer01 = AVHanLayer(d_model, nhead, dim_feedforward, dropout)
        self.han_layer02 = AVHanLayer(d_model, nhead, dim_feedforward, dropout)

        self.gate_weights = nn.Parameter(torch.zeros(1))

    def forward(self, src_v, src_a, src_t, reserve_modality="audio"):

        if reserve_modality == "audio":
            mem_v_for_t = self.han_layer01(src_v, src_t)
            mem_v_for_a = self.han_layer02(src_v, src_a)
            return self.gate_weights * mem_v_for_t + (1 - self.gate_weights) * mem_v_for_a
        
        elif reserve_modality == "visual":
            mem_a_for_t = self.han_layer01(src_a, src_t)
            mem_a_for_v = self.han_layer02(src_a, src_v)
            return self.gate_weights * mem_a_for_t + (1 - self.gate_weights) * mem_a_for_v

        else:
            print("memory error!!!")



class ST_CrossScale_Memory(nn.Module):
    def __init__(self, N_layer_compress, scales, channel, d_model, nhead, dim_feedforward, dropout):

        super(ST_CrossScale_Memory, self).__init__()

        self.spatial_compress = EncoderCompress(N_layer=N_layer_compress)
        self.temporal_scale = Temp_multiscale_clue(scales, channel)
        self.dual_reco = Daul_Reco(d_model, nhead, dim_feedforward, dropout)
        self.block = SelfAttention(d_model, nhead, dim_feedforward, dropout)

        self.recall = AVHanLayer(d_model, nhead, dim_feedforward, dropout)

        self.gate_weights = nn.Parameter(torch.zeros(scales))
        self.gate_weights_audio = nn.Parameter(torch.zeros(scales))


    def forward(self, src_v, src_a, src_t, reserve_modality="audio"):

        if reserve_modality == "audio":
            
            B, T, H, W, C = src_v.shape
            mem_visual = src_v

            compress_visual = self.spatial_compress(src_v)
            visual_feats = self.temporal_scale(compress_visual)
            gate = F.softmax(self.gate_weights, dim=-1)
            visual_feats = [w * f for w,f in zip(gate, visual_feats)]
            concatenated = torch.cat(visual_feats, dim=2)
            scale_visual = concatenated.permute(0, 2, 1)

            scale_visual = self.dual_reco(scale_visual, src_a, src_t, reserve_modality="audio")
            scale_fusion = self.block(scale_visual, scale_visual)

            mem_visual = rearrange(mem_visual, 'b t h w c -> b (t h w) c')
            recall_visual = self.recall(mem_visual, scale_fusion)

            return recall_visual

        elif reserve_modality == "visual":
            
            B, T, H, W, C = src_v.shape
            mem_audio = src_a

            compress_visual = self.spatial_compress(src_v)
            audio_feats = self.temporal_scale(src_a)
            audio_feats = [w * f for w,f in zip(self.gate_weights_audio, audio_feats)]
            concatenated = torch.cat(audio_feats, dim=2)
            scale_audio = concatenated.permute(0, 2, 1)
            scale_audio = self.dual_reco(compress_visual, scale_audio, src_t, reserve_modality="visual")

            scale_fusion = self.block(scale_audio, scale_audio)
            recall_audio = self.recall(mem_audio, scale_fusion)
            return recall_audio

        else:
            print("memory error!!!")


class AVQA_Reconstruct(nn.Module):
    
    def __init__( self, encoder_dim=512, scales=4, num_mem_token=10, norm_layer=nn.LayerNorm, **kwargs ):
        
        super().__init__()

        self.modality = ["audio", "visual"]
        w, h = 14, 14
        self.modality_param = nn.ParameterDict({
            "audio": nn.Parameter(torch.zeros(1, num_mem_token, w // 2, h // 2, encoder_dim)),
            "visual": nn.Parameter(torch.zeros(1, num_mem_token, 512)),
        })
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)
        self.fc_audio = nn.Linear(128, 512)
        self.pool2d = nn.AdaptiveAvgPool2d((7, 7))

        self.memory_unity = nn.ModuleDict({
            "audio": ST_CrossScale_Memory(2, scales, 512, d_model=512, nhead=1, dim_feedforward=2048, dropout=0.),
            "visual": ST_CrossScale_Memory(2, scales, 512, d_model=512, nhead=1, dim_feedforward=2048, dropout=0.),
        })

        self.out_block = SelfAttention(512, 1, 2048, dropout=0.1)
        self.layer_norm = norm_layer(encoder_dim)
        
        self.rec_audio_attn = AVHanLayer(512, 1, 2048, 0.1)
        self.rec_audio = nn.Sequential(
            nn.Linear(512, 128),
        )
        self.multiplier = nn.Parameter(torch.tensor(3.0))

        self.mse_loss = nn.MSELoss()

    def forward(self, reserve_modality="audio", topk=1, flag=None, ques_len=None, **inputs):
        
        ###################################################################
        # Data preparation phase
        ###################################################################
        Batchsize, Tseg, _ = inputs["audio"].shape
        text = inputs["text"]
        word_feat, sentence_feat = self.question_encoder(text, ques_len)
        word_feat = rearrange(word_feat, "t b c -> b t c")
        
        data_list_inp = {}
        data_list_inp["text_inp"] = word_feat

        ###################################################################
        # Dense Temporal-scale Reconstruction && Multimodal Dependency Modeling
        ###################################################################
        if reserve_modality == "audio":
            audio = inputs[reserve_modality]
            audio = self.fc_audio(audio)

            mem_visual = self.modality_param[reserve_modality]
            mem_visual = repeat(mem_visual, 'one t h w c -> (b one) t h w c', b=Batchsize)
            mem_visual = self.memory_unity["audio"](mem_visual, audio, word_feat, reserve_modality="audio")
            
            data_list_inp["audio_inp"] = audio
            data_list_inp["visual_inp"] = mem_visual
        
        elif reserve_modality == "visual":
            mem_audio = self.modality_param[reserve_modality]
            mem_audio = repeat(mem_audio, 'one t c -> (b one) t c', b=Batchsize)

            visual = inputs[reserve_modality]
            visual = rearrange(visual, 'b t w h c -> (b t) w h c')
            visual = visual.permute(0, 3, 1, 2)
            visual = self.pool2d(visual)
            visual = rearrange(visual, '(b t) c h w -> b t c h w', b=Batchsize)  
            visual = visual.permute(0, 1, 3, 4, 2)

            mem_audio = self.memory_unity["visual"](visual, mem_audio, word_feat, reserve_modality="visual")
            data_list_inp["audio_inp"] = mem_audio
            visual = rearrange(visual, 'b t h w c -> b (t h w) c')
            data_list_inp["visual_inp"] = visual
        
        else:
            print("network err!!!") 
        
        gene_mask = self.make_mask(ques_len)
        x_output = self.encoder(data_list_inp, mask=gene_mask)
        
        ###################################################################
        # Cross-Sample Relation-based Pseudo-label Learning &&
        # Cross-Modal Relation-based Contrastive Learning
        ###################################################################

        flag = flag.unsqueeze(-1).to(torch.int).to(device)
        flag_matrix = torch.bitwise_or(flag, flag.T)
        diagonal_indices = torch.arange(flag_matrix.shape[0])
        flag_matrix[diagonal_indices, diagonal_indices] = 0

        if reserve_modality == "audio":
            audio_sim = data_list_inp["audio_inp"]
            audio_sim = audio_sim.mean(dim=1) 
            norm_audio = F.normalize(audio_sim, p=2, dim=1)
            cos_audio_inBatch = 1 + torch.matmul(norm_audio, norm_audio.T)
            
            ###################################################################
            # Pseudo Label Step
            ###################################################################
            cos_audio_mask = cos_audio_inBatch * flag_matrix

            _, cos_audio_indices = torch.topk(cos_audio_mask, k=topk, largest=True, dim=1)

            input_visual = inputs["visual"]
            input_visual = reduce(input_visual, 'b t (h1 h2) (w1 w2) c -> b t h1 w1 c', 'mean', h2=2, w2=2)
            input_visual = rearrange(input_visual, 'b t c w h -> b (t w h c)')

            selected_visual = input_visual[cos_audio_indices]
            aggregated_visual = selected_visual.mean(dim=1)
            target = input_visual
            bool_flag = ( flag == 0 )
            target[bool_flag.squeeze(1)] = aggregated_visual[bool_flag.squeeze(1)]
            x_out_visual = x_output[:, :Tseg*7*7, :]

            recall_visual = x_out_visual
            pool_recall = recall_visual.mean(dim=1)

            pool_audio = data_list_inp["audio_inp"].mean(dim=1)
            pool_text = data_list_inp["text_inp"].mean(dim=1)
            
            reco_loss_item = contrastive_loss(pool_recall, pool_audio) + contrastive_loss(pool_recall, pool_text)

            x_out_visual = rearrange(x_out_visual, 'b twh c -> b (twh c)')
            mse_item = self.mse_loss(x_out_visual, target)

            return mse_item, x_out_visual, reco_loss_item
            
            
        elif reserve_modality == "visual":

            visual_sim = data_list_inp["visual_inp"]
            visual_sim = visual_sim.mean(dim=(1)) 
            norm_visual = F.normalize(visual_sim, p=2, dim=1)
            cos_visual_inBatch = 1+ torch.matmul(norm_visual, norm_visual.T)

            ###################################################################
            # Pseudo Label Step
            ###################################################################
            
            cos_visual_mask = cos_visual_inBatch * flag_matrix
            _, cos_visual_indices = torch.topk(cos_visual_mask, k=topk, largest=True, dim=1)

            input_audio = inputs["audio"]
            input_audio = input_audio.reshape(Batchsize, -1)
            selected_audio = input_audio[cos_visual_indices]
            aggregated_audio = selected_audio.mean(dim=1)
            target = input_audio
            bool_flag = ( flag == 0 )
            target[bool_flag.squeeze(1)] = aggregated_audio[bool_flag.squeeze(1)]

            x_out_audio = x_output[:, :Tseg*7*7+10, :]
            recall_audio = x_out_audio
            pool_recall = recall_audio.mean(dim=1)
            pool_visual = data_list_inp["visual_inp"].mean(dim=1)
            pool_text = data_list_inp["text_inp"].mean(dim=1)
            reco_loss_item = contrastive_loss(pool_recall, pool_visual) + contrastive_loss(pool_recall, pool_text)

            x_out_audio = x_out_audio.reshape(Batchsize, 10, -1, 512).sum(dim=2)
            x_out_audio = self.rec_audio(x_out_audio)
            x_out_audio = x_out_audio.reshape(Batchsize, -1)
            mse_item = self.mse_loss(x_out_audio, target)

            return mse_item, x_out_audio, reco_loss_item
        else:
            print("network: reserve_modality!!!")

    def encoder(self, data_list_inp, mask):
        
        Batch, Tseg, _ = data_list_inp["audio_inp"].shape
        text_data = data_list_inp["text_inp"]
        audio_data = data_list_inp["audio_inp"]
        visual_data = data_list_inp["visual_inp"]
        x_data = torch.cat((visual_data, audio_data, text_data), dim=1)
        x_data = self.out_block(x_data, x_data)

        return x_data


    def make_mask(self, seq_length):

        mask = torch.ones(len(seq_length), max(seq_length) + 10 + 490).to(device)
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        
        mask = Variable(mask)
        mask = mask.to(torch.float)
        return mask