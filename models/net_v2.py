''' Define the Transformer model '''
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers_v2 import DecoderLayer,  TBIFormerBlock
from models import GlobalInteractor_hivt
from models import LocalEncoder_hivt
from models import MLPDecoder_hivt
from utils import TemporalData

class Tem_ID_Encoder(nn.Module):
    def __init__(self, d_model, dropout=0.1,
                 max_t_len=200, max_a_len=20):
        super(Tem_ID_Encoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        ie = self.build_id_enc(max_a_len)
        self.register_buffer('ie', ie)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        return pe

    def build_id_enc(self, max_len):
        ie = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ie[:, 0::2] = torch.sin(position * div_term)
        ie[:, 1::2] = torch.cos(position * div_term)
        ie = ie.unsqueeze(0)
        return ie

    def get_pos_enc(self, num_bn, num_p, num_t, t_offset):
        pe = self.pe[:, t_offset: num_t + t_offset]
        pe = pe.repeat(1, num_bn*num_p, 1)   # TODO: 
        return pe

    def get_id_enc(self, num_p, num_t, i_offset, id_enc_shuffle):

        ie = self.ie[:, id_enc_shuffle]
        ie = ie.repeat_interleave(num_p*num_t, dim=1)
        return ie

    def forward(self, x, num_p, num_t, t_offset=0, i_offset=0):
        ''' 
            [num_p] number of body parts, 
            [num_t] length of time, 
        '''
        BN = x.shape[0]
        index = list(np.arange(0, num_p))
        id_enc_shuffle = random.choices(index, k=BN)
        pos_enc = self.get_pos_enc(num_p, num_t, t_offset)
        id_enc = self.get_id_enc(num_p, num_t, i_offset, id_enc_shuffle)
        x = x + pos_enc + id_enc     #  Temporal Encoding + Identity Encoding
        return self.dropout(x)


class TBIFormerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=1000, device='cuda', kernel_size=10):
        super().__init__()
        self.embeddings = Tem_ID_Encoder(d_model, dropout=dropout,
                                        max_t_len=n_position, max_a_len=10)   #  temporal encodings + identity encodings
        self.layer_stack = nn.ModuleList([
            TBIFormerBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.embeddings_table = nn.Embedding(10, d_k * n_head)



    def forward(self, src, return_attns=False):
        '''
            src: B,N,T,D
        '''


        enc_attn_list = []
        bn, p, t, d = src.size()
        src = src.reshape(bn, -1, d)

        enc_in = self.embeddings(src, p, t)  # temporal encodings + identity encodings

        enc_output = (enc_in)

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(
                enc_output, self.embeddings_table.weight)
            enc_attn_list += [enc_attn] if return_attns else []


        if return_attns:
            return enc_output, enc_attn_list

        return enc_output



class Decoder(nn.Module):

    def __init__(
            self,  n_layers, n_head, d_k, d_v,
            d_model, d_inner,  dropout=0.1, device='cuda'):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.device = device

    def forward(self, trg_seq, enc_output, return_attns=False):

        dec_enc_attn_list = []
        dec_output = (trg_seq)  # bs * person, 3 * person + input_frames, dim=128
        layer=0
        for dec_layer in self.layer_stack:
            layer+=1
            dec_output, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_enc_attn_list
        return dec_output


def body_partition(mydata, index):   # Body Partition
    bn, seq_len, _ = mydata.shape
    mydata = mydata.reshape(bn, seq_len, -1, 3)  # 96, 50, 15, 3
    out = torch.zeros(bn, seq_len, len(index), 3).to(mydata.device)  # x, 12, 3, 35
    for i in range(len(index)):
        temp1 = mydata[:, :, index[i], :].reshape(-1, len(index[i]), 3).transpose(1,2)
        # temp2 = torch.mean(temp1, dim=-1, keepdim=True)
        temp2 = F.avg_pool1d(temp1, kernel_size=5, padding=1)
        temp2 = temp2.transpose(1, 2).reshape(bn, seq_len, -1, 3)
        out[:, :, i, :] = temp2[:, :, 0, :]
    return out


class TBIFormer(nn.Module):

    def __init__(
            self, input_dim=128, d_model=512, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2,
            device='cuda', kernel_size=10, d_traj_query=64, opt=None):

        super().__init__()
        self.kernel_size = opt.kernel_size
        self.device = device
        self.d_model = d_model

        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=input_dim, kernel_size=(1, opt.kernel_size), stride=(1, 1), bias=False),
                                nn.ReLU(inplace=False))

        self.encoder = TBIFormerEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            dropout=dropout, device=self.device, kernel_size=kernel_size)

        self.decoder = Decoder(d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, device=self.device)

        
        kernel_size1 = int(kernel_size/2+1)
        if kernel_size%2==0:
            kernel_size2 =  int(kernel_size/2)
        else:
            kernel_size2 =  int(kernel_size/2+1)
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=45, out_channels=d_model, kernel_size=kernel_size1,
                                             bias=False),
                                   nn.ReLU(inplace=False),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size2,
                                             bias=False),
                                   nn.ReLU(inplace=False))        
        
        self.proj_inverse=nn.Linear(d_model, 42)
        self.l1=nn.Linear(d_model+d_traj_query, d_model*8)
        self.l2=nn.Linear(d_model*8, d_model*25)
        
        # HiVT components
        historical_steps = opt.input_time
        self.future_steps = opt.output_time
        node_dim, edge_dim = 3, 3 # number of dimensions
        num_heads, hivt_dropout, num_temporal_layers, local_radius, parallel, num_modes = 8, 0.1, 4, None, False, 6
        num_global_layers, rotate = 3, True
        embed_dim = opt.hivt_embed_dim
        
        self.local_encoder_traj = LocalEncoder_hivt(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=hivt_dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor_traj = GlobalInteractor_hivt(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=hivt_dropout,
                                                  rotate=rotate)
        self.decoder_traj = MLPDecoder_hivt(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=self.future_steps,
                                  num_modes=num_modes,
                                  uncertain=False)
        
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == input_dim, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'


    def forward(self, src, traj_query):
        '''
        src_seq:  B*N, T, J*3
        '''
        num_modes = traj_query.shape[0]
        bn = src.shape[0]
        # bs = int(bn / n_person)

        # ====== Temporal Body Partition Module =========
        index = [[8, 9, 10], [11, 12, 13], [1, 2, 3], [4, 5, 6], [0, 7, 14]]  # 5 body parts
        part_seq = body_partition(src, index).permute(0, 3, 2, 1)
        mpbp_seq = self.conv2d(part_seq).permute(0, 2, 3, 1).reshape(bn, 5, -1, 128)    #  multi-person body parts sequence
 
      
        # ======= TBIFormer Encoder ============
        enc_out = self.encoder(mpbp_seq)

        
        # ======= Transformer Decoder ============
        src_query = src.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # the last sub-sequence for query
        global_body_query = self.mlp(src_query).reshape(bs, n_person, -1)
        global_body_query = global_body_query.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        enc_out = enc_out.unsqueeze(0).repeat(num_modes,1,1,1)
        traj_query = traj_query.reshape(num_modes, bs, n_person, -1)
        new_query = torch.cat((global_body_query, traj_query), dim=-1)
        dec_output = self.decoder(new_query, enc_out, False)
        dec_output = dec_output.reshape(num_modes, bn, 1, -1)

        # =======  FC ============
        dec_output = self.l1(dec_output)
        dec_output = self.l2(dec_output)
        dec_output = dec_output.view(num_modes, bn, self.future_steps, -1)
        dec_out = self.proj_inverse(dec_output)
        return dec_out
    
    def forward_original(self, src, n_person, trj_dist):
        '''
        src_seq:  B*N, T, J*3
        '''

        bn = src.shape[0]
        bs = int(bn / n_person)

        # ====== Temporal Body Partition Module =========
        index = [[8, 9, 10], [11, 12, 13], [1, 2, 3], [4, 5, 6], [0, 7, 14]]  # 5 body parts
        part_seq = body_partition(src, index).permute(0, 3, 2, 1)
        mpbp_seq = self.conv2d(part_seq).permute(0, 2, 3, 1).reshape(bs, n_person, 5, -1, 128)    #  multi-person body parts sequence
 
      
        # ======= TBIFormer Encoder ============
        enc_out = self.encoder(mpbp_seq, trj_dist, n_person)

        
        # ======= Transformer Decoder ============
        src_query = src.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # the last sub-sequence for query
        global_body_query = self.mlp(src_query).reshape(bs, n_person, -1)
        dec_output = self.decoder(global_body_query, enc_out, False)
        dec_output = dec_output.reshape(bn, 1, -1)

        # =======  FC ============
        dec_output = self.l1(dec_output)
        dec_output = self.l2(dec_output)
        dec_output = dec_output.view(bn, 25, -1)
        dec_out = self.proj_inverse(dec_output)
        return dec_out


    def traj_forward(self, input_traj_temporalData):
        local_embed = self.local_encoder_traj(data=input_traj_temporalData)
        global_embed = self.global_interactor_traj(data=input_traj_temporalData, local_embed=local_embed)
        y_hat, pi, out_feature = self.decoder_traj(local_embed=local_embed, global_embed=global_embed)
        return y_hat, out_feature