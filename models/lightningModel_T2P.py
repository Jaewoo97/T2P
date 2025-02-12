''' Define the Transformer model '''
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers import DecoderLayer, TBIFormerBlock
from models import GlobalInteractor_hivt
from models import LocalEncoder_hivt
from models import MLPDecoder_hivt
from utils import TemporalData

from utils_.iRPE import piecewise_index
import itertools
import torch
import numpy as np
import torch_dct as dct  # https://github.com/zh217/torch-dct

def temporal_partition(src, opt):
    src = src[:, :, 1:]
    B, N, L, _ = src.size()
    stride = 1
    fn = int((L - opt.kernel_size) / stride + 1)
    idx = np.expand_dims(np.arange(opt.kernel_size), axis=0) + \
          np.expand_dims(np.arange(fn), axis=1) * stride
    return idx      
       

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

    def get_pos_enc(self, num_a, num_p, num_t, t_offset):
        pe = self.pe[:, t_offset: num_t + t_offset]
        pe = pe.repeat(1, num_a*num_p, 1)
        return pe

    def get_id_enc(self, num_p, num_t, i_offset, id_enc_shuffle):

        ie = self.ie[:, id_enc_shuffle]
        ie = ie.repeat_interleave(num_p*num_t, dim=1)
        return ie

    def forward(self, x, num_a, num_p, num_t, t_offset=0, i_offset=0):
        ''' 
            [num_a] number of person, 
            [num_p] number of body parts, 
            [num_t] length of time, 
        '''
        index = list(np.arange(0, num_p))
        id_enc_shuffle = random.choices(index, k=num_a)
        # id_enc_shuffle = random.sample(index, num_a)
        pos_enc = self.get_pos_enc(num_a, num_p, num_t, t_offset)
        id_enc = self.get_id_enc(num_p, num_t, i_offset, id_enc_shuffle)
        # import pdb;pdb.set_trace()
        x = x + pos_enc + id_enc     #  Temporal Encoding + Identity Encoding
        
        return self.dropout(x)


class TBIFormerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=1000, device='cuda', kernel_size=10):
        super().__init__()
        self.embeddings = Tem_ID_Encoder(d_model, dropout=dropout,
                                        max_t_len=n_position, max_a_len=20)   #  temporal encodings + identity encodings
        self.layer_stack = nn.ModuleList([
            TBIFormerBlock(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.embeddings_table = nn.Embedding(10, d_k * n_head)



    def forward(self, src, n_person, return_attns=False):
        '''
            src: B,N,T,D
        '''


        enc_attn_list = []
        sz_b, n, p, t, d = src.size()

        src = src.reshape(sz_b, -1, d)

        enc_in = self.embeddings(src, n, p, t)  # temporal encodings + identity encodings

        enc_output = (enc_in)
        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(
                enc_output, n_person, self.embeddings_table.weight)
            enc_attn_list += [enc_attn] if return_attns else []


        if return_attns:
            return enc_output, enc_attn_list

        return enc_output



class Decoder(nn.Module):

    def __init__(
            self,  n_layers, n_head, d_k, d_v,
            d_model, d_inner, d_traj_query=64, dropout=0.1, device='cuda'):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, d_traj_query=d_traj_query, dropout=dropout)
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


class T2P(nn.Module):

    def __init__(
            self, input_dim=128, d_model=512, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64, dropout=0.2,
            device='cuda', kernel_size=10, d_traj_query=64, opt=None):

        super().__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.kernel_size = opt.kernel_size
        self.device = device
        self.d_model = d_model
        self.output_time = opt.output_time
        self.input_time = opt.input_time
        self.num_joints = opt.num_joints
        self.dataset = opt.dataset
        self.sampling_method = opt.sampling_method
        
        self.conv2d = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=input_dim, kernel_size=(1, opt.kernel_size), stride=(1, 1), bias=False),
                                nn.ReLU(inplace=False))

        self.encoder = TBIFormerEncoder(n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner,
            dropout=dropout, device=self.device, kernel_size=kernel_size)

        self.decoder = Decoder(d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_traj_query=d_traj_query, dropout=dropout, device=self.device)

        
        kernel_size1 = int(kernel_size/2+1)
        if kernel_size%2==0:
            kernel_size2 =  int(kernel_size/2)
        else:
            kernel_size2 =  int(kernel_size/2+1)
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=self.num_joints*3, out_channels=d_model, kernel_size=kernel_size1,
                                             bias=False),
                                   nn.ReLU(inplace=False),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size2,
                                             bias=False),
                                   nn.ReLU(inplace=False))

        
        
        self.proj_inverse=nn.Linear(d_model, (self.num_joints-1)*3)
        self.l1=nn.Linear(d_model, (d_model//4)*self.output_time)              
        self.l2=nn.Linear(d_model//4, d_model)          
        self.query_linear=nn.Linear(d_model+d_traj_query, d_model)          
        # HiVT components
        historical_steps = opt.input_time
        self.future_steps = opt.output_time
        node_dim, edge_dim = 3, 3 # number of dimensions
        num_heads, hivt_dropout, num_temporal_layers, local_radius, parallel, num_modes = 8, 0.1, 4, 50, False, opt.num_modes
        num_global_layers, rotate = 3, True
        embed_dim = opt.hivt_embed_dim
        reshape_dim = (self.input_time-self.kernel_size)*5
        self.local_encoder_traj = LocalEncoder_hivt(reshape_dim=reshape_dim, 
                                                    historical_steps=historical_steps,
                                                    node_dim=node_dim,
                                                    edge_dim=edge_dim,
                                                    embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=hivt_dropout,
                                                    num_temporal_layers=num_temporal_layers,
                                                    local_radius=local_radius,
                                                    parallel=parallel,
                                                    enc_feat_dim=input_dim)
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


    def forward(self, batch_data, mode):
        input_seq, output_seq = batch_data.input_seq.clone(), batch_data.output_seq.clone()
        if len(input_seq.shape) == 3:
            bn, t, d = input_seq.shape
            input_seq = input_seq.reshape(bn//25, 25, t, d)
        if len(output_seq.shape)==3:
            bn, t, d = output_seq.shape
            output_seq = output_seq.reshape(bn//25, 25, t, d)
        B, N, _, D = input_seq.shape
        hip_joint_idx = 0       # Out of 15 joints, hip joint is idx 0
        
        input_ = input_seq.view(-1, self.input_time, input_seq.shape[-1])
        output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])
        
        input_ = input_.reshape(-1, self.opt.input_time, self.num_joints, 3)        #   Relative position to hip joint
        input_hip = input_[:,:,0,:].unsqueeze(-2).repeat(1,1,self.num_joints,1)
        offset = input_ - input_hip
        
        offset = offset.reshape(-1, self.opt.input_time, input_seq.shape[-1])    
        offset = offset[:, 1:self.opt.input_time, :] - offset[:, :self.opt.input_time-1, :] # as temporal displacement
        src = dct.dct(offset)
        enc_feat = self.forward_encode_body(src, N)
        #################### Trajectory prediction ####################
        pred_trajectory, pi, traj_feats = self.traj_forward(batch_data, enc_feat)
        num_modes = traj_feats.shape[0]
        inverse_rotMat = torch.linalg.inv(batch_data.rotate_mat)
        inverse_rotMat = inverse_rotMat.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        predicted_motion = torch.matmul(pred_trajectory, inverse_rotMat)
        predicted_motion = (predicted_motion + batch_data.positions[:,self.opt.input_time-1].unsqueeze(0).unsqueeze(-2)).unsqueeze(-2).repeat(1,1,1,self.num_joints,1)
        ###############################################################
        
        traj_feats = traj_feats.clone().detach()
        
        #################### Local pose prediction ####################
        rec_ = self.forward_local(src, N, traj_feats, enc_feat)
        rec = dct.idct(rec_)
        rec = rec.reshape(num_modes, N*B, self.opt.output_time, self.num_joints-1, 3)
        
        gt_trajectory = batch_data.y.unsqueeze(0)
        output_ = output_[:,:,:].reshape(-1, self.opt.output_time+1, self.num_joints, 3)        #   Relative position to hip joint
        output_hip = output_[:,:,0,:].unsqueeze(-2).repeat(1,1,self.num_joints,1)
        offset_output = output_ - output_hip
        
        if mode == 'eval':
            if self.sampling_method == 'ade':
                l2 = torch.norm(gt_trajectory - pred_trajectory, p=2, dim=-1)
                mask_ = ~batch_data.padding_mask[:, -self.output_time:]
                masked_l2 = l2 * mask_.unsqueeze(0)
                sum_masked_l2 = masked_l2.sum(dim=2)
                sum_mask = mask_.sum(dim=1)
                sum_mask[sum_mask==0] += 1
                average_masked_l2 = sum_masked_l2 / sum_mask.unsqueeze(0)
                made_idcs = torch.argmin(average_masked_l2, dim=0)
            elif self.sampling_method == 'fde':
                l2 = torch.norm(gt_trajectory - pred_trajectory, p=2, dim=-1)
                mask_ = ~batch_data.padding_mask[:, -self.output_time:]
                
                for agentIdx in range(mask_.shape[0]):
                    if sum(mask_[agentIdx]) == 0: continue
                    max_idx = torch.where(mask_[agentIdx]==True)[0].max()
                    mask_[agentIdx, :] = False
                    mask_[agentIdx, max_idx] = True
                
                masked_l2 = l2 * mask_.unsqueeze(0)
                sum_masked_l2 = masked_l2.sum(dim=2)
                sum_mask = mask_.sum(dim=1)
                sum_mask[sum_mask==0] += 1
                average_masked_l2 = sum_masked_l2 / sum_mask.unsqueeze(0)
                made_idcs = torch.argmin(average_masked_l2, dim=0)
                
            offset_output = offset_output.clone().detach()
            offset_output = offset_output.unsqueeze(0).repeat(num_modes, 1, 1, 1, 1)
            results = offset_output[:, :, :1, 1:]
            for i in range(1, self.opt.output_time+1):
                results = torch.cat([results, offset_output[:, :, :1, 1:] + torch.sum(rec[:, :, :i, :], dim=2, keepdim=True)],dim=2)

            predicted_motion[:,:,:,1:,:] = predicted_motion[:,:,:,1:,:] + results[:,:,1:]
            predicted_motion = predicted_motion.reshape(num_modes, B*N, self.opt.output_time, self.num_joints, 3)
            predicted_motion = predicted_motion[made_idcs, torch.arange(B*N)]
            gt = output_.view(B, N, -1, self.num_joints, 3)[:,:,1:,...]
        
            return predicted_motion, gt.reshape(B*N, self.opt.output_time, self.num_joints, 3)
        elif mode == 'train':
            return pred_trajectory, gt_trajectory, rec, offset_output
    
    def forward_encode_body(self, src, n_person):
        '''
        src_seq:  B*N, T, J*3
        '''
        bn = src.shape[0]
        bs = int(bn / n_person)

        # ====== Temporal Body Partition Module =========
        if self.dataset == 'cmu_umpm': index = [[9, 10, 11], [12, 13, 14], [1, 2, 3], [4, 5, 6], [0, 7, 8]]  # 5 body parts
        elif 'jrdb' in self.dataset: index = [[9, 10, 11], [12, 13, 14], [1, 2, 3], [4, 5, 6], [0, 7, 8]]  # 5 body parts
        # if self.dataset == 'cmu_umpm': index = [[8, 9, 10], [11, 12, 13], [1, 2, 3], [4, 5, 6], [0, 7, 14]]  # 5 body parts (original code)
        elif self.dataset == '3dpw': index = [[0, 2, 4], [1, 3, 5], [8, 10, 12], [7, 9, 11], [6, 7, 8]]  # 5 body parts
        part_seq = body_partition(src, index).permute(0, 3, 2, 1)
        mpbp_seq = self.conv2d(part_seq).permute(0, 2, 3, 1).reshape(bs, n_person, 5, -1, 128)    #  multi-person body parts sequence

        # ======= TBIFormer Encoder ============
        enc_out = self.encoder(mpbp_seq, n_person)
        return enc_out
    
    def forward_local(self, src, n_person, traj_query, enc_out):
        '''
        src_seq:  B*N, T, J*3
        '''
        num_modes = traj_query.shape[0]
        bn = src.shape[0]
        bs = int(bn / n_person)
        
        # ======= Transformer Decoder ============
        src_query = src.transpose(1, 2)[:, :, -self.kernel_size:].clone()  # the last sub-sequence for query
        global_body_query = self.mlp(src_query).reshape(bs, n_person, -1)
        global_body_query = global_body_query.unsqueeze(0).repeat(num_modes, 1, 1, 1)
        enc_out = enc_out.unsqueeze(0).repeat(num_modes,1,1,1)
        traj_query = traj_query.reshape(num_modes, bs, n_person, -1)
        new_query = torch.cat((global_body_query, traj_query), dim=-1)
        new_query = self.query_linear(new_query)
        dec_output = self.decoder(new_query, enc_out, False)
        dec_output = dec_output.reshape(num_modes, bn, 1, -1)
        # import pdb;pdb.set_trace()

        # =======  FC ============
        dec_output = self.l1(dec_output)
        dec_output = dec_output.view(num_modes, bn, self.future_steps, -1)
        dec_output = self.l2(dec_output)
        dec_out = self.proj_inverse(dec_output)
        return dec_out
    
    def traj_forward(self, input_traj_temporalData, enc_feat):
        local_embed = self.local_encoder_traj(data=input_traj_temporalData, enc_feat=enc_feat)
        global_embed = self.global_interactor_traj(data=input_traj_temporalData, local_embed=local_embed)
        y_hat, pi, out_feature = self.decoder_traj(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi, out_feature
