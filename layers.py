''' Define the Layers '''
import torch.nn as nn
from .sublayers import PositionwiseFeedForward, SBI_MSA, MultiHeadAttention, PositionwiseFeedForward_decoder
# from .sublayers import PositionwiseFeedForward, SBI_MSA, MultiHeadAttention



class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, d_traj_query=64, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, d_traj_query=d_traj_query, dropout=dropout)
        # self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward_decoder(d_model, d_inner, d_traj_query=d_traj_query, dropout=dropout)

    def forward(self, dec_input, enc_output):
        dec_output, dec_enc_attn = self.enc_attn(
            dec_input, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn

class TBIFormerBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TBIFormerBlock, self).__init__()
        self.sbi_msa = SBI_MSA(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, enc_input, n_person, emb):
        enc_output, residual, enc_slf_attn = self.sbi_msa(
            enc_input, n_person, emb)
        enc_output = self.pos_ffn(enc_output)
        enc_output += residual
        return enc_output, enc_slf_attn

