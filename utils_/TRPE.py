
from .soft_dtw_cuda import SoftDTW
from .iRPE import piecewise_index
import itertools
import torch
import numpy as np

def temporal_partition(src, opt):
    src = src[:, :, 1:]
    B, N, L, _ = src.size()
    stride = 1
    fn = int((L - opt.kernel_size) / stride + 1)
    idx = np.expand_dims(np.arange(opt.kernel_size), axis=0) + \
          np.expand_dims(np.arange(fn), axis=1) * stride
    return idx      
       
     
       
def bulding_TRPE_matrix(input_seq, opt):      
    # Build the distance matrix for TRPE
    bs, n, t, jn, d = input_seq.shape
    input_ = input_seq.reshape(bs * n, opt.input_time, -1)
    dist_matrix = torch.zeros(bs, n, n)
    id_lis = list(itertools.combinations(list(np.arange(0, n)), 2))  # list of pairs

    if opt.device == 'cpu':
        sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    else:
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    #  Filter useless values to reduce the computational burden
    torso = input_.clone().reshape(bs, n, t, -1)[:, :, :, :3]
    temporal_idx = temporal_partition(torso, opt)
    trj_seq = torso[:, :, temporal_idx].permute(0, 2, 1, 3, 4).reshape(-1, n, opt.kernel_size, 3)
    fn = len(temporal_idx)
    dist_matrix = torch.tensor([opt.theta]).repeat(bs, fn * n, fn * n)
    torch.set_printoptions(profile="full")
    for i in range(len(id_lis)):
        res = sdtw(trj_seq[:, id_lis[i][0], :, :], trj_seq[:, id_lis[i][1], :, :])
        res_ = res.unsqueeze(-1).reshape(bs, -1)
        for k in range(n):
            dist_matrix[:, k * fn:k * fn + fn, k * fn:k * fn + fn] = torch.zeros([bs, fn, fn])
        for j in range(fn):
            dist_matrix[:, id_lis[i][0] * fn + j, id_lis[i][1] * fn + j] = res_[:, j]
            dist_matrix[:, id_lis[i][1] * fn + j, id_lis[i][0] * fn + j] = res_[:, j]
    trj_dist = piecewise_index(torch.abs(dist_matrix), 1, 9, opt.theta, dtype=int)

    return trj_dist