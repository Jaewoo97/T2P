import numpy as np
import torch

def get_dct_matrix(N):
    """Calculates DCT Matrix of size N."""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def keypoint_mse(output, target, mask=None):
    """Implement 2D and 3D MSE loss
    Arguments:
    output -- tensor of predicted keypoints (B, ..., K, d)
    target -- tensor of ground truth keypoints (B, ..., K, d)
    mask   -- (optional) tensor of shape (B, ..., K, 1)
    """
    assert output.shape == target.shape
    assert len(output.shape) >= 3

    B = output.shape[0]
    K = output.shape[-2]

    dims = len(output.shape)

    if mask is None:
        mask = torch.ones(B, *[1] * (dims - 1)).float().to(output.device)
        valids = torch.ones(B, *[1] * (dims - 3)).to(output.device) * K

    else:
        if len(mask.shape) != len(output.shape):  # i.e. shape is (B, ..., K)
            mask = mask.unsqueeze(-1)

        assert mask.shape[:-1] == output.shape[:-1]

        valids = torch.sum(mask.squeeze(), dim=-1)
    import pdb;pdb.set_trace()
    norm = torch.norm(output * mask - target * mask, p=2, dim=-1)
    mean_K = torch.sum(norm, dim=-1) / valids
    mean_B = torch.mean(mean_K)

    return mean_B