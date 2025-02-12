import torch
import numpy as np

def APE(V_trgt, V_pred, frame_idx):
    scale = 1000
    if V_pred.dim() == V_trgt.dim():
        V_pred = V_pred - V_pred[:, :, :, 0:1, :]
        V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]

        err = torch.norm(V_trgt-V_pred, p=2, dim=-1)
        err_frames = err[:,:,torch.tensor(frame_idx)-1,:].mean(0).mean(0).mean(-1)
        err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,None,:,None]).mean(0).mean(0).mean(-1)
        err_overall = err_overall[torch.tensor(frame_idx)-1]

    else:
        assert V_pred.dim() == V_trgt.dim()+1
        assert V_pred.shape[1:] == V_trgt.shape
        N_MODES, B, N, T, N_KP, DIM = V_pred.shape

        V_pred = V_pred - V_pred[:, :, :, :, 0:1, :]
        V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]
        l2 = torch.norm(V_trgt.unsqueeze(0)-V_pred, dim=-1, p=2)
        l2 = l2.reshape(N_MODES, B*N, -1).mean(-1)
        
        best_idcs = torch.argmin(l2, dim=0)
        V_pred_best = V_pred.reshape(N_MODES, B*N, T, N_KP, DIM)
        V_pred_best = V_pred_best[best_idcs, torch.arange(B*N)].reshape(B, N, T, N_KP, DIM)

        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred_best[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()

    return err_frames.cpu().data.numpy() * scale, err_overall.cpu().data.numpy() * scale


def JPE(V_trgt, V_pred, frame_idx):
    scale = 1000
    if V_pred.dim() == V_trgt.dim():
        err = np.arange(len(frame_idx), dtype=np.float_)
        err_overall = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()
            err_overall[idx] = torch.mean(torch.mean(torch.mean(torch.norm(V_trgt[:, :, :frame_idx[idx]-1, :, :] - V_pred[:, :, :frame_idx[idx]-1, :, :], dim=-1), dim=3),dim=2),dim=1).cpu().data.numpy().mean()
    else:
        assert V_pred.dim() == V_trgt.dim()+1
        assert V_pred.shape[1:] == V_trgt.shape
        N_MODES, B, N, T, N_KP, DIM = V_pred.shape

        l2 = torch.norm(V_trgt.unsqueeze(0)-V_pred, dim=-1, p=2)
        l2 = l2.reshape(N_MODES, B*N, -1).mean(-1)
        
        best_idcs = torch.argmin(l2, dim=0)
        V_pred_best = V_pred.reshape(N_MODES, B*N, T, N_KP, DIM)
        V_pred_best = V_pred_best[best_idcs, torch.arange(B*N)].reshape(B, N, T, N_KP, DIM)

        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred_best[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()

    return err * scale, err_overall * scale


# def ADE(V_pred, V_trgt, frame_idx):
#     scale = 1000
#     err = np.arange(len(frame_idx), dtype=np.float_)
#     for idx in range(len(frame_idx)):
#         err[idx] = torch.linalg.norm(V_trgt[:, :, :frame_idx[idx], :, :] - V_pred[:, :, :frame_idx[idx], :, :], dim=-1).mean(1).mean()
#     return err * scale


def FDE(V_trgt, V_pred, frame_idx):
    scale = 1000
    if V_pred.dim() == V_trgt.dim():
        err = np.arange(len(frame_idx), dtype=np.float_)
        err_overall = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()
            err_overall[idx] = torch.linalg.norm(V_trgt[:, :, :frame_idx[idx], : 1, :] - V_pred[:, :, :frame_idx[idx], : 1, :], dim=-1).mean(1).mean(1).mean()
    else:
        assert V_pred.dim() == V_trgt.dim()+1
        assert V_pred.shape[1:] == V_trgt.shape
        N_MODES, B, N, T, N_KP, DIM = V_pred.shape

        hip_pred = V_pred[..., 0, :]
        hip_trgt = V_trgt[..., 0, :]
        l2 = torch.norm(hip_trgt.unsqueeze(0)-hip_pred, dim=-1, p=2)
        l2 = l2.reshape(N_MODES, B*N, -1).mean(-1)
        
        best_idcs = torch.argmin(l2, dim=0)
        V_pred_best = V_pred.reshape(N_MODES, B*N, T, N_KP, DIM)
        V_pred_best = V_pred_best[best_idcs, torch.arange(B*N)].reshape(B, N, T, N_KP, DIM)

        err = np.arange(len(frame_idx), dtype=np.float_)
        for idx in range(len(frame_idx)):
            err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred_best[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()

    return err * scale, err_overall * scale