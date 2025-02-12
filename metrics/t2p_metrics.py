from typing import Any, Callable, Dict, Optional
import torch
import numpy as np
from torchmetrics import Metric

class APE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        super(APE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target):
        # Shape: B*N, T, 15, 3
        outputs = outputs - outputs[ :, :, 0:1, :]
        target = target - target[ :, :, 0:1, :]

        err = torch.norm(target-outputs, p=2, dim=-1)
        err_frames = err[:,torch.tensor(self.frame_idx)-1,:].mean(-1)
        # err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,None,:,None]).mean(0).mean(0).mean(-1)
        # err_overall = err_overall[torch.tensor(self.frame_idx)-1]
        self.sum += err_frames.sum()
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale

class APE_overall(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        super(APE_overall, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target):
        # Shape: B*N, T, 15, 3
        outputs = outputs - outputs[ :, :, 0:1, :]
        target = target - target[ :, :, 0:1, :]

        err = torch.norm(target-outputs, p=2, dim=-1)
        err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,:,None]).mean(-1)
        err_overall = err_overall[:, torch.tensor(self.frame_idx)-1]
        self.sum += err_overall.sum()
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale


class JPE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        super(JPE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target):
        # Shape: B*N, T, 15, 3

        err = torch.norm(target-outputs, p=2, dim=-1)
        err_frames = err[:,torch.tensor(self.frame_idx)-1,:].mean(-1)
        # err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,None,:,None]).mean(0).mean(0).mean(-1)
        # err_overall = err_overall[torch.tensor(self.frame_idx)-1]
        self.sum += err_frames.sum()
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale

class JPE_overall(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        super(JPE_overall, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target):
        # Shape: B*N, T, 15, 3

        err = torch.norm(target-outputs, p=2, dim=-1)
        err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,:,None]).mean(-1)
        err_overall = err_overall[:, torch.tensor(self.frame_idx)-1]
        self.sum += err_overall.sum()
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale


class FDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        frame_idx=-1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        self.scale = 1000
        super(FDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target):
        # Shape: B*N, T, 15, 3

        err = torch.norm(target[:,self.frame_idx-1:self.frame_idx,:1,:]-outputs[:,self.frame_idx-1:self.frame_idx,:1,:], p=2, dim=-1)[:,0,0]
        # err_frames = err[:,torch.tensor(self.frame_idx)-1,:].mean(-1)
        self.sum += err.sum()
        self.count += target.size(0)

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale

# class FDE_overall(Metric):
#     full_state_update: Optional[bool] = False
#     higher_is_better: Optional[bool] = False

#     def __init__(
#         self,
#         frame_idx=-1,
#         compute_on_step: bool = True,
#         dist_sync_on_step: bool = False,
#         process_group: Optional[Any] = None,
#         dist_sync_fn: Callable = None,
#     ) -> None:
#         scale = 1000
#         super(FDE_overall, self).__init__(
#             compute_on_step=compute_on_step,
#             dist_sync_on_step=dist_sync_on_step,
#             process_group=process_group,
#             dist_sync_fn=dist_sync_fn,
#         )
#         self.frame_idx = frame_idx
#         self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
#     def update(self, outputs, target):
#         # Shape: B*N, T, 15, 3
#         outputs = outputs - outputs[ :, :, 0:1, :]
#         target = target - target[ :, :, 0:1, :]

#         err = torch.norm(target-outputs, p=2, dim=-1)
#         err_overall = (torch.cumsum(err, dim=-2) / (torch.arange(err.size(-2), device=err.device)+1)[None,:,None]).mean(-1)
#         err_overall = err_overall[:, torch.tensor(self.frame_idx)-1]
#         self.sum += err_overall.sum()
#         self.count += target.size(0)

#     def compute(self) -> torch.Tensor:
#         return self.sum / self.count

class ADE_TR(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(ADE_TR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        pred, target = pred[:,:,0,:], target[:,:,0,:]
        ade = torch.norm(target - pred, p=2, dim=-1)
        ade = ade.mean(-1)
        
        self.sum += ade.sum()
        self.count += ade.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

class FDE_TR(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(FDE_TR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        
        pred, target = pred[:,:,0,:], target[:,:,0,:]
        l2 = torch.norm(target - pred, p=2, dim=-1)
        fde = l2[-1]
        
        self.sum += fde.sum()
        self.count += fde.shape[0]
        
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
    
class MR_TR(Metric):

    def __init__(self,
                 miss_threshold: float = 2.0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(MR_TR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.miss_threshold = miss_threshold

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               padding_mask: torch.Tensor) -> None:
        l2 = torch.norm(target - pred, p=2, dim=-1)
        padding_mask_sum = padding_mask.sum(dim=-1)
        padding_mask_bool = padding_mask_sum == 0
        padding_mask_sum[padding_mask_sum==0] += 1
        ade = (l2*padding_mask).sum(dim=-1)/padding_mask_sum
        made_idcs = torch.argmin(ade, dim=0)
        
        frame_mask = torch.zeros((padding_mask.shape[0], padding_mask.shape[1])).cuda()
        frame_mask[:, -1] = 1
        padding_mask = torch.logical_and(padding_mask, frame_mask.bool())
        fde = (l2*padding_mask).sum(dim=-1)
        fde = fde[made_idcs, torch.arange(l2.size(1))]
        fde_padding_mask_bool = padding_mask[:,-1] == 0
        fde_ = fde[~fde_padding_mask_bool]
        
        
        self.sum += (fde_ > self.miss_threshold).sum()
        self.count += fde_.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count