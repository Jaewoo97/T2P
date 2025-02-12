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
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        outputs = outputs - outputs[ :, :, 0:1, :]
        target = target - target[ :, :, 0:1, :]
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx-1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        
        err = torch.norm(target[padding_mask]-outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

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
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        outputs = outputs - outputs[ :, :, 0:1, :]
        target = target - target[ :, :, 0:1, :]
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, :self.frame_idx] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        
        err = torch.norm(target[padding_mask]-outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]
        
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
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx-1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        
        err = torch.norm(target[padding_mask]-outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

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
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, :self.frame_idx] = 1
        # import pdb;pdb.set_trace()
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        
        err = torch.norm(target[padding_mask]-outputs[padding_mask], p=2, dim=-1)
        err_frames = err.mean(-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]

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
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, self.frame_idx-1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target, outputs = target[padding_mask], outputs[padding_mask]
        err_frames = torch.norm(target[:, 0]-outputs[:, 0], p=2, dim=-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]
                
    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale


class ADE_TR(Metric):
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
        self.scale = 1
        super(ADE_TR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        frame_mask[:, :self.frame_idx] = 1
        if self.frame_idx == -1: frame_mask[:, self.frame_idx] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target = target[padding_mask]
        outputs = outputs[padding_mask]
        err = torch.norm(target[:, 0, :2]-outputs[:, 0, :2], p=2, dim=-1)
        self.sum += err.sum()
        self.count += err.shape[0]

    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale


class FDE_TR(Metric):
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
        self.scale = 1
        super(FDE_TR, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.frame_idx = frame_idx
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, outputs, target, padding_mask):
        # Shape: B*N, T, 15, 3
        frame_mask = torch.zeros((outputs.shape[0], outputs.shape[1])).cuda()
        if self.frame_idx != -1: frame_mask[:, self.frame_idx-1] = 1
        else: frame_mask[:, -1] = 1
        padding_mask = torch.logical_and(~padding_mask, frame_mask.bool())
        target, outputs = target[padding_mask], outputs[padding_mask]
        err_frames = torch.norm(target[:, 0, :2]-outputs[:, 0, :2], p=2, dim=-1)
        self.sum += err_frames.sum()
        self.count += err_frames.shape[0]
                
    def compute(self) -> torch.Tensor:
        return (self.sum / self.count)*self.scale
