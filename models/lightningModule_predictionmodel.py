import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

# from metrics import APE_Mean, APE_T
from metrics.t2p_metrics_v2 import APE, JPE, FDE, APE_overall, JPE_overall
from hydra.utils import instantiate
from utils_.viz import *


class PredictionModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        output_time: int = 50,
        dataset: str = 'cmu_umpm',
        batch_size: int = 0,
        is_baseline: bool = False,
        num_joints: int = 15,
        viz_traj: bool = False,
        viz_joint: bool = False,
        viz_joint_jansang: bool = False
    ) -> None:
        super(PredictionModel, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.output_time = output_time
        self.batch_size = batch_size
        self.is_baseline = is_baseline
        self.num_joints = num_joints
        self.viz_traj = viz_traj
        self.viz_joint = viz_joint
        self.viz_joint_jansang = viz_joint_jansang
        self.save_hyperparameters()
        # self.net = instantiate(net)

        if self.dataset == 'cmu_umpm':
            if output_time >= 50:
                metrics = MetricCollection(
                    {
                        "APE_1000ms": APE(frame_idx=25),
                        "APE_overall_1000ms": APE_overall(frame_idx=25),
                        "JPE_1000ms": JPE(frame_idx=25),
                        "JPE_overall_1000ms": JPE_overall(frame_idx=25),
                        "FDE_1000ms": FDE(frame_idx=25),
                        
                        "APE_2000ms": APE(frame_idx=50),
                        "APE_overall_2000ms": APE_overall(frame_idx=50),
                        "JPE_2000ms": JPE(frame_idx=50),
                        "JPE_overall_2000ms": JPE_overall(frame_idx=50),
                        "FDE_2000ms": FDE(frame_idx=50),
                    }
                )
            else: 
                metrics = MetricCollection(
                    {
                        "APE_200ms": APE(frame_idx=5),"APE_overall_200ms": APE_overall(frame_idx=5),"JPE_200ms": JPE(frame_idx=5),"JPE_overall_200ms": JPE_overall(frame_idx=5),"FDE_200ms": FDE(frame_idx=5),
                        "APE_400ms": APE(frame_idx=10),"APE_overall_400ms": APE_overall(frame_idx=10),"JPE_400ms": JPE(frame_idx=10),"JPE_overall_400ms": JPE_overall(frame_idx=10),"FDE_400ms": FDE(frame_idx=10),
                        "APE_600ms": APE(frame_idx=15),"APE_overall_600ms": APE_overall(frame_idx=15),"JPE_600ms": JPE(frame_idx=15),"JPE_overall_600ms": JPE_overall(frame_idx=15),"FDE_600ms": FDE(frame_idx=15),
                        "APE_800ms": APE(frame_idx=20),"APE_overall_800ms": APE_overall(frame_idx=20),"JPE_800ms": JPE(frame_idx=20),"JPE_overall_800ms": JPE_overall(frame_idx=20),"FDE_800ms": FDE(frame_idx=20),
                        "APE_1000ms": APE(frame_idx=25),
                        "APE_overall_1000ms": APE_overall(frame_idx=25),
                        "JPE_1000ms": JPE(frame_idx=25),
                        "JPE_overall_1000ms": JPE_overall(frame_idx=25),
                        "FDE_1000ms": FDE(frame_idx=25),
                    }
                )
        elif self.dataset == '3dpw':
            if output_time == 20:
                metrics = MetricCollection(
                    {
                        "APE_400ms": APE(frame_idx=5),
                        "APE_800ms": APE(frame_idx=10),
                        "APE_1200ms": APE(frame_idx=15),
                        "APE_1600ms": APE(frame_idx=20),
                        "APE_overall_800ms": APE_overall(frame_idx=10),
                        "APE_overall_1600ms": APE_overall(frame_idx=20),
                        
                        "JPE_400ms": JPE(frame_idx=5),
                        "JPE_800ms": JPE(frame_idx=10),
                        "JPE_1200ms": JPE(frame_idx=15),
                        "JPE_1600ms": JPE(frame_idx=20),
                        "JPE_overall_800ms": JPE_overall(frame_idx=10),
                        "JPE_overall_1600ms": JPE_overall(frame_idx=20),
                        
                        "FDE_400ms": FDE(frame_idx=5),
                        "FDE_800ms": FDE(frame_idx=10),
                        "FDE_1200ms": FDE(frame_idx=15),
                        "FDE_1600ms": FDE(frame_idx=20),
                    }
                )
        elif 'jrdb' in self.dataset:
            if output_time <= 30:
                metrics = MetricCollection(
                    {
                        "APE_1000ms": APE(frame_idx=15),
                        "APE_overall_1000ms": APE_overall(frame_idx=15),
                        "JPE_1000ms": JPE(frame_idx=15),
                        "JPE_overall_1000ms": JPE_overall(frame_idx=15),
                        "FDE_1000ms": FDE(frame_idx=15),
                        
                        "APE_2000ms": APE(frame_idx=30),
                        "APE_overall_2000ms": APE_overall(frame_idx=30),
                        "JPE_2000ms": JPE(frame_idx=30),
                        "JPE_overall_2000ms": JPE_overall(frame_idx=30),
                        "FDE_2000ms": FDE(frame_idx=30),
                    }
                )
            elif output_time <= 60:
                metrics = MetricCollection(
                    {
                        "APE_1000ms": APE(frame_idx=15),
                        "APE_overall_1000ms": APE_overall(frame_idx=15),
                        "JPE_1000ms": JPE(frame_idx=15),
                        "JPE_overall_1000ms": JPE_overall(frame_idx=15),
                        "FDE_1000ms": FDE(frame_idx=15),
                        
                        "APE_2000ms": APE(frame_idx=30),
                        "APE_overall_2000ms": APE_overall(frame_idx=30),
                        "JPE_2000ms": JPE(frame_idx=30),
                        "JPE_overall_2000ms": JPE_overall(frame_idx=30),
                        "FDE_2000ms": FDE(frame_idx=30),
                        
                        "APE_4000ms": APE(frame_idx=60), "APE_overall_4000ms": APE_overall(frame_idx=60), "JPE_4000ms": JPE(frame_idx=60), "JPE_overall_4000ms": JPE_overall(frame_idx=60), "FDE_4000ms": FDE(frame_idx=60),
                    }
                )
            elif output_time <= 75:
                metrics = MetricCollection(
                    {                           
                        "APE_2500ms": APE(frame_idx=37), "APE_overall_2500ms": APE_overall(frame_idx=37), "JPE_2500ms": JPE(frame_idx=37), "JPE_overall_2500ms": JPE_overall(frame_idx=37), "FDE_2500ms": FDE(frame_idx=37),
                        "APE_5000ms": APE(frame_idx=75), "APE_overall_5000ms": APE_overall(frame_idx=75), "JPE_5000ms": JPE(frame_idx=75), "JPE_overall_5000ms": JPE_overall(frame_idx=75), "FDE_5000ms": FDE(frame_idx=75),
                    }
                )
            elif output_time <= 90:
                metrics = MetricCollection(
                    {
                        "APE_1000ms": APE(frame_idx=15),
                        "APE_overall_1000ms": APE_overall(frame_idx=15),
                        "JPE_1000ms": JPE(frame_idx=15),
                        "JPE_overall_1000ms": JPE_overall(frame_idx=15),
                        "FDE_1000ms": FDE(frame_idx=15),
                        
                        "APE_2000ms": APE(frame_idx=30),
                        "APE_overall_2000ms": APE_overall(frame_idx=30),
                        "JPE_2000ms": JPE(frame_idx=30),
                        "JPE_overall_2000ms": JPE_overall(frame_idx=30),
                        "FDE_2000ms": FDE(frame_idx=30),
                                                
                        "APE_3000ms": APE(frame_idx=45), "APE_overall_3000ms": APE_overall(frame_idx=45), "JPE_3000ms": JPE(frame_idx=45), "JPE_overall_3000ms": JPE_overall(frame_idx=45), "FDE_3000ms": FDE(frame_idx=45),
                        "APE_6000ms": APE(frame_idx=90), "APE_overall_6000ms": APE_overall(frame_idx=90), "JPE_6000ms": JPE(frame_idx=90), "JPE_overall_6000ms": JPE_overall(frame_idx=90), "FDE_6000ms": FDE(frame_idx=90),
                    }
                )
        # self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics = metrics.clone()
        for k, v in self.val_metrics.items():
            self.val_metrics[k] = v.cuda()
        self.output_dir = None
        # self.historical_steps = historical_steps
        # self.future_steps = future_steps

    def forward(self, data, mode):
        if mode == 'train':
            pred_traj, gt_traj, rec, offset = self.net(data, mode)
            return pred_traj, gt_traj, rec, offset
        elif mode == 'eval':
            pred_traj, gt_traj = self.net(data, mode)
            return pred_traj, gt_traj
            
            
    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, gt_traj, pred_traj, rec, offset, data):
        if not self.is_baseline:
            l2 = torch.norm(gt_traj - pred_traj, p=2, dim=-1)
            ade = l2.mean(-1)
            made_idcs = torch.argmin(ade, dim=0)
            
            traj_loss = l2[made_idcs, torch.arange(l2.size(1))]
            traj_loss = traj_loss[~data.padding_mask[:,-self.output_time:]].mean()
            
            joint_loss = torch.norm((offset[:,1:self.output_time+1] - offset[:,:self.output_time])[:,:,1:] - rec[made_idcs, torch.arange(rec.size(1))], p=2, dim=-1)
            joint_loss = joint_loss[~data.padding_mask[:,-self.output_time:]].mean()
            
            # pad_mask = ~data.padding_mask[:,-self.output_time:]
            # frame_mask = torch.zeros((pad_mask.shape[0], pad_mask.shape[1])).cuda()
            # frame_mask[:, self.output_time-1] = 1
            # pad_mask = torch.logical_and(pad_mask, frame_mask.bool())
            # fde_loss = l2[made_idcs, torch.arange(l2.size(1))]
            # fde_loss = fde_loss[pad_mask].mean()
            
            # loss = traj_loss + joint_loss + (fde_loss*2)
            loss = traj_loss + joint_loss
            # loss = joint_loss + fde_loss
            # loss = traj_loss
            
            return {
                "loss": loss,
                # "fde_loss": fde_loss.item(),
                "traj_loss": traj_loss.item(),
                "joint_loss": joint_loss.item(),
            }
        else:
            N_MODES, NB, _, _, _ = rec.shape
            output_compare = offset[:, 1:self.output_time+1, :] - offset[:, :self.output_time, :]
            gt_rec = output_compare.unsqueeze(0).reshape(1,NB,self.output_time,self.num_joints,3)
            l2 = torch.norm(rec - gt_rec, p=2, dim=-1)
            ade = l2.mean(-1).mean(-1)
            made_idcs = torch.argmin(ade, dim=0)
            rec = rec[made_idcs, torch.arange(rec.size(1))].reshape(NB,-1,self.num_joints*3)
            rec_loss = torch.norm(rec[:, :self.output_time, :] - (offset[:, 1:self.output_time+1, :] - offset[:, :self.output_time, :]), p=2, dim=-1) 
            rec_loss = rec_loss[~data.padding_mask[:,-self.output_time:]].mean()
            # N_MODES, NB, _, _ = offset.shape
            # results = offset[:, :, :1, :]
            # for i in range(1, self.output_time+1):
            #     results = torch.cat(
            #         [results, offset[:, :, :1, :] + torch.sum(rec[:, :, :i, :], dim=2, keepdim=True)],
            #         dim=2)
            # results = results[:, :, 1:, :]  # num_mode 3 15 45

            # prediction = results.view(N_MODES, NB, -1, self.num_joints, 3)
            # gt = offset.view(N_MODES, NB, -1, self.num_joints, 3)[:,:,1:,...]
            
            # l2 = torch.norm(prediction - gt, p=2, dim=-1)
            # ade = l2.mean(-1).mean(-1)
            # # ade = l2.mean(-1)[...,-1]
            # made_idcs = torch.argmin(ade, dim=0)
            # rec = rec[made_idcs, torch.arange(rec.size(1))].reshape(NB,-1,self.num_joints*3)
            
            # rec_loss = (rec[:, :self.output_time, :] - (offset[0, :, 1:self.output_time+1, :] - offset[0, :, :self.output_time, :])) ** 2
            # rec_loss = torch.mean(rec_loss[~data.padding_mask[:,-self.output_time:]])
            
            # # rec_loss = torch.mean((rec[:, :self.output_time, :] - (offset[0, :, 1:self.output_time+1, :] - offset[0, :, :self.output_time, :])) ** 2)
                        
            return {
                "loss": rec_loss,
                "rec_loss": rec_loss.item(),
            }

    def training_step(self, data, batch_idx):
        pred_traj, gt_traj, rec, offset = self(data, mode='train')
        losses = self.cal_loss(gt_traj, pred_traj, rec, offset, data)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        # if batch_idx in [328, 664, 652, 220, 216]:
        # if batch_idx in [664]:
        # if batch_idx >620 and batch_idx < 640:
        # if batch_idx in [462]:
        out, gt = self(data, mode='eval')
        padding_mask = data.padding_mask[:,-self.output_time:]
        
        ######## For evaluation of fine-grained samples ###########
        # fut_disp = torch.norm((gt[:,0,0]-gt[:,-1,0]), p=2, dim=-1)
        # mask = fut_disp < 0.2
        # out, gt, padding_mask = out[mask], gt[mask], padding_mask[mask]
        # if mask.sum() > 0:
        #     metrics = self.val_metrics(out, gt, padding_mask)
        ###########################################################
            
        metrics = self.val_metrics(out, gt, padding_mask)
        if batch_idx != 0: 
            if self.viz_traj:
                viz_trajectory(out, gt, data, self.output_dir, batch_idx)
            
            if self.viz_joint:
                viz_joint(out, gt, data, self.output_dir, batch_idx)
                # viz_joint_JRDB_v2(out, gt, data, self.output_dir, batch_idx)
            
            if self.viz_joint_jansang:
                if batch_idx < 10:
                    viz_joint_jansang_v2(out, gt, data, self.output_dir, batch_idx)
            
    
    def on_validation_epoch_end(self):
        final_metrics = self.val_metrics.compute()
        # print('Printing actual results!')
        # for n in final_metrics.keys():
        #     print(f'{n}: {final_metrics[n]}')
        # print('Final results done')
        for k, v in self.val_metrics.items():
            self.log(
                f"val/{k}",
                v,
                on_step=False,  
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=1
            )
            # self.logger.log_metrics()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(self.epochs*1.1), eta_min=0.0)

        return [optimizer], [scheduler]
