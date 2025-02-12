import os
import os.path as osp
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F

from shutil import copyfile

PLOT_COLOR = {0: {'past': 'r', 'fut': 'cyan', 'pred': 'g', 'g_gt': 'm', 'g_pred': 'y'}, 1: {'past': 'r', 'fut': 'b', 'pred': 'lime', 'g_gt': 'pink', 'g_pred': 'lemonchiffon'}}
ACTOR_CATEGORY = {'vehicle.bus.bendy': 0, 'vehicle.bus.rigid': 1, 'vehicle.car':2, 'vehicle.construction':3, 
                    'vehicle.emergency.ambulance':4, 'vehicle.emergency.police':5, 'vehicle.motorcycle': 6, 'vehicle.trailer':7, 
                    'vehicle.truck':8, 'vehicle.bicycle': 9, 'others': 10}

def save_modules(log_dir: str, config_fn: str, cfg: str):
    modules_dir = osp.join(log_dir, 'modules')
    viz_dir = osp.join(log_dir, 'viz')
    viz_test_dir = osp.join(log_dir, 'viz_test')
    os.makedirs(modules_dir)
    os.makedirs(viz_dir)
    os.makedirs(viz_test_dir)

    explore_dict(cfg, modules_dir)


def viz_result_batch(self, log_dir, data, output, batch_idx, stage, is_segment=False, is_gtabs=False):

    for batch_n in torch.unique(data.batch):

        batch_mask = data.batch == batch_n
        
        valid_agents = torch.where(batch_mask)[0]

        has_goal_batch = data.has_goal[valid_agents].bool()

        # # reg, batch 둘 다 해당하는지
        # regbatch_mask = output['reg_mask'].sum(-1).bool()[batch_mask]
        
        rotate_mat = torch.empty(valid_agents.size(0), 2, 2, device=self.device)
        sin_vals = torch.sin(data['rotate_angles'][valid_agents])
        cos_vals = torch.cos(data['rotate_angles'][valid_agents])
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = sin_vals
        rotate_mat[:, 1, 0] = -sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        gt_paths = torch.bmm(data.y[valid_agents], rotate_mat)

        pred_paths, pred_scales = torch.chunk(output['loc'][:,valid_agents], 2, -1)
        pred_probs = output['pi'][valid_agents]
        k_,n_,ts_,x_ = pred_paths.shape
        pred_paths = pred_paths.transpose(1,0).reshape(n_,-1,x_)
        pred_paths = torch.bmm(pred_paths, rotate_mat).reshape(n_,k_,ts_,x_).transpose(1,0)

        pasts = data.positions[valid_agents].cpu()
        categorys = data.category[valid_agents].cpu()
        assert data.av_index.size(0) == len(torch.unique(data.batch)) == torch.unique(data.batch)[-1]+1, 'av index 구하는것 다름'
        av_index = data.av_index[batch_n]
        # target_pasts = data.positions[]
        agent_masks = ~data['padding_mask'][valid_agents].cpu()
        regmask_batch = output['reg_mask'][valid_agents].cpu()

        
        
        gt_paths = gt_paths.cpu()
        pred_paths = pred_paths.detach().cpu()
        pred_scales = pred_scales.detach().cpu()
        pred_probs = F.softmax(pred_probs, -1).cpu()

        if not is_gtabs:
            gt_paths = torch.cumsum(gt_paths, dim=1)
            pred_paths = torch.cumsum(pred_paths, dim=2)
        gt_paths = torch.cat((torch.zeros_like(gt_paths)[:,:1,:], gt_paths), dim=1)
        pred_paths = torch.cat((torch.zeros_like(pred_paths)[:,:,:1,:], pred_paths), dim=2)

        valid_lane_mask = torch.isin(data.lane_actor_index[1,:], valid_agents)
        valid_lane_idxs = data.lane_actor_index[0,valid_lane_mask]
        valid_lane_idxs = torch.unique(valid_lane_idxs)

        valid_lanes = torch.index_select(data.lane_positions, 0, valid_lane_idxs).cpu()

        if stage == 'test':
            instance_token, sample_token = data['seq_id'][av_index].split('_')

            theta_av = data['theta'][batch_n]
            origin_av = data['origin'][batch_n].to(gt_paths.device)
            sin_vals = torch.sin(theta_av)
            cos_vals = torch.cos(theta_av)
            inv_rot_mat = torch.empty(gt_paths.size(0), 2, 2, device=gt_paths.device)
            inv_rot_mat[:, 0, 0] = cos_vals
            inv_rot_mat[:, 0, 1] = sin_vals
            inv_rot_mat[:, 1, 0] = -sin_vals
            inv_rot_mat[:, 1, 1] = cos_vals

            pasts = torch.bmm(pasts, inv_rot_mat)
            gt_paths = torch.bmm(gt_paths, inv_rot_mat)
            pred_paths = pred_paths.transpose(1,0).reshape(n_,-1,2)
            pred_paths = torch.bmm(pred_paths, inv_rot_mat).reshape(n_,k_,-1,2).transpose(1,0)
            valid_lanes = torch.bmm(valid_lanes, inv_rot_mat.repeat(valid_lanes.size(0),1,1))

            # pasts, gt_paths, pred_paths, valid_lanes = pasts+origin_av[None,None,:], gt_paths+origin_av[None,None,:], pred_paths+origin_av[None,None,None,:], valid_lanes+origin_av[None,None,:]
            # gt path 와 pred path 는 뒤에서 visualize 해줄때 past 의 시작점을 더해주기 때문에 생략
            pasts, valid_lanes = pasts+origin_av[None,None,:], valid_lanes+origin_av[None,None,:]

        fig, ax = plt.subplots(1,1, figsize=(15,15))
        ax.set_aspect('equal')
        if is_segment:
            lane_paddings = (1-data['lane_paddings']).bool()
            lane_paddings = torch.index_select(lane_paddings, 0, valid_lane_idxs).cpu()
            for li, vl in enumerate(valid_lanes):
                vp = lane_paddings[li]
                vl = vl[vp]
                ax.plot(vl[:,0], vl[:,1], c='0.8')
        else:
            ax.scatter(valid_lanes[:,0], valid_lanes[:,1], s=1, c='0.8')

        for i, has_goal in enumerate(has_goal_batch):
            if not has_goal:
                continue
            agent_i = valid_agents[i]

            almask = data['lane_actor_index'][1] == agent_i
            agent_goal_prob = output['goal_prob'] * almask.float()
            agent_goal_idx = torch.argmax(agent_goal_prob)
            agent_goal_idx = data['lane_actor_index'][0][agent_goal_idx]
            goal_pos = data.lane_positions[agent_goal_idx]
            goal_msk = (1-data['lane_paddings']).bool()[agent_goal_idx]
            goal_pos = goal_pos[goal_msk].cpu()

            is_av = agent_i in data.agent_index
            ax.plot(goal_pos[:,0], goal_pos[:,1], linewidth=3, c=PLOT_COLOR[is_av]['g_pred'])

            gt_goal = data['goal_idcs'] * almask
            if gt_goal.sum() > 0:
                goal_idx = torch.nonzero(gt_goal)[0].item()
                goal_idx = data['lane_actor_index'][0,:][goal_idx]

                goal_pos_gt = data.lane_positions[goal_idx]
                goal_msk = (1-data['lane_paddings']).bool()[goal_idx]
                goal_pos_gt = goal_pos_gt[goal_msk].cpu()
                ax.scatter(goal_pos_gt[:,0], goal_pos_gt[:,1], c=PLOT_COLOR[is_av]['g_gt'])

        for i, agent_i in enumerate(valid_agents):
            past = pasts[i,:self.ref_time+1]
            category = categorys[i]
            category = list(ACTOR_CATEGORY.keys())[category]
            is_av = agent_i.item() == av_index.item()
            gt_path = gt_paths[i] + past[self.ref_time]
            pred_prob = pred_probs[i]
            pred_path = pred_paths[:,i] + past[self.ref_time]

            agent_mask = agent_masks[i]
            mask_past, mask_future = agent_mask[:self.ref_time+1], agent_mask[self.ref_time:]

            past = past[mask_past]
            gt_path = gt_path[mask_future]
            pred_path = pred_path[:,mask_future]

            if stage == 'test':
                num_samples2show = 10
            else:    
                num_samples2show = 3

            if self.pretrain:
                pred_diff = torch.norm(pred_path - gt_path, dim=-1).sum(-1)
                topk_idx = torch.topk(pred_diff,num_samples2show, largest=False)[1]
            else:
                topk_idx = torch.topk(pred_prob,num_samples2show)[1]

            pred_path = pred_path[topk_idx]
            pred_scale = pred_scales[:,i]
            pred_scale = pred_scale[topk_idx, -1]

            if regmask_batch[i,self.ref_time+1]: ## reg_mask에 해당하는 것만 show
                for k in range(num_samples2show):
                    ax.plot(pred_path[k,:,0], pred_path[k,:,1], c=PLOT_COLOR[is_av]['pred'])
                    if mask_future[-1]: ax.scatter(pred_path[k,-1,0], pred_path[k,-1,1], s=15, c=PLOT_COLOR[is_av]['pred']) 

                    ellipse = Ellipse((pred_path[k,-1,0],pred_path[k,-1,1]), pred_scale[k,0], pred_scale[k,1], facecolor='none', edgecolor='k')
                    ax.add_patch(ellipse)

            ax.plot(past[:,0], past[:,1], c=PLOT_COLOR[is_av]['past'])

            ax.plot(gt_path[:,0], gt_path[:,1], c=PLOT_COLOR[is_av]['fut'])
            if mask_future[-1]: ax.scatter(gt_path[-1,0], gt_path[-1,1], s=15, c=PLOT_COLOR[is_av]['fut'])

            if is_av:
                ax.set_title(f'{category}')

        if stage == 'test':
            plt.xlim([origin_av[0]-50, origin_av[0]+50])
            plt.ylim([origin_av[1]-50, origin_av[1]+50])   
            img_fn = osp.join(log_dir, 'viz_test', f'{instance_token}_{sample_token}.jpg')
        else:    
            img_fn = osp.join(log_dir, 'viz', f'Out_{stage}_B{batch_idx}_S{batch_n}.jpg')
    
        plt.savefig(img_fn)
        plt.close()
        plt.clf()

def viz_data_goal(onesample, viz_dir):
    ref_time = 4

    fig, ax = plt.subplots(figsize=(30,30))
    agent_index = onesample['agent_index']
    ax.scatter(onesample['lane_positions'][:,:,0], onesample['lane_positions'][:,:,1], c='gray', s=1)
    for iidx, vid in enumerate(range(len(onesample['y']))):
        positions = onesample['positions'][vid]
        x, y = positions[:ref_time+1], positions[ref_time:]
        padding = onesample['padding_mask'][vid]
        padding_past, padding_fut = padding[:ref_time+1], padding[ref_time:]
        valid_x, valid_y = x[~padding_past], y[~padding_fut]

        ax.plot(valid_x[:,0], valid_x[:,1], c='g')
        ax.text(valid_x[-1,0], valid_x[-1,1], str(iidx))
        color='r' if vid == agent_index else 'b'
        ax.plot(valid_y[:,0], valid_y[:,1], c=color)
        if ~padding_fut[-1]:
            ax.scatter(valid_y[-1,0], valid_y[-1,1], c=color)

    goal_lane_ids = onesample['lane_actor_index'][0][onesample['goal_idcs'].bool()]
    goal_actor_idcs = onesample['lane_actor_index'][1][onesample['goal_idcs'].bool()]
    for gidx, goal_lane_id in enumerate(goal_lane_ids):
        goal_cls = onesample['lane_positions'][goal_lane_id]
        goal_cl_mask = (1-onesample['lane_paddings'][goal_lane_id]).bool()
        goal_cls = goal_cls[goal_cl_mask]
        ax.scatter(goal_cls[:,0], goal_cls[:,1], c='y')
        ax.text(goal_cls[0,0], goal_cls[0,1]-1, str(goal_actor_idcs[gidx].item()), c='r')

    ax.set_aspect('equal')
    plt.savefig(os.path.join(viz_dir, onesample['seq_id']+'.jpg'))
    plt.close()
    plt.clf()

def figure_to_array(fig, ax, shape='hwc'):
    #Image from plot
    # ax.axis('off')
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = torch.frombuffer(fig.canvas.tostring_rgb(), dtype=torch.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if shape == 'hwc':
        pass
    elif shape == 'cwh':
        image_from_plot = image_from_plot.permute(2,0,1)
    else:
        raise KeyError('shape should be hwc or cwh')
    return image_from_plot

def explore_dict(d, save_dir):
    """
    A recursive function to explore all items of a nested dictionary.

    Args:
    - d: a dictionary with nested dictionaries as items

    Returns:
    - None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            explore_dict(value, save_dir)
        elif isinstance(value, str):
            if value.endswith('.py') and os.path.isfile(value):
                fn = Path(value).name
                copyfile(value, osp.join(save_dir, fn))