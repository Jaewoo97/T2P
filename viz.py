import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import glob
from PIL import Image
from utils_.opt import Options
import os
HUMAN_COLORS = ['orangered', 'limegreen', 'deepskyblue', 'cyan', 'skyblue', 'navy', 'magenta', 'darkturquoise', 'olive', 'dimgray', 'darkorange', 'lightcoral', 'lime', 'yellowgreen', 'peru', 'chocolate', 'orangered', 'navy', 'mediumturquoise', 'crimson', 'red', 'green', 'blue', 'yellow', 'cyan', 'skyblue', 'olive', 'dimgray', 'darkorange',]
TBI15_BONES = np.array(
    [[0, 1], [1, 2], [2, 3], [0, 4],
        [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
)
EDGES_3DPW = np.array([(0, 1), (1, 8), (8, 7), (7, 0),
			 (0, 2), (2, 4),
			 (1, 3), (3, 5),
			 (7, 9), (9, 11),
			 (8, 10), (10, 12),
			 (6, 7), (6, 8)])

def viz_trajectory(pred, gt, data, output_dir, batch_idx):
    print(f'Visualizing trajectory of batch {batch_idx}')
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'trajectory')): os.makedirs(os.path.join(output_dir, 'viz_results', 'trajectory'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt[:,:,0].reshape(B, N, T-1, 3).cpu().numpy()
    y_pred = pred[:, :, 0].reshape(B, N, T-1, 3).cpu().detach().numpy()
    
    for scene_idx in range(B):
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        xy_mean = np.zeros((0, 2))
        for agent_idx in range(N):
            if padding_mask_fut[scene_idx, agent_idx].sum() <= 1: continue
            x_gt_masked = x_gt[scene_idx, agent_idx][padding_mask_past[scene_idx, agent_idx]]
            y_gt_masked = y_gt[scene_idx, agent_idx][padding_mask_fut[scene_idx, agent_idx]]
            y_pred_masked = y_pred[scene_idx, agent_idx][padding_mask_fut[scene_idx, agent_idx]]
            
            ax.plot(x_gt_masked[:, 0], x_gt_masked[:, 1], 'ko-', linewidth=0.25, markersize=0.5)
            ax.plot(y_gt_masked[:, 0], y_gt_masked[:, 1], 'bo-', linewidth=0.25, markersize=0.5)
            ax.plot(y_pred_masked[:, 0], y_pred_masked[:, 1], 'ro-', linewidth=0.25, markersize=0.5)
            xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt_masked.mean(0)[:2], axis=0)), axis=0)
        ax.set_xlim([xy_mean.mean(0)[0]-5, xy_mean.mean(0)[0]+5])
        ax.set_ylim([xy_mean.mean(0)[1]-5, xy_mean.mean(0)[1]+5])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(f'{output_dir}/viz_results/trajectory/batch_{batch_idx}_scene_{scene_idx}.png')
        plt.close()
        plt.cla()
    # import pdb;pdb.set_trace()
    
# def viz_pose_sequence_3d(output_dir, y_pred, data, batch_idx, comment=None, bones=None):
def viz_joint(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}'))
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'gifs')): os.makedirs(os.path.join(output_dir, 'viz_results', 'gifs'))
    
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            # Plot trajectory of agents (only past GT)
            frame_idx_past = min(frame_idx_, data.input_seq.shape[2]-1)
            if padding_mask_past[scene_idx, :, frame_idx_past].sum() < 1: continue
            for agent_idx in range(x_gt.shape[1]):
                for traj_frame_idx in range(0,frame_idx_past+1):
                    ax.scatter3D(x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
            # Plot body motion
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                    ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=1.0, s=5)
                    
                    if bones is None:
                        if x_gt.shape[3] == 15: bones = TBI15_BONES
                        if x_gt.shape[3] == 13: bones = EDGES_3DPW
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
                    for traj_frame_idx in range(0,frame_idx_-1):
                        ax.scatter3D(x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                
                # Plot pred/gt trajectory
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    for traj_frame_idx in range(0,data.output_seq.shape[2]-1):
                        ax.scatter3D(y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=HUMAN_COLORS[agent_idx], alpha=1.0, s=5)
                        ax.scatter3D(y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
                
                # plot pred motion
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_pred[scene_idx, agent_idx, frame_idx, :, 0], y_pred[scene_idx, agent_idx, frame_idx, :, 1], y_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)
                    # for traj_frame_idx in range(0,frame_idx-1):
                    #     ax.scatter3D(y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=HUMAN_COLORS[agent_idx], alpha=1.0, s=5)
                
                # plot GT motion
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_gt[scene_idx, agent_idx, frame_idx, :, 0], y_gt[scene_idx, agent_idx, frame_idx, :, 1], y_gt[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=0.7, s=5)
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black', alpha=0.65)
                        ax.add_line(line)
                    # for traj_frame_idx in range(0,frame_idx_-1):
                    #     ax.scatter3D(y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [y_gt[scene_idx, :, :, :, 0].mean(), y_gt[scene_idx, :, :, :, 0].mean()]
            scene_max = np.max((np.abs(y_gt[scene_idx, :, :, :, 0]).max(), np.abs(y_gt[scene_idx, :, :, :, 1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max*1.2), scene_center[0]+(scene_max*1.2)])
            ax.set_ylim3d([scene_center[1]-(scene_max*1.2), scene_center[1]+(scene_max*1.2)])
            ax.set_zlim3d([0, 3])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.close()
            plt.cla()
            sdf=1
        gif_save_dir = os.path.join(output_dir, 'viz_results', 'gifs')
        frame_save_dir = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')
        save_as_gif_v2(gif_save_dir, frame_save_dir, TOTAL_T, batch_idx, scene_idx_list)

JRDB_COLORS = ['crimson', 'royalblue', 'darkturquoise', 'limegreen', 'black']
def viz_joint_JRDB(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    JRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/JRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    TBI_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/TBI/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    MRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/MRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    JRT_pred, TBI_pred, MRT_pred = JRT_pred.unsqueeze(0).cpu(), TBI_pred.unsqueeze(0).cpu(), MRT_pred.unsqueeze(0).cpu()
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB', f'batch_{batch_idx}'))
    
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(-1,TOTAL_T,15):
            if frame_idx_ == -1: frame_idx_=0
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                    ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=0.5, s=5)
                    
                    if bones is None:
                        if x_gt.shape[3] == 15: bones = TBI15_BONES
                        if x_gt.shape[3] == 13: bones = EDGES_3DPW
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='dimgray')
                        ax.add_line(line)
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    # ax.scatter3D(y_pred[scene_idx, agent_idx, frame_idx, :, 0], y_pred[scene_idx, agent_idx, frame_idx, :, 1], y_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=JRDB_COLORS[0], alpha=0.8)
                        ax.add_line(line)
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(y_pred[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                    for traj_frame_idx in range(0,frame_idx-1):
                        ax.scatter3D(y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=JRDB_COLORS[0], alpha=1.0, s=5)
                    
                    # JRT    
                    
                    # ax.scatter3D(JRT_pred[scene_idx, agent_idx, frame_idx, :, 0], JRT_pred[scene_idx, agent_idx, frame_idx, :, 1], JRT_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [JRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], JRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [JRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], JRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [JRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], JRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=JRDB_COLORS[1], alpha=0.8)
                        ax.add_line(line)
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(JRT_pred[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                    for traj_frame_idx in range(0,frame_idx-1):
                        ax.scatter3D(JRT_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], JRT_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=JRDB_COLORS[1], alpha=1.0, s=5)
                    
                    # MRT
                    # ax.scatter3D(MRT_pred[scene_idx, agent_idx, frame_idx, :, 0], MRT_pred[scene_idx, agent_idx, frame_idx, :, 1], MRT_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [MRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], MRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [MRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], MRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [MRT_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], MRT_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=JRDB_COLORS[2], alpha=0.8)
                        ax.add_line(line)
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(MRT_pred[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                    for traj_frame_idx in range(0,frame_idx-1):
                        ax.scatter3D(MRT_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], MRT_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=JRDB_COLORS[2], alpha=1.0, s=5)
                    
                        
                    # TBI
                    # ax.scatter3D(TBI_pred[scene_idx, agent_idx, frame_idx, :, 0], TBI_pred[scene_idx, agent_idx, frame_idx, :, 1], TBI_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [TBI_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], TBI_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [TBI_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], TBI_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [TBI_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], TBI_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=JRDB_COLORS[3], alpha=0.8)
                        ax.add_line(line)
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(TBI_pred[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                    for traj_frame_idx in range(0,frame_idx-1):
                        ax.scatter3D(TBI_pred[scene_idx, agent_idx, traj_frame_idx, 0, 0], TBI_pred[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=JRDB_COLORS[3], alpha=1.0, s=5)
                
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # if y_gt[scene_idx, agent_idx, frame_idx, :, 0].sum() < 0.001: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    # ax.scatter3D(y_gt[scene_idx, agent_idx, frame_idx, :, 0], y_gt[scene_idx, agent_idx, frame_idx, :, 1], y_gt[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black', alpha=0.8)
                        ax.add_line(line)
                    for traj_frame_idx in range(0,frame_idx-1):
                        if padding_mask_fut[scene_idx, agent_idx, traj_frame_idx] == False: continue
                        ax.scatter3D(y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], y_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [xy_mean[:,0].mean(), xy_mean[:,1].mean()]
            scene_max = np.max((np.abs(xy_mean[:,0]-scene_center[0]).max(), np.abs(xy_mean[:,1]-scene_center[1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(int(scene_center[0])-10, int(scene_center[0])+11), range(int(scene_center[1])-10, int(scene_center[1])+11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max), scene_center[0]+(scene_max)])
            ax.set_ylim3d([scene_center[1]-(scene_max), scene_center[1]+(scene_max)])
            ax.set_zlim3d([0, 2.5])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_JRDB', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            # plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=150)
            plt.close()
            plt.cla()

def viz_joint_JRDB_v2(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    JRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference_v2/JRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    TBI_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference_v2/TBI/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    MRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference_v2/MRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    JRT_pred, TBI_pred, MRT_pred = JRT_pred.unsqueeze(0).cpu(), TBI_pred.unsqueeze(0).cpu(), MRT_pred.unsqueeze(0).cpu()
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v2')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v2'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v2', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v2', f'batch_{batch_idx}'))
    predTypes = {'t2p': y_pred, 'jrt':JRT_pred, 'mrt': MRT_pred, 'tbi':TBI_pred, 'gt':y_gt}

    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        for predType in predTypes.keys():
            pred2print = predTypes[predType]
            if padding_mask_fut[scene_idx].sum() <= 1: continue
            for frame_idx_ in range(-1,TOTAL_T,1):
                xy_mean = np.zeros((0, 2))
                if frame_idx_ < data.input_seq.shape[2]: # T < current time
                    if predType != 't2p': continue
                # if frame_idx_ not in [0, 29, 66,67,103, 104]: continue
                if frame_idx_%2==0: continue
                if frame_idx_ == -1: frame_idx_=0
                plt.close()
                plt.cla()
                fig = plt.figure(figsize=(20, 9))
                ax = fig.add_subplot(111, projection='3d')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
                if frame_idx_ < data.input_seq.shape[2]: # T < current time
                    if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                    for agent_idx in range(x_gt.shape[1]):
                        if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                        xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                        ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=1.0, s=5)
                        
                        if bones is None:
                            if x_gt.shape[3] == 15: bones = TBI15_BONES
                            if x_gt.shape[3] == 13: bones = EDGES_3DPW
                        for edge in bones:
                            x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                            y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                            z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c='black')
                            ax.add_line(line)
                        for traj_frame_idx in range(0,frame_idx_-1):
                            ax.scatter3D(x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
            
                else: # T > current time
                    frame_idx = frame_idx_ - data.input_seq.shape[2]
                    
                    # JRT post-processing
                    # if predType == 'jrt':
                    #     pred2print[scene_idx, agent_idx, frame_idx, 0, :]
                    
                    if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                    for agent_idx in range(pred2print.shape[1]):
                        if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                        xy_mean = np.concatenate((xy_mean, np.expand_dims(pred2print[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                        ax.scatter3D(pred2print[scene_idx, agent_idx, frame_idx, :, 0], pred2print[scene_idx, agent_idx, frame_idx, :, 1], pred2print[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                        if bones is None: bones = TBI15_BONES
                        for edge in bones:
                            x_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 0], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                            y_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 1], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                            z_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 2], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx], alpha=0.8)
                            ax.add_line(line)
                        for traj_frame_idx in range(0,frame_idx-1):
                            ax.scatter3D(pred2print[scene_idx, agent_idx, traj_frame_idx, 0, 0], pred2print[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=HUMAN_COLORS[agent_idx], alpha=1.0, s=5)
                        
                ax.set_title(f'frame: {frame_idx_}')
                
                scene_center = [xy_mean[:,0].mean(), xy_mean[:,1].mean()]
                # scene_max = np.max((np.abs(xy_mean[:,0]-scene_center[0]).max(), np.abs(xy_mean[:,1]-scene_center[1]).max()))
                scene_max=3.5
                # Plot horizontal plane
                xx, yy = np.meshgrid(range(int(scene_center[0])-10, int(scene_center[0])+11), range(int(scene_center[1])-10, int(scene_center[1])+11))
                z = (9 - xx - yy) *0
                ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
                
                # Plot xy grid lines
                for ii in range(-8,10,2):
                    line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                    ax.add_line(line_x)
                for jj in range(-8,10,2):
                    line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                    ax.add_line(line_x)
                ax.set_xlim3d([scene_center[0]-(scene_max), scene_center[0]+(scene_max)])
                ax.set_ylim3d([scene_center[1]-(scene_max), scene_center[1]+(scene_max)])
                ax.set_zlim3d([0, 2.5])
                # ax.set_xlabel("x")
                # ax.set_ylabel("y")
                # ax.set_zlabel("z")
                frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_JRDB_v2', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}_{predType}.png'
                plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
                # plt.savefig(frame_save_name, bbox_inches='tight', dpi=150)
                plt.close()
                plt.cla()
                
def viz_joint_JRDB_v3(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    JRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/JRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    TBI_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/TBI/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    MRT_pred = torch.load(f'/ssd4tb/jaewoo/t2p/t2p/jrdb_2_5_inference/MRT/PRED_scene_{str(batch_idx).zfill(4)}.pt')
    JRT_pred, TBI_pred, MRT_pred = JRT_pred.unsqueeze(0).cpu(), TBI_pred.unsqueeze(0).cpu(), MRT_pred.unsqueeze(0).cpu()
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v3')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v3'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v3', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_JRDB_v3', f'batch_{batch_idx}'))
    predTypes = {'t2p': y_pred, 'jrt':JRT_pred, 'mrt': MRT_pred, 'tbi':TBI_pred, 'gt':y_gt}

    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        for predType in predTypes.keys():
            pred2print = predTypes[predType]
            xy_mean = np.zeros((0, 2))
            if padding_mask_fut[scene_idx].sum() <= 1: continue
            for frame_idx_ in range(-1,TOTAL_T,1):
                for azimuth in range(0,360,10):
                    if azimuth != 260: continue
                    if frame_idx_ not in [0, 24, 68, 102]: continue
                    if frame_idx_ < data.input_seq.shape[2]: # T < current time
                        if predType != 't2p': continue
                    if frame_idx_ == -1: frame_idx_=0
                    plt.close()
                    plt.cla()
                    fig = plt.figure(figsize=(20, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.grid(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                
                    if frame_idx_ < data.input_seq.shape[2]: # T < current time
                        if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                        for agent_idx in range(x_gt.shape[1]):
                            if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                            xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                            ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=1.0, s=5)
                            
                            if bones is None:
                                if x_gt.shape[3] == 15: bones = TBI15_BONES
                                if x_gt.shape[3] == 13: bones = EDGES_3DPW
                            for edge in bones:
                                x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                                y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                                z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                                line = Line3D(x_, y_, z_, c='black')
                                ax.add_line(line)
                            for traj_frame_idx in range(0,frame_idx_-1):
                                ax.scatter3D(x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 0], x_gt[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c='black', alpha=1.0, s=5)
                
                    else: # T > current time
                        frame_idx = frame_idx_ - data.input_seq.shape[2]
                        
                        # JRT post-processing
                        # if predType == 'jrt':
                        #     pred2print[scene_idx, agent_idx, frame_idx, 0, :]
                        
                        if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                        for agent_idx in range(pred2print.shape[1]):
                            if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                            xy_mean = np.concatenate((xy_mean, np.expand_dims(pred2print[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                            ax.scatter3D(pred2print[scene_idx, agent_idx, frame_idx, :, 0], pred2print[scene_idx, agent_idx, frame_idx, :, 1], pred2print[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1.0, s=5)
                            if bones is None: bones = TBI15_BONES
                            for edge in bones:
                                x_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 0], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                                y_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 1], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                                z_ = [pred2print[scene_idx, agent_idx, frame_idx, edge[0], 2], pred2print[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                                line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx], alpha=0.8)
                                ax.add_line(line)
                            for traj_frame_idx in range(0,frame_idx-1):
                                ax.scatter3D(pred2print[scene_idx, agent_idx, traj_frame_idx, 0, 0], pred2print[scene_idx, agent_idx, traj_frame_idx, 0, 1], 0, c=HUMAN_COLORS[agent_idx], alpha=1.0, s=5)
                            
                    ax.set_title(f'frame: {frame_idx_}')
                    
                    scene_center = [xy_mean[:,0].mean(), xy_mean[:,1].mean()]
                    scene_max = np.max((np.abs(xy_mean[:,0]-scene_center[0]).max(), np.abs(xy_mean[:,1]-scene_center[1]).max()))
                    # Plot horizontal plane
                    xx, yy = np.meshgrid(range(int(scene_center[0])-10, int(scene_center[0])+11), range(int(scene_center[1])-10, int(scene_center[1])+11))
                    z = (9 - xx - yy) *0
                    ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
                    scene_max = 2.1
                    # Plot xy grid lines
                    for ii in range(-8,10,2):
                        line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                        ax.add_line(line_x)
                    for jj in range(-8,10,2):
                        line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                        ax.add_line(line_x)
                    ax.set_xlim3d([scene_center[0]-(scene_max), scene_center[0]+(scene_max)])
                    ax.set_ylim3d([scene_center[1]-(scene_max), scene_center[1]+(scene_max)])
                    ax.set_zlim3d([0, 2.5])
                    # ax.set_xlabel("x")
                    # ax.set_ylabel("y")
                    # ax.set_zlabel("z")
                    ax.azim=azimuth
                    ax.elev=44
                    frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_JRDB_v3', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}_{predType}_azimuth{azimuth}.png'
                    plt.savefig(frame_save_name, bbox_inches='tight', dpi=150)
                    # plt.savefig(frame_save_name, bbox_inches='tight', dpi=150)
                    plt.close()
                    plt.cla()

def viz_joint_onlyPred(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}'))
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'gifs')): os.makedirs(os.path.join(output_dir, 'viz_results', 'gifs'))
    
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                    ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=0.5, s=5)
                    
                    if bones is None:
                        if x_gt.shape[3] == 15: bones = TBI15_BONES
                        if x_gt.shape[3] == 13: bones = EDGES_3DPW
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_pred[scene_idx, agent_idx, frame_idx, :, 0], y_pred[scene_idx, agent_idx, frame_idx, :, 1], y_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=0.5, s=5)
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)
                
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [y_gt[scene_idx, :, :, :, 0].mean(), y_gt[scene_idx, :, :, :, 0].mean()]
            scene_max = np.max((np.abs(y_gt[scene_idx, :, :, :, 0]).max(), np.abs(y_gt[scene_idx, :, :, :, 1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max*1.2), scene_center[0]+(scene_max*1.2)])
            ax.set_ylim3d([scene_center[1]-(scene_max*1.2), scene_center[1]+(scene_max*1.2)])
            ax.set_zlim3d([0, 3])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}') + f'/PRED_scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.close()
            plt.cla()
        gif_save_dir = os.path.join(output_dir, 'viz_results', 'gifs')
        frame_save_dir = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')
        save_as_gif_v2(gif_save_dir, frame_save_dir, TOTAL_T, batch_idx, scene_idx_list)

def viz_joint_onlyGT(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint'))
    # pred: B*N, T, 15, 3
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}'))
    # if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'gifs')): os.makedirs(os.path.join(output_dir, 'viz_results', 'gifs'))
    
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                continue
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_gt[scene_idx, agent_idx, frame_idx, :, 0], y_gt[scene_idx, agent_idx, frame_idx, :, 1], y_gt[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=0.5, s=5)
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [y_gt[scene_idx, :, :, :, 0].mean(), y_gt[scene_idx, :, :, :, 0].mean()]
            scene_max = np.max((np.abs(y_gt[scene_idx, :, :, :, 0]).max(), np.abs(y_gt[scene_idx, :, :, :, 1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max*1.2), scene_center[0]+(scene_max*1.2)])
            ax.set_ylim3d([scene_center[1]-(scene_max*1.2), scene_center[1]+(scene_max*1.2)])
            ax.set_zlim3d([0, 3])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}') + f'/GT_scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.close()
            plt.cla()
        # gif_save_dir = os.path.join(output_dir, 'viz_results', 'gifs')
        # frame_save_dir = os.path.join(output_dir, 'viz_results', 'joint', f'batch_{batch_idx}')
        # save_as_gif_v2(gif_save_dir, frame_save_dir, TOTAL_T, batch_idx, scene_idx_list)
    
def viz_joint_jansang(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    # pred: B*N, T, 15, 3
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_jansang')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_jansang'))
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}'))
    if bones is None:
        if x_gt.shape[3] == 15: bones = TBI15_BONES
        if x_gt.shape[3] == 13: bones = EDGES_3DPW
    scene_idx_list = []
    for scene_idx in range(B):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(0, TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                    ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=0.5, s=5)
                    
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
                        
                    loop_frame_idx = frame_idx_-1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75,0,np.min((max_loop, frame_idx_)))
                    while loop_idx < 10 and loop_frame_idx >=0:
                        if padding_mask_fut[scene_idx, agent_idx, loop_frame_idx] == False: continue
                        ax.scatter3D(x_gt[scene_idx, agent_idx, loop_frame_idx, :, 0], x_gt[scene_idx, agent_idx, loop_frame_idx, :, 1], x_gt[scene_idx, agent_idx, loop_frame_idx, :, 2], c='black', alpha=loop_alpha[loop_idx], s=5)
                        for edge in bones:
                            x_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 0], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 0]]
                            y_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 1], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 1]]
                            z_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 2], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c='black', alpha=loop_alpha[loop_idx])
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_pred[scene_idx, agent_idx, frame_idx, :, 0], y_pred[scene_idx, agent_idx, frame_idx, :, 1], y_pred[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1, s=5)
                    
                    for edge in bones:
                        x_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)
                        
                    loop_frame_idx = frame_idx-1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75,0,np.min((max_loop, frame_idx)))
                    while loop_idx < 10 and loop_frame_idx >=0:
                        if padding_mask_fut[scene_idx, agent_idx, loop_frame_idx] == False: continue
                        ax.scatter3D(y_pred[scene_idx, agent_idx, loop_frame_idx, :, 0], y_pred[scene_idx, agent_idx, loop_frame_idx, :, 1], y_pred[scene_idx, agent_idx, loop_frame_idx, :, 2], c='black', alpha=loop_alpha[loop_idx], s=5)
                        for edge in bones:
                            x_ = [y_pred[scene_idx, agent_idx, loop_frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, loop_frame_idx, edge[1], 0]]
                            y_ = [y_pred[scene_idx, agent_idx, loop_frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, loop_frame_idx, edge[1], 1]]
                            z_ = [y_pred[scene_idx, agent_idx, loop_frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, loop_frame_idx, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx], alpha=loop_alpha[loop_idx])
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1
                
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [y_gt[scene_idx, :, :, :, 0].mean(), y_gt[scene_idx, :, :, :, 0].mean()]
            scene_max = np.max((np.abs(y_gt[scene_idx, :, :, :, 0]).max(), np.abs(y_gt[scene_idx, :, :, :, 1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max*1.2), scene_center[0]+(scene_max*1.2)])
            ax.set_ylim3d([scene_center[1]-(scene_max*1.2), scene_center[1]+(scene_max*1.2)])
            ax.set_zlim3d([0, 3])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.close()
            plt.cla()
            
def viz_joint_jansang_v2(pred, gt, data, output_dir, batch_idx, bones=None):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    # pred: B*N, T, 15, 3
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_jansang')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_jansang'))
    B, N, T, XYZ_3 = data.output_seq.shape
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3)[:,:,:,0].cpu().numpy()
    y_gt = gt.reshape(B, N, T-1, XYZ_3//3, 3).cpu().numpy()
    y_pred = pred.reshape(B, N, T-1, XYZ_3//3, 3).cpu().detach().numpy()
    
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    TOTAL_T = data.input_seq.shape[2] + data.output_seq.shape[2]-1
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3//3, 3).detach().cpu().numpy()[:,:,1:]
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}'))
    if bones is None:
        if x_gt.shape[3] == 15: bones = TBI15_BONES
        if x_gt.shape[3] == 13: bones = EDGES_3DPW
    scene_idx_list = []
    for scene_idx in range(B):
        # if scene_idx != 6: continue
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx_ in range(0, TOTAL_T):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            if frame_idx_ < data.input_seq.shape[2]: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx_].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx_] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx_, 0, :][:2], axis=0)), axis=0)
                    ax.scatter3D(x_gt[scene_idx, agent_idx, frame_idx_, :, 0], x_gt[scene_idx, agent_idx, frame_idx_, :, 1], x_gt[scene_idx, agent_idx, frame_idx_, :, 2], c='black', alpha=0.5, s=5)
                    
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx_, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx_, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
                        
                    loop_frame_idx = frame_idx_-1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75,0,np.min((max_loop, frame_idx_)))
                    while loop_idx < 10 and loop_frame_idx >=0:
                        if padding_mask_fut[scene_idx, agent_idx, loop_frame_idx] == False: continue
                        ax.scatter3D(x_gt[scene_idx, agent_idx, loop_frame_idx, :, 0], x_gt[scene_idx, agent_idx, loop_frame_idx, :, 1], x_gt[scene_idx, agent_idx, loop_frame_idx, :, 2], c='black', alpha=loop_alpha[loop_idx], s=5)
                        for edge in bones:
                            x_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 0], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 0]]
                            y_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 1], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 1]]
                            z_ = [x_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 2], x_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c='black', alpha=loop_alpha[loop_idx])
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1
            
            else: # T > current time
                frame_idx = frame_idx_ - data.input_seq.shape[2]
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter3D(y_gt[scene_idx, agent_idx, frame_idx, :, 0], y_gt[scene_idx, agent_idx, frame_idx, :, 1], y_gt[scene_idx, agent_idx, frame_idx, :, 2], c='black', alpha=1, s=5)
                    
                    for edge in bones:
                        x_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)

                    loop_frame_idx = frame_idx-1
                    max_loop = 10
                    loop_idx = 0
                    loop_alpha = np.linspace(0.75,0,np.min((max_loop, frame_idx)))
                    while loop_idx < 10 and loop_frame_idx >=0:
                        if padding_mask_fut[scene_idx, agent_idx, loop_frame_idx] == False: continue
                        ax.scatter3D(y_gt[scene_idx, agent_idx, loop_frame_idx, :, 0], y_gt[scene_idx, agent_idx, loop_frame_idx, :, 1], y_gt[scene_idx, agent_idx, loop_frame_idx, :, 2], c='black', alpha=loop_alpha[loop_idx], s=5)
                        for edge in bones:
                            x_ = [y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 0]]
                            y_ = [y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 1]]
                            z_ = [y_gt[scene_idx, agent_idx, loop_frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, loop_frame_idx, edge[1], 2]]
                            line = Line3D(x_, y_, z_, c='black', alpha=loop_alpha[loop_idx])
                            ax.add_line(line)
                        loop_frame_idx -= 1
                        loop_idx += 1
                
            ax.set_title(f'frame: {frame_idx_}')
            
            scene_center = [y_gt[scene_idx, :, :, :, 0].mean(), y_gt[scene_idx, :, :, :, 0].mean()]
            scene_max = np.max((np.abs(y_gt[scene_idx, :, :, :, 0]).max(), np.abs(y_gt[scene_idx, :, :, :, 1]).max()))
            # Plot horizontal plane
            xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))
            z = (9 - xx - yy) *0
            ax.plot_surface(xx, yy, z, color='0.5', alpha=0.06, zorder=0)
            
            # Plot xy grid lines
            for ii in range(-8,10,2):
                line_x = Line3D([scene_center[0]+ii,scene_center[0]+ii], [scene_center[1]-9,scene_center[1]+9], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            for jj in range(-8,10,2):
                line_x = Line3D([scene_center[0]-9,scene_center[0]+9], [scene_center[1]+jj,scene_center[1]+jj], [0,0], c='grey', alpha=0.5, zorder=1)
                ax.add_line(line_x)
            ax.set_xlim3d([scene_center[0]-(scene_max*1.2), scene_center[0]+(scene_max*1.2)])
            ax.set_ylim3d([scene_center[1]-(scene_max*1.2), scene_center[1]+(scene_max*1.2)])
            ax.set_zlim3d([0, 3])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_jansang', f'batch_{batch_idx}') + f'/scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx_).zfill(3)}.png'
            plt.savefig(frame_save_name, bbox_inches='tight', dpi=300)
            plt.close()
            plt.cla()
    

def viz_pose_sequence_3d(output_dir, y_pred, data, batch_idx, comment=None, bones=None):
    ''' y_pred: B X N X T X J X 3'''
    print(f'Visualizing pose sequence of batch {batch_idx}')
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    B, N, T, XYZ_3 = data.output_seq.shape
    xy_sizes = [4.5]
    x_gt = data.input_seq.reshape(B, N, -1, XYZ_3/3, 3).detach().cpu().numpy()
    y_gt = data.output_seq.reshape(B, N, -1, XYZ_3/3, 3).detach().cpu().numpy()
    padding_mask_past = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,:-(T-1)].cpu().numpy()
    padding_mask_fut = ~data.padding_mask.reshape(B, N, data.padding_mask.shape[1])[:,:,-(T-1):].cpu().numpy()
    y_pred = y_pred.reshape(B, N, T-1, XYZ_3//3, 3)
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'joint_frames', f'batch_{batch_idx}')): os.makedirs(os.path.join(output_dir, 'viz_results', 'joint_frames', f'batch_{batch_idx}'))
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'gifs')): os.makedirs(os.path.join(output_dir, 'viz_results', 'gifs'))
    
    scene_idx_list = []
    for scene_idx in range(0, B, 2):
        scene_idx_list.append(scene_idx)
        xy_mean = np.zeros((0, 2))
        if padding_mask_fut[scene_idx].sum() <= 1: continue
        for frame_idx in range(y_pred.shape[2]):
            plt.close()
            plt.cla()
            fig = plt.figure(figsize=(20, 9))
            ax = fig.add_subplot(111, projection='3d')
            x_axis_x, x_axis_y, x_axis_z = [0,1.5], [0,0], [0,0]
            y_axis_x, y_axis_y, y_axis_z = [0,0], [0,1.5], [0,0]
            z_axis_x, z_axis_y, z_axis_z = [0,0], [0,0], [0,1.5]
            line_x = Line3D(x_axis_x, x_axis_y, x_axis_z, c='red')
            line_y = Line3D(y_axis_x, y_axis_y, y_axis_z, c='green')
            line_z = Line3D(z_axis_x, z_axis_y, z_axis_z, c='blue')
            ax.add_line(line_x)
            ax.add_line(line_y)
            ax.add_line(line_z)
            
            if frame_idx < T-1: # T < current time
                if padding_mask_past[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(x_gt.shape[1]):
                    if padding_mask_past[scene_idx, agent_idx, frame_idx] == False: continue
                    xy_mean = np.concatenate((xy_mean, np.expand_dims(x_gt[scene_idx, agent_idx, frame_idx, 0, :][:2], axis=0)), axis=0)
                    ax.scatter(x_gt[scene_idx, agent_idx, frame_idx, :, 0], x_gt[scene_idx, agent_idx, frame_idx, :, 1], x_gt[scene_idx, agent_idx, frame_idx, :, 2], 'k.')
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [x_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], x_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [x_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], x_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [x_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], x_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c='black')
                        ax.add_line(line)
            
            else: # T > current time
                if padding_mask_fut[scene_idx, :, frame_idx].sum() < 1: continue
                for agent_idx in range(y_pred.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter(y_pred[scene_idx, agent_idx, frame_idx, :, 0], y_pred[scene_idx, agent_idx, frame_idx, :, 1], y_pred[scene_idx, agent_idx, frame_idx, :, 2], 'k.')
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 0], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 1], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_pred[scene_idx, agent_idx, frame_idx, edge[0], 2], y_pred[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)
                
                for agent_idx in range(y_gt.shape[1]):
                    if padding_mask_fut[scene_idx, agent_idx, frame_idx] == False: continue
                    # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
                    ax.scatter(y_gt[scene_idx, agent_idx, frame_idx, :, 0], y_gt[scene_idx, agent_idx, frame_idx, :, 1], y_gt[scene_idx, agent_idx, frame_idx, :, 2], 'k.')
                    
                    if bones is None: bones = TBI15_BONES
                    for edge in bones:
                        x_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 0], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 0]]
                        y_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 1], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 1]]
                        z_ = [y_gt[scene_idx, agent_idx, frame_idx, edge[0], 2], y_gt[scene_idx, agent_idx, frame_idx, edge[1], 2]]
                        line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
                        ax.add_line(line)
            
            for xy_size in xy_sizes:                    
                ax.set_xlim3d([xy_mean.mean(0)[0]-xy_size, xy_mean.mean(0)[0]+xy_size])
                ax.set_ylim3d([xy_mean.mean(0)[1]-xy_size, xy_mean.mean(0)[1]+xy_size])
                ax.set_zlim3d([0, 3])
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                frame_save_name = os.path.join(output_dir, 'viz_results', 'joint_frames', f'batch_{batch_idx}') + f'/xySize_{xy_size}_scene_{str(scene_idx).zfill(3)}_frame_{str(frame_idx).zfill(3)}_{comment}.png'
                plt.savefig(frame_save_name, bbox_inches='tight')
        gif_save_dir = os.path.join(output_dir, 'viz_results', 'gifs')
        frame_save_dir = os.path.join(output_dir, 'viz_results', 'joint_frames', f'batch_{batch_idx}')
        save_as_gif(gif_save_dir, frame_save_dir, None, batch_idx, scene_idx_list)
        

def viz_pose_3d(y_pred, comment, frame, bones=None):
    ''' Input shape: N X T X J X 3'''
    # new_order = [2,0,1]
    # y_pred = y_pred[...,new_order]
    # Feet on ground
    # for agent_idx in range(y_pred.shape[0]):
    #     y_pred[agent_idx,:,2] = y_pred[agent_idx,:,2] - y_pred[agent_idx,:,2].min())
    y_pred = y_pred[:,frame,:,:]
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = y_pred[:,:,0], y_pred[:,:,1], y_pred[:,:,2]
    for agent_idx in range(y_pred.shape[0]):
        # ax.scatter(z[agent_idx], x[agent_idx], y[agent_idx], 'y.')
        ax.scatter(x[agent_idx], y[agent_idx], z[agent_idx], 'y.')
        for joint_idx in range(x.shape[1]):
            ax.text(x[agent_idx, joint_idx], y[agent_idx, joint_idx], z[agent_idx, joint_idx], str(joint_idx), None)
        
        x_axis_x, x_axis_y, x_axis_z = [0,1.5], [0,0], [0,0]
        y_axis_x, y_axis_y, y_axis_z = [0,0], [0,1.5], [0,0]
        z_axis_x, z_axis_y, z_axis_z = [0,0], [0,0], [0,1.5]
        line_x = Line3D(x_axis_x, x_axis_y, x_axis_z, c='red')
        line_y = Line3D(y_axis_x, y_axis_y, y_axis_z, c='green')
        line_z = Line3D(z_axis_x, z_axis_y, z_axis_z, c='blue')
        ax.add_line(line_x)
        ax.add_line(line_y)
        ax.add_line(line_z)

        if bones is None: bones = TBI15_BONES
        for edge in bones:
            x_ = [y_pred[agent_idx, edge[0], 0], y_pred[agent_idx, edge[1], 0]]
            y_ = [y_pred[agent_idx, edge[0], 1], y_pred[agent_idx, edge[1], 1]]
            z_ = [y_pred[agent_idx, edge[0], 2], y_pred[agent_idx, edge[1], 2]]
            line = Line3D(x_, y_, z_, c=HUMAN_COLORS[agent_idx])
            ax.add_line(line)
            # ax.plot(x_, y_, z_, zdir='z', c='black')
    
    ax.set_xlim3d([-3, 3])
    ax.set_ylim3d([-3, 3])
    ax.set_zlim3d([0, 3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if isinstance(comment, list):
        for i in range(1,len(comment)):
            comment[0] += f'_{comment[i]}'
        comment = comment[0]
    plt.savefig(f'/ssd4tb/jaewoo/t2p/t2p/traj_viz/1019test/{comment}_frame_{frame}.png', bbox_inches='tight')
    plt.cla()
    
def save_as_gif(gif_save_dir, frame_save_dir, gifFrames, batch_idx, scene_idx_list):
    imgs = glob.glob(frame_save_dir+'/*.png')
    imgs = sorted(imgs)
    sizes = [int(os.basename(imgName)[0]) for imgName in imgs]
    sizes = np.unique(np.array(sizes))
    frames_ = [[] for i in sizes]
    for img in imgs:
        temp = Image.open(img)
        image = temp.copy()
        for size_idx, size in enumerate(sizes):
            if f'xySize_{size}' in img: frames_[size_idx].append(image)
        temp.close()

    for size_idx, size in enumerate(sizes):
        frames = frames_[size_idx]
        if gifFrames is None: gifFrames = len(frames)
        for batchIdx in range(len(frames)//gifFrames):
            gifFilename = f'batch_{batch_idx}_{size}by{size}_frames_{batchIdx*gifFrames}_{(batchIdx+1)*gifFrames}.gif'
            frames[0 + int(batchIdx*gifFrames)].save(os.path.join(gif_save_dir, gifFilename), save_all=True, append_images=frames[(batchIdx*gifFrames)+1:(batchIdx+1)*gifFrames], optimize=False, duration=100, loop=0)

def save_as_gif_v2(gif_save_dir, frame_save_dir, gifFrames, batch_idx, scene_idx_list):
    imgs = glob.glob(frame_save_dir+'/*.png')
    imgs = sorted(imgs)
    frames_ = []
    for img in imgs:
        temp = Image.open(img)
        image = temp.copy()
        frames_.append(image)
        temp.close()

    frames = frames_
    if gifFrames is None: gifFrames = len(frames)
    for batchIdx in range(len(frames)//gifFrames):
        gifFilename = f'batch_{batch_idx}_frames_{batchIdx*gifFrames}_{(batchIdx+1)*gifFrames}.gif'
        frames[0 + int(batchIdx*gifFrames)].save(os.path.join(gif_save_dir, gifFilename), save_all=True, append_images=frames[(batchIdx*gifFrames)+1:(batchIdx+1)*gifFrames], optimize=False, duration=100, loop=0)
