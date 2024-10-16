import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append('/mnt/jaewoo4tb/t2p/')
from utils.utils import *
from itertools import permutations
from itertools import product
from scipy.spatial.transform import Rotation as R_
import os
from HiVT.utils_hivt import TemporalData



class PreProcess:
    def __init__(self, agent_data: dict, scene_name: str, object_data: dict=None):
        self.MAX_OBJECT_NUM = 20
        self.MAX_CLUSTER_NUM = 25
        self.FRAME_LENGTH = 20
        self.FRAME_STEP = 6
        self.PAST_FRAME = 8
        self.FUT_FRAME = 12
        self.MAX_NUM_AGENTS = 40
        self.TEMP_NUM_AGENTS = 100
        self.scene_name = scene_name

        self.agent_data = agent_data
        self.object_data = object_data
        
    def __call__(self, initial_frame):
        # print(f"start preprocessing from frame {initial_frame} to {initial_frame + (self.FRAME_LENGTH - 1) * self.FRAME_STEP}")
        scene = {}

        agent_position = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 2))
        robot_position = torch.zeros(size=(1, self.FRAME_LENGTH, 2))
        agent_pose = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 72))
        agent_pose_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        agent_mask = torch.ones(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        b_ori = torch.zeros(size=(self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, 2))

        agent_id_to_idx = {}

        frames = torch.IntTensor(size=(self.FRAME_LENGTH, ))
        
        for frame in range(self.FRAME_LENGTH):
            actual_frame = initial_frame + frame * self.FRAME_STEP
            frames[frame] = actual_frame
            for agent_id, value in self.agent_data[actual_frame].items():
                robot_position[0, frame] = torch.tensor(value['robot_pos'][:2])
                if agent_id not in agent_id_to_idx:
                    agent_id_to_idx[agent_id] = len(agent_id_to_idx) 
                agent_save_idx = agent_id_to_idx[agent_id]

                agent_position[agent_save_idx, frame, :] = torch.from_numpy(value["global_position"][:2])
                agent_mask[agent_save_idx, frame] = False

                if value["pose"] is None:
                    agent_pose[agent_save_idx, frame, :] = torch.zeros(size=(72,))
                else:
                    agent_pose[agent_save_idx, frame, :] = torch.from_numpy(value["pose"]["smpl_thetas"])
                    b_ori[agent_save_idx, frame, :] = value["pose"]["b_ori"].cpu()[0,:2]
                    agent_pose_mask[agent_save_idx, frame] = False

            # check if all frame have 3 agnets with pose
            if torch.sum(agent_pose_mask[:, frame]) < 2:
                print(f"Invalid data frames: only {torch.sum(agent_pose_mask[frame])} pose data at frame {actual_frame}")
                return
            
        x, positions = agent_position.clone(), agent_position.clone()
        
        padding_mask_all = agent_mask
        
        bos_mask = torch.zeros(self.TEMP_NUM_AGENTS, self.PAST_FRAME, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask_all[:, 0]
        bos_mask[:, 1: self.PAST_FRAME] = padding_mask_all[:, : self.PAST_FRAME-1] & ~padding_mask_all[:, 1: self.PAST_FRAME]
        
        rotate_angles = torch.zeros(self.TEMP_NUM_AGENTS, dtype=torch.float)
        valid_indices = [torch.nonzero(row[:self.PAST_FRAME], as_tuple=True)[0] for row in ~padding_mask_all]
        valid_pose_indices = [torch.nonzero(row[:self.PAST_FRAME], as_tuple=True)[0] for row in ~agent_pose_mask]
        
        for actor_id in range(self.TEMP_NUM_AGENTS):
            if len(valid_indices[actor_id]) < 2: continue   # If zero or one frame of observation, skip
            if (valid_indices[actor_id][1:]-valid_indices[actor_id][:-1]).min()>1: continue # If no consecutive frames, skip
            if len(valid_pose_indices[actor_id]) > 0 and valid_pose_indices[actor_id].max() > self.PAST_FRAME-3:
                heading_vector = b_ori[actor_id, valid_pose_indices[actor_id].max()]
                rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0])
            else:
                heading_vector = x[actor_id, valid_indices[actor_id][-1]] - x[actor_id, valid_indices[actor_id][-2]]
                rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0]) 
                
        x[:,self.PAST_FRAME:] = x[:,self.PAST_FRAME:] - x[:,self.PAST_FRAME-1].unsqueeze(-2)
        x[:,1:self.PAST_FRAME] = x[:,1:self.PAST_FRAME] - x[:,:self.PAST_FRAME-1]
        x[:,0] = torch.zeros(x.shape[0], 2)
        padding_mask_all_temp = padding_mask_all.clone()
        padding_mask_all_temp[:,1:self.PAST_FRAME] = padding_mask_all[:,1:self.PAST_FRAME] | padding_mask_all[:,:self.PAST_FRAME-1]
        x[padding_mask_all_temp] = 0
        y = x[:,self.PAST_FRAME:]
        
        rotate_mat = torch.empty(self.TEMP_NUM_AGENTS, 3, 3)
        sin_vals = torch.sin(rotate_angles)
        cos_vals = torch.cos(rotate_angles)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals      # original: -
        rotate_mat[:, 0, 2] = 0
        rotate_mat[:, 1, 0] = sin_vals     # original: +
        rotate_mat[:, 1, 1] = cos_vals
        rotate_mat[:, 1, 2] = 0
        rotate_mat[:, 2, 0] = 0
        rotate_mat[:, 2, 1] = 0
        rotate_mat[:, 2, 2] = 1
        if y is not None:
            y = torch.bmm(y, rotate_mat[:, :2, :2]) 
        
        agent_mask_true = (~agent_mask[:,:self.PAST_FRAME]).sum(-1) > (self.PAST_FRAME//2)      # Enough frames for past (True is valid data)
        num_valid_agents = agent_mask_true.sum()
        rotate_mat_ = torch.empty(num_valid_agents, 3, 3)
        positions_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        x_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        y_ = torch.Tensor(size=(num_valid_agents, self.FUT_FRAME, 2))
        agent_pose_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 72))
        agent_pose_mask_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        padding_mask_all_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        bos_mask_ = torch.zeros(num_valid_agents, self.PAST_FRAME, dtype=torch.bool)
        rotate_angles_ = torch.zeros(num_valid_agents, dtype=torch.float)
        
        rotate_mat_[:agent_mask_true.sum()] = rotate_mat[agent_mask_true]
        rotate_angles_[:agent_mask_true.sum()] = rotate_angles[agent_mask_true]
        positions_[:agent_mask_true.sum()] = positions[agent_mask_true]
        # agent_position_[:agent_mask_true.sum()] = agent_position[agent_mask_true]
        agent_pose_[:agent_mask_true.sum()] = agent_pose[agent_mask_true]
        agent_pose_mask_[:agent_mask_true.sum()] = agent_pose_mask[agent_mask_true]
        padding_mask_all_[:agent_mask_true.sum()] = padding_mask_all[agent_mask_true]
        bos_mask_[:agent_mask_true.sum()] = bos_mask[agent_mask_true]
        x_[:agent_mask_true.sum()] = x[agent_mask_true]
        y_[:agent_mask_true.sum()] = y[agent_mask_true]
        
        edge_index = torch.LongTensor(list(permutations(range(num_valid_agents), 2))).t().contiguous()
        
        scene = {
                # "frames": frames, 
                'num_nodes': agent_mask_true.sum(),
                'rotate_mat': rotate_mat_,
                'scene': self.scene_name,
                'x': x_[:,:self.PAST_FRAME],
                'x_pose': agent_pose_[:, :self.PAST_FRAME],
                'x_pose_mask': agent_pose_mask_[:, :self.PAST_FRAME],
                'positions': positions_,
                'rotate_angles': rotate_angles_,
                'padding_mask': padding_mask_all_, # position masking
                'edge_index': edge_index,
                'bos_mask': bos_mask_,
                'y': y_,
                'y_pose': agent_pose_[:, self.PAST_FRAME:],
                'y_pose_mask': agent_pose_mask_[:, self.PAST_FRAME:],
                'robot_pos': robot_position,
                }

        data = TemporalData(**scene)
        return data, (agent_mask_true).sum()
        # return scene, (agent_mask_true).sum()
        
    def rotate_root(pose, rotation_angle_degrees, axis):
        # Convert the angle to radians
        if torch.is_tensor(rotation_angle_degrees): 
            if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
        pose = pose.squeeze()
        rotation_angle_radians = rotation_angle_degrees
        rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
        root_rotation_matrix = R_.from_rotvec(pose[:, :3]).as_matrix()
        new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
        new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
        new_pose = np.copy(pose)
        new_pose[:, :3] = new_root_rotation_vector
        return np.expand_dims(new_pose, 0)

if __name__ == "__main__":
    from glob import glob
    parser = argparse.ArgumentParser(description="Preprocessing script for dataset.")
    
    parser.add_argument("--scene_idx_start", type=int, default=0,
                        help="Scene idx out of 27 to start.")
    parser.add_argument("--scene_idx_end", type=int, default=27,
                        help="Scene idx out of 27 to end.")
    args = parser.parse_args()

    print(f"starting 2nd preprocessing, start: {args.scene_idx_start} / end: {args.scene_idx_end}.")
    save_root = "/mnt/jaewoo4tb/t2p/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot"
    datas = glob("/mnt/jaewoo4tb/t2p/preprocessed_1st/jrdb_v1/*.pt")
    datas.sort()
    max_num_agents = 0
    for scene_count, i in enumerate(range(len(datas))):
        if scene_count < args.scene_idx_start or scene_count > args.scene_idx_end: continue
        agent_data = torch.load(datas[i])
        scene_name = datas[i][43:-3]
        print(f"Processing scene {scene_name}")
        
        preprocess = PreProcess(agent_data, scene_name)
        frame_range = list(agent_data.keys())
        frame_range.sort()
        
        try:
            os.makedirs(f"{save_root}/{scene_name}")
        except:
            pass
        
        for initial_frame in tqdm(range(frame_range[-1] - preprocess.FRAME_STEP * (preprocess.FRAME_LENGTH - 1))):
            result, num_agents = preprocess(initial_frame)
            if num_agents > max_num_agents: max_num_agents = num_agents
            if result is not None:
                torch.save(result, f"{save_root}/{scene_name}/{initial_frame}.pt")
        print(f"Max num agents: {max_num_agents}")