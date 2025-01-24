import os
import os.path as osp
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union
import random
random.seed(4)

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import torch

from multiprocessing import Process
from multiprocessing import Pool
from itertools import repeat

from scipy.spatial.distance import cdist
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.dataset import files_exist
from tqdm import tqdm
import pickle as pkl
# from shapely.geometry import LineString, Point
import random
import glob

import sys
sys.path.append('/ssd4tb/jaewoo/t2p/t2p')
from utils import TemporalData
from debug_util import *

PED_CLASS = {}
VEH_CLASS = {}
SPLIT_NAME = {'mocap_UMPM': {'train': 'train', 'val': 'val', 'test': 'val'}, 'jrdb': {'train': 'train', 'val': 'val', 'test': 'val'}}
RAW_FILE_NAMES_JRDB = {'train': '/ssd4tb/jaewoo/t2p/parsed_jrdb/jrdb_bev_v2', 'val': '/ssd4tb/jaewoo/t2p/parsed_jrdb/jrdb_bev_v2', 'test': '/ssd4tb/jaewoo/t2p/parsed_jrdb/jrdb_bev_v2'}
RAW_FILE_NAMES_UMPM = {'train': 'data/Mocap_UMPM/train_3_75_mocap_umpm.npy', 'val': 'data/Mocap_UMPM/test_3_75_mocap_umpm.npy', 'test': 'data/Mocap_UMPM/test_3_75_mocap_umpm.npy'}
RAW_FILE_NAMES_MUPOTS = {'train': 'data/Mocap_UMPM/mupots_150_3persons.npy', 'val': 'data/Mocap_UMPM/test_3_75_mocap_umpm.npy', 'test': 'data/Mocap_UMPM/test_3_75_mocap_umpm.npy'}

class jrdb(Dataset):

    def __init__(self,
                 split: str,
                 root: str,
                 process_dir: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 process: bool = False,
                 spec_args: Dict = None) -> None:
        self._split = split
        self._local_radius = local_radius
        for k,v in spec_args.items():
            self.__setattr__(k, v)

        self._directory = SPLIT_NAME[self.dataset][split]
        self.process_dir = process_dir

        self._raw_file_names = sorted(glob.glob(RAW_FILE_NAMES_JRDB[split] + '/*.pt'))
        random.shuffle(self._raw_file_names)
        print(self._raw_file_names[:10])
        if split == 'train': self._raw_file_names = self._raw_file_names[:(len(self._raw_file_names)*85)//100]  # 85% as train
        else: self._raw_file_names = self._raw_file_names[(len(self._raw_file_names)*85)//100 :]                # 15% as val
            
        self._processed_file_names = [str(f) + '.pt' for f in range(len(self._raw_file_names))]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(jrdb, self).__init__(root, transform=transform)

    def _download(self):
        return
    
    # @property
    # def raw_dir(self) -> str:
    #     return os.path.join(self.root, self.version)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.process_dir, self._directory)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def _process(self):

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Found processed files')
            return

        print('Processing...', file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print('Done!', file=sys.stderr)

    def process(self) -> None:

        if self.n_jobs > 1:
            raw_file_list = []
            total_num = len(self._raw_file_names)
            num_per_proc = int(np.ceil(total_num / self.n_jobs))
            for proc_id in range(self.n_jobs):
                start = proc_id*num_per_proc
                end = min((proc_id+1)*num_per_proc, total_num)
                raw_file_list.append(self._raw_file_names[start:end])

            procs = []
            for proc_id in range(self.n_jobs):
                process = Process(target=self.process_jrdb, args=(raw_file_list[proc_id],self._local_radius))
                process.daemon = True
                process.start()
                procs.append(process)

            for proc in procs:
                proc.join()

        else:
            self.process_jrdb(self._raw_file_names, self._local_radius)

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(self.processed_paths[idx])
        if self._split == 'train':
            data = self.augment(data) 
        return data

    def augment(self, data):
        if self.random_flip:
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([-1,1])
                data.y = data.y * torch.tensor([-1,1])
                data.positions = data.positions * torch.tensor([-1,1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(theta_y, -1*theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(angle_y, -1*angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(lane_angle_y, -1*lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([-1,1])
                data.lane_vectors = data.lane_vectors * torch.tensor([-1,1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([-1,1])
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([1,-1])
                data.y = data.y * torch.tensor([1,-1])
                data.positions = data.positions * torch.tensor([1,-1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(-1*theta_y, theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(-1*angle_y, angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(-1*lane_angle_y, lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([1,-1])
                data.lane_vectors = data.lane_vectors * torch.tensor([1,-1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([1,-1])

        return data

    
    def process_jrdb(self, tokens: str,
                         radius: float) -> Dict:

        for token in tqdm(tokens):  # token: N_sequences, N_agents, T, 45 에서 N_sequences 의 index
            raw_file = torch.load(token)
            N, T, N_JOINTS, _ = raw_file['data'].shape
            seq_data = raw_file['data'].float()
            body_xyz = seq_data.clone()
            # seq_data = seq_data[:,:,:,:2]       # Only x,y (2D)
            # input_traj, output_traj = torch.tensor(seq_data[:, :self.ref_time, 0], dtype=torch.float), torch.tensor(seq_data[:, self.ref_time:, 0], dtype=torch.float)     # Only hip joint info
            input_traj, output_traj = seq_data[:, :self.ref_time, 0], seq_data[:, self.ref_time:, 0]     # Only hip joint info
            edge_index = torch.LongTensor(list(permutations(range(N), 2))).t().contiguous()
            # edge_index = edge_index.unsqueeze(0).repeat(num_batches, 1, 1)
            x = torch.cat((input_traj, output_traj), dim= 1)
            positions = x.clone()
            x[:,self.ref_time:] = x[:,self.ref_time:] - x[:,self.ref_time-1].unsqueeze(-2)
            x[:,1:self.ref_time] = x[:,1:self.ref_time] - x[:,:self.ref_time-1]
            x[:,0] = torch.zeros(N, 3)
            y = x[:,self.ref_time:]
            
            padding_mask = raw_file['padding']
            bos_mask = torch.zeros(N, self.ref_time, dtype=torch.bool).cuda()
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1: self.ref_time] = padding_mask[:, : self.ref_time-1] & ~padding_mask[:, 1: self.ref_time]
            
            rotate_angles = torch.zeros(N, dtype=torch.float)

            for actor_id in range(N):
                heading_vector = x[actor_id, self.ref_time] - x[actor_id, self.ref_time-1]
                rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0])
            
            if self.rotate:
                rotate_mat = torch.empty(N, 3, 3)
                sin_vals = torch.sin(rotate_angles)
                cos_vals = torch.cos(rotate_angles)
                rotate_mat[:, 0, 0] = cos_vals
                rotate_mat[:, 0, 1] = -sin_vals
                rotate_mat[:, 0, 2] = 0
                rotate_mat[:, 1, 0] = sin_vals
                rotate_mat[:, 1, 1] = cos_vals
                rotate_mat[:, 1, 2] = 0
                rotate_mat[:, 2, 0] = 0
                rotate_mat[:, 2, 1] = 0
                rotate_mat[:, 2, 2] = 1
                if y is not None:
                    y = torch.bmm(y, rotate_mat)
            else:
                rotate_mat = None
            
            processed = {
                'body_xyz': body_xyz,
                'x': x[:,:self.ref_time],
                'positions': positions,
                'rotate_angles': rotate_angles,
                'padding_mask': padding_mask,
                'edge_index': edge_index,
                'bos_mask': bos_mask,
                'y': y,
                'num_nodes': N,
                'rotate_mat': rotate_mat
            }
            data = TemporalData(**processed)
            
            torch.save(data, os.path.join(self.processed_dir, os.path.split(token)[1]))
        return 

    
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def normalize_angle(angle):
    if angle < -torch.pi:
        angle += 2*torch.pi
    elif angle > torch.pi:
        angle -= 2*torch.pi
    return angle

if __name__ == '__main__':
    # spec_args = {'dataset': 'nuScenes', 'n_jobs': 32, 't_h': 2, 't_f': 6, 'res': 2, 'ref_time':4, 'lseg_len': 10, 'lseg_angle_thres': 30, 'lseg_dist_thres': 2.5}
    # A1D = nuScenesDataset('mini_val', root='data/nuScenes', process_dir='preprocessed/nuScenes_frm',
    #                          version='v1.0-mini', process=True, spec_args=spec_args)

    spec_args = {'dataset': 'jrdb', 'n_jobs': 0, 't_h': 2, 't_f': 6, 'res': 2, 'ref_time':15, 'lseg_len': 10, 'lseg_angle_thres': 30, 'lseg_dist_thres': 2.5, 'random_flip': True, 'rotate': True}
    # A1D = mocap_UMPM('val', process_dir='preprocessed/mocap_UMPM_input25', root='data/Mocap_UMPM',
    #                          process=True, spec_args=spec_args)
    A1D = jrdb('train', process_dir='/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_input25', root='/ssd4tb/jaewoo/t2p/parsed_jrdb/jrdb_bev_v2',
                             process=True, spec_args=spec_args)
    
    from torch_geometric.data.batch import Batch
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    # import warnings
    # warnings.filterwarnings('error')
    
    # for data in tqdm(A1D):
    #     pass

    # bs=64
    # for i in tqdm(range(0, len(A1D), bs)):
    #     try:
    #         Batch.from_data_list(A1D[i:i+bs])
    #     except:
    #         for data in A1D[i:i+bs]:
    #             print(data)
    #         a=1

    # bs=64
    # for i in tqdm(range(0, len(A1D), bs)):
    #     Batch.from_data_list(A1D[i:i+bs])

    # Dloader = DataLoader(A1D, batch_size=bs, shuffle=False,
    #                       num_workers=1, pin_memory=True,
    #                       persistent_workers=False)
    # for data in tqdm(Dloader):
    #     a=1
    #     pass

    
    # from debug_util import viz_data_goal
    # # data0 = A1D[0]
    # # viz_data_goal(data0, 'tmp_viz')
    # for data0 in A1D:
    #     viz_data_goal(data0, 'tmp_viz')