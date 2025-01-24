from torch_geometric.data import Dataset
import torch
import numpy as np
import copy
import open3d as o3d
import glob

class T2PDataset(Dataset):
    def __init__(self, dataset, mode=0, device='cuda', transform=None, input_time=None):
        self.dataset = dataset
        if dataset == "mocap_umpm":
            self.num_person = 3
            if mode==0: # Train
                if input_time == 50: 
                    self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/mocap_UMPM/train/*.pt')
                elif input_time == 25:
                    self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/mocap_UMPM_input25_v2/train/*.pt')
            else:   # Val, Test
                if input_time == 50:
                    self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/mocap_UMPM/val/*.pt')
                elif input_time == 25:
                    self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/mocap_UMPM_input25_v2/val/*.pt')
        elif dataset == "3dpw":
            self.num_person = 2
            if input_time != 10: raise Exception('Input time step other than 10 is not implemented yet')
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/3dpw_input10_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/3dpw_input10_v2/val/*.pt')
        elif dataset == "jrdb_BEV":
            self.num_person = 25    # Max num person
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/cmhtp/t2p/preprocessed/jrdb_bev_v3_input_3.2_output_4.8_frameRep_6_FPS_2_5_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/cmhtp/t2p/preprocessed/jrdb_bev_v3_input_3.2_output_4.8_frameRep_6_FPS_2_5_v2/val/*.pt')
        elif dataset == "jrdb":
            self.num_person = 20    # Max num person
            if input_time != 15: raise Exception('Input time step other than 25 is not implemented yet')
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_input15_maxPerson20/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_input15_maxPerson20/val/*.pt')              
        elif dataset == "jrdb_1_2_25":
            self.num_person = 25    # Max num person
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_1_output_2_frameRep_1_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_1_output_2_frameRep_1_v2/val/*.pt')              
        elif dataset == "jrdb_1_2_25_15":
            self.num_person = 25    # Max num person
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_1_output_2_frameRep_15_v3/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_1_output_2_frameRep_15_v3/val/*.pt')
        elif dataset == "jrdb_2_4_25":
            self.num_person = 25    # Max num person
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_2_output_4_frameRep_1_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_2_output_4_frameRep_1_v2/val/*.pt')
        elif dataset == "jrdb_2_5_25":
            self.num_person = 25    # Max num person
            if mode==0:
                # self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_2_output_5_frameRep_3_test/train/*.pt')
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v3_input_2_output_5_frameRep_3/train/*.pt')
            elif mode==1:
                # self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_2_output_5_frameRep_3_test/val/*.pt')
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v3_input_2_output_5_frameRep_3/val/*.pt')
        elif dataset == "jrdb_3_6_25":
            self.num_person = 25    # Max num person
            if mode==0:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_3_output_6_frameRep_1_v2/train/*.pt')
            elif mode==1:
                self.data = glob.glob('/ssd4tb/jaewoo/t2p/t2p/preprocessed/jrdb_bev_v2_input_3_output_6_frameRep_1_v2/val/*.pt')
        elif dataset == "mupots":  # two modes both for evaluation
            raise Exception('Not implemented yet!')
            if mode == 0:
                self.data = np.load(
                    'data/MuPoTs3D/mupots_150_2persons.npy')[:,:,::2,:]
            if mode==1:
                self.data = np.load('data/MuPoTs3D/mupots_150_3persons.npy')[:,:,::2,:]
        # if dataset == "3dpw":
        #     if mode == 1:
        #         self.data = np.load(
        #             '/home/ericpeng/DeepLearning/Projects/MotionPrediction/MRT_nips2021/pose3dpw/test_2_3dpw.npy')
        elif dataset == "mix1":
            raise Exception('Not implemented yet!')
            if mode == 1:
                self.data = np.load('data/mix1_6persons.npy')
        elif dataset == "mix2":
            raise Exception('Not implemented yet!')
            if mode == 1:
                self.data = np.load('data/mix2_10persons.npy')
        self.data = sorted(self.data)
        self.len_ = len(self.data)
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.input_time = input_time
        super(T2PDataset, self).__init__(transform=transform)

    def get(self, idx):
        data = torch.load(self.data[idx])
        if self.transform:   # radomly rotate the scene for augmentation
            idx_ = np.random.randint(0, 3)
            rot = [np.pi, np.pi/2, np.pi/4, np.pi*2]
            points = data.body_xyz.numpy().reshape(-1, 3)
            # 读取点
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 点旋转
            pcd_EulerAngle = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((0, rot[idx_], 0))
            pcd_EulerAngle.rotate(R1)  # 不指定旋转中心
            pcd_EulerAngle.paint_uniform_color([0, 0, 1])
            data['body_xyz'] = torch.tensor(np.asarray(pcd_EulerAngle.points).reshape(-1, 75, 45))
            # data = np.asarray(pcd_EulerAngle.points).reshape(-1, 75, 45)

        input_seq = data.body_xyz[:, :self.input_time, :]
        output_seq = data.body_xyz[:, self.input_time:, :]

        input_seq = torch.as_tensor(input_seq, dtype=torch.float32)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32)
        # input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(self.device)
        # output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(self.device)
        last_input = input_seq[:, -1:, :]
        output_seq = torch.cat([last_input, output_seq], dim=1)
        data['input_seq'] = input_seq.reshape(input_seq.shape[0]//self.num_person, self.num_person, input_seq.shape[1], -1)
        data['output_seq'] = output_seq.reshape(output_seq.shape[0]//self.num_person, self.num_person, output_seq.shape[1], -1)
        
        data.bos_mask = data.bos_mask.cpu()
        # for key in data.keys:
        #     if torch.is_tensor(data[key]):
        #         data[key] = data[key].cuda()
        return data

    def len(self):
        return self.len_
    
    