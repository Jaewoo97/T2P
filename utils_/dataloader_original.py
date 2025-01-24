import torch.utils.data as data
import torch
import numpy as np
import copy
import open3d as o3d


class Data(data.Dataset):
    def __init__(self, dataset, mode=0, device='cuda', transform=False, opt=None):
        if dataset == "mocap_umpm":
            if mode==0:
                self.data = np.load('data/Mocap_UMPM/train_3_75_mocap_umpm.npy')
            else:
                self.data = np.load('data/Mocap_UMPM/test_3_75_mocap_umpm.npy')
        if dataset == "mupots":  # two modes both for evaluation
            if mode == 0:
                self.data = np.load(
                    'data/MuPoTs3D/mupots_150_2persons.npy')[:,:,::2,:]
            if mode==1:
                self.data = np.load('data/MuPoTs3D/mupots_150_3persons.npy')[:,:,::2,:]
        # if dataset == "3dpw":
        #     if mode == 1:
        #         self.data = np.load(
        #             '/home/ericpeng/DeepLearning/Projects/MotionPrediction/MRT_nips2021/pose3dpw/test_2_3dpw.npy')
        if dataset == "mix1":
            if mode == 1:
                self.data = np.load('data/mix1_6persons.npy')
        if dataset == "mix2":
            if mode == 1:
                self.data = np.load('data/mix2_10persons.npy')

        self.len = len(self.data)
        self.device = device
        self.dataset = dataset
        self.transform = transform
        self.input_time = opt.input_time


    def __getitem__(self, index):
        data = self.data[index]
        
        if self.transform:   # radomly rotate the scene for augmentation
            idx = np.random.randint(0, 3)
            rot = [np.pi, np.pi/2, np.pi/4, np.pi*2]
            points = self.data[index].reshape(-1, 3)
            # 读取点
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # 点旋转
            pcd_EulerAngle = copy.deepcopy(pcd)
            R1 = pcd.get_rotation_matrix_from_xyz((0, rot[idx], 0))
            pcd_EulerAngle.rotate(R1)  # 不指定旋转中心
            pcd_EulerAngle.paint_uniform_color([0, 0, 1])
            data = np.asarray(pcd_EulerAngle.points).reshape(-1, 75, 45)

        input_seq = data[:, :self.input_time, ]
        output_seq = data[:, self.input_time:, :]

        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(self.device)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(self.device)
        last_input = input_seq[:, -1:, :]
        output_seq = torch.cat([last_input, output_seq], dim=1)
        input_seq = input_seq.reshape(input_seq.shape[0], input_seq.shape[1], -1)
        output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], -1)

        return input_seq, output_seq

    def __len__(self):
        return self.len



