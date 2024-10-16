import numpy as np
from scipy.spatial.transform import Rotation as R_
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
from PIL import Image
from io import BytesIO
import sys
sys.path.append('/mnt/jaewoo4tb/textraj')
from bev.model import BEV
from bev.cfg import bev_settings
from bev.post_parser import *
from preprocess_utils import *

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
rotate_x = np.radians(90)
rotate_y = np.radians(90)
rotate_z = np.radians(90)
rotate_z_ = np.radians(72)
RX = Rx(rotate_x)
RZ = Rz(rotate_z)
RY = Ry(rotate_y)
RZ_ = Rz(rotate_z_)
R = RZ
# R = RX * RZ

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

def rotate_coordinates(local_coords, theta):
    """
    Rotates the local coordinates of a point based on the robot's rotation around the z-axis.

    :param local_coords: tuple (x', y'), the local coordinates of the person
    :param theta: float, rotation angle in degrees
    :return: tuple (x, y), the global coordinates after rotation
    """
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta)

    # Rotation matrix for counterclockwise rotation around the z-axis
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Convert local coordinates to a numpy array
    local_coords_array = np.array(local_coords)

    # Calculate global coordinates after rotation
    new_local_coords = np.dot(rotation_matrix, local_coords_array)

    return new_local_coords

class Agent:
    def __init__(self, id):
        self.id = id
        self.type = None
        self.local_pos = np.array([0, 0], dtype=np.float32)
        self.global_pos = np.array([0, 0, 0], dtype=np.float32)
        self.bbox = [0, 0, 0, 0]
        self.caption = None
        self.rot_z = 0
        self.pose = None        
        self.visible = None
        self.robot_ori = None
        self.robot_pos = None

class Socials:
    def __init__(self):
        self.clusters = {}  # key: cluster ID, value: (caption |str|, list of agent IDs |int|)
        self.interactions = {} # key: (agent_id, agent_id), value: |str| of interaction description. Order of agent_id doesn't matter
        
class Pose3DEngine:
    def __init__(self):
        self.HORIZONTAL_MARGIN = 100
        self.CROP_IMG_DIR = "/mnt/jaewoo4tb/textraj/temp/crop_imgs"
        
        print("Loading 3D pose regressor")
        default_cfg = bev_settings()
        self.pose_model = BEV(default_cfg)
        self.smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)
        
        print('Caption3DPoseEngine initialized.')        


    def _crop_img_single(self, img, bbox, draw_bbox=True, hori_margin=None):
        h, w, c = img.shape
        if draw_bbox:
            cv2.rectangle(img, bbox[:2], [bbox[0] + bbox[2], bbox[1] + bbox[3]], (0, 0, 255), 2)

        big_img = np.hstack([img, img, img]) # hstack to handle bbox that overflows the boundary, may need to modify images for smooth stitching
        if hori_margin is None:
            x_start = bbox[0] + w - self.HORIZONTAL_MARGIN
            x_end = x_start + bbox[2] + 2 * self.HORIZONTAL_MARGIN
        else:
            x_start = bbox[0] + w - hori_margin
            x_end = x_start + bbox[2] + 2 * hori_margin
        
        return big_img[:, x_start:x_end]

    def load_files(self, label_2d, label_3d, img_dir, social):
        self.label_2d = json.load(open(label_2d))["labels"]
        self.label_3d = json.load(open(label_3d))["labels"]
        self.label_social = json.load(open(social))["labels"]
        self.img_dir = img_dir
        
    
    def preprocess_frame(self, target_frame, global_position, global_ori):
        ''' Process location '''
        self.target_frame = f'{target_frame:06d}'
        self.img = cv2.imread(self.img_dir + self.target_frame + ".jpg")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.agents_temp, self.agents, self.socials = {}, {}, Socials()
        frame_2d = self.label_2d[self.target_frame + ".jpg"]
        frame_3d = self.label_3d[self.target_frame + ".pcd"]
        frame_3d_dict = {}
        for frame_3d_i in frame_3d:
            frame_3d_dict[int(frame_3d_i['label_id'].split(":")[1])] = frame_3d_i

        # Process visible agents
        self.visible_agents = []
        for agent_2d in frame_2d:
            if agent_2d["attributes"]["occlusion"] == "Severely_occluded" or agent_2d["attributes"]["occlusion"] == "Fully_occluded": #neglect occluded agents
                continue
            
            id = int(agent_2d["label_id"].split(":")[1])
            self.visible_agents.append(id)
            bbox = agent_2d["box"]
            temp = Agent(id)
            temp.type = agent_2d["label_id"].split(":")[0]
            temp.visible = True
            temp.bbox = bbox
            self.agents_temp[id] = temp
                
        for temp_id in self.agents_temp.keys():
            if temp_id not in frame_3d_dict.keys(): continue
            temp = self.agents_temp[temp_id]
            agent_3d = frame_3d_dict[temp.id]
            local_pos_temp = np.array((agent_3d["box"]["cx"], agent_3d["box"]["cy"]), dtype=np.float32)
            local_pos_temp = rotate_coordinates(local_pos_temp, global_ori)
            temp.local_pos = local_pos_temp
            temp.global_pos[:2] = local_pos_temp + global_position[:2]
            # print(f'id: {temp.id}, pos: {temp.global_pos[:2]}')
            temp.rot_z = agent_3d["box"]["rot_z"]
            temp.observation_angle = agent_3d["observation_angle"]
            temp.robot_ori = global_ori
            temp.robot_pos = global_position[:2]
            self.agents[temp_id] = temp
        

    def regress_3dpose(self):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            if int(agent.id) in self.visible_agents and np.linalg.norm(agent.local_pos)<5 and np.linalg.norm(agent.local_pos)>0.75:
                bbox = agent.bbox
                temp_img = self.img.copy()
                crop_img = self._crop_img_single(temp_img, bbox, draw_bbox=False, hori_margin=35)
                pose = self.pose_model.forward_parse(crop_img)
                if pose is not None:
                    if pose['verts'].shape[0] > 1: 
                        choose_agent = np.argmin(np.abs(pose['cam'][:,0]))
                        pose['smpl_thetas'] = np.expand_dims(pose['smpl_thetas'][choose_agent], 0)
                        pose['smpl_betas'] = np.expand_dims(pose['smpl_betas'][choose_agent], 0)
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -90, 'x')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    b_ori_theta = torch.atan2(b_ori[0, 1], b_ori[0, 0]) * (180/np.pi)
                    rot_z = (agent.rot_z * (180/np.pi))
                    robot_ori = agent.robot_ori
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta - rot_z + robot_ori + 180, 'z')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    pose.update({'verts': verts, 'joints': joints, 'smpl_face':face, 'b_ori':b_ori})
                    agent.pose = pose
                    agent.global_pos = agent.global_pos - [0, 0, joints[:,:,2].min().cpu().numpy()]

def get_b_ori(joints):
    x_axis = joints[:, 2, :] - joints[:, 1, :]
    z_axis = joints[:, 0, :] - joints[:, 12, :]
    # x_axis[:, -1] = 0
    # z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device='cuda').repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(x_axis.cuda(), z_axis.cuda(), dim=-1)
    b_ori = y_axis[:, :3]  # body forward dir of GAMMA is y axis
    return b_ori

def rotate_smpl_thetas_toLidar(smpl_thetas):
    smpl_thetas[0][0] = smpl_thetas[0][0]-(np.pi/2)
    return smpl_thetas

def rotate_smpl_thetas_rotateZ(smpl_thetas, b_ori, rot_z, robot_ori):
    robot_ori = robot_ori / 180.0
    b_ori = torch.atan2(b_ori[0, 1], b_ori[0, 0])
    smpl_thetas[0][1] = smpl_thetas[0][1]-np.pi
    return smpl_thetas

def rotate_root(pose, rotation_angle_degrees, axis):
    # Convert the angle to radians
    if torch.is_tensor(rotation_angle_degrees): 
        if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
    pose = pose.squeeze()
    rotation_angle_radians = np.radians(rotation_angle_degrees)
    rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
    root_rotation_matrix = R_.from_rotvec(pose[:3]).as_matrix()
    new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
    new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
    new_pose = np.copy(pose)
    new_pose[:3] = new_root_rotation_vector
    return np.expand_dims(new_pose, 0)

def to_verb(verb):
    if verb == 'conversation': return 'having conversation'
    else: return verb

if __name__ == "__main__":
    target_frame = 20
    target_label_2d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_2d_stitched/bytes-cafe-2019-02-07_0.json"
    target_label_3d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_3d/bytes-cafe-2019-02-07_0.json"
    img_dir = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/images/image_stitched/bytes-cafe-2019-02-07_0/"
    test = Pose3DEngine()
    test.load_files(target_label_2d, target_label_3d, img_dir)
    test.preprocess_frame(target_frame) 

    test.caption_single()
    test.caption_dual()

    for agent in test.agents:  
        print(str(agent.id), agent.caption)
    
    for pair in test.pairs:
        print(str(pair["pair"][0].id), "<->", str(pair["pair"][1].id), pair["caption"])