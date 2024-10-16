# function for suppress print
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from typing import List, Optional, Tuple

from torch_geometric.data import Data
import torch.nn.functional as F
# from torch_geometric.data import Data

# function for suppress print
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
# from torch_geometric.data import Data
def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 
        
# Rotation stuff
def axis_angle_to_matrix(axis_angle):
    axis = axis_angle[:3]
    angle = np.linalg.norm(axis)
    if angle > 0:
        axis = axis / angle  # Normalize the axis
    return R.from_rotvec(axis * angle).as_matrix()

def get_heading_direction(theta):
    """
    Get the heading direction from the SMPL theta parameters.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.

    Returns:
        heading_direction: (3,) array representing the heading direction.
    """
    theta = theta.reshape(-1, 3)
    # Extract the root joint's rotation (usually the first joint in SMPL)
    root_rotation_axis_angle = theta[0]
    
    # Convert the root joint's axis-angle representation to a rotation matrix
    root_rotation_matrix = axis_angle_to_matrix(root_rotation_axis_angle)
    
    # Define the forward direction vector (assume [1, 0, 0] for X-forward)
    forward_vector = np.array([1, 0, 0])
    
    # Apply the root joint's rotation to the forward vector
    heading_direction = root_rotation_matrix @ forward_vector
    
    return heading_direction.reshape(-1)

def matrix_to_axis_angle(matrix):
    rot = R.from_matrix(matrix)
    return rot.as_rotvec()

def apply_z_rotation_on_theta(theta, z_rotation_angle):
    """
    Apply a rotation around the Z-axis to the SMPL theta parameter.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.
        z_rotation_angle: rotation angle around the Z-axis in radians.

    Returns:
        (N, 3) array of updated axis-angle representations.
    """
    theta = theta.reshape(-1, 3)
    z_rotation_matrix = R.from_euler('z', z_rotation_angle).as_matrix()
    
    updated_theta = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        rotation_matrix = axis_angle_to_matrix(theta[i])
        new_rotation_matrix = z_rotation_matrix @ rotation_matrix
        new_axis_angle = matrix_to_axis_angle(new_rotation_matrix)
        updated_theta[i] = new_axis_angle
    
    return updated_theta.reshape(-1)



def stopPrint(func, *args, **kwargs):
    with open(os.devnull,"w") as devNull:
        original = sys.stdout
        sys.stdout = devNull
        func(*args, **kwargs)
        sys.stdout = original 
        
# Rotation stuff
def axis_angle_to_matrix(axis_angle):
    axis = axis_angle[:3]
    angle = np.linalg.norm(axis)
    if angle > 0:
        axis = axis / angle  # Normalize the axis
    return R.from_rotvec(axis * angle).as_matrix()

def get_heading_direction(theta):
    """
    Get the heading direction from the SMPL theta parameters.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.

    Returns:
        heading_direction: (3,) array representing the heading direction.
    """
    theta = theta.reshape(-1, 3)
    # Extract the root joint's rotation (usually the first joint in SMPL)
    root_rotation_axis_angle = theta[0]
    
    # Convert the root joint's axis-angle representation to a rotation matrix
    root_rotation_matrix = axis_angle_to_matrix(root_rotation_axis_angle)
    
    # Define the forward direction vector (assume [1, 0, 0] for X-forward)
    forward_vector = np.array([1, 0, 0])
    
    # Apply the root joint's rotation to the forward vector
    heading_direction = root_rotation_matrix @ forward_vector
    
    return heading_direction.reshape(-1)

def matrix_to_axis_angle(matrix):
    rot = R.from_matrix(matrix)
    return rot.as_rotvec()

def apply_z_rotation_on_theta(theta, z_rotation_angle):
    """
    Apply a rotation around the Z-axis to the SMPL theta parameter.

    Args:
        theta: (N, 3) array of axis-angle representations for N joints.
        z_rotation_angle: rotation angle around the Z-axis in radians.

    Returns:
        (N, 3) array of updated axis-angle representations.
    """
    theta = theta.reshape(-1, 3)
    z_rotation_matrix = R.from_euler('z', z_rotation_angle).as_matrix()
    
    updated_theta = np.zeros_like(theta)
    for i in range(theta.shape[0]):
        rotation_matrix = axis_angle_to_matrix(theta[i])
        new_rotation_matrix = z_rotation_matrix @ rotation_matrix
        new_axis_angle = matrix_to_axis_angle(new_rotation_matrix)
        updated_theta[i] = new_axis_angle
    
    return updated_theta.reshape(-1)

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from romp.vis_human.vis_utils import *
import copy
import trimesh
from PIL import Image
import pyrender
from pyrender import OffscreenRenderer, PerspectiveCamera, PointLight
from matplotlib import cm as cmx
from matplotlib import colors
from matplotlib.image import imsave
jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

frame_idx = "001000"
target_label_2d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_2d_stitched/bytes-cafe-2019-02-07_0.json"
target_label_3d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_3d/bytes-cafe-2019-02-07_0.json"
target_img = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/images/image_stitched/bytes-cafe-2019-02-07_0/" + frame_idx + ".jpg"

LIGHT_POSE_1 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_2 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5],
    [1.0, 0.0,           0.0,           5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_3 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, -5],
    [1.0, 0.0,           0.0,           5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_4 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5],
    [1.0, 0.0,           0.0,           -5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_5 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, -5],
    [1.0, 0.0,           0.0,           -5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSES = [LIGHT_POSE_1, LIGHT_POSE_2, LIGHT_POSE_3, LIGHT_POSE_4, LIGHT_POSE_5]

def create_thick_line_prism(start, end, thickness=0.1):
    # Compute direction vector and perpendicular vector
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    direction /= length  # Normalize direction
    perpendicular1 = np.array([-direction[1], direction[0], 0])  # Perpendicular vector in the XY plane
    perpendicular2 = np.array([direction[1], -direction[0], 0])  # Perpendicular vector in the XY plane
    
    # Define the half-thickness offset
    offset = thickness / 2
    
    # Vertices of the rectangular prism
    vertices = np.array([
        start + perpendicular1 * offset,  # Corner 1
        start - perpendicular1 * offset,  # Corner 2
        end - perpendicular1 * offset,    # Corner 3
        end + perpendicular1 * offset,     # Corner 4
        start + perpendicular2 * offset,  # Corner 5
        start - perpendicular2 * offset,  # Corner 6
        end - perpendicular2 * offset,    # Corner 7
        end + perpendicular2 * offset     # Corner 8
    ])
    
    # Define the faces of the rectangular prism
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Side faces
        [0, 1, 3], [0, 3, 2],  # Top face
        [4, 5, 6], [4, 6, 7],  # Bottom face
        [0, 1, 5], [0, 5, 4],  # Side faces
        [2, 3, 7], [2, 7, 6]   # Side faces
    ])
    
    # Create and return the Trimesh object
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def create_grid_lines_mesh(size=10, divisions=10, robot_pos=None):   
    # Initialize arrays for vertices and indices
    vertices = []
    faces = []
    start_x = int(robot_pos[0] / 10) * 10
    start_y = int(robot_pos[1] / 10) * 10
    # Create lines parallel to the X-axis
    for i in range(divisions + 1):
        # y = i * (size / divisions) + start_y
        y = i * (size / divisions) - size / 2 + start_y
        if abs(robot_pos[1] - (y + start_y))<5.01:
            start = [robot_pos[0] - 5, y + start_y, 0]
            end = [robot_pos[0] + 5, y + start_y, 0]
            prism = create_thick_line_prism(start, end, 0.05)
            start_index = len(vertices)
            vertices.extend(prism.vertices)
            # Adjust faces indices
            for face in prism.faces:
                faces.append([index + start_index for index in face])
    
    # Create lines parallel to the Y-axis
    for i in range(divisions + 1):
        # x = i * (size / divisions) + start_x
        x = i * (size / divisions) - size / 2 + start_x
        if abs(robot_pos[0] - (x + start_x))<5.01:
            start = [x + start_x, robot_pos[1] - 5, 0]
            end = [x + start_x, robot_pos[1] + 5, 0]
            prism = create_thick_line_prism(start, end, 0.05)
            start_index = len(vertices)
            vertices.extend(prism.vertices)
            # Adjust faces indices
            for face in prism.faces:
                faces.append([index + start_index for index in face])
                
    vertices = np.array(vertices)
    faces = np.array(faces)
    grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    grid_mesh.visual.vertex_colors[:, 3] = 255
    grid_mesh.visual.vertex_colors[:, :3] = [0,0,0]
    return grid_mesh

def create_arrow(start, end, shaft_radius=0.05, shaft_resolution=20, head_radius=0.1, head_length=0.2):
    vector = end - start
    if vector.shape[0] == 2: vector = np.append(vector, 0)
    else: vector = vector.squeeze()
    if torch.is_tensor(vector): vector = vector.cpu().numpy()
    length = np.linalg.norm(vector)

    # Create the shaft of the arrow
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=length - head_length, sections=shaft_resolution) 
    
    # Create the arrowhead
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=shaft_resolution)
    
    # Translate the arrowhead to the end of the shaft
    head.apply_translation([0, 0, length - head_length / 2])

    # Combine shaft and head
    arrow = trimesh.util.concatenate([shaft, head])

    # Rotate the arrow to align with the vector
    axis = np.array([0, 0, 1])  # Default arrow direction along z-axis
    if not np.allclose(vector, axis):  # Check if the vector is not aligned with z-axis
        rot_vector = np.cross(axis, vector)
        rot_vector = rot_vector / np.linalg.norm(rot_vector)
        angle = np.arccos(np.dot(axis, vector) / length)
        rotation = trimesh.transformations.rotation_matrix(angle, rot_vector)
        arrow.apply_transform(rotation)
    
    # Translate arrow to the start position
    start_ = start
    start_[2] = 0
    translation = trimesh.transformations.translation_matrix(start_)
    translation_matrix = np.eye(4)  # Start with an identity matrix
    translation_matrix[:2, 3] = translation[:2,2]
    arrow.apply_transform(translation_matrix)

    return arrow

def plot_2d_bbox(frame_idx, annot_2d, img, ax):

    try:
        frame_data_2d  = annot_2d["labels"][str(frame_idx).zfill(6) + ".jpg"]
    except:
        frame_data_2d  = annot_2d[str(frame_idx).zfill(6) + ".jpg"]
    # img = np.hstack([img, img])
    h, w, c = img.shape

    ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
    # ax.axis("off")

    occlusion_hide = True

    for person in frame_data_2d:
        box_coord = person["box"]
        caption = person["label_id"].split(":")[1]
        # caption = ""

        if occlusion_hide and (person["attributes"]["occlusion"] == "Severely_occluded" or person["attributes"]["occlusion"] == "Fully_occluded"):
            continue
    
        ax.add_patch(
        patches.Rectangle(
        box_coord[:2],                   # (x, y)
        box_coord[2],                    # width
        box_coord[3],                    # height
        edgecolor = 'red',
        fill=False,
        ))
        if box_coord[0] + box_coord[2] > w: # when bbox overflows the image
            ax.add_patch(
            patches.Rectangle(
            [box_coord[0] - w, box_coord[1]],                   # (x, y)
            box_coord[2],                    # width
            box_coord[3],                    # height
            edgecolor = 'red',
            fill=False,
            ))

        ax.text(box_coord[0], box_coord[1] - 10, caption, color='red', fontsize=5)

def plot_3d_pos(frame_idx, annot_3d, ax):
    try:
        frame_data_3d = annot_3d["labels"][str(frame_idx).zfill(6) + ".pcd"]
    except:
        frame_data_3d = annot_3d[str(frame_idx).zfill(6) + ".pcd"]

    cxs = []
    cys = []

    max = 0

    for person in frame_data_3d:
        cx, cy = person["box"]["cx"], person["box"]["cy"]
        caption = person["label_id"]
        ax.text(cx, cy + 0.3, caption, color="black", fontsize=5)
        cxs.append(cx)
        cys.append(cy)
        max_t = np.max(np.abs([cx, cy]))
        if max < max_t:
            max = max_t

    max += 0.5
    ax.scatter(cxs, cys, color="red")
    ax.set_xlim(-max, max)
    ax.set_ylim(-max, max)
    ax.set_aspect('equal')

def plot_text_annot(frame_idx, person, interaction, ax):
    x_position = 0.1
    y_start = 0.1
    line_spacing = 0.1
    fontsize = 7
    for line_idx, agent_id in enumerate(person.keys()):
        agent_descript = person[agent_id][frame_idx]
        agent_sentence = f'Agent {agent_id}: {agent_descript}'
        ax.text(x_position, y_start + line_idx * line_spacing, agent_sentence, fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
    # Set axis limits
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, line_spacing * len(person.keys()))
    ax.axis('off')

def plot_2d_human(frame_idx, person, img, save_dir=None):
    pass

def plot_3d_human(draw_frame_idx, person, ax, save_dir=None):
    scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 0.1]))
    offscreen_r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
    triangles, verts_tran = [], None
    draw_person_idx = 0
    assert draw_frame_idx in person.keys()
    person_count = 0
    if len(person[draw_frame_idx].keys()) == 0: return None
    for person_id in person[draw_frame_idx].keys():
        if person[draw_frame_idx][person_id]['pose'] == None: continue
        draw_person_idx += 1
        verts_tran = person[draw_frame_idx][person_id]['pose']['verts'].cpu()
        verts_tran[...,:3] = verts_tran[...,:3] + torch.tensor(person[draw_frame_idx][person_id]['global_position'][:3]).unsqueeze(0).unsqueeze(0)
        # else:
        #     verts_tran_ = person[draw_frame_idx][person_id]['pose']['verts']
        #     verts_tran_[...,:2] = verts_tran_[...,:2] - torch.tensor(person[draw_frame_idx][person_id]['global_position'][:2]).unsqueeze(0).unsqueeze(0)
        #     verts_tran = torch.cat((verts_tran, verts_tran_), dim=0)
        triangles.append(person[draw_frame_idx][person_id]['pose']['smpl_face'])
        
        # Add humans
        body_meshes = []
        m = trimesh.Trimesh(vertices=verts_tran[0], faces=triangles[0])
        m.visual.vertex_colors[:, 3] = 255
        colors = np.asarray(scalarMap.to_rgba(draw_person_idx / verts_tran.shape[0])[:3]) * 255
        colors = np.clip(colors * 1.5, 0, 255)  # Increase brightness by 50%, clamp to [0, 255]
        m.visual.vertex_colors[:, :3] = colors.astype(np.uint8)        
        body_meshes.append(m)
        body_mesh = pyrender.Mesh.from_trimesh(body_meshes, smooth=False)
        body_node = pyrender.Node(mesh=body_mesh, name='body')
        scene.add_node(body_node)
        person_count += 1
        # arrow = create_arrow(person[draw_frame_idx][person_id]['global_position'][:3], person[draw_frame_idx][person_id]['global_position'][:3] + \
        #     (person[draw_frame_idx][person_id]['pose']['b_ori'] / torch.norm(person[draw_frame_idx][person_id]['pose']['b_ori'])).cpu().numpy())
        # arrow_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(arrow), name='arrow')
        # scene.add_node(arrow_node)
        # Create a sphere using trimesh
        material = pyrender.MetallicRoughnessMaterial( metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=[1.0, 0.0, 0.0, 1.0])
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere_mesh_start = pyrender.Mesh.from_trimesh(sphere, material=material)
        translation_matrix_start = np.array([
            [1.0, 0.0, 0.0, person[draw_frame_idx][person_id]['global_position'][0]],  # x-axis translation
            [0.0, 1.0, 0.0, person[draw_frame_idx][person_id]['global_position'][1]],  # y-axis translation
            [0.0, 0.0, 1.0, person[draw_frame_idx][person_id]['global_position'][2]],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(sphere_mesh_start, pose = translation_matrix_start)
        material2 = pyrender.MetallicRoughnessMaterial( metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=[1.0, 1.0, 0.0, 1.0])
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere_mesh_end = pyrender.Mesh.from_trimesh(sphere, material=material2)
        person[draw_frame_idx][person_id]['pose']['b_ori'] = person[draw_frame_idx][person_id]['pose']['b_ori'].cpu()
        translation_matrix_end = np.array([
            [1.0, 0.0, 0.0, person[draw_frame_idx][person_id]['global_position'][0] + (person[draw_frame_idx][person_id]['pose']['b_ori'] / torch.norm(person[draw_frame_idx][person_id]['pose']['b_ori']))[0,0]],  # x-axis translation
            [0.0, 1.0, 0.0, person[draw_frame_idx][person_id]['global_position'][1] + (person[draw_frame_idx][person_id]['pose']['b_ori'] / torch.norm(person[draw_frame_idx][person_id]['pose']['b_ori']))[0,1]],  # y-axis translation
            [0.0, 0.0, 1.0, person[draw_frame_idx][person_id]['global_position'][2] + (person[draw_frame_idx][person_id]['pose']['b_ori'] / torch.norm(person[draw_frame_idx][person_id]['pose']['b_ori']))[0,2]],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(sphere_mesh_end, pose = translation_matrix_end)
        
    # Add floor
    floor = trimesh.creation.box(extents=np.array([10, 10, 0.02]),
                        transform=np.array([[1.0, 0.0, 0.0, person[draw_frame_idx][person_id]['robot_pos'][0]],
                                            [0.0, 1.0, 0.0, person[draw_frame_idx][person_id]['robot_pos'][1]],
                                            [0.0, 0.0, 1.0, -0.05],
                                            [0.0, 0.0, 0.0, 1.0],
                                            ]),
                        )
    floor.visual.vertex_colors = [0.3, 0.3, 0.3]
    floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
    # scene.add_node(floor_node)
    
    # Add other stuff
    point_l = PointLight(color=np.ones(3), intensity=40.0)
    for light_pose in LIGHT_POSES:
        light_pose_ = copy.deepcopy(light_pose)
        light_pose_[0, 3] = light_pose_[0, 3] + person[draw_frame_idx][person_id]['robot_pos'][0]
        light_pose_[1, 3] = light_pose_[1, 3] + person[draw_frame_idx][person_id]['robot_pos'][1]
        _ = scene.add(point_l, pose=light_pose_)
    cam = PerspectiveCamera(yfov=(np.pi / 3))
    cam_pose = np.array([
        [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5.5+person[draw_frame_idx][person_id]['robot_pos'][0]],
        [1.0, 0.0,           0.0,           0.0+person[draw_frame_idx][person_id]['robot_pos'][1]],
        [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 5],
        [0.0,  0.0,           0.0,          1.0]
    ])
    cam_node = scene.add(cam, pose=cam_pose)
    axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
    robot_position = np.array([
        [1.0, 0.0, 0.0, person[draw_frame_idx][person_id]['robot_pos'][0]],  # x-axis translation
        [0.0, 1.0, 0.0, person[draw_frame_idx][person_id]['robot_pos'][1]],  # y-axis translation
        [0.0, 0.0, 1.0, 0],  # z-axis translation
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # Add grid on floor
    # TODO: need to manage disappearing lines
    grid_lines_mesh = create_grid_lines_mesh(size=20, divisions=20, robot_pos=person[draw_frame_idx][person_id]['robot_pos'])
    scene.add(pyrender.Mesh.from_trimesh(grid_lines_mesh))

    scene.add(axis_mesh, pose=robot_position)
    color, depth = offscreen_r.render(scene)
    # print(f'Number of agents in frame {draw_frame_idx} is {person_count}')
    if save_dir != None:
        imsave(f"{save_dir}/frame_{draw_frame_idx}.jpeg", color)
        
    
def visualize_scene(frame_idx, img, annot_2d, annot_3d, person, interaction, save_dir):
    # fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    fig, axs = plt.subplots(4, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 2, 2, 2]})
    plot_2d_bbox(frame_idx, annot_2d, img, axs[0])
    plot_3d_pos(frame_idx, annot_3d, axs[1])
    plot_text_annot(frame_idx, person, interaction, axs[2])
    plot_3d_human(frame_idx, person, axs[3])
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir, dpi=300)

class TemporalData(Data):

    def __init__(self,
                num_nodes=None,
                rotate_mat=None,
                scene=None,
                x=None,
                x_pose=None,
                x_pose_mask=None,
                x_text=None,
                x_text_mask=None,
                x_interaction=None,
                x_interaction_mask=None,
                positions=None,
                rotate_angles=None,
                padding_mask=None,
                padding_mask_total=None,
                edge_index=None,
                bos_mask=None,
                y=None,
                y_pose=None,
                y_pose_mask=None,
                y_text=None,
                y_text_mask=None,
                y_interaction=None,
                y_interaction_mask=None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(num_nodes=num_nodes, rotate_mat=rotate_mat, scene=scene, x=x, x_pose=x_pose, x_pose_mask=x_pose_mask,
                x_text=x_text, x_text_mask=x_text_mask, x_interaction=x_interaction, x_interaction_mask=x_interaction_mask, positions=positions,
                rotate_angles=rotate_angles, padding_mask=padding_mask, padding_mask_total=padding_mask_total, edge_index=edge_index, bos_mask=bos_mask,
                y=y, y_pose=y_pose, y_pose_mask=y_pose_mask, y_text=y_text, y_text_mask=y_text_mask, y_interaction=y_interaction, y_interaction_mask=y_interaction_mask, **kwargs)
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    f1 = open(target_label_2d)
    annot_2d = json.load(f1)
    f2 = open(target_label_3d)
    annot_3d = json.load(f2)
    img = cv2.imread(target_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plot_2d_bbox(frame_idx, annot_2d, img, axs[0])
    plot_3d_pos(frame_idx, annot_3d, axs[1])

    plt.tight_layout() 
    plt.show()

    plt.savefig("/mnt/jaewoo4tb/textraj/temp/crop_imgs/test.png", dpi=300)
    print('finished.')
    
    
    