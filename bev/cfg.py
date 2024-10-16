import sys
import argparse
import os.path as osp
import os
import torch
from romp.utils import download_model

model_dict = {
    1: 'BEV_ft_agora.pth',
    2: 'BEV.pth',
}
model_id = 2
conf_dict = {1:[0.25, 20, 2], 2:[0.1, 20, 1.6]}
long_conf_dict = {1:[0.12, 20, 1.5, 0.46], 2:[0.08, 20, 1.6, 0.8]}


def bev_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description = 'ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('-m', '--mode', type=str, default='image', help = 'Inferece mode, including image, video, webcam')
    parser.add_argument('--model_id', type=int, default=2, help = 'Whether to process the input as a long image, sliding window way')
    parser.add_argument('-i', '--input', type=str, default='/ssd4tb/jaewoo/t2p/jrdb/train_dataset/images/image_0/jordan-hall-2019-04-22_0/000092.jpg', help = 'Path to the input image / video')
    parser.add_argument('-o', '--save_path', type=str, default=osp.join('/mnt/jaewoo4tb/textraj/romp/files/','BEV_results'), help = 'Path to save the results')
    parser.add_argument('--crowd', action='store_false', help = 'Whether to process the input as a long image, sliding window way')
    parser.add_argument('--GPU', type=int, default=0, help = 'The gpu device number to run the inference on. If GPU=-1, then running in cpu mode')

    parser.add_argument('--overlap_ratio', type=float, default=long_conf_dict[model_id][3], help = 'The frame_rate of saved video results')
    parser.add_argument('--center_thresh', type=float, default=conf_dict[model_id][0], help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--nms_thresh', type=float, default=conf_dict[model_id][1], help = 'The 2D-pose-projection similarity threshold of suppressing overlapping predictions.')
    parser.add_argument('--relative_scale_thresh', type=float, default=conf_dict[model_id][2], help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--show_largest', action='store_true', help = 'Whether to show the largest person only')
    parser.add_argument('--show_patch_results', action='store_true', help = 'During processing long image, whether to show the results of intermediate results of each patch.')
    parser.add_argument('--calc_smpl', action='store_false', help = 'Whether to calculate the smpl mesh from estimated SMPL parameters')
    parser.add_argument('--renderer', type=str, default='sim3dr', help = 'Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast), open3d (webcam)')
    parser.add_argument('--render_mesh', action='store_false', help = 'Whether to render the estimated 3D mesh mesh to image')
    parser.add_argument('--show', action='store_true', help = 'Whether to show the rendered results')
    parser.add_argument('--show_items', type=str, default='mesh,mesh_bird_view,pj2d', help = 'The items to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf,rotate_mesh. splited with ,')
    parser.add_argument('--save_video', action='store_true', help = 'Whether to save the video results')
    parser.add_argument('--frame_rate', type=int, default=24, help = 'The frame_rate of saved video results')
    parser.add_argument('--smpl_path', type=str, default=osp.join('/mnt/jaewoo4tb/textraj/romp/files/.romp','SMPLA_NEUTRAL.pth'), help = 'The path of SMPL-A model file')
    parser.add_argument('--smil_path', type=str, default=osp.join('/mnt/jaewoo4tb/textraj/romp/files/.romp','smil_packed_info.pth'), help = 'The path of SMIL model file')
    parser.add_argument('--model_path', type=str, default=osp.join('/mnt/jaewoo4tb/textraj/romp/files/.romp',model_dict[model_id]), help = 'The path of BEV checkpoint')

    # not support temporal processing now
    parser.add_argument('-t', '--temporal_optimize', action='store_true', help = 'Whether to use OneEuro filter to smooth the results')
    parser.add_argument('-sc','--smooth_coeff', type=float, default=3., help = 'The smoothness coeff of OneEuro filter, the smaller, the smoother.')
    parser.add_argument('--webcam_id',type=int, default=0, help = 'The Webcam ID.')

    # Parsing args
    parser.add_argument('--imgs_dir', type=str, default='/ssd4tb/jaewoo/t2p/jrdb/train_dataset/images/image_0/', help = 'Path to the input image / video')
    parser.add_argument('--parsed_dir', type=str, default='/ssd4tb/jaewoo/t2p/preprocessed/jrdb', help = 'Path to save parsed data')
    parser.add_argument('--split_by', type=int, default=3)
    parser.add_argument('--split_by_idx', type=int, default=0)
    parser.add_argument('--scene_idx', type=int, default=0)
    args = parser.parse_args(input_args)
    
    if args.model_id != 2:
        args.model_path = osp.join('/mnt/jaewoo4tb/textraj/romp/files/','.romp',model_dict[args.model_id])
        args.center_thresh = conf_dict[args.model_id][0]
        args.nms_thresh = conf_dict[args.model_id][1]
        args.relative_scale_thresh = conf_dict[model_id][2]
    if not torch.cuda.is_available():
        args.GPU = -1
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True
    if not os.path.exists(args.smpl_path):
        print('please prepare SMPL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation')
        #smpl_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/smpla_packed_info.pth'
        #download_model(smpl_url, args.smpl_path, 'SMPL-A')
    if not os.path.exists(args.smil_path):
        print('please prepare SMIL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation')
        #smil_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/smil_packed_info.pth'
        #download_model(smil_url, args.smil_path, 'SMIL')
    if not os.path.exists(args.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/'+model_dict[model_id]
        download_model(romp_url, args.model_path, 'BEV')
    if args.crowd:
        args.center_thresh = long_conf_dict[args.model_id][0]
        args.nms_thresh = long_conf_dict[args.model_id][1]
        args.relative_scale_thresh = long_conf_dict[model_id][2]
        args.overlap_ratio = long_conf_dict[args.model_id][3]
    
    return args