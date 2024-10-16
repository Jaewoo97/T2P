import torch
import torch.nn as nn
import sys
sys.path.append('/ssd4tb/jaewoo/t2p/ROMP/simple_romp')
sys.path.append('/home/user/anaconda3/envs/textraj/lib/python3.8/site-packages/bev/')
import numpy as np
from romp.model import HigherResolutionNet, BasicBlock
from bev.post_parser import CenterMap3D

from bev.post_parser import SMPLA_parser, body_mesh_projection2image, pack_params_dict,\
    suppressing_redundant_prediction_via_projection, remove_outlier, denormalize_cam_params_to_trans
from romp.utils import img_preprocess, create_OneEuroFilter, check_filter_state, \
    time_cost, download_model, determine_device, ResultSaver, WebcamVideoStream, \
    wait_func, collect_frame_path, progress_bar, smooth_results, convert_tensor2numpy, save_video_results
    
BN_MOMENTUM = 0.1

def get_3Dcoord_maps_halfz(size, z_base):
    range_arr = torch.arange(size, dtype=torch.float32)
    z_len = len(z_base)
    Z_map = z_base.reshape(1,z_len,1,1,1).repeat(1,1,size,size,1)
    Y_map = range_arr.reshape(1,1,size,1,1).repeat(1,z_len,1,size,1) / size * 2 -1
    X_map = range_arr.reshape(1,1,1,size,1).repeat(1,z_len,size,1,1) / size * 2 -1

    out = torch.cat([Z_map,Y_map,X_map], dim=-1)
    return out

def conv3x3_1D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_1D(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

def conv3x3_3D(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock_3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock_3D, self).__init__()
        self.conv1 = conv3x3_3D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_3D(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out

def get_cam3dmap_anchor(FOV, centermap_size):
    depth_level = np.array([1, 10, 20, 100], dtype=np.float32)
    map_coord_range_each_level = (np.array([2/64., 25/64., 3/64., 2/64.], dtype=np.float32) * centermap_size).astype(int)
    scale_level = 1/np.tan(np.radians(FOV/2.))/depth_level
    cam3dmap_anchor = []
    scale_cache = 8
    for scale, coord_range in zip(scale_level, map_coord_range_each_level):
        cam3dmap_anchor.append(scale_cache-np.arange(1,coord_range+1)/coord_range*(scale_cache-scale))
        scale_cache = scale
    cam3dmap_anchor = np.concatenate(cam3dmap_anchor)
    return cam3dmap_anchor

def convert_cam_params_to_centermap_coords(cam_params, cam3dmap_anchor):
    center_coords = torch.ones_like(cam_params)
    center_coords[:,1:] = cam_params[:,1:].clone()
    cam3dmap_anchors = cam3dmap_anchor.to(cam_params.device)[None]
    scale_num = len(cam3dmap_anchor)
    if len(cam_params) != 0:
        center_coords[:,0] = torch.argmin(torch.abs(cam_params[:,[0]].repeat(1, scale_num) - cam3dmap_anchors), dim=1).float()/128 * 2. - 1.
    
    return center_coords

def denormalize_center(center, size=128):
    center = (center+1)/2*size
    center = torch.clamp(center, 1, size-1).long()
    return center

class BEVv1(nn.Module):
    def __init__(self, **kwargs):
        super(BEVv1, self).__init__()
        print('Using BEV.')
        self.backbone = HigherResolutionNet()
        self._build_head()
        self._build_parser(conf_thresh=kwargs.get('center_thresh', 0.1))
    
    def _build_parser(self, conf_thresh=0.12):
        self.centermap_parser = CenterMap3D(conf_thresh=conf_thresh)

    def _build_head(self):
        params_num, cam_dim = 3+22*6+11, 3
        self.outmap_size = 128 
        self.output_cfg = {'NUM_PARAMS_MAP':params_num-cam_dim, 'NUM_CENTER_MAP':1, 'NUM_CAM_MAP':cam_dim}
        
        self.head_cfg = {'NUM_BASIC_BLOCKS':1, 'NUM_CHANNELS': 128}
        self.bv_center_cfg = {'NUM_DEPTH_LEVEL': self.outmap_size//2, 'NUM_BLOCK': 2}
        
        self.backbone_channels = self.backbone.backbone_channels
        self.transformer_cfg = {'INPUT_C':self.head_cfg['NUM_CHANNELS'], 'NUM_CHANNELS': 512}
        self._make_transformer()
        
        self.cam3dmap_anchor = torch.from_numpy(get_cam3dmap_anchor(60, self.outmap_size)).float()
        self.register_buffer('coordmap_3d', get_3Dcoord_maps_halfz(self.outmap_size, z_base=self.cam3dmap_anchor))
        self._make_final_layers(self.backbone_channels)
    
    def _make_transformer(self, drop_ratio=0.2):
        self.position_embeddings = nn.Embedding(self.outmap_size, self.transformer_cfg['INPUT_C'], padding_idx=0)
        self.transformer = nn.Sequential(
            nn.Linear(self.transformer_cfg['INPUT_C'],self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.transformer_cfg['NUM_CHANNELS']),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_ratio),
            nn.Linear(self.transformer_cfg['NUM_CHANNELS'],self.output_cfg['NUM_PARAMS_MAP']))

    def _make_final_layers(self, input_channels):
        self.det_head = self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])
        self.param_head = self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP'], with_outlayer=False)
        
        self._make_bv_center_layers(input_channels,self.bv_center_cfg['NUM_DEPTH_LEVEL']*2)
        self._make_3D_map_refiner()
    
    def _make_head_layers(self, input_channels, output_channels, num_channels=None, with_outlayer=True):
        head_layers = []
        if num_channels is None:
            num_channels = self.head_cfg['NUM_CHANNELS']

        for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
            head_layers.append(nn.Sequential(
                    BasicBlock(input_channels, num_channels,downsample=nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0))))
            input_channels = num_channels
        if with_outlayer:
            head_layers.append(nn.Conv2d(in_channels=num_channels,\
                out_channels=output_channels,kernel_size=1,stride=1,padding=0))

        return nn.Sequential(*head_layers)

    def _make_bv_center_layers(self, input_channels, output_channels):
        num_channels = self.outmap_size // 8
        self.bv_pre_layers = nn.Sequential(
                    nn.Conv2d(in_channels=input_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,stride=1,padding=1),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True),\
                    nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=1,stride=1,padding=0),\
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),\
                    nn.ReLU(inplace=True))
        
        input_channels = (num_channels + self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP'])*self.outmap_size
        inter_channels = 512
        self.bv_out_layers = nn.Sequential(
                    BasicBlock_1D(input_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, inter_channels),\
                    BasicBlock_1D(inter_channels, output_channels))

    def _make_3D_map_refiner(self):
        self.center_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CENTER_MAP'], self.output_cfg['NUM_CENTER_MAP']))
        self.cam_map_refiner = nn.Sequential(BasicBlock_3D(self.output_cfg['NUM_CAM_MAP'], self.output_cfg['NUM_CAM_MAP']))
    
    def fv_conditioned_bv_estimation(self, x, center_maps_fv, cam_maps_offset):
        img_feats = self.bv_pre_layers(x)
        summon_feats = torch.cat([center_maps_fv, cam_maps_offset, img_feats], 1).view(img_feats.size(0), -1, self.outmap_size)
        
        outputs_bv = self.bv_out_layers(summon_feats)
        center_maps_bv = outputs_bv[:, :self.bv_center_cfg['NUM_DEPTH_LEVEL']]
        cam_maps_offset_bv = outputs_bv[:, self.bv_center_cfg['NUM_DEPTH_LEVEL']:]
        center_map_3d = center_maps_fv.repeat(1,self.bv_center_cfg['NUM_DEPTH_LEVEL'],1,1) * \
                        center_maps_bv.unsqueeze(2).repeat(1,1,self.outmap_size,1)
        return center_map_3d, cam_maps_offset_bv
    
    def coarse2fine_localization(self, x):
        maps_fv = self.det_head(x)
        center_maps_fv = maps_fv[:,:self.output_cfg['NUM_CENTER_MAP']]
        # predict the small offset from each anchor at 128 map to meet the real 2D image map: map from 0~1 to 0~4 image coordinates
        cam_maps_offset = maps_fv[:,self.output_cfg['NUM_CENTER_MAP']:self.output_cfg['NUM_CENTER_MAP']+self.output_cfg['NUM_CAM_MAP']]
        center_maps_3d, cam_maps_offset_bv = self.fv_conditioned_bv_estimation(x, center_maps_fv, cam_maps_offset)

        center_maps_3d = self.center_map_refiner(center_maps_3d.unsqueeze(1)).squeeze(1)
        # B x 3 x H x W -> B x 1 x H x W x 3  |  B x 3 x D x W -> B x D x 1 x W x 3
        # B x D x H x W x 3 + B x 1 x H x W x 3 + B x D x 1- x W x 3  .to(cam_maps_offset.device)
        cam_maps_3d = self.coordmap_3d + \
                        cam_maps_offset.unsqueeze(-1).transpose(4,1).contiguous()
        # cam_maps_offset_bv adjust z-wise only
        cam_maps_3d[:,:,:,:,2] = cam_maps_3d[:,:,:,:,2] + cam_maps_offset_bv.unsqueeze(2).contiguous()
        cam_maps_3d = self.cam_map_refiner(cam_maps_3d.unsqueeze(1).transpose(5,1).squeeze(-1))
        
        return center_maps_3d, cam_maps_3d, center_maps_fv
    
    def differentiable_person_feature_sampling(self, feature, pred_czyxs, pred_batch_ids):
        cz, cy, cx = pred_czyxs[:,0], pred_czyxs[:,1], pred_czyxs[:,2]
        position_encoding = self.position_embeddings(cz)
        feature_sampled = feature[pred_batch_ids, :, cy, cx]

        input_features = feature_sampled + position_encoding
        return input_features
    
    def mesh_parameter_regression(self, fv_f, cams_preds, pred_batch_ids):
        cam_czyx = denormalize_center(convert_cam_params_to_centermap_coords(cams_preds.clone(), self.cam3dmap_anchor), size=self.outmap_size)
        feature_sampled = self.differentiable_person_feature_sampling(fv_f, cam_czyx, pred_batch_ids)
        params_preds = self.transformer(feature_sampled)
        params_preds = torch.cat([cams_preds, params_preds], 1)
        return params_preds, cam_czyx
    
    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        center_maps_3d, cam_maps_3d, center_maps_fv = self.coarse2fine_localization(x)
        
        center_preds_info_3d = self.centermap_parser.parse_3dcentermap(center_maps_3d)
        if len(center_preds_info_3d[0])==0:
            # print('No person detected!')
            return None
        pred_batch_ids, pred_czyxs, center_confs = center_preds_info_3d
        cams_preds = cam_maps_3d[pred_batch_ids,:,pred_czyxs[:,0],pred_czyxs[:,1],pred_czyxs[:,2]]

        front_view_features = self.param_head(x)
        params_preds, cam_czyx = self.mesh_parameter_regression(front_view_features, cams_preds, pred_batch_ids)
        
        output = {'params_pred':params_preds.float(), 'cam_czyx':cam_czyx.float(), 
                'center_map':center_maps_fv.float(),'center_map_3d':center_maps_3d.float().squeeze(),
                'pred_batch_ids':pred_batch_ids, 'pred_czyxs':pred_czyxs, 'center_confs':center_confs}
        return output

class BEV(nn.Module):
    def __init__(self, romp_settings):
        super(BEV, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()
        print("Model built and initialized.")
    
    def _build_model_(self):
        model = BEVv1(center_thresh=self.settings.center_thresh).eval()
        model.load_state_dict(torch.load(self.settings.model_path, map_location=self.tdevice), strict=False)
        model = model.to(self.tdevice)
        self.model = nn.DataParallel(model)

    def _initilization_(self):        
        if self.settings.calc_smpl:
            self.smpl_parser = SMPLA_parser(self.settings.smpl_path, self.settings.smil_path).to(self.tdevice)
        
        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_(self.settings.smooth_coeff)

        # if self.settings.render_mesh or self.settings.mode == 'webcam':
            # self.renderer = setup_renderer(name=self.settings.renderer)
        self.visualize_items = self.settings.show_items.split(',')
        self.result_keys = ['smpl_thetas', 'smpl_betas', 'cam','cam_trans', 'params_pred', 'center_confs', 'pred_batch_ids']
    
    def _initialize_optimization_tools_(self, smooth_coeff):
        self.OE_filters = {}
        if not self.settings.show_largest:
            from tracker.byte_tracker_3dcenter import Tracker
            self.tracker = Tracker(det_thresh=0.12, low_conf_det_thresh=0.05, track_buffer=60, match_thresh=300, frame_rate=30)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        parsed_results = self.model(input_image.to(self.tdevice))
        if parsed_results is None:
            return None, image_pad_info
        parsed_results.update(pack_params_dict(parsed_results['params_pred']))
        parsed_results.update({'cam_trans':denormalize_cam_params_to_trans(parsed_results['cam'])})

        all_result_keys = list(parsed_results.keys())
        for key in all_result_keys:
            if key not in self.result_keys:
                del parsed_results[key]
        return parsed_results, image_pad_info
        
    @time_cost('BEV')
    @torch.no_grad()
    def forward(self, image, signal_ID=0, **kwargs):
        # if image.shape[1] / image.shape[0] >= 2 and self.settings.crowd:
        #     outputs = self.process_long_image(image, show_patch_results=self.settings.show_patch_results)
        # else:
        outputs = self.process_normal_image(image, signal_ID)
        if outputs is None:
            return None
        
        if self.settings.render_mesh:
            mesh_color_type = 'identity' if self.settings.mode!='webcam' and not self.settings.save_video else 'same'
            rendering_cfgs = {'mesh_color':mesh_color_type, 'items': self.visualize_items, 'renderer': self.settings.renderer}
            outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
        # if 
        
        # if self.settings.render_mesh:
        #     mesh_color_type = 'identity' if self.settings.mode!='webcam' and not self.settings.save_video else 'same'
        #     rendering_cfgs = {'mesh_color':mesh_color_type, 'items': self.visualize_items, 'renderer': self.settings.renderer}
        #     # outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
        # if self.settings.show:
        #     h, w = outputs['rendered_image'].shape[:2]
        #     show_image = outputs['rendered_image'] if h<=1080 else cv2.resize(outputs['rendered_image'], (int(w*(1080/h)), 1080))
        #     cv2.imwrite('/ssd4tb/jaewoo/t2p/viz_0913/test_1.png', show_image)
        #     wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs)
    
    
    @torch.no_grad()
    def forward_parse(self, image, signal_ID=0, **kwargs):
        # if image.shape[1] / image.shape[0] >= 2 and self.settings.crowd:
        #     outputs = self.process_long_image(image, show_patch_results=self.settings.show_patch_results)
        # else:
        outputs = self.process_normal_image(image, signal_ID)
        if outputs is None:
            return None
        return convert_tensor2numpy(outputs)
        
    def process_normal_image(self, image, signal_ID):
        outputs, image_pad_info = self.single_image_forward(image)
        meta_data = {'input2org_offsets': image_pad_info}
        
        if outputs is None:
            return None
        
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
            if outputs is None:
                return None
            outputs.update({'cam_trans':denormalize_cam_params_to_trans(outputs['cam'])})
        
        if self.settings.calc_smpl:
            verts, joints, face = self.smpl_parser(outputs['smpl_betas'], outputs['smpl_thetas']) 
            outputs.update({'verts': verts, 'joints': joints, 'smpl_face':face})
            if self.settings.render_mesh:
                meta_data['vertices'] = outputs['verts']
            projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
            outputs.update(projection)
            
            outputs = suppressing_redundant_prediction_via_projection(outputs,image.shape, thresh=self.settings.nms_thresh)
            outputs = remove_outlier(outputs,relative_scale_thresh=self.settings.relative_scale_thresh)
        return outputs
    
    #@time_cost('BEV')
    def process_long_image(self, full_image, show_patch_results=False):
        print('processing in crowd mode')
        from bev.split2process import get_image_split_plan, convert_crop_cam_params2full_image,\
            collect_outputs, exclude_boudary_subjects, padding_image_overlap
        full_image_pad, image_pad_info, pad_length = padding_image_overlap(full_image, overlap_ratio=self.settings.overlap_ratio)
        meta_data = {'input2org_offsets': image_pad_info}
        
        fh, fw = full_image_pad.shape[:2]
        # please crop the human area out from the huge/long image to facilitate better predictions.
        crop_boxes = get_image_split_plan(full_image_pad,overlap_ratio=self.settings.overlap_ratio)

        croped_images, outputs_list = [], []
        for cid, crop_box in enumerate(crop_boxes):
            l,r,t,b = crop_box
            croped_image = full_image_pad[t:b, l:r]
            crop_outputs, image_pad_info = self.single_image_forward(croped_image)
            if crop_outputs is None:
                outputs_list.append(crop_outputs)
                continue
            verts, joints, face = self.smpl_parser(crop_outputs['smpl_betas'], crop_outputs['smpl_thetas']) 
            crop_outputs.update({'verts': verts, 'joints': joints, 'smpl_face':face})
            outputs_list.append(crop_outputs)
            croped_images.append(croped_image)
        
        # exclude the subjects in the overlapping area, the right of this crop
        for cid in range(len(crop_boxes)):
            this_outs = outputs_list[cid]
            if this_outs is not None:
                if cid != len(crop_boxes) - 1:
                    this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid+1, 0]
                    drop_boundary_ratio = (this_right - next_left) / fh / 2
                    exclude_boudary_subjects(this_outs, drop_boundary_ratio, ptype='left', torlerance=0)
                ch, cw = croped_images[cid].shape[:2]
                projection = body_mesh_projection2image(this_outs['joints'], this_outs['cam'], vertices=this_outs['verts'], input2org_offsets=torch.Tensor([0, ch, 0, cw, ch, cw]))
                this_outs.update(projection)
                
        # exclude the subjects in the overlapping area, the left of next crop
        for cid in range(1,len(crop_boxes)-1):
            this_outs, next_outs = outputs_list[cid], outputs_list[cid+1]
            this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid+1, 0]
            drop_boundary_ratio = (this_right - next_left) / fh / 2 
            if next_outs is not None:
                exclude_boudary_subjects(next_outs, drop_boundary_ratio, ptype='right', torlerance=0) 
        
        for cid, crop_image in enumerate(croped_images):
            this_outs = outputs_list[cid]
            ch, cw = croped_images[cid].shape[:2]
            this_outs = suppressing_redundant_prediction_via_projection(this_outs, [ch, cw], thresh=self.settings.nms_thresh,conf_based=True)
            this_outs = remove_outlier(this_outs, scale_thresh=1, relative_scale_thresh=self.settings.relative_scale_thresh)
        
        if show_patch_results:
            rendering_cfgs = {'mesh_color':'identity', 'items':['mesh','center_conf','pj2d'], 'renderer':self.settings.renderer}
            for cid, crop_image in enumerate(croped_images):
                this_outs = outputs_list[cid]
                # this_outs = rendering_romp_bev_results(self.renderer, this_outs, crop_image, rendering_cfgs)
                # saver = ResultSaver(self.settings.mode, self.settings.save_path)
                # saver(this_outs, 'crop.jpg', prefix=f'{self.settings.center_thresh}_{cid}')     
        
        outputs = {}
        for cid, crop_box in enumerate(crop_boxes):
            crop_outputs = outputs_list[cid]
            if crop_outputs is None:
                continue
            crop_box[:2] -= pad_length
            crop_outputs['cam'] = convert_crop_cam_params2full_image(crop_outputs['cam'], crop_box, full_image.shape[:2])
            collect_outputs(crop_outputs, outputs)
        
        if self.settings.render_mesh:
            meta_data['vertices'] = outputs['verts']
        projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
        outputs.update(projection)
        outputs = suppressing_redundant_prediction_via_projection(outputs, full_image.shape, thresh=self.settings.nms_thresh,conf_based=True)
        outputs = remove_outlier(outputs, scale_thresh=0.5, relative_scale_thresh=self.settings.relative_scale_thresh)

        return outputs
    
    def temporal_optimization(self, outputs, signal_ID, image_scale=128, depth_scale=30):
        check_filter_state(self.OE_filters, signal_ID, self.settings.show_largest, self.settings.smooth_coeff)
        if self.settings.show_largest:
            max_id = torch.argmax(outputs['cam'][:,0])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = \
                smooth_results(self.OE_filters[signal_ID], \
                    outputs['smpl_thetas'][max_id], outputs['smpl_betas'][max_id], outputs['cam'][max_id])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = outputs['smpl_thetas'].unsqueeze(0), outputs['smpl_betas'].unsqueeze(0), outputs['cam'].unsqueeze(0)
        else:
            cam_trans = outputs['cam_trans'].cpu().numpy()
            cams = outputs['cam'].cpu().numpy()
            det_confs = outputs['center_confs'].cpu().numpy()
            tracking_points = np.concatenate([(cams[:,[2,1]]+1)*image_scale, cam_trans[:,[2]]*depth_scale, cams[:,[0]]*image_scale/2],1)
            tracked_ids, results_inds = self.tracker.update(tracking_points, det_confs)
            if len(tracked_ids) == 0:
                return None

            for key in self.result_keys:
                outputs[key] = outputs[key][results_inds]

            for ind, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(self.settings.smooth_coeff)
                outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind] = \
                    smooth_results(self.OE_filters[signal_ID][tid], \
                    outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind])
            outputs['track_ids'] = np.array(tracked_ids).astype(np.int32)
        return outputs
