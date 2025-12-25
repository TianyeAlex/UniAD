"""
Custom collate function for UniAD to support batch_size > 1
"""
from mmcv.parallel.data_container import DataContainer
import torch


def _get_data(item):
    """Helper function to safely extract data from DataContainer or raw value"""
    if isinstance(item, DataContainer):
        return item.data
    else:
        return item


def _to_tensor(data):
    """Convert numpy arrays to torch tensors for GPU transfer"""
    import numpy as np
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, list):
        return [_to_tensor(item) for item in data]
    else:
        return data


def uniad_collate_fn(batch):
    """UniAD custom collate function supporting batch_size > 1
    
    Args:
        batch: List of samples from dataset, length = batch_size
        
    Returns:
        dict: Batched data with proper structure
    """
    batch_size = len(batch)
    
    # Helper function for safe data extraction
    def get_data(key):
        """Extract data from all samples in batch for given key"""
        return [_get_data(sample[key]) for sample in batch]
    
    # 1. Stack images: [B, queue_length, num_cam, 3, H, W]
    imgs = torch.stack([_get_data(sample['img']) for sample in batch], dim=0)
    
    # 2. Collect img_metas as list
    # Each sample's img_metas is {0: {...}, 1: {...}, ...}
    # Result: [{0: {...}, 1: {...}}, {0: {...}, 1: {...}}, ...]
    img_metas_list = get_data('img_metas')
    
    # 3. Collect GT data (each is a list of tensors/boxes for each frame)
    gt_labels_3d_list = get_data('gt_labels_3d')
    gt_bboxes_3d_list = get_data('gt_bboxes_3d')
    gt_inds_list = get_data('gt_inds')
    
    # 4. Collect temporal data
    l2g_r_mat_list = get_data('l2g_r_mat')
    l2g_t_list = get_data('l2g_t')
    timestamp_list = get_data('timestamp')
    
    # 5. Collect trajectory data
    gt_past_traj_list = get_data('gt_past_traj')
    gt_past_traj_mask_list = get_data('gt_past_traj_mask')
    gt_fut_traj_list = get_data('gt_fut_traj')
    gt_fut_traj_mask_list = get_data('gt_fut_traj_mask')
    
    # 6. Collect SDC (ego vehicle) data
    gt_sdc_bbox_list = get_data('gt_sdc_bbox')
    gt_sdc_label_list = get_data('gt_sdc_label')
    gt_sdc_fut_traj_list = get_data('gt_sdc_fut_traj')
    gt_sdc_fut_traj_mask_list = get_data('gt_sdc_fut_traj_mask')
    
    # 7. Collect map/segmentation data
    gt_lane_labels_list = get_data('gt_lane_labels')
    gt_lane_bboxes_list = get_data('gt_lane_bboxes')
    gt_lane_masks_list = get_data('gt_lane_masks')
    
    # 8. Collect occupancy data
    gt_segmentation_list = get_data('gt_segmentation')
    gt_instance_list = get_data('gt_instance')
    gt_centerness_list = get_data('gt_centerness')
    gt_offset_list = get_data('gt_offset')
    gt_flow_list = get_data('gt_flow')
    gt_backward_flow_list = get_data('gt_backward_flow')
    gt_occ_has_invalid_frame_list = get_data('gt_occ_has_invalid_frame')
    gt_occ_img_is_valid_list = get_data('gt_occ_img_is_valid')
    
    # 9. Collect planning data
    gt_future_boxes_list = get_data('gt_future_boxes')
    gt_future_labels_list = get_data('gt_future_labels')
    sdc_planning_list = get_data('sdc_planning')
    sdc_planning_mask_list = get_data('sdc_planning_mask')
    command_list = get_data('command')
    
    # 10. Assemble return dict
    # NOTE: Critical fix for distributed training with batch_size > 1:
    # 
    # Problem: mmcv's scatter_gather.py recursively processes lists:
    #   if isinstance(obj, list) and len(obj) > 0:
    #       out = list(map(list, zip(*map(scatter_map, obj))))
    # This flattens List[List[Tensor]] (our batch structure) → List[Tensor] (broken!)
    #
    # Root cause: Even with DataContainer(cpu_only=True), after returning obj.data,
    # if data is a list, scatter continues to process it recursively.
    #
    # Solution: Wrap the entire batch list as a tuple (batch,) to prevent scatter
    # from treating it as a list to be distributed. Tuple of length 1 is kept as-is.
    # Then unwrap in forward_track_train: gt_labels_3d = gt_labels_3d[0]
    return {
        'img': DataContainer(imgs, stack=True, cpu_only=False),
        'img_metas': DataContainer((img_metas_list,), cpu_only=True),  # Tuple wrapper
        
        # Wrap each list in a tuple to prevent scatter flattening
        'gt_labels_3d': DataContainer((gt_labels_3d_list,), cpu_only=True),
        'gt_bboxes_3d': DataContainer((gt_bboxes_3d_list,), cpu_only=True),
        'gt_inds': DataContainer((gt_inds_list,), cpu_only=True),
        
        'l2g_r_mat': DataContainer((l2g_r_mat_list,), cpu_only=True),
        'l2g_t': DataContainer((l2g_t_list,), cpu_only=True),
        'timestamp': DataContainer((timestamp_list,), cpu_only=True),
        
        'gt_past_traj': DataContainer((gt_past_traj_list,), cpu_only=True),
        'gt_past_traj_mask': DataContainer((gt_past_traj_mask_list,), cpu_only=True),
        'gt_fut_traj': DataContainer((gt_fut_traj_list,), cpu_only=True),
        'gt_fut_traj_mask': DataContainer((gt_fut_traj_mask_list,), cpu_only=True),
        
        'gt_sdc_bbox': DataContainer((gt_sdc_bbox_list,), cpu_only=True),
        'gt_sdc_label': DataContainer((gt_sdc_label_list,), cpu_only=True),
        'gt_sdc_fut_traj': DataContainer((gt_sdc_fut_traj_list,), cpu_only=True),
        'gt_sdc_fut_traj_mask': DataContainer((gt_sdc_fut_traj_mask_list,), cpu_only=True),
        
        'gt_lane_labels': DataContainer((gt_lane_labels_list,), cpu_only=True),
        'gt_lane_bboxes': DataContainer((gt_lane_bboxes_list,), cpu_only=True),
        'gt_lane_masks': DataContainer((gt_lane_masks_list,), cpu_only=True),
        
        'gt_segmentation': DataContainer((gt_segmentation_list,), cpu_only=True),
        'gt_instance': DataContainer((gt_instance_list,), cpu_only=True),
        'gt_centerness': DataContainer((gt_centerness_list,), cpu_only=True),
        'gt_offset': DataContainer((gt_offset_list,), cpu_only=True),
        'gt_flow': DataContainer((gt_flow_list,), cpu_only=True),
        'gt_backward_flow': DataContainer((gt_backward_flow_list,), cpu_only=True),
        'gt_occ_has_invalid_frame': DataContainer((gt_occ_has_invalid_frame_list,), cpu_only=True),
        'gt_occ_img_is_valid': DataContainer((gt_occ_img_is_valid_list,), cpu_only=True),
        
        'gt_future_boxes': DataContainer((gt_future_boxes_list,), cpu_only=True),
        'gt_future_labels': DataContainer((gt_future_labels_list,), cpu_only=True),
        'sdc_planning': DataContainer((sdc_planning_list,), cpu_only=True),
        'sdc_planning_mask': DataContainer((sdc_planning_mask_list,), cpu_only=True),
        'command': DataContainer((command_list,), cpu_only=True),
    }
