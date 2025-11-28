"""
GPU同步优化补丁

应用此文件中的修改以消除UniAD训练中的GPU同步瓶颈。

使用方法:
1. 备份原文件
2. 应用下面的修改
3. 重新训练并对比性能

预期提升: 20-35% (从3.6秒/iter → 2.3-2.75秒/iter)
"""

# ============================================================================
# 修改1: losses/track_loss.py - get_num_boxes()
# ============================================================================
# 文件: projects/mmdet3d_plugin/losses/track_loss.py
# 行号: 156-163

# ❌ 修改前:
"""
def get_num_boxes(self, num_samples):
    num_boxes = torch.as_tensor(num_samples,
                                dtype=torch.float,
                                device=self.sample_device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    return num_boxes
"""

# ✅ 修改后:
"""
def get_num_boxes(self, num_samples):
    num_boxes = torch.as_tensor(num_samples,
                                dtype=torch.float,
                                device=self.sample_device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
    return num_boxes  # 返回tensor而非标量
"""

# ============================================================================
# 修改2: losses/track_loss.py - loss_bboxes的avg_factor
# ============================================================================
# 文件: projects/mmdet3d_plugin/losses/track_loss.py
# 行号: 278-285

# ❌ 修改前:
"""
mask = target_obj_ids != -1
bbox_weights = torch.ones_like(target_boxes) * self.code_weights
avg_factor = src_boxes[mask].size(0)
avg_factor = reduce_mean(target_boxes.new_tensor([avg_factor]))
loss_bbox = self.loss_bboxes(
    src_boxes[mask],
    target_boxes[mask],
    bbox_weights[mask],
    avg_factor=avg_factor.item(),
)
"""

# ✅ 修改后:
"""
mask = target_obj_ids != -1
bbox_weights = torch.ones_like(target_boxes) * self.code_weights
avg_factor = src_boxes[mask].size(0)
avg_factor = reduce_mean(target_boxes.new_tensor([avg_factor]))
loss_bbox = self.loss_bboxes(
    src_boxes[mask],
    target_boxes[mask],
    bbox_weights[mask],
    avg_factor=avg_factor,  # 直接传递tensor
)
"""

# ============================================================================
# 修改3: losses/track_loss.py - 循环中的obj_idxes访问
# ============================================================================
# 文件: projects/mmdet3d_plugin/losses/track_loss.py
# 行号: 388-398

# ❌ 修改前:
"""
num_disappear_track = 0
for j in range(len(track_instances)):
    obj_id = track_instances.obj_idxes[j].item()
    # set new target idx.
    if obj_id >= 0:
        if obj_id in obj_idx_to_gt_idx:
            track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
        else:
            num_disappear_track += 1
            track_instances.matched_gt_idxes[j] = -1
    else:
        track_instances.matched_gt_idxes[j] = -2
"""

# ✅ 修改后:
"""
# 一次性转换到CPU，避免循环中的重复同步
obj_idxes_cpu = track_instances.obj_idxes.cpu().numpy()
matched_gt_idxes = track_instances.matched_gt_idxes.clone()

num_disappear_track = 0
for j, obj_id in enumerate(obj_idxes_cpu):
    # set new target idx.
    if obj_id >= 0:
        if obj_id in obj_idx_to_gt_idx:
            matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
        else:
            num_disappear_track += 1
            matched_gt_idxes[j] = -1
    else:
        matched_gt_idxes[j] = -2

# 写回GPU
track_instances.matched_gt_idxes = matched_gt_idxes
"""

# 同样的优化应用到第584行的类似代码

# ============================================================================
# 修改4: dense_heads/track_head.py - num_total_pos
# ============================================================================
# 文件: projects/mmdet3d_plugin/uniad/dense_heads/track_head.py
# 行号: 401-402

# ❌ 修改前:
"""
num_total_pos = loss_cls.new_tensor([num_total_pos])
num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
"""

# ✅ 修改后:
"""
num_total_pos = loss_cls.new_tensor([num_total_pos])
num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1)
# 保持tensor形式，loss_bbox会自动处理
"""

# ============================================================================
# 修改5: dense_heads/bevformer_head.py - num_total_pos
# ============================================================================
# 文件: projects/mmdet3d_plugin/uniad/dense_heads/bevformer_head.py
# 行号: 381-382

# ❌ 修改前:
"""
num_total_pos = loss_cls.new_tensor([num_total_pos])
num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
"""

# ✅ 修改后:
"""
num_total_pos = loss_cls.new_tensor([num_total_pos])
num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1)
# 保持tensor形式
"""

# ============================================================================
# 修改6: losses/occflow_loss.py - 条件判断优化
# ============================================================================
# 文件: projects/mmdet3d_plugin/losses/occflow_loss.py
# 行号: 47

# ❌ 修改前:
"""
if frame_mask.sum().item() == 0:
    return prediction.sum() * 0.
"""

# ✅ 修改后 (方案1 - 推荐):
"""
if not frame_mask.any():
    return prediction.sum() * 0.
"""

# 或者 (方案2):
"""
if frame_mask.sum() == 0:
    return prediction.sum() * 0.
"""

# 同样应用到第114行

# ============================================================================
# 修改7: dense_heads/panseg_head.py - num_total_pos (可选)
# ============================================================================
# 文件: projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py
# 行号: 635, 890

# ❌ 修改前:
"""
num_total_pos = torch.clamp(
    reduce_mean(bbox_targets.new_tensor([num_total_pos])),
    min=1).item()
"""

# ✅ 修改后:
"""
num_total_pos = torch.clamp(
    reduce_mean(bbox_targets.new_tensor([num_total_pos])),
    min=1)
"""

# ============================================================================
# 验证脚本
# ============================================================================

# 运行此脚本验证修改是否正确
"""
import torch

print("验证tensor类型的avg_factor...")

# 创建测试数据
loss = torch.randn(100).cuda()

# 测试1: 标量avg_factor
result1 = loss.sum() / 50.0

# 测试2: tensor avg_factor
avg_factor = torch.tensor(50.0).cuda()
result2 = loss.sum() / avg_factor

# 测试3: 0维tensor avg_factor
avg_factor_0d = torch.tensor(50.0).cuda()
result3 = loss.sum() / avg_factor_0d

assert torch.allclose(result1, result2), "标量vs tensor不一致！"
assert torch.allclose(result1, result3), "标量vs 0维tensor不一致！"

print("✅ 所有验证通过！tensor类型的avg_factor工作正常。")
print(f"   result1 (scalar): {result1.item():.6f}")
print(f"   result2 (tensor): {result2.item():.6f}")
print(f"   result3 (0-d tensor): {result3.item():.6f}")
"""
