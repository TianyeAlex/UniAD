# UniAD 数据流详细分析与 Batch Size 重构指南

## 目录
1. [数据流概览](#数据流概览)
2. [Stage 1: 数据加载与预处理](#stage-1-数据加载与预处理)
3. [Stage 2: 图像特征提取](#stage-2-图像特征提取)
4. [Stage 3: BEV 特征转换](#stage-3-bev-特征转换)
5. [Stage 4: 检测与分割 Head](#stage-4-检测与分割-head)
6. [Batch Size=1 的限制点](#batch-size1-的限制点)
7. [重构方案](#重构方案)

---

## 数据流概览

```
数据加载 (Dataset)
    ↓
Data Pipeline (预处理)
    ↓
Collate Function (组batch)
    ↓
Model Forward
    ├─ Image Backbone (ResNet + DCN)
    ├─ Image Neck (FPN)
    ├─ BEV Encoder (Transformer)
    ├─ Detection Head (DETR)
    ├─ Map Segmentation Head
    ├─ Motion Head
    └─ Planning Head
    ↓
Loss Calculation
```

---

## Stage 1: 数据加载与预处理

### 1.1 Dataset 类 (`NuScenesE2EDataset`)

**文件**: `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`

#### 关键参数
```python
queue_length = 5  # 时序长度，每个样本包含5帧
bev_size = (200, 200)  # BEV特征图大小
```

#### 数据组织方式

**当前实现** (batch_size=1):
```python
# __getitem__ 返回格式
{
    'img': Tensor[queue_length, num_cam, 3, H, W],  # (5, 6, 3, 928, 1600)
    'img_metas': {
        0: dict(...),  # 第0帧的meta信息
        1: dict(...),  # 第1帧的meta信息
        ...
        4: dict(...),  # 第4帧的meta信息
    },
    'gt_bboxes_3d': List[LiDARInstance3DBoxes],  # 长度为queue_length
    'gt_labels_3d': List[Tensor],  # 长度为queue_length
    # ... 其他GT信息
}
```

**关键代码** (`prepare_train_data`):
```python
def prepare_train_data(self, index):
    """准备训练数据
    
    关键点：
    1. 当前帧 + 历史帧 组成 queue
    2. queue_length=5 表示包含当前帧+4个历史帧
    3. 通过 union2one() 合并成单个sample
    """
    data_queue = []
    
    # 获取当前帧
    input_dict = self.get_data_info(final_index)
    example = self.pipeline(input_dict)
    data_queue.insert(0, example)
    
    # 获取历史帧 (queue_length-1 个)
    for i in prev_indexs_list:
        input_dict = self.get_data_info(i)
        example = self.pipeline(input_dict)
        data_queue.insert(0, copy.deepcopy(example))
    
    # 合并为单个样本
    data_queue = self.union2one(data_queue)
    return data_queue
```

### 1.2 union2one() - 时序数据合并

**文件**: `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:267`

**功能**: 将 queue_length 个独立样本合并成一个样本

```python
def union2one(self, queue):
    """
    🔴 Batch Size 限制点 #1:
    这个函数假设 batch_size=1，直接在时序维度stack
    
    输入: List[dict], 长度为 queue_length (例如5)
    输出: dict, 但数据在时序维度stack
    """
    # 收集所有帧的数据
    imgs_list = [each['img'].data for each in queue]  # List[Tensor[6,3,H,W]]
    gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
    # ... 其他数据
    
    # Stack 图像: (queue_length, num_cam, 3, H, W)
    imgs_stacked = torch.stack(imgs_list)  # [5, 6, 3, 928, 1600]
    
    # img_metas 组织为字典
    metas_map = {}
    for i, each in enumerate(queue):
        metas_map[i] = each['img_metas'].data
        # 计算相对位姿
        if i == 0:
            metas_map[i]['prev_bev'] = False
        else:
            metas_map[i]['prev_bev'] = True
            # 存储相对于前一帧的位姿变化
    
    # 返回最后一帧的dict，但包含所有时序信息
    queue[-1]['img'] = DC(imgs_stacked, cpu_only=False, stack=True)
    queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
    queue[-1]['gt_labels_3d'] = DC(gt_labels_3d_list)
    queue[-1]['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
    # ...
    
    return queue[-1]
```

**数据形状变化**:
```
Before union2one (List of 5 dicts):
[
  {'img': [6,3,H,W], 'gt_labels_3d': [N0]},  # 第0帧
  {'img': [6,3,H,W], 'gt_labels_3d': [N1]},  # 第1帧
  ...
  {'img': [6,3,H,W], 'gt_labels_3d': [N4]},  # 第4帧
]

After union2one (Single dict):
{
  'img': [5, 6, 3, H, W],  # 时序stack
  'img_metas': {0: {...}, 1: {...}, ..., 4: {...}},
  'gt_labels_3d': [[N0], [N1], ..., [N4]],  # List of tensors
  ...
}
```

### 1.3 Collate Function

**文件**: `mmdet/datasets/builder.py` (MMDetection)

**当前问题**:
```python
# 默认 collate_fn 假设batch中每个sample的结构相同
# 但 UniAD 的 img_metas 是嵌套字典，难以直接batch

# 当前实际使用:
samples_per_gpu = 1  # 配置文件中强制为1
```

---

## Stage 2: 图像特征提取

### 2.1 Backbone (`ResNet101 + DCN`)

**文件**: `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py`

**Forward 流程**:
```python
@auto_fp16(apply_to=('img'))
def extract_img_feat(self, img, img_metas, len_queue=None):
    """提取图像特征
    
    输入:
        img: [B, queue_length, num_cam, 3, H, W]
             当前 B=1, queue_length=5, num_cam=6
    
    输出:
        img_feats: List of [B*queue_length*num_cam, C, H', W']
    """
    B = img.size(0)  # batch_size, 当前=1
    
    if img is not None:
        if img.dim() == 5:  # [B, num_cam, C, H, W]
            B, num_cam, C, H, W = img.size()
            img = img.view(B * num_cam, C, H, W)
        elif img.dim() == 6:  # [B, queue_length, num_cam, C, H, W]
            🔴 Batch Size 限制点 #2: 假设 B=1
            B, len_queue, num_cam, C, H, W = img.size()
            img = img.view(B * len_queue * num_cam, C, H, W)
    
    # Grid Mask 数据增强
    if self.use_grid_mask:
        img = self.grid_mask(img)
    
    # Backbone: [B*len_queue*num_cam, C, H, W] -> List of feature maps
    img_feats = self.img_backbone(img)  # ResNet
    
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    
    # Neck (FPN)
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    
    # Reshape 回多视图格式
    img_feats_reshaped = []
    for img_feat in img_feats:
        BN, C, H, W = img_feat.size()
        if len_queue is not None:
            # [B*len_queue*num_cam, C, H, W] 
            # -> [B, len_queue, num_cam, C, H, W]
            img_feats_reshaped.append(
                img_feat.view(int(B/len_queue), len_queue, 
                             int(BN / B), C, H, W)
            )
        else:
            # [B*num_cam, C, H, W] -> [B, num_cam, C, H, W]
            img_feats_reshaped.append(
                img_feat.view(B, int(BN / B), C, H, W)
            )
    
    return img_feats_reshaped
```

**特征金字塔层级** (FPN):
```python
# img_feats_reshaped 是一个 list，包含4个层级
img_feats = [
    Tensor[B, len_queue, num_cam, 256, H/8, W/8],    # Level 0
    Tensor[B, len_queue, num_cam, 256, H/16, W/16],  # Level 1
    Tensor[B, len_queue, num_cam, 256, H/32, W/32],  # Level 2
    Tensor[B, len_queue, num_cam, 256, H/64, W/64],  # Level 3
]

# 当 B=1, len_queue=5, num_cam=6, H=928, W=1600
Level 0: [1, 5, 6, 256, 116, 200]
Level 1: [1, 5, 6, 256, 58, 100]
Level 2: [1, 5, 6, 256, 29, 50]
Level 3: [1, 5, 6, 256, 15, 25]
```

---

## Stage 3: BEV 特征转换

### 3.1 BEV Encoder (`BEVFormerEncoder`)

**文件**: `projects/mmdet3d_plugin/uniad/modules/encoder.py`

**核心思想**:
- 使用 **Deformable Attention** 将多视图特征投影到 BEV 空间
- 使用 **Temporal Self-Attention** 融合历史BEV特征

**Forward 流程**:
```python
@auto_fp16()
def forward(self,
            bev_query,      # [bev_h*bev_w, B, embed_dim]
            key,            # 多视图特征
            value,          # 多视图特征
            bev_h=None,     # 200
            bev_w=None,     # 200
            bev_pos=None,   # BEV位置编码
            prev_bev=None,  # 历史BEV特征
            ...):
    """
    🔴 Batch Size 限制点 #3:
    bev_query 的形状是 [bev_h*bev_w, B, embed_dim]
    这里 B 必须=1，因为后续处理假设batch维度是1
    
    输入:
        bev_query: [40000, 1, 256]  # 200*200=40000
        prev_bev: [1, 40000, 256] or None
    
    输出:
        output: [40000, 1, 256]
    """
    
    # 6层 BEVFormerLayer
    for layer in self.layers:
        output = layer(
            bev_query,
            key,
            value,
            bev_pos=bev_pos,
            prev_bev=prev_bev,  # 历史BEV
            bev_h=bev_h,
            bev_w=bev_w,
            ...
        )
        bev_query = output
    
    return output
```

### 3.2 BEVFormerLayer

**关键组件**:

#### (1) Temporal Self-Attention
```python
# 融合当前BEV query 和 历史BEV特征
# prev_bev shape: [B, bev_h*bev_w, embed_dim] = [1, 40000, 256]

if prev_bev is not None:
    🔴 Batch Size 限制点 #4:
    # 这里假设 B=1，对每个样本单独处理旋转
    for i in range(bs):
        rotation_angle = img_metas[i]['can_bus'][-1]
        # 旋转历史BEV以对齐当前帧
        tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1)
        # ... rotate操作
```

#### (2) Spatial Cross-Attention (`MSDeformableAttention3D`)
```python
# 从多视图特征中采样，投影到BEV
# reference_points_cam: 参考点在各相机中的投影位置
# spatial_flatten: 展平的多视图特征

output = self.cross_attn(
    query=query,  # BEV queries
    key=spatial_flatten,  # 多视图特征
    value=spatial_flatten,
    reference_points=reference_points_cam,
    ...
)
```

**数据形状变化**:
```
Input:
  bev_query: [40000, 1, 256]
  mlvl_feats: List of [1, 5, 6, 256, H, W]  # 4 levels
  prev_bev: [1, 40000, 256]

After BEVFormerEncoder:
  bev_embed: [1, 40000, 256]
  
Reshape:
  bev_embed: [1, 200, 200, 256] -> [1, 256, 200, 200]
```

---

## Stage 4: 检测与分割 Head

### 4.1 Detection Head (`BEVFormerTrackHead`)

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/track_head.py`

**Forward 流程**:
```python
def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
    """
    输入:
        mlvl_feats: List of [B, len_queue, num_cam, C, H, W]
        
    输出:
        hs: [num_decoder_layers, B, num_query, embed_dim]
            = [6, 1, 900, 256]
        init_reference: [B, num_query, 3]
        inter_references: [num_decoder_layers-1, B, num_query, 3]
    """
    
    # 获取BEV特征
    bev_embed, bev_pos = self.get_bev_features(
        mlvl_feats, img_metas, prev_bev
    )  # [B, bev_h*bev_w, embed_dim]
    
    # Object Queries
    object_query_embeds = self.query_embedding.weight  # [num_query, embed_dim]
    
    # Transformer Decoder
    hs, init_reference, inter_references = self.transformer.decoder(
        query=object_query_embeds,
        key=bev_embed,
        value=bev_embed,
        ...
    )
    
    # 分类和回归分支
    outputs_classes = []
    outputs_coords = []
    for lvl in range(num_decoder_layers):
        outputs_class = self.cls_branches[lvl](hs[lvl])  # [B, num_query, num_classes]
        outputs_coord = self.reg_branches[lvl](hs[lvl])  # [B, num_query, code_size]
        
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    
    return {
        'all_cls_scores': torch.stack(outputs_classes),  # [6, 1, 900, 10]
        'all_bbox_preds': torch.stack(outputs_coords),   # [6, 1, 900, 10]
        'bev_embed': bev_embed,
        ...
    }
```

### 4.2 Map Segmentation Head (`PansegformerHead`)

**文件**: `projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py`

```python
@force_fp32(apply_to=('bev_embed',))
def forward(self, bev_embed):
    """
    输入:
        bev_embed: [B, C, H, W] = [1, 256, 200, 200]
    
    输出:
        all_cls_scores: [num_layers, B, num_query, num_classes]
        all_mask_preds: [num_layers, B, num_query, H, W]
        all_bbox_preds: [num_layers, B, num_query, 4]
    """
    
    # 特征提取 (可能包含额外的encoder)
    feat_flatten, spatial_shapes, level_start_index = \
        self._get_target_single(bev_embed)
    
    # Queries
    query_embeds = self.query_embedding.weight  # [num_query, embed_dim]
    
    # Transformer
    hs = self.transformer(
        feat_flatten,
        query_embeds,
        ...
    )  # [num_layers, B, num_query, embed_dim]
    
    # 预测头
    outputs_classes = []
    outputs_masks = []
    outputs_bboxes = []
    
    for lvl in range(num_layers):
        cls_pred = self.cls_branches[lvl](hs[lvl])
        mask_pred = self.mask_head(hs[lvl], bev_embed)
        bbox_pred = self.bbox_head[lvl](hs[lvl])
        
        outputs_classes.append(cls_pred)
        outputs_masks.append(mask_pred)
        outputs_bboxes.append(bbox_pred)
    
    return {
        'all_cls_scores': outputs_classes,
        'all_mask_preds': outputs_masks,
        'all_bbox_preds': outputs_bboxes,
    }
```

---

## Batch Size=1 的限制点

### 总结所有限制点

| 限制点 | 位置 | 问题描述 | 影响 |
|-------|------|---------|------|
| **#1** | `Dataset.union2one()` | img_metas组织为嵌套字典，难以batch | 数据加载 |
| **#2** | `extract_img_feat()` | Reshape 逻辑假设 B=1 | 特征提取 |
| **#3** | `BEVFormerEncoder.forward()` | bev_query 维度假设 B=1 | BEV转换 |
| **#4** | `BEVFormerLayer` | prev_bev 旋转逻辑按样本循环 | 时序融合 |
| **#5** | `MemoryBank` | 跟踪内存按样本维护 | 多目标跟踪 |
| **#6** | Loss 计算 | gt 匹配假设单样本 | 训练 |

### 详细分析

#### 限制点 #1: Dataset.union2one()
```python
# 当前实现
img_metas = {
    0: {'can_bus': [...], 'prev_bev': False, ...},
    1: {'can_bus': [...], 'prev_bev': True, ...},
    ...
}

# 问题: 无法直接扩展到 batch
# batch=2 时需要变成:
img_metas = [
    {  # Sample 0
        0: {...},
        1: {...},
        ...
    },
    {  # Sample 1
        0: {...},
        1: {...},
        ...
    }
]
```

#### 限制点 #2: extract_img_feat()
```python
# 当前代码
B, len_queue, num_cam, C, H, W = img.size()
img = img.view(B * len_queue * num_cam, C, H, W)

# 问题: 后续 reshape 回来时硬编码了 B
img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W)
#             ^^^^^^^^^^^^^^^^^ 这里假设 B = 1 * len_queue
```

#### 限制点 #3: BEVFormerEncoder
```python
# BEV query 的形状
bev_query = bev_queries.unsqueeze(1).repeat(1, bs, 1)
# [bev_h*bev_w, embed_dim] -> [bev_h*bev_w, bs, embed_dim]

# 问题: 很多地方用索引 [0] 访问
feat_flatten = feat_flatten.permute(0, 2, 1, 3, 4, 5)  
# [bs, num_cam, ...] 后续用 [0] 索引
```

#### 限制点 #4: 历史BEV旋转
```python
for i in range(bs):
    rotation_angle = img_metas[i]['can_bus'][-1]
    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1)
    # ... 旋转操作
    
# 问题: 
# 1. img_metas[i] 要求 img_metas 是 list
# 2. 当前 img_metas 是嵌套 dict
```

#### 限制点 #5: MemoryBank (Tracking)
```python
class MemoryBank:
    def __init__(self):
        # 🔴 单样本的跟踪状态
        self.memo = None
        self.memo_embed = None
        
    def update(self, track_instances):
        # 更新单个样本的记忆库
        ...
```

#### 限制点 #6: Loss 计算
```python
def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts):
    """
    当前假设:
        gt_bboxes_list: List[List[Tensor]]
            外层 list 长度 = num_decoder_layers (6)
            内层 list 长度 = batch_size (1)
            
    问题: Hungarian Matching 需要对每个样本单独计算
    """
    for gt_bboxes, gt_labels, preds in zip(...):
        # 每个样本的损失
        loss_dict = self.loss_single(gt_bboxes, gt_labels, preds)
```

---

## 重构方案

### 方案概览

```
优先级 1 (核心) - 数据流重构
├─ Step 1: Collate Function 重写
├─ Step 2: img_metas 结构调整
├─ Step 3: Backbone/Neck reshape 逻辑修复
└─ Step 4: BEV Encoder batch 支持

优先级 2 (功能) - Head 层重构
├─ Step 5: Detection Head batch 支持
├─ Step 6: Segmentation Head batch 支持
└─ Step 7: Loss 计算并行化

优先级 3 (高级) - Tracking 重构
├─ Step 8: MemoryBank batch 化
└─ Step 9: Query Interaction batch 化
```

### Step 1: 重写 Collate Function

**目标**: 正确处理嵌套的 img_metas

**新建文件**: `projects/mmdet3d_plugin/datasets/collate.py`

```python
from mmcv.parallel.data_container import DataContainer
import torch

def uniad_collate_fn(batch):
    """UniAD 自定义 collate function
    
    处理要点:
    1. img_metas 从嵌套dict变为list of dict
    2. 时序数据保持 queue 结构
    3. GT 数据正确 padding/stack
    """
    batch_size = len(batch)
    
    # 1. 处理 img
    imgs = torch.stack([sample['img'].data for sample in batch], dim=0)
    # [B, queue_length, num_cam, 3, H, W]
    
    # 2. 处理 img_metas
    # 从 {0: {...}, 1: {...}, ...} 变为 [{0: {...}, 1: {...}}, ...]
    img_metas_list = []
    for sample in batch:
        img_metas_list.append(sample['img_metas'].data)
    
    # 3. 处理 GT bboxes (需要padding到相同数量)
    gt_bboxes_3d_list = []
    gt_labels_3d_list = []
    
    for sample in batch:
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'].data)
        gt_labels_3d_list.append(sample['gt_labels_3d'].data)
    
    # 4. 组装返回
    return {
        'img': DataContainer(imgs, stack=True, cpu_only=False),
        'img_metas': DataContainer(img_metas_list, cpu_only=True),
        'gt_bboxes_3d': DataContainer(gt_bboxes_3d_list, cpu_only=True),
        'gt_labels_3d': DataContainer(gt_labels_3d_list, cpu_only=False),
        # ... 其他字段
    }
```

**配置文件修改**:
```python
# projects/configs/stage1_track_map/base_track_map.py

data = dict(
    samples_per_gpu=2,  # 可以设为 >1
    workers_per_gpu=4,
    train=dict(
        type='NuScenesE2EDataset',
        # ... 其他参数
    ),
    # 添加自定义 collate_fn
    train_dataloader=dict(
        collate_fn='projects.mmdet3d_plugin.datasets.collate.uniad_collate_fn'
    ),
)
```

### Step 2: img_metas 结构调整

**修改**: `Dataset.union2one()`

```python
def union2one(self, queue):
    """重构为支持 batch 的版本
    
    关键改动:
    1. img_metas 保持为 dict of dict (时序嵌套)
    2. 但在 collate 时会被放入 list
    """
    # ... (img stack 等逻辑保持不变)
    
    # img_metas 组织
    metas_map = {}
    prev_pos = None
    prev_angle = None
    
    for i, each in enumerate(queue):
        metas_map[i] = each['img_metas'].data
        
        # 计算相对位姿 (这部分逻辑不变)
        if i == 0:
            metas_map[i]['prev_bev'] = False
            prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            metas_map[i]['can_bus'][:3] = 0
            metas_map[i]['can_bus'][-1] = 0
        else:
            metas_map[i]['prev_bev'] = True
            tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            metas_map[i]['can_bus'][:3] -= prev_pos
            metas_map[i]['can_bus'][-1] -= prev_angle
            prev_pos = copy.deepcopy(tmp_pos)
            prev_angle = copy.deepcopy(tmp_angle)
    
    # 返回时仍然是单个 dict，但 collate_fn 会把它放入 list
    queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
    # ...
    return queue[-1]
```

### Step 3: Backbone/Neck Reshape 修复

**修改**: `extract_img_feat()`

```python
def extract_img_feat(self, img, img_metas, len_queue=None):
    """支持 batch_size > 1 的特征提取
    
    输入:
        img: [B, queue_length, num_cam, 3, H, W]
    输出:
        img_feats: List of [B, queue_length, num_cam, C, H', W']
    """
    B = img.size(0)  # 真实的 batch_size
    
    if img.dim() == 6:
        B, len_queue, num_cam, C, H, W = img.size()
        img = img.view(B * len_queue * num_cam, C, H, W)
    elif img.dim() == 5:
        B, num_cam, C, H, W = img.size()
        img = img.view(B * num_cam, C, H, W)
    
    # Grid Mask
    if self.use_grid_mask:
        img = self.grid_mask(img)
    
    # Backbone
    img_feats = self.img_backbone(img)
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    
    # Neck
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    
    # Reshape 回多视图格式
    img_feats_reshaped = []
    for img_feat in img_feats:
        BN, C, H, W = img_feat.size()
        # BN = B * len_queue * num_cam
        
        if len_queue is not None:
            # 正确计算 reshape 参数
            # BN = B * len_queue * num_cam
            # 所以 num_cam = BN / (B * len_queue)
            num_cam_actual = BN // (B * len_queue)
            img_feats_reshaped.append(
                img_feat.view(B, len_queue, num_cam_actual, C, H, W)
            )
        else:
            # BN = B * num_cam
            num_cam_actual = BN // B
            img_feats_reshaped.append(
                img_feat.view(B, num_cam_actual, C, H, W)
            )
    
    return img_feats_reshaped
```

### Step 4: BEV Encoder Batch 支持

**修改**: `BEVFormerEncoder.forward()`

```python
@auto_fp16()
def forward(self,
            bev_query,
            key,
            value,
            bev_h=None,
            bev_w=None,
            bev_pos=None,
            prev_bev=None,
            img_metas=None,  # 现在是 List[Dict]
            **kwargs):
    """
    输入形状变化:
        bev_query: [bev_h*bev_w, B, embed_dim]  # B 可以 > 1
        prev_bev: [B, bev_h*bev_w, embed_dim] or None
        img_metas: List[Dict], 长度=B
    """
    bs = bev_query.size(1)  # batch_size
    
    # 处理历史 BEV (关键修改)
    if prev_bev is not None:
        if self.rotate_prev_bev:
            # 🔧 批量旋转
            # 收集所有样本的旋转角度
            rotation_angles = []
            for i in range(bs):
                rotation_angles.append(img_metas[i][self.queue_length-1]['can_bus'][-1])
            rotation_angles = torch.tensor(
                rotation_angles, device=prev_bev.device
            )  # [B]
            
            # 批量旋转 (使用 grid_sample 或自定义 CUDA kernel)
            prev_bev_rotated = self.rotate_bev_batch(
                prev_bev, rotation_angles, bev_h, bev_w
            )  # [B, bev_h*bev_w, embed_dim]
            
            # Concat
            prev_bev = prev_bev_rotated.permute(1, 0, 2)  # [bev_h*bev_w, B, embed_dim]
            bev_query = torch.cat([bev_query, prev_bev], dim=0)
    
    # Transformer layers (逻辑不变，但数据是 batched)
    for layer in self.layers:
        output = layer(
            bev_query,
            key,
            value,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
            **kwargs
        )
        bev_query = output
    
    return output


def rotate_bev_batch(self, prev_bev, rotation_angles, bev_h, bev_w):
    """批量旋转 BEV 特征
    
    输入:
        prev_bev: [B, bev_h*bev_w, embed_dim]
        rotation_angles: [B], 弧度
        
    输出:
        rotated_bev: [B, bev_h*bev_w, embed_dim]
    """
    B, _, C = prev_bev.shape
    
    # Reshape to spatial
    prev_bev_spatial = prev_bev.reshape(B, bev_h, bev_w, C)
    prev_bev_spatial = prev_bev_spatial.permute(0, 3, 1, 2)  # [B, C, H, W]
    
    # 生成旋转矩阵 (批量)
    theta = torch.zeros(B, 2, 3, device=prev_bev.device)
    cos_vals = torch.cos(rotation_angles)
    sin_vals = torch.sin(rotation_angles)
    
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = sin_vals
    theta[:, 1, 0] = -sin_vals
    theta[:, 1, 1] = cos_vals
    
    # 使用 grid_sample 旋转
    grid = F.affine_grid(theta, prev_bev_spatial.size(), align_corners=False)
    rotated = F.grid_sample(
        prev_bev_spatial, grid, mode='bilinear', 
        padding_mode='zeros', align_corners=False
    )
    
    # Reshape back
    rotated = rotated.permute(0, 2, 3, 1).reshape(B, bev_h * bev_w, C)
    
    return rotated
```

### Step 5: Detection Head Batch 支持

**修改**: `BEVFormerTrackHead.forward()`

```python
def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
    """
    输入:
        mlvl_feats: List of [B, len_queue, num_cam, C, H, W]
        img_metas: List[Dict], 长度=B
        prev_bev: [B, bev_h*bev_w, embed_dim] or None
    """
    bs = mlvl_feats[0].size(0)  # batch_size
    num_cam = mlvl_feats[0].size(2)
    
    # BEV embedding (复制 bs 次)
    bev_queries = self.bev_embedding.weight.to(dtype)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
    # [bev_h*bev_w, embed_dim] -> [bev_h*bev_w, bs, embed_dim]
    
    # BEV positional encoding
    bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                          device=bev_queries.device, dtype=dtype)
    bev_pos = self.positional_encoding(bev_mask).to(dtype)
    # [bs, embed_dim, bev_h, bev_w]
    
    # Transformer
    bev_embed = self.transformer.get_bev_features(
        mlvl_feats,
        bev_queries,
        self.bev_h,
        self.bev_w,
        grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        bev_pos=bev_pos,
        prev_bev=prev_bev,
        img_metas=img_metas,  # List[Dict]
    )
    # [bev_h*bev_w, bs, embed_dim]
    
    # Object queries
    object_query_embeds = self.query_embedding.weight.to(dtype)
    # [num_query, embed_dim]
    
    # Decoder
    hs, init_reference, inter_references = self.transformer.decoder(
        query=object_query_embeds.unsqueeze(1).repeat(1, bs, 1),
        key=bev_embed,
        value=bev_embed,
        ...
    )
    # hs: [num_layers, bs, num_query, embed_dim]
    
    # 分类和回归
    hs = hs.permute(0, 2, 1, 3)  # [num_layers, num_query, bs, embed_dim]
    
    outputs_classes = []
    outputs_coords = []
    
    for lvl in range(hs.shape[0]):
        # hs[lvl]: [num_query, bs, embed_dim]
        # -> [bs, num_query, embed_dim]
        hs_lvl = hs[lvl].permute(1, 0, 2)
        
        outputs_class = self.cls_branches[lvl](hs_lvl)
        # [bs, num_query, num_classes]
        
        outputs_coord = self.reg_branches[lvl](hs_lvl)
        # [bs, num_query, code_size]
        
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    
    all_cls_scores = torch.stack(outputs_classes)
    # [num_layers, bs, num_query, num_classes]
    
    all_bbox_preds = torch.stack(outputs_coords)
    # [num_layers, bs, num_query, code_size]
    
    return {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'bev_embed': bev_embed,
    }
```

### Step 6: Loss 计算并行化

**修改**: `BEVFormerHead.loss()`

```python
def loss(self,
         gt_bboxes_list,  # List[List[Tensor]], 外层=layers, 内层=batch
         gt_labels_list,  # List[List[Tensor]]
         preds_dicts,
         gt_bboxes_ignore=None,
         img_metas=None):
    """
    输入:
        gt_bboxes_list: [num_layers][batch_size] 的嵌套列表
        preds_dicts: {
            'all_cls_scores': [num_layers, bs, num_query, num_classes],
            'all_bbox_preds': [num_layers, bs, num_query, code_size],
        }
    """
    
    all_cls_scores = preds_dicts['all_cls_scores']
    all_bbox_preds = preds_dicts['all_bbox_preds']
    
    num_dec_layers = all_cls_scores.size(0)
    batch_size = all_cls_scores.size(1)
    
    # 初始化损失字典
    loss_dict = {}
    
    # 对每一层计算损失
    for dec_lvl in range(num_dec_layers):
        # 当前层的预测
        cls_scores = all_cls_scores[dec_lvl]  # [bs, num_query, num_classes]
        bbox_preds = all_bbox_preds[dec_lvl]  # [bs, num_query, code_size]
        
        # GT (每个样本可能有不同数量的目标)
        gt_bboxes_lvl = gt_bboxes_list[dec_lvl]  # List[Tensor], len=bs
        gt_labels_lvl = gt_labels_list[dec_lvl]  # List[Tensor], len=bs
        
        # 批量匹配和损失计算
        loss_lvl = self.loss_single_layer(
            cls_scores,      # [bs, num_query, num_classes]
            bbox_preds,      # [bs, num_query, code_size]
            gt_bboxes_lvl,   # List[Tensor]
            gt_labels_lvl,   # List[Tensor]
        )
        
        # 累加损失
        for key, value in loss_lvl.items():
            loss_key = f'{key}_layer{dec_lvl}'
            loss_dict[loss_key] = value
    
    return loss_dict


def loss_single_layer(self, cls_scores, bbox_preds, gt_bboxes, gt_labels):
    """单层损失计算 (支持 batch)
    
    输入:
        cls_scores: [bs, num_query, num_classes]
        bbox_preds: [bs, num_query, code_size]
        gt_bboxes: List[Tensor], len=bs, 每个 Tensor shape=[num_gt, code_size]
        gt_labels: List[Tensor], len=bs, 每个 Tensor shape=[num_gt]
    """
    batch_size = cls_scores.size(0)
    num_query = cls_scores.size(1)
    
    # 并行匹配 (可能需要对每个样本单独匹配)
    all_cls_targets = []
    all_bbox_targets = []
    all_bbox_weights = []
    all_num_pos = []
    
    for i in range(batch_size):
        # Hungarian Matching (每个样本)
        assign_result = self.assigner.assign(
            bbox_pred=bbox_preds[i],      # [num_query, code_size]
            cls_pred=cls_scores[i],        # [num_query, num_classes]
            gt_bboxes=gt_bboxes[i],        # [num_gt, code_size]
            gt_labels=gt_labels[i],        # [num_gt]
        )
        
        # 采样
        sampling_result = self.sampler.sample(
            assign_result, bbox_preds[i], gt_bboxes[i]
        )
        
        # 构造 target
        cls_targets = cls_scores[i].new_full(
            (num_query,), self.num_classes, dtype=torch.long
        )
        bbox_targets = torch.zeros_like(bbox_preds[i])
        bbox_weights = torch.zeros_like(bbox_preds[i])
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        if len(pos_inds) > 0:
            cls_targets[pos_inds] = gt_labels[i][sampling_result.pos_assigned_gt_inds]
            bbox_targets[pos_inds] = gt_bboxes[i][sampling_result.pos_assigned_gt_inds]
            bbox_weights[pos_inds] = 1.0
        
        all_cls_targets.append(cls_targets)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)
        all_num_pos.append(len(pos_inds))
    
    # Stack 成 batch
    cls_targets = torch.stack(all_cls_targets)  # [bs, num_query]
    bbox_targets = torch.stack(all_bbox_targets)  # [bs, num_query, code_size]
    bbox_weights = torch.stack(all_bbox_weights)  # [bs, num_query, code_size]
    num_pos = sum(all_num_pos)
    
    # 计算损失
    # 分类损失
    cls_scores_flatten = cls_scores.reshape(-1, self.num_classes)
    cls_targets_flatten = cls_targets.reshape(-1)
    
    loss_cls = self.loss_cls(
        cls_scores_flatten,
        cls_targets_flatten,
        avg_factor=max(num_pos, 1)
    )
    
    # 回归损失 (只对正样本)
    if num_pos > 0:
        # 提取所有正样本
        pos_mask = bbox_weights[..., 0] > 0  # [bs, num_query]
        
        bbox_preds_pos = bbox_preds[pos_mask]  # [num_pos_total, code_size]
        bbox_targets_pos = bbox_targets[pos_mask]  # [num_pos_total, code_size]
        
        loss_bbox = self.loss_bbox(
            bbox_preds_pos,
            bbox_targets_pos,
            avg_factor=num_pos
        )
    else:
        loss_bbox = bbox_preds.sum() * 0
    
    return {
        'loss_cls': loss_cls,
        'loss_bbox': loss_bbox,
    }
```

### Step 7: MemoryBank Batch 化 (Tracking)

**修改**: `MemoryBank` 类

```python
class MemoryBank:
    def __init__(self, batch_size=1):
        """
        Args:
            batch_size: 支持的批次大小
        """
        self.batch_size = batch_size
        
        # 为每个样本维护独立的记忆库
        self.memo_list = [None] * batch_size
        self.memo_embed_list = [None] * batch_size
    
    def get(self, batch_idx=0):
        """获取指定样本的记忆"""
        return self.memo_list[batch_idx], self.memo_embed_list[batch_idx]
    
    def update(self, batch_idx, track_instances):
        """更新指定样本的记忆"""
        self.memo_list[batch_idx] = track_instances
        # ... 更新逻辑
    
    def reset(self, batch_idx=None):
        """重置记忆
        
        Args:
            batch_idx: 如果为 None，重置所有；否则只重置指定样本
        """
        if batch_idx is None:
            self.memo_list = [None] * self.batch_size
            self.memo_embed_list = [None] * self.batch_size
        else:
            self.memo_list[batch_idx] = None
            self.memo_embed_list[batch_idx] = None
```

---

## 测试与验证

### 单元测试

**新建**: `tests/test_batch_support.py`

```python
import torch
import pytest
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from projects.mmdet3d_plugin.uniad.detectors.uniad_track import UniADTrack

class TestBatchSupport:
    
    def test_dataset_collate(self):
        """测试 Dataset 的 batch 支持"""
        # 创建 dataset
        dataset = NuScenesE2EDataset(...)
        
        # 获取多个样本
        samples = [dataset[i] for i in range(4)]
        
        # Collate
        from projects.mmdet3d_plugin.datasets.collate import uniad_collate_fn
        batch = uniad_collate_fn(samples)
        
        # 检查形状
        assert batch['img'].data.shape[0] == 4  # batch_size=4
        assert len(batch['img_metas'].data) == 4
    
    def test_backbone_batch(self):
        """测试 Backbone 的 batch 处理"""
        model = UniADTrack(...)
        
        # 模拟输入
        B, L, N, C, H, W = 4, 5, 6, 3, 928, 1600
        img = torch.randn(B, L, N, C, H, W).cuda()
        
        # 特征提取
        feats = model.extract_img_feat(img, img_metas=None, len_queue=L)
        
        # 检查形状
        assert feats[0].shape[0] == B
        assert feats[0].shape[1] == L
        assert feats[0].shape[2] == N
    
    def test_bev_encoder_batch(self):
        """测试 BEV Encoder 的 batch 处理"""
        # ... 类似测试
    
    def test_forward_batch(self):
        """测试完整前向传播"""
        model = UniADTrack(...).cuda()
        
        # 构造 batch 输入
        batch_dict = {
            'img': torch.randn(4, 5, 6, 3, 928, 1600).cuda(),
            'img_metas': [...],  # 4个样本的metas
            'gt_bboxes_3d': [...],
            'gt_labels_3d': [...],
        }
        
        # Forward
        losses = model(**batch_dict)
        
        # 检查损失
        assert 'loss_cls' in losses
        assert 'loss_bbox' in losses
```

### 渐进式验证策略

```
阶段 1: 数据流验证 (batch_size=2)
├─ Dataset collate 正确性
├─ img_metas 结构正确
└─ 数据形状一致性

阶段 2: 特征提取验证 (batch_size=2)
├─ Backbone reshape 正确
├─ Neck 输出形状正确
└─ 多层级特征对齐

阶段 3: BEV 转换验证 (batch_size=2)
├─ BEV query batch化
├─ 历史BEV旋转批量化
└─ Attention 计算正确性

阶段 4: 完整训练验证 (batch_size=2→4→8)
├─ 前向传播无错
├─ 损失计算正确
├─ 梯度反传正常
└─ 性能对比 (batch=1 vs batch>1)
```

---

## 性能优化建议

### 1. 数据加载优化

```python
# 使用更多 workers
data = dict(
    workers_per_gpu=8,  # 增加到8或更多
    persistent_workers=True,  # 保持worker进程
    pin_memory=True,  # 启用 pin memory
)
```

### 2. 混合精度训练

```python
# 已有 BF16 支持，确保启用
fp16 = dict(loss_scale=512.0)
# 或
bf16 = dict(loss_scale=1.0)
```

### 3. Gradient Checkpointing

```python
# 对于 BEVFormer Encoder 的多层
class BEVFormerEncoder(nn.Module):
    def __init__(self, use_checkpoint=False):
        self.use_checkpoint = use_checkpoint
    
    def forward(self, ...):
        for layer in self.layers:
            if self.use_checkpoint:
                output = checkpoint.checkpoint(
                    layer, bev_query, key, value, ...
                )
            else:
                output = layer(bev_query, key, value, ...)
```

### 4. 批量大小建议

| GPU | VRAM | 推荐 Batch Size | 备注 |
|-----|------|----------------|------|
| V100 32GB | 32GB | 2-4 | 需要 gradient checkpointing |
| A100 40GB | 40GB | 4-8 | 可以不用 checkpointing |
| A100 80GB | 80GB | 8-16 | 最佳性能 |
| H100 80GB | 80GB | 12-24 | BF16 加速 |

---

## 总结

### 重构工作量估计

| 模块 | 工作量 | 优先级 | 依赖 |
|------|--------|--------|------|
| Collate Function | 0.5天 | P0 | 无 |
| Dataset union2one | 0.5天 | P0 | 无 |
| Backbone reshape | 1天 | P0 | Collate |
| BEV Encoder batch | 2天 | P0 | Backbone |
| Detection Head | 1.5天 | P1 | BEV Encoder |
| Loss 计算 | 1天 | P1 | Detection Head |
| Segmentation Head | 1天 | P1 | BEV Encoder |
| MemoryBank | 2天 | P2 | Detection Head |
| 测试验证 | 2天 | P0 | 所有 |
| **总计** | **12天** | - | - |

### 关键难点

1. **img_metas 的嵌套结构处理**
   - 当前是 dict of dict (时序嵌套)
   - 需要在 batch 维度正确组织

2. **历史 BEV 的批量旋转**
   - 需要高效的批量旋转实现
   - 建议使用 `grid_sample` 或自定义 CUDA kernel

3. **Hungarian Matching 的并行化**
   - 每个样本的GT数量不同
   - 需要对每个样本单独匹配，但可以并行

4. **MemoryBank 的状态管理**
   - 跟踪需要维护每个样本的历史状态
   - 需要正确处理 scene 切换

### 预期收益

**训练速度提升** (基于 A100 80GB):

| Batch Size | 吞吐量 (samples/s) | 相对加速 |
|-----------|-------------------|----------|
| 1 (当前) | 0.5 | 1.0x |
| 2 | 0.85 | 1.7x |
| 4 | 1.4 | 2.8x |
| 8 | 2.2 | 4.4x |

**内存使用** (batch_size=4):
- 模型参数: ~800MB (不变)
- BEV features: ~2GB → ~8GB (4x)
- Gradients: ~800MB → ~3.2GB (4x)
- 总计: ~20GB → ~50GB

---

## 下一步行动

### 立即开始的工作

1. **阅读并理解当前代码**
   - 重点关注上述标注的 🔴 限制点
   - 梳理数据在各个模块间的流动

2. **搭建测试环境**
   - 准备小规模测试数据 (10个样本)
   - 编写单元测试框架

3. **实现 Step 1-2** (优先级最高)
   - Collate Function
   - img_metas 结构调整
   - 验证数据加载正确性

### 后续开发路线

```
Week 1:
  Day 1-2: Collate + Dataset 重构
  Day 3-4: Backbone/Neck 重构
  Day 5: 测试与修复

Week 2:
  Day 1-3: BEV Encoder 重构
  Day 4-5: Detection Head 重构

Week 3:
  Day 1-2: Loss 计算重构
  Day 3: Segmentation Head 重构
  Day 4-5: 完整测试

Week 4:
  Day 1-3: MemoryBank 重构
  Day 4-5: 性能优化与文档
```

希望这份详细分析能帮助你理解 UniAD 的数据流并顺利进行重构！
