# UniAD 模型训练前向传播流程

本文档详细描述了UniAD模型在训练过程中的完整前向传播流程，包括从图像输入到多任务输出的所有步骤。

## 目录

1. [整体架构概述](#整体架构概述)
2. [图像特征提取](#图像特征提取)
3. [BEV特征生成](#bev特征生成)
4. [多任务Head前向传播](#多任务head前向传播)
5. [完整数据流](#完整数据流)

---

## 整体架构概述

UniAD是一个端到端的自动驾驶感知和规划统一框架，采用**分层架构**：

```
输入: 多视角图像 (6个相机)
    ↓
图像特征提取 (ResNet101-DCN + FPN)
    ↓
BEV特征生成 (BEVFormer Encoder)
    ↓
多任务Head处理
    ├─> Detection & Tracking (BEVFormer Decoder)
    ├─> Map Segmentation (PansegFormer)
    ├─> Motion Prediction (Motion Transformer)
    ├─> Occupancy Prediction (Occ Transformer)
    └─> Planning (Planning MLP)
    ↓
输出: 多任务预测结果
```

### 核心组件

| 组件 | 类名 | 功能 |
|------|------|------|
| **主模型** | `UniAD` | 端到端多任务模型 |
| **跟踪基类** | `UniADTrack` | 检测和跟踪功能 |
| **图像编码器** | `ResNet101-DCN` | 提取图像特征 |
| **图像颈部** | `FPN` | 多尺度特征融合 |
| **BEV生成** | `BEVFormerTrackHead` | 生成BEV特征 |
| **分割Head** | `PansegformerHead` | 地图分割 |
| **运动Head** | `MotionHead` | 运动预测 |
| **占用Head** | `OccHead` | 占用预测 |
| **规划Head** | `PlanningHeadSingleMode` | 轨迹规划 |

---

## 图像特征提取

### 1. 输入数据格式

训练时的输入图像格式：

```python
# 输入shape
img: Tensor [B, L, N, C, H, W]
# B: batch size (通常为1)
# L: queue_length (时序帧数，例如5)
# N: num_cameras (6个相机)
# C: channels (3, RGB)
# H, W: 图像高度和宽度 (例如 900, 1600)

# 示例
img.shape = torch.Size([1, 5, 6, 3, 900, 1600])
```

### 2. Grid Mask数据增强

```python
class GridMask:
    """网格遮罩增强，随机遮挡图像的某些区域"""
    def __init__(self, 
                 use_h=True, 
                 use_w=True, 
                 rotate=1, 
                 offset=False, 
                 ratio=0.5, 
                 mode=1, 
                 prob=0.7):
        ...
```

**作用**：在训练时以0.7的概率对图像应用网格遮罩，增强模型鲁棒性。

### 3. 图像Backbone: ResNet101-DCN

```python
def extract_img_feat(self, img, len_queue=None):
    """提取图像特征
    Args:
        img: [B*L, N, C, H, W] 或 [B, N, C, H, W]
        len_queue: 时序长度
    Returns:
        img_feats_reshaped: List[Tensor]
            每个scale的特征: [B, L, N, C, H, W] 或 [B, N, C, H, W]
    """
    # 1. Reshape: 合并batch和时序维度
    B, N, C, H, W = img.size()
    img = img.reshape(B * N, C, H, W)  # [B*N, C, H, W]
    
    # 2. Grid Mask增强 (训练时)
    if self.use_grid_mask:
        img = self.grid_mask(img)
    
    # 3. ResNet101-DCN提取特征
    img_feats = self.img_backbone(img)
    # img_feats是list或dict，包含多个尺度的特征
    # 例如: [feat_level0, feat_level1, feat_level2, feat_level3]
    
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    
    # 4. FPN颈部网络
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)
    
    # 5. Reshape回原始维度
    img_feats_reshaped = []
    for img_feat in img_feats:
        _, c, h, w = img_feat.size()
        if len_queue is not None:
            # [B*L*N, C, H, W] -> [B, L, N, C, H, W]
            img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
        else:
            # [B*N, C, H, W] -> [B, N, C, H, W]
            img_feat_reshaped = img_feat.view(B, N, c, h, w)
        img_feats_reshaped.append(img_feat_reshaped)
    
    return img_feats_reshaped
```

**输出**：多尺度特征列表

```python
# 4个尺度的特征
img_feats_reshaped = [
    feat_level0,  # [B, L, N, 256, H/8, W/8]
    feat_level1,  # [B, L, N, 256, H/16, W/16]
    feat_level2,  # [B, L, N, 256, H/32, W/32]
    feat_level3,  # [B, L, N, 256, H/64, W/64]
]
```

### 4. ResNet101-DCN结构

```
Input: [B*N, 3, 900, 1600]
    ↓
Conv1 + MaxPool
    ↓
Layer1 (ResBlock × 3)  → output: [B*N, 256, 225, 400]
    ↓
Layer2 (ResBlock × 4)  → output: [B*N, 512, 113, 200]  → FPN Level 0
    ↓
Layer3 (ResBlock × 23, with DCN) → output: [B*N, 1024, 57, 100] → FPN Level 1
    ↓
Layer4 (ResBlock × 3, with DCN) → output: [B*N, 2048, 29, 50]  → FPN Level 2
```

**DCN (Deformable Convolution v2)**：
- 在Layer3和Layer4中使用
- 允许卷积核根据内容自适应调整采样位置
- 提高对形变物体的特征提取能力

### 5. FPN (Feature Pyramid Network)

```python
# FPN配置
img_neck=dict(
    type="FPN",
    in_channels=[512, 1024, 2048],  # 从ResNet Layer2,3,4
    out_channels=256,                # 统一输出通道数
    start_level=0,
    add_extra_convs="on_output",     # 在输出上添加额外卷积层
    num_outs=4,                      # 输出4个尺度
    relu_before_extra_convs=True,
)
```

**FPN流程**：

```
ResNet输出:
    Layer2: [512, H/8, W/8]
    Layer3: [1024, H/16, W/16]
    Layer4: [2048, H/32, W/32]
        ↓
    1×1卷积统一到256通道
        ↓
    自顶向下融合
        ↓
FPN输出 (4个尺度):
    P2: [256, H/8, W/8]
    P3: [256, H/16, W/16]
    P4: [256, H/32, W/32]
    P5: [256, H/64, W/64]  (额外添加)
```

---

## BEV特征生成

BEV (Bird's Eye View) 特征是UniAD的核心，所有下游任务都基于BEV特征进行。

### ⚠️ 重要：BEV特征共享机制

**UniAD中BEV特征只在Track阶段生成一次，所有后续任务（seg_head、motion_head、occ_head、planning_head）都复用同一个BEV特征图。**

#### 为什么这样设计？

1. **计算效率高**
   - BEVFormer Encoder是最耗时的部分（需要多尺度可变形注意力）
   - 只计算一次BEV特征，所有任务共享，大幅节省计算量
   - 避免重复的图像特征提取和BEV转换

2. **统一的场景表示**
   - 所有任务在同一个BEV空间下工作
   - 保证了不同任务之间的空间一致性
   - 便于任务间的信息融合

3. **任务间信息流动**
   - 虽然BEV特征共享，但任务之间还有额外的信息传递
   - Track输出的query embedding传递给Motion
   - Motion输出的轨迹信息传递给Occupancy和Planning

#### BEV特征生成与复用流程

```python
# 1. Track阶段：生成BEV特征（唯一一次）
losses_track, outs_track = self.forward_track_train(...)
bev_embed = outs_track["bev_embed"]  # [H*W, B, C] = [40000, 1, 256]
bev_pos = outs_track["bev_pos"]      # [B, C, H, W] = [1, 256, 200, 200]

# 2. 所有后续任务直接复用bev_embed
# Map Segmentation
losses_seg, outs_seg = self.seg_head.forward_train(
    bev_embed,  # ← 复用BEV特征
    img_metas, gt_lane_labels, gt_lane_bboxes, gt_lane_masks
)

# Motion Prediction
ret_dict_motion = self.motion_head.forward_train(
    bev_embed,  # ← 复用BEV特征
    gt_bboxes_3d, gt_labels_3d, gt_fut_traj, gt_fut_traj_mask,
    outs_track=outs_track,  # + Track的query信息
    outs_seg=outs_seg       # + Seg的地图信息
)

# Occupancy Prediction
losses_occ = self.occ_head.forward_train(
    bev_embed,  # ← 复用BEV特征
    outs_motion,  # + Motion的轨迹信息
    gt_segmentation=gt_segmentation, ...
)

# Planning
outs_planning = self.planning_head.forward_train(
    bev_embed,  # ← 复用BEV特征
    outs_motion,  # + Motion的agent信息
    sdc_planning, sdc_planning_mask, command, ...
)
```

#### 数据流示意图

```
输入图像 [B, L, N, 3, H, W]
        ↓
   ╔════════════════════════════╗
   ║   Track Head 前向传播      ║
   ╠════════════════════════════╣
   ║ 1. 图像特征提取            ║
   ║    ResNet101-DCN + FPN     ║
   ║ 2. BEV特征生成 ⭐          ║
   ║    BEVFormer Encoder       ║
   ║    (最耗时的部分！)         ║
   ║ 3. 目标检测与跟踪          ║
   ║    BEVFormer Decoder       ║
   ╚════════════════════════════╝
        ↓
   输出: bev_embed (共享) + track_query + bbox
        ↓
   ┌────┴────┬─────────┬─────────┬─────────┐
   ↓         ↓         ↓         ↓         ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Seg  │ │Motion│ │ Occ  │ │Plan  │ │ ...  │
│ Head │ │ Head │ │ Head │ │ Head │ │      │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘
   ↑         ↑         ↑         ↑         ↑
   └─────────┴─────────┴─────────┴─────────┘
         所有任务都使用同一个 bev_embed
```

#### 推理时也是相同的机制

```python
# forward_test中的流程
result_track = self.simple_test_track(...)
bev_embed = result_track[0]["bev_embed"]  # 提取BEV特征

# 所有任务复用
result_seg = self.seg_head.forward_test(bev_embed, ...)
result_motion, outs_motion = self.motion_head.forward_test(bev_embed, ...)
outs_occ = self.occ_head.forward_test(bev_embed, ...)
result_planning = self.planning_head.forward_test(bev_embed, ...)
```

### 1. BEV生成入口

```python
def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
    """生成BEV特征
    Args:
        imgs: [B, N, C, H, W] 当前帧图像
        img_metas: 图像元数据
        prev_bev: 历史BEV特征 (用于时序建模)
    Returns:
        bev_embed: [H*W, B, C] BEV特征
        bev_pos: [B, C, H, W] BEV位置编码
    """
    # 1. 提取图像特征
    img_feats = self.extract_img_feat(img=imgs)
    
    # 2. 调用BEVFormer生成BEV
    if self.freeze_bev_encoder:
        with torch.no_grad():
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, 
                img_metas=img_metas, 
                prev_bev=prev_bev)
    else:
        bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
            mlvl_feats=img_feats, 
            img_metas=img_metas, 
            prev_bev=prev_bev)
    
    # 3. Reshape BEV特征
    if bev_embed.shape[1] == self.bev_h * self.bev_w:
        bev_embed = bev_embed.permute(1, 0, 2)  # [B, H*W, C] -> [H*W, B, C]
    
    assert bev_embed.shape[0] == self.bev_h * self.bev_w
    return bev_embed, bev_pos
```

### 2. BEVFormer Encoder详细流程

```python
def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
    """
    Args:
        mlvl_feats: List[Tensor] 多尺度图像特征
            每个元素shape: [B, N, C, H, W]
        img_metas: List[dict] 图像元数据
        prev_bev: Tensor [B, H*W, C] 历史BEV特征
    Returns:
        bev_embed: [H*W, B, C] BEV特征
        bev_pos: [B, C, H, W] BEV位置编码
    """
    bs, num_cam, _, _, _ = mlvl_feats[0].shape
    dtype = mlvl_feats[0].dtype
    
    # 1. 初始化BEV queries
    bev_queries = self.bev_embedding.weight.to(dtype)  # [H*W, C]
    
    # 2. 生成BEV位置编码
    bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                           device=bev_queries.device).to(dtype)
    bev_pos = self.positional_encoding(bev_mask).to(dtype)  # [B, C, H, W]
    
    # 3. Transformer编码
    bev_embed = self.transformer.get_bev_features(
        mlvl_feats,      # 多尺度图像特征
        bev_queries,     # BEV查询
        self.bev_h,
        self.bev_w,
        grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        bev_pos=bev_pos,
        img_metas=img_metas,
        prev_bev=prev_bev,  # 历史BEV特征（如果有）
    )
    
    return bev_embed, bev_pos
```

#### 历史BEV特征的生成：`get_history_bev`

在训练时，如果需要使用历史帧信息，会通过 `get_history_bev` 函数**递归生成历史BEV特征**：

```python
def get_history_bev(self, imgs_queue, img_metas_list):
    """
    以节省显存的方式递归生成历史帧的BEV特征
    
    Args:
        imgs_queue: [B, len_queue, N, C, H, W] 历史帧图像队列
        img_metas_list: 历史帧元数据列表
    Returns:
        prev_bev: 最后一帧的BEV特征，用于当前帧
    """
    self.eval()  # 切换到评估模式
    with torch.no_grad():  # ⚠️ 不计算梯度，节省显存
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
        
        # 提取所有历史帧的图像特征
        img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
        
        # 逐帧迭代生成BEV（每一帧都使用前一帧的BEV）
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            
            # 使用前一帧的BEV生成当前帧的BEV
            prev_bev, _ = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, 
                img_metas=img_metas, 
                prev_bev=prev_bev  # 时序递归：Frame_i 使用 BEV_(i-1)
            )
    
    self.train()  # 恢复训练模式
    return prev_bev  # 返回最后一帧的BEV，作为当前帧的历史上下文
```

**时序递归过程**：

```
历史帧序列: [Frame_0, Frame_1, Frame_2]

Step 1: prev_bev = None
        → 生成 bev_0 (无历史信息)

Step 2: prev_bev = bev_0
        → 生成 bev_1 (融合了 Frame_0 的信息)

Step 3: prev_bev = bev_1
        → 生成 bev_2 (融合了 Frame_0 + Frame_1 的信息)

返回: bev_2 作为当前帧的 prev_bev
```

**调用流程**：

```python
def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
    """生成当前帧的BEV特征"""
    
    # 如果提供了历史图像，先生成历史BEV
    if prev_img is not None and prev_img_metas is not None:
        assert prev_bev is None
        prev_bev = self.get_history_bev(prev_img, prev_img_metas)
    
    # 提取当前帧图像特征
    img_feats = self.extract_img_feat(img=imgs)
    
    # 生成当前帧BEV（使用历史BEV作为时序上下文）
    bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
        mlvl_feats=img_feats, 
        img_metas=img_metas, 
        prev_bev=prev_bev  # 历史BEV特征
    )
    
    return bev_embed, bev_pos
```

**为什么使用 `torch.no_grad()`？**

1. **节省显存**：历史帧只用于提供上下文，不需要反向传播
2. **加速计算**：不保存中间激活值和梯度信息
3. **训练稳定**：历史帧的特征提取不影响当前帧的梯度更新

**训练时的具体使用场景**：

```python
# forward_track_train 中，逐帧处理视频序列
for i in range(num_frame):  # 例如处理5帧序列
    # 获取当前帧之前的所有历史帧
    prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
    prev_img_metas = copy.deepcopy(img_metas)
    img_single = torch.stack([img_[i] for img_ in img], dim=0)
    
    # 调用单帧前向传播
    frame_res = self._forward_single_frame_train(
        img_single,      # 当前帧
        img_metas_single,
        track_instances,
        prev_img,        # 历史帧 [Frame_0, ..., Frame_(i-1)]
        prev_img_metas,  # 历史帧元数据
        ...
    )
    # 在 _forward_single_frame_train 内部会调用 get_bevs
    # get_bevs 会调用 get_history_bev 处理 prev_img
```

**示例**：

```
训练序列: [Frame_0, Frame_1, Frame_2, Frame_3, Frame_4]

处理 Frame_2:
├─ prev_img = [Frame_0, Frame_1]  (前2帧)
├─ 调用 get_history_bev([Frame_0, Frame_1])
│   ├─ i=0: prev_bev=None → 生成 bev_0
│   └─ i=1: prev_bev=bev_0 → 生成 bev_1
├─ 返回 prev_bev = bev_1
└─ 使用 bev_1 生成 Frame_2 的BEV (融合了历史信息)

处理 Frame_3:
├─ prev_img = [Frame_0, Frame_1, Frame_2]  (前3帧)
├─ 调用 get_history_bev([Frame_0, Frame_1, Frame_2])
│   ├─ i=0: prev_bev=None → 生成 bev_0
│   ├─ i=1: prev_bev=bev_0 → 生成 bev_1
│   └─ i=2: prev_bev=bev_1 → 生成 bev_2
├─ 返回 prev_bev = bev_2
└─ 使用 bev_2 生成 Frame_3 的BEV
```

---

### 完整示例：5帧连续训练的BEV生成过程

假设训练时使用 `queue_length=5` 的连续帧序列，下面详细说明整个BEV生成过程：

#### 输入数据

```python
# 训练输入
img: Tensor [B, L, N, C, H, W]
# B=1 (batch size)
# L=5 (len_queue，5帧连续图像)
# N=6 (6个相机)
# C=3 (RGB通道)
# H, W=900, 1600 (图像尺寸)

# 示例
img.shape = torch.Size([1, 5, 6, 3, 900, 1600])
# 5帧时序连续：[Frame_0, Frame_1, Frame_2, Frame_3, Frame_4]
```

#### 训练主循环：逐帧处理

```python
def forward_track_train(self, img, ...):
    """
    关键：虽然输入是5帧，但在训练中是逐帧处理的
    """
    track_instances = self._generate_empty_tracks()
    num_frame = img.size(1)  # num_frame = 5
    
    # 初始化每帧的GT
    gt_instances_list = []
    for i in range(num_frame):  # 为每一帧准备GT
        gt_instances = Instances((1, 1))
        boxes = normalize_bbox(gt_bboxes_3d[0][i].tensor, self.pc_range)
        gt_instances.boxes = boxes
        gt_instances.labels = gt_labels_3d[0][i]
        gt_instances.obj_ids = gt_inds[0][i]
        gt_instances_list.append(gt_instances)
    
    # ========== 核心：逐帧前向传播 ==========
    for i in range(num_frame):  # i = 0, 1, 2, 3, 4
        # 1. 准备历史帧（第i帧之前的所有帧）
        prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
        # i=0: [Frame_0]  (特殊处理)
        # i=1: [Frame_0]
        # i=2: [Frame_0, Frame_1]
        # i=3: [Frame_0, Frame_1, Frame_2]
        # i=4: [Frame_0, Frame_1, Frame_2, Frame_3]
        
        # 2. 提取当前帧
        img_single = torch.stack([img_[i] for img_ in img], dim=0)
        # [1, 6, 3, 900, 1600]
        
        # 3. 单帧前向传播
        frame_res = self._forward_single_frame_train(
            img_single,      # 当前帧
            img_metas_single,
            track_instances,
            prev_img,        # 历史帧
            prev_img_metas,
            ...
        )
        
        track_instances = frame_res["track_instances"]
    
    # 返回最后一帧（Frame_4）的结果
    out["bev_embed"] = frame_res["bev_embed"]  # Frame_4 的 BEV
    return losses, out
```

#### 每一帧的详细BEV生成过程

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**  
**Frame 0 (i=0)：第一帧**  
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

```python
# 输入
prev_img = img[:, :1, ...]  # [1, 1, 6, 3, 900, 1600] (只有Frame_0自己)
img_single = img[:, 0, ...]  # [1, 6, 3, 900, 1600] (Frame_0)

# 调用 get_bevs → get_history_bev
def get_history_bev(imgs_queue, ...):  # imgs_queue=[1,1,6,3,900,1600]
    len_queue = 1  # 只有1帧
    prev_bev = None
    
    # 循环1次
    for i in range(1):
        img_feats = extract_img_feat(Frame_0)
        prev_bev, _ = get_bev_features(
            img_feats, 
            prev_bev=None  # ⚠️ 第一帧没有历史
        )
    return prev_bev  # bev_0

# 回到 get_bevs
prev_bev = bev_0  # Frame_0 的 BEV

# 生成当前帧 Frame_0 的最终 BEV
img_feats = extract_img_feat(Frame_0)
bev_embed, bev_pos = get_bev_features(
    img_feats,
    prev_bev=bev_0  # 使用自己作为"历史"
)
# bev_embed: Frame_0 的 BEV (Temporal Self-Attention 对自己)
```

**结果**：Frame_0 的 BEV 没有真正的历史信息，只包含 Frame_0 自身。

---

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**  
**Frame 1 (i=1)：第二帧**  
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

```python
# 输入
prev_img = img[:, :1, ...]  # [1, 1, 6, 3, 900, 1600] (Frame_0)
img_single = img[:, 1, ...]  # [1, 6, 3, 900, 1600] (Frame_1)

# 调用 get_history_bev
def get_history_bev(imgs_queue, ...):  # imgs_queue=[1,1,6,3,900,1600]
    len_queue = 1
    prev_bev = None
    
    # 循环1次：处理 Frame_0
    for i in range(1):
        img_feats = extract_img_feat(Frame_0)
        prev_bev, _ = get_bev_features(
            img_feats,
            prev_bev=None  # Frame_0 没有历史
        )
    return prev_bev  # bev_0

# 回到 get_bevs
prev_bev = bev_0  # Frame_0 的 BEV

# 生成 Frame_1 的 BEV
img_feats = extract_img_feat(Frame_1)
bev_embed, bev_pos = get_bev_features(
    img_feats,          # Frame_1 的特征
    prev_bev=bev_0      # ✅ Frame_0 的 BEV
)

# 在 Temporal Self-Attention 中
query = Frame_1_BEV_queries
key = cat([Frame_1_BEV_queries, bev_0])  # 拼接当前和历史
value = cat([Frame_1_BEV_queries, bev_0])
# Frame_1 的每个位置关注 Frame_1 和 Frame_0
```

**结果**：Frame_1 的 BEV 融合了 Frame_0 的时序信息。

---

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**  
**Frame 2 (i=2)：第三帧**  
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

```python
# 输入
prev_img = img[:, :2, ...]  # [1, 2, 6, 3, 900, 1600] (Frame_0, Frame_1)
img_single = img[:, 2, ...]  # [1, 6, 3, 900, 1600] (Frame_2)

# 调用 get_history_bev (递归处理2帧历史)
def get_history_bev(imgs_queue, ...):  # imgs_queue=[1,2,6,3,900,1600]
    len_queue = 2
    prev_bev = None
    
    # 第1次循环：处理 Frame_0
    img_feats = extract_img_feat(Frame_0)
    prev_bev, _ = get_bev_features(
        img_feats,
        prev_bev=None  # Frame_0 没有历史
    )
    # prev_bev = bev_0
    
    # 第2次循环：处理 Frame_1
    img_feats = extract_img_feat(Frame_1)
    prev_bev, _ = get_bev_features(
        img_feats,
        prev_bev=bev_0  # ✅ 使用 Frame_0 的 BEV
    )
    # prev_bev = bev_1 (已融合 Frame_0)
    
    return prev_bev  # bev_1

# 回到 get_bevs
prev_bev = bev_1  # Frame_1 的 BEV (包含 Frame_0 信息)

# 生成 Frame_2 的 BEV
img_feats = extract_img_feat(Frame_2)
bev_embed, bev_pos = get_bev_features(
    img_feats,          # Frame_2 的特征
    prev_bev=bev_1      # ✅ Frame_1 的 BEV (已包含 Frame_0)
)

# Temporal Self-Attention
query = Frame_2_BEV_queries
key = cat([Frame_2_BEV_queries, bev_1])  # bev_1 包含 Frame_0 信息
# Frame_2 的 BEV 融合 Frame_1 (间接融合 Frame_0)
```

**结果**：Frame_2 的 BEV 融合了 Frame_0, Frame_1, Frame_2 的信息（通过递归传递）。

---

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**  
**Frame 3 (i=3)：第四帧**  
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

```python
prev_img = [Frame_0, Frame_1, Frame_2]  # 3帧历史

# get_history_bev 递归处理3帧
├─ i=0: prev_bev=None → bev_0
├─ i=1: prev_bev=bev_0 → bev_1 (融合 Frame_0)
└─ i=2: prev_bev=bev_1 → bev_2 (融合 Frame_0, 1)

prev_bev = bev_2  # 包含 Frame_0, 1 信息

# 生成 Frame_3 的 BEV
bev_embed = get_bev_features(Frame_3, prev_bev=bev_2)
# 融合 Frame_0, 1, 2, 3
```

---

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**  
**Frame 4 (i=4)：第五帧（最后一帧）⭐**  
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

```python
prev_img = [Frame_0, Frame_1, Frame_2, Frame_3]  # 4帧历史

# get_history_bev 递归处理4帧
├─ i=0: prev_bev=None → bev_0
├─ i=1: prev_bev=bev_0 → bev_1
├─ i=2: prev_bev=bev_1 → bev_2
└─ i=3: prev_bev=bev_2 → bev_3 (融合 Frame_0, 1, 2)

prev_bev = bev_3  # 包含 Frame_0, 1, 2 信息

# 生成 Frame_4 的 BEV（最终输出）
img_feats = extract_img_feat(Frame_4)
bev_embed, bev_pos = get_bev_features(
    img_feats,
    prev_bev=bev_3  # 使用包含前4帧信息的 BEV
)
# bev_embed: [40000, 1, 256]
# 融合了所有 5 帧的时序信息
```

**结果**：Frame_4 的 BEV 是最终返回的特征，包含了完整的时序信息。

---

#### 完整流程可视化

```
输入: [Frame_0, Frame_1, Frame_2, Frame_3, Frame_4]
      ↓ 逐帧处理
      
┌─────────────────────────────────────────────────────────┐
│ 循环 i=0: Frame_0                                       │
├─────────────────────────────────────────────────────────┤
│ prev_img = [Frame_0]  (特殊处理)                        │
│   ↓ get_history_bev                                     │
│   └─ i=0: prev_bev=None → bev_0                         │
│ prev_bev = bev_0                                        │
│ 生成: bev_0' (Temporal Self-Attn 对自己)                │
│ 信息: Frame_0                                           │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 循环 i=1: Frame_1                                       │
├─────────────────────────────────────────────────────────┤
│ prev_img = [Frame_0]                                    │
│   ↓ get_history_bev                                     │
│   └─ i=0: prev_bev=None → bev_0                         │
│ prev_bev = bev_0                                        │
│ 生成: bev_1 (融合 bev_0)                                │
│ 信息: Frame_0, Frame_1                                  │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 循环 i=2: Frame_2                                       │
├─────────────────────────────────────────────────────────┤
│ prev_img = [Frame_0, Frame_1]                           │
│   ↓ get_history_bev (递归)                              │
│   ├─ i=0: prev_bev=None → bev_0                         │
│   └─ i=1: prev_bev=bev_0 → bev_1                        │
│ prev_bev = bev_1 (包含 Frame_0)                         │
│ 生成: bev_2 (融合 bev_1)                                │
│ 信息: Frame_0, Frame_1, Frame_2                         │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 循环 i=3: Frame_3                                       │
├─────────────────────────────────────────────────────────┤
│ prev_img = [Frame_0, Frame_1, Frame_2]                  │
│   ↓ get_history_bev (递归)                              │
│   ├─ i=0: prev_bev=None → bev_0                         │
│   ├─ i=1: prev_bev=bev_0 → bev_1                        │
│   └─ i=2: prev_bev=bev_1 → bev_2                        │
│ prev_bev = bev_2 (包含 Frame_0, 1)                      │
│ 生成: bev_3 (融合 bev_2)                                │
│ 信息: Frame_0, 1, 2, 3                                  │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 循环 i=4: Frame_4 (最后一帧) ⭐                         │
├─────────────────────────────────────────────────────────┤
│ prev_img = [Frame_0, Frame_1, Frame_2, Frame_3]         │
│   ↓ get_history_bev (递归)                              │
│   ├─ i=0: prev_bev=None → bev_0                         │
│   ├─ i=1: prev_bev=bev_0 → bev_1                        │
│   ├─ i=2: prev_bev=bev_1 → bev_2                        │
│   └─ i=3: prev_bev=bev_2 → bev_3                        │
│ prev_bev = bev_3 (包含 Frame_0, 1, 2)                   │
│ 生成: bev_4 (融合 bev_3)                                │
│ 信息: Frame_0, 1, 2, 3, 4 (完整时序)                    │
└─────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────┐
│ 返回最终结果                                            │
├─────────────────────────────────────────────────────────┤
│ out["bev_embed"] = bev_4  [40000, 1, 256]               │
│ out["bev_pos"] = bev_pos_4  [1, 256, 200, 200]          │
│   ↓                                                     │
│ 传递给所有下游任务                                       │
│ ├─ Seg Head                                             │
│ ├─ Motion Head                                          │
│ ├─ Occ Head                                             │
│ └─ Planning Head                                        │
└─────────────────────────────────────────────────────────┘
```

#### 关键要点

1. **逐帧处理，非并行**：
   - 虽然输入包含5帧，但训练时是在 `for` 循环中**逐帧处理**
   - 每一帧都会生成独立的 BEV 特征

2. **时序信息递归累积**：
   - Frame_0: 无历史
   - Frame_1: 融合 Frame_0
   - Frame_2: 融合 Frame_0, 1
   - Frame_3: 融合 Frame_0, 1, 2
   - Frame_4: 融合 Frame_0, 1, 2, 3 (完整时序)

3. **显存优化策略**：
   - `get_history_bev` 使用 `torch.no_grad()`
   - 历史帧的 BEV 生成不计算梯度
   - 只有当前帧的 BEV 生成参与反向传播

4. **梯度流动**：
   ```
   Frame_0: 不计算梯度 (在 get_history_bev 中)
   Frame_1: 不计算梯度 (在 get_history_bev 中)
   Frame_2: 不计算梯度 (在 get_history_bev 中)
   Frame_3: 不计算梯度 (在 get_history_bev 中)
   Frame_4: ✅ 计算梯度 (在 get_bevs 中)
   ```

5. **最终输出**：
   - 返回的 `bev_embed` 是 **Frame_4 的 BEV 特征**
   - 它通过 Temporal Self-Attention 融合了所有 5 帧的信息
   - 这个 BEV 特征被所有下游任务共享使用

6. **计算复杂度**：
   - 每一帧都需要调用 `get_history_bev` 递归处理历史帧
   - Frame_4 的处理最耗时：需要递归处理 4 帧历史
   - 但通过 `no_grad` 避免了梯度计算的开销

### 3. BEVFormer Transformer结构

BEVFormer Transformer包含**Encoder**和**Decoder**两部分，BEV生成主要使用Encoder。

#### Encoder配置

```python
encoder=dict(
    type="BEVFormerEncoder",
    num_layers=6,  # 6层Encoder
    pc_range=point_cloud_range,
    num_points_in_pillar=4,  # 每个pillar采样4个点
    return_intermediate=False,
    transformerlayers=dict(
        type="BEVFormerLayer",
        attn_cfgs=[
            # 1. Temporal Self-Attention
            dict(
                type="TemporalSelfAttention", 
                embed_dims=256, 
                num_levels=1
            ),
            # 2. Spatial Cross-Attention
            dict(
                type="SpatialCrossAttention",
                pc_range=point_cloud_range,
                deformable_attention=dict(
                    type="MSDeformableAttention3D",
                    embed_dims=256,
                    num_points=8,  # 每个查询采样8个点
                    num_levels=4,  # 4个特征层级
                ),
                embed_dims=256,
            ),
        ],
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=(
            "self_attn",   # Temporal自注意力
            "norm",
            "cross_attn",  # Spatial交叉注意力
            "norm",
            "ffn",         # 前馈网络
            "norm",
        ),
    ),
)
```

#### BEVFormer Encoder Layer流程

```python
class BEVFormerLayer(BaseTransformerLayer):
    def forward(self,
                query,           # BEV queries [H*W, B, C]
                key,             # 图像特征 (多尺度)
                value,           # 图像特征 (多尺度)
                bev_pos,         # BEV位置编码
                ref_2d,          # 2D参考点
                ref_3d,          # 3D参考点
                bev_h, bev_w,
                spatial_shapes,  # 每个尺度的空间形状
                level_start_index,
                prev_bev=None,   # 历史BEV
                **kwargs):
        """
        操作顺序: self_attn -> norm -> cross_attn -> norm -> ffn -> norm
        """
        # 1. Temporal Self-Attention (如果有历史BEV)
        if self.operation_order[0] == 'self_attn':
            if prev_bev is not None:
                # 将当前BEV和历史BEV拼接
                query_cat = torch.cat([query, prev_bev], dim=0)
            else:
                query_cat = query
            
            # 执行自注意力
            query = self.attentions[0](
                query,
                query_cat,
                query_cat,
                identity=identity,
                query_pos=bev_pos,
                **kwargs
            )
            query = self.norms[0](query)
        
        # 2. Spatial Cross-Attention
        if self.operation_order[2] == 'cross_attn':
            query = self.attentions[1](
                query,
                key,     # 图像特征
                value,   # 图像特征
                identity=identity,
                query_pos=bev_pos,
                reference_points=ref_3d,  # 3D参考点
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
            query = self.norms[1](query)
        
        # 3. FFN
        query = self.ffns[0](query, identity=identity)
        query = self.norms[2](query)
        
        return query
```

### 4. Temporal Self-Attention机制

```python
class TemporalSelfAttention(nn.Module):
    """时序自注意力，用于融合历史BEV信息"""
    
    def forward(self,
                query,        # 当前BEV [H*W, B, C]
                key,          # 当前BEV + 历史BEV [2*H*W, B, C]
                value,        # 当前BEV + 历史BEV [2*H*W, B, C]
                query_pos,    # BEV位置编码
                **kwargs):
        """
        通过自注意力机制，让当前BEV的每个位置关注:
        1. 当前BEV的其他位置
        2. 历史BEV的对应位置 (考虑车辆运动的位移)
        """
        # 1. 加入位置编码
        query = query + query_pos
        key = key + query_pos
        
        # 2. 多头自注意力
        # Q: [H*W, B, C]
        # K, V: [2*H*W, B, C] (包含历史)
        output = self.multihead_attn(
            query=query,
            key=key,
            value=value,
        )
        
        return output
```

**作用**：
- 融合当前帧和历史帧的BEV信息
- 利用时序连续性，提高特征质量
- 通过CAN Bus信息对齐不同时刻的BEV

### 5. Spatial Cross-Attention机制

```python
class SpatialCrossAttention(nn.Module):
    """空间交叉注意力，从图像特征中采样信息到BEV"""
    
    def forward(self,
                query,              # BEV queries [H*W, B, C]
                key,                # 图像特征 (多尺度flatten)
                value,              # 图像特征 (多尺度flatten)
                reference_points,   # 3D参考点 [H*W, B, Z, 2]
                spatial_shapes,     # 每个尺度的[H, W]
                **kwargs):
        """
        对于BEV的每个位置(x, y):
        1. 根据相机参数，投影到各个相机的图像平面
        2. 在多个高度z上采样 (num_points_in_pillar=4)
        3. 从多尺度图像特征中采样 (MSDeformableAttention)
        4. 聚合采样的特征
        """
        # 1. 生成采样点
        # reference_points: [H*W, B, num_cams, Z, 2]
        # Z个高度，每个高度在图像上有(u,v)坐标
        
        # 2. 可变形注意力采样
        output = self.deformable_attention(
            query=query,
            key=key,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )
        
        return output
```

**采样点生成过程**：

```
BEV位置 (x, y) 在地面上
    ↓
在不同高度z采样: [z1, z2, z3, z4]
    ↓
3D点: [(x,y,z1), (x,y,z2), (x,y,z3), (x,y,z4)]
    ↓
投影到6个相机: 使用lidar2img矩阵
    ↓
图像坐标: (u, v) for each camera
    ↓
从多尺度特征图采样: 使用双线性插值
    ↓
聚合特征: 加权求和
```

### 6. BEV特征输出

经过6层Encoder后：

```python
# 输出BEV特征
bev_embed: Tensor [H*W, B, C]
    # H = bev_h = 200
    # W = bev_w = 200
    # C = embed_dims = 256
    # 总共40000个BEV位置，每个位置256维特征

bev_pos: Tensor [B, C, H, W]
    # BEV位置编码
    # 用于后续任务的位置信息
```

**BEV特征表示**：
- 俯视图的栅格特征
- 每个栅格对应地面上0.5m × 0.5m的区域
- 覆盖范围：[-51.2m, 51.2m] × [-51.2m, 51.2m]

---

## 多任务Head前向传播

基于BEV特征，UniAD执行5个任务的前向传播。

### 1. Detection & Tracking Head

#### 1.1 Track Instances初始化

```python
def _generate_empty_tracks(self):
    """初始化跟踪实例"""
    track_instances = Instances((1, 1))
    num_queries = 901  # 900个物体query + 1个ego query
    dim = 256 * 2  # 512
    device = self.query_embedding.weight.device
    
    # Query embedding
    query = self.query_embedding.weight  # [901, 512]
    track_instances.query = query
    
    # 参考点 (normalized coordinates)
    track_instances.ref_pts = self.reference_points(query[..., :dim//2])
    # [901, 3] (x, y, z)
    
    # 初始化预测框
    track_instances.pred_boxes = torch.zeros((901, 10))
    # 10维: (x, y, w, l, z, h, sin(θ), cos(θ), vx, vy)
    
    # 其他属性
    track_instances.obj_idxes = torch.full((901,), -1, dtype=torch.long)
    track_instances.scores = torch.zeros(901)
    track_instances.pred_logits = torch.zeros((901, num_classes))
    
    # Memory bank (用于时序跟踪)
    track_instances.mem_bank = torch.zeros((901, mem_bank_len, 256))
    track_instances.mem_padding_mask = torch.ones((901, mem_bank_len), dtype=torch.bool)
    
    return track_instances
```

#### 1.2 Detection Head前向传播

```python
def get_detections(self, 
                   bev_embed,              # [H*W, B, C] BEV特征
                   object_query_embeds,    # [901, 512] Query embedding
                   ref_points,             # [901, 3] 参考点
                   img_metas):
    """
    使用BEVFormer Decoder进行目标检测
    """
    # 1. Decoder前向传播
    hs, references = self.transformer.decoder(
        query=object_query_embeds[:, :256],  # [901, 256]
        key=bev_embed,                        # [H*W, B, 256]
        value=bev_embed,
        query_pos=object_query_embeds[:, 256:],  # [901, 256]
        reference_points=ref_points,         # [901, 3]
        spatial_shapes=torch.tensor([[bev_h, bev_w]]),
        **kwargs
    )
    # hs: [num_layers, B, 901, 256] 每层的输出
    # references: [num_layers, B, 901, 3] 每层的参考点
    
    # 2. 分类和回归头
    all_cls_scores = []
    all_bbox_preds = []
    all_past_traj_preds = []
    
    for lvl in range(hs.shape[0]):  # 遍历每一层
        # 分类
        cls_scores = self.cls_branches[lvl](hs[lvl])  # [B, 901, num_classes]
        
        # 回归
        bbox_preds = self.reg_branches[lvl](hs[lvl])  # [B, 901, 10]
        
        # 历史轨迹
        past_traj = self.past_traj_reg_branches[lvl](hs[lvl])  
        # [B, 901, (past_steps+fut_steps)*2]
        
        all_cls_scores.append(cls_scores)
        all_bbox_preds.append(bbox_preds)
        all_past_traj_preds.append(past_traj)
    
    # 3. 堆叠所有层的输出
    all_cls_scores = torch.stack(all_cls_scores)  # [num_layers, B, 901, 10]
    all_bbox_preds = torch.stack(all_bbox_preds)  # [num_layers, B, 901, 10]
    all_past_traj_preds = torch.stack(all_past_traj_preds)
    
    # 4. 获取最后一层的参考点和特征
    last_ref_pts = references[-1]  # [B, 901, 3]
    query_feats = hs  # [num_layers, B, 901, 256]
    
    return {
        "all_cls_scores": all_cls_scores,
        "all_bbox_preds": all_bbox_preds,
        "all_past_traj_preds": all_past_traj_preds,
        "last_ref_points": last_ref_pts,
        "query_feats": query_feats,
    }
```

#### 1.3 Query Interaction Module (QIM)

QIM用于在时序帧之间更新和融合query：

```python
class QueryInteractionModule:
    def forward(self, 
                track_instances,      # 当前跟踪实例
                gt_instances=None):   # GT实例 (训练时)
        """
        1. 匹配当前检测和历史跟踪
        2. 更新query embedding
        3. 处理新出现和消失的物体
        """
        # 匹配
        matched_indices = self.matcher(track_instances, gt_instances)
        
        # 更新embedding
        track_instances.query = self.update_query(
            track_instances.query,
            track_instances.output_embedding,
            matched_indices
        )
        
        return track_instances
```

### 2. Map Segmentation Head

```python
class PansegformerHead:
    def forward_train(self,
                      bev_embed,      # [H*W, B, C] BEV特征
                      img_metas,
                      gt_lane_labels,
                      gt_lane_bboxes,
                      gt_lane_masks):
        """
        地图分割：将BEV特征分割为不同的地图元素
        """
        # 1. Reshape BEV特征
        bs = bev_embed.shape[1]
        bev_embed = bev_embed.permute(1, 2, 0)  # [B, C, H*W]
        bev_embed = bev_embed.view(bs, -1, self.bev_h, self.bev_w)  
        # [B, 256, 200, 200]
        
        # 2. 提取多尺度特征 (通过卷积下采样)
        mlvl_feats = self.extract_feat(bev_embed)
        # List of [B, 256, H_i, W_i]
        
        # 3. Transformer Encoder
        memory = self.transformer.encoder(mlvl_feats)
        
        # 4. Transformer Decoder
        # 初始化query
        query_embeds = self.query_embedding.weight  # [num_queries, 256]
        
        hs = self.transformer.decoder(
            query=query_embeds,
            key=memory,
            value=memory,
            **kwargs
        )
        # hs: [num_layers, B, num_queries, 256]
        
        # 5. 分类、回归、mask预测
        outputs_classes = []
        outputs_coords = []
        outputs_masks = []
        
        for lvl in range(hs.shape[0]):
            # 分类 (thing类别)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            
            # 边界框回归
            outputs_coord = self.reg_branches[lvl](hs[lvl])
            
            # Mask预测 (使用mask head)
            outputs_mask = self.mask_head(hs[lvl], memory)
            
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_masks.append(outputs_mask)
        
        # 6. 计算损失
        losses = self.loss(
            outputs_classes,
            outputs_coords,
            outputs_masks,
            gt_lane_labels,
            gt_lane_bboxes,
            gt_lane_masks
        )
        
        return losses, outputs
```

**Segmentation Head输出**：

```python
outputs = {
    "semantic_masks": [B, num_classes, H, W],  # 语义分割
    "instance_masks": [B, num_instances, H, W],  # 实例分割
    "forward_masks": [B, num_classes, H, W],  # 方向掩码 (前向)
    "backward_masks": [B, num_classes, H, W],  # 方向掩码 (后向)
}
```

### 3. Motion Prediction Head

```python
class MotionHead:
    def forward_train(self,
                      bev_embed,          # [H*W, B, C] BEV特征
                      gt_bboxes_3d,       # GT 3D框
                      gt_labels_3d,       # GT标签
                      gt_fut_traj,        # GT未来轨迹
                      gt_fut_traj_mask,   # GT轨迹mask
                      outs_track,         # Track输出
                      outs_seg):          # Seg输出
        """
        运动预测：预测检测到的物体的未来轨迹
        """
        # 1. 获取track query
        track_query = outs_track['track_query_embeddings'][-1]  
        # [B, num_valid_queries, 256]
        
        # 2. 匹配track query和GT
        matched_idxes = outs_track['track_query_matched_idxes'][-1]
        
        # 3. Agent特征编码
        agent_features = self.agent_encoder(track_query)
        
        # 4. 多模态轨迹预测 Transformer
        traj_query = self.traj_query_embedding.weight  
        # [num_modes, 256] (例如6个模式)
        
        # Decoder
        traj_hs = self.transformer(
            query=traj_query,
            key=bev_embed,
            value=bev_embed,
            agent_features=agent_features,  # 注入agent信息
            **kwargs
        )
        # [num_layers, B, num_agents, num_modes, 256]
        
        # 5. 轨迹回归
        all_traj_preds = []
        all_mode_probs = []
        
        for lvl in range(traj_hs.shape[0]):
            # 轨迹预测
            traj_pred = self.traj_reg_branches[lvl](traj_hs[lvl])
            # [B, num_agents, num_modes, fut_steps, 2]
            
            # 模式概率
            mode_prob = self.mode_cls_branches[lvl](traj_hs[lvl])
            # [B, num_agents, num_modes]
            
            all_traj_preds.append(traj_pred)
            all_mode_probs.append(mode_prob)
        
        # 6. 计算损失
        losses = self.loss(
            all_traj_preds,
            all_mode_probs,
            gt_fut_traj,
            gt_fut_traj_mask,
            matched_idxes
        )
        
        return {
            "losses": losses,
            "outs_motion": {
                "traj_preds": all_traj_preds[-1],
                "mode_probs": all_mode_probs[-1],
                "track_query": track_query,
                ...
            }
        }
```

**Motion Head输出**：

```python
outs_motion = {
    "traj_preds": [B, num_agents, num_modes, 12, 2],  # 12步未来轨迹
    "mode_probs": [B, num_agents, 6],  # 6个模式的概率
    "track_query": [B, num_agents, 256],  # Agent特征
}
```

### 4. Occupancy Prediction Head

```python
class OccHead:
    def forward_train(self,
                      bev_embed,         # [H*W, B, C] BEV特征
                      outs_motion,       # Motion输出
                      gt_segmentation,   # GT占用栅格
                      gt_instance,       # GT实例ID
                      gt_img_is_valid):  # 有效帧标志
        """
        占用预测：预测未来时刻的BEV占用栅格和光流
        """
        # 1. 获取track和trajectory query
        track_query = outs_motion['track_query']  # [B, N, 256]
        traj_query = outs_motion['traj_query']    # [L, B, N, M, 256]
        
        # 2. 构建实例query
        # 结合track query和trajectory query
        instance_query = self.build_instance_query(
            track_query, traj_query
        )  # [B, N, 256]
        
        # 3. BEV投影层
        bev_feat = self.bev_proj(bev_embed)  # [H*W, B, 256]
        bev_feat = bev_feat.permute(1, 2, 0).view(bs, 256, H, W)
        
        # 4. Transformer Decoder (预测未来占用)
        occ_queries = self.occ_query_embedding.weight  # [H*W, 256]
        
        future_states = []
        for t in range(self.n_future):  # 预测n_future个未来时刻
            # Decoder
            occ_hs = self.transformer.decoder(
                query=occ_queries,
                key=bev_feat,
                value=bev_feat,
                instance_query=instance_query,  # 注入实例信息
                **kwargs
            )
            
            # 占用预测
            occ_pred = self.occ_head(occ_hs)  # [B, H, W]
            
            # 光流预测
            flow_pred = self.flow_head(occ_hs)  # [B, 2, H, W]
            
            future_states.append({
                "occ": occ_pred,
                "flow": flow_pred
            })
            
            # 使用光流warp特征到下一时刻
            bev_feat = self.flow_warp(bev_feat, flow_pred)
        
        # 5. 计算损失
        losses = self.loss(
            future_states,
            gt_segmentation,
            gt_instance,
            gt_img_is_valid
        )
        
        return losses
```

**Occupancy Head输出**：

```python
future_states = [
    {
        "occ": [B, H, W],      # 占用预测 (0/1)
        "flow": [B, 2, H, W],  # 光流 (vx, vy)
    }
    for t in range(n_future)  # 例如4个未来时刻
]
```

### 5. Planning Head

```python
class PlanningHeadSingleMode:
    def forward_train(self,
                      bev_embed,        # [H*W, B, C] BEV特征
                      outs_motion,      # Motion输出
                      sdc_planning,     # GT规划轨迹
                      sdc_planning_mask,
                      command,          # 高层指令
                      gt_future_boxes): # 未来帧的障碍物
        """
        规划：预测自车的未来轨迹
        """
        # 1. 获取SDC (self-driving car) embedding
        sdc_embedding = outs_motion.get('sdc_embedding')  # [B, 256]
        
        # 2. 提取规划相关特征
        # BEV特征
        bev_feat = bev_embed.permute(1, 2, 0).view(bs, 256, H, W)
        
        # Agent特征 (检测到的其他车辆)
        agent_feat = outs_motion['track_query']  # [B, N, 256]
        
        # 3. 特征融合
        # Command embedding
        cmd_embed = self.command_embedding(command)  # [B, 256]
        
        # 融合SDC、command和BEV特征
        planning_feat = torch.cat([
            sdc_embedding,
            cmd_embed,
            bev_feat.mean(dim=[2, 3]),  # 全局BEV特征
        ], dim=-1)  # [B, 768]
        
        # 4. MLP预测轨迹
        traj_pred = self.planning_mlp(planning_feat)
        # [B, planning_steps * 2] = [B, 12] (6步，每步(x,y))
        
        traj_pred = traj_pred.view(bs, self.planning_steps, 2)
        # [B, 6, 2]
        
        # 5. 碰撞损失 (可选)
        if self.use_col_optim:
            collision_loss = self.collision_loss(
                traj_pred,
                gt_future_boxes,  # 未来时刻的障碍物框
            )
        
        # 6. L2损失
        planning_loss = F.l1_loss(
            traj_pred, 
            sdc_planning, 
            reduction='none'
        )
        planning_loss = (planning_loss * sdc_planning_mask.unsqueeze(-1)).mean()
        
        losses = {
            "loss_planning": planning_loss,
        }
        if self.use_col_optim:
            losses["loss_collision"] = collision_loss
        
        return {
            "losses": losses,
            "traj_pred": traj_pred,
        }
```

**Planning Head输出**：

```python
outputs = {
    "traj_pred": [B, 6, 2],  # 6步未来轨迹 (3秒)
    "command": [B],           # 执行的指令
}
```

---

## 完整数据流

### 训练阶段完整前向传播

```python
def forward_train(self, 
                  img,              # [B, L, N, C, H, W]
                  gt_bboxes_3d,     # 检测GT
                  gt_labels_3d,
                  gt_lane_labels,   # 地图GT
                  gt_lane_masks,
                  gt_fut_traj,      # 运动GT
                  gt_segmentation,  # 占用GT
                  sdc_planning,     # 规划GT
                  command,
                  **kwargs):
    """
    完整的前向传播流程
    """
    losses = dict()
    
    # ============ 1. Track (Detection + Tracking) ============
    # 时序处理：逐帧前向传播
    len_queue = img.size(1)  # 5帧
    
    losses_track, outs_track = self.forward_track_train(
        img, 
        gt_bboxes_3d, 
        gt_labels_3d, 
        ...
    )
    # outs_track包含:
    #   - bev_embed: [H*W, B, C]
    #   - bev_pos: [B, C, H, W]
    #   - track_query_embeddings: List of [B, 901, 256]
    #   - track_bbox_results: 检测框
    
    losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
    losses.update(losses_track)
    
    # Upsample BEV (如果使用tiny模型)
    outs_track = self.upsample_bev_if_tiny(outs_track)
    
    # 提取BEV特征
    bev_embed = outs_track["bev_embed"]  # [40000, 1, 256]
    bev_pos = outs_track["bev_pos"]      # [1, 256, 200, 200]
    
    # 使用最后一帧的img_metas
    img_metas = [each[len_queue-1] for each in img_metas]
    
    # ============ 2. Map Segmentation ============
    outs_seg = dict()
    if self.with_seg_head:
        losses_seg, outs_seg = self.seg_head.forward_train(
            bev_embed, 
            img_metas,
            gt_lane_labels, 
            gt_lane_bboxes, 
            gt_lane_masks
        )
        losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
        losses.update(losses_seg)
    
    # ============ 3. Motion Prediction ============
    outs_motion = dict()
    if self.with_motion_head:
        ret_dict_motion = self.motion_head.forward_train(
            bev_embed,
            gt_bboxes_3d, 
            gt_labels_3d, 
            gt_fut_traj, 
            gt_fut_traj_mask, 
            outs_track=outs_track, 
            outs_seg=outs_seg
        )
        losses_motion = ret_dict_motion["losses"]
        outs_motion = ret_dict_motion["outs_motion"]
        outs_motion['bev_pos'] = bev_pos
        
        losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
        losses.update(losses_motion)
    
    # ============ 4. Occupancy Prediction ============
    if self.with_occ_head:
        # 处理空query情况
        if outs_motion['track_query'].shape[1] == 0:
            device, dtype = bev_embed.device, bev_embed.dtype
            outs_motion['track_query'] = torch.zeros((1, 1, 256), device=device, dtype=dtype)
            outs_motion['track_query_pos'] = torch.zeros((1, 1, 256), device=device, dtype=dtype)
            outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256), device=device, dtype=dtype)
        
        losses_occ = self.occ_head.forward_train(
            bev_embed,
            outs_motion,
            gt_inds_list=gt_inds,
            gt_segmentation=gt_segmentation,  
            gt_instance=gt_instance, 
            gt_img_is_valid=gt_occ_img_is_valid,
        )
        losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
        losses.update(losses_occ)
    
    # ============ 5. Planning ============
    if self.with_planning_head:
        outs_planning = self.planning_head.forward_train(
            bev_embed, 
            outs_motion, 
            sdc_planning, 
            sdc_planning_mask, 
            command, 
            gt_future_boxes
        )
        losses_planning = outs_planning['losses']
        losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
        losses.update(losses_planning)
    
    # ============ 6. 处理NaN ============
    for k, v in losses.items():
        losses[k] = torch.nan_to_num(v)
    
    return losses
```

### 数据流可视化

```
输入图像 [1, 5, 6, 3, 900, 1600]
    ↓
ResNet101-DCN
    ↓
FPN多尺度特征 [1, 5, 6, 256, H_i, W_i] × 4
    ↓
BEVFormer Encoder (6层)
    ├─> Temporal Self-Attention (融合历史BEV)
    ├─> Spatial Cross-Attention (图像→BEV)
    └─> FFN
    ↓
BEV特征 [40000, 1, 256]
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│             │             │             │             │             │
▼             ▼             ▼             ▼             ▼             ▼
Track       Map Seg      Motion        Occupancy     Planning
(BEVFormer  (Panseg-     (Motion       (Occ          (MLP)
Decoder)    former)      Transformer)  Transformer)
│             │             │             │             │
▼             ▼             ▼             ▼             ▼
检测框       地图分割      轨迹预测      占用预测      规划轨迹
[B,N,10]    [B,3,H,W]    [B,N,M,T,2]  [B,T,H,W]    [B,6,2]
│             │             │             │             │
▼             ▼             ▼             ▼             ▼
损失计算 ← task_loss_weight × (track:1.0, map:1.0, motion:1.0, occ:1.0, planning:1.0)
```

### 关键尺寸总结

| 数据 | Shape | 说明 |
|------|-------|------|
| **输入** |
| img | [1, 5, 6, 3, 900, 1600] | B, L, N_cam, C, H, W |
| **图像特征** |
| FPN Level 0 | [1, 5, 6, 256, 113, 200] | 1/8分辨率 |
| FPN Level 1 | [1, 5, 6, 256, 57, 100] | 1/16分辨率 |
| FPN Level 2 | [1, 5, 6, 256, 29, 50] | 1/32分辨率 |
| FPN Level 3 | [1, 5, 6, 256, 15, 25] | 1/64分辨率 |
| **BEV特征** |
| bev_embed | [40000, 1, 256] | H×W=200×200 |
| bev_pos | [1, 256, 200, 200] | 位置编码 |
| **Track输出** |
| track_query | [1, 901, 256] | 901个query |
| bbox_pred | [1, 901, 10] | 检测框 |
| cls_scores | [1, 901, 10] | 分类得分 |
| **Map输出** |
| semantic_masks | [1, 3, 200, 200] | 3个类别 |
| instance_masks | [1, N_inst, 200, 200] | 实例分割 |
| **Motion输出** |
| traj_pred | [1, N_agents, 6, 12, 2] | 6模式×12步 |
| mode_prob | [1, N_agents, 6] | 模式概率 |
| **Occupancy输出** |
| occ_pred | [1, 4, 200, 200] | 4个未来时刻 |
| flow_pred | [1, 4, 2, 200, 200] | 光流 |
| **Planning输出** |
| traj_pred | [1, 6, 2] | 6步轨迹 |

---

## 总结

UniAD的训练前向传播流程可以总结为：

### 1. **层次化特征提取**
- **图像层**：ResNet101-DCN + FPN提取多尺度图像特征
- **BEV层**：BEVFormer将多视角图像特征转换为统一的BEV表示
- **任务层**：各个任务Head基于BEV特征进行专门处理

### 2. **时序信息融合**
- **Temporal Self-Attention**：在BEV生成阶段融合历史帧
- **Query Interaction**：在检测跟踪中维护时序一致的query
- **Memory Bank**：保存历史query信息用于长期跟踪

### 3. **多任务协同**
- **Track → Motion**：检测结果提供agent特征
- **Track + Seg → Motion**：结合检测和地图信息预测轨迹
- **Motion → Occ**：轨迹信息指导占用预测
- **Motion + Occ → Planning**：综合所有信息进行规划

### 4. **关键创新点**
- **统一BEV表示**：所有任务共享BEV特征
- **Query-based设计**：灵活的query机制支持多任务
- **端到端训练**：联合优化所有任务
- **时序建模**：充分利用视频的时序信息

### 5. **计算复杂度**
- **最耗时部分**：BEVFormer Encoder的空间交叉注意力
- **参数量最大**：ResNet101-DCN backbone
- **推理瓶颈**：多个Transformer Decoder的级联

这个架构设计使得UniAD能够在单一框架内完成从感知到规划的所有任务，实现真正的端到端自动驾驶。
