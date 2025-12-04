# UniAD Dataset Processing Pipeline

本文档详细描述了UniAD数据集处理流程，包括数据读取、目标检测和地图分割等的数据处理及真值生成过程。

## 目录

1. [概述](#概述)
2. [数据加载流程](#数据加载流程)
3. [数据增强Pipeline详解](#数据增强pipeline详解)
4. [目标检测真值生成](#目标检测真值生成)
5. [地图分割真值生成](#地图分割真值生成)
6. [运动预测真值生成](#运动预测真值生成)
7. [规划真值生成](#规划真值生成)
8. [占用预测真值生成](#占用预测真值生成)
9. [多帧融合机制union2one](#多帧融合机制union2one)
10. [DataContainer机制](#datacontainer机制)
11. [数据结构总结](#数据结构总结)
12. [常见问题FAQ](#常见问题faq)

---

## 概述

### 整体架构

UniAD使用`NuScenesE2EDataset`类继承自`NuScenesDataset`，实现了端到端自动驾驶的多任务数据处理。主要特点：

- **时序数据处理**：`queue_length=5`，包含当前帧和4个历史帧
- **多任务标注**：目标检测、地图分割、运动预测、占用预测、规划
- **坐标系统**：Lidar → Ego → Global 多层级坐标变换

### 关键参数

```python
queue_length = 5              # 时序帧数（当前帧 + 4历史帧）
bev_size = (200, 200)         # BEV特征图大小
patch_size = (102.4, 102.4)   # 地图patch大小（米）
canvas_size = (200, 200)      # 地图画布大小（像素）
map_num_classes = 3           # 地图类别数
thickness = 2                 # 地图线条粗细（200x200画布）
```

---

## 数据加载流程

### 1. 入口函数：`__getitem__()`

```python
def __getitem__(self, idx):
    if self.test_mode:
        return self.prepare_test_data(idx)
    else:
        return self.prepare_train_data(idx)
```

**功能**：根据索引获取数据
- 训练模式：调用`prepare_train_data()`
- 测试模式：调用`prepare_test_data()`

---

### 2. 训练数据准备：`prepare_train_data()`

```python
def prepare_train_data(self, index):
    """
    返回结构:
        img: [queue_length, 6, 3, H, W]  # 时序多视角图像
        img_metas: list of dict           # 每帧的元数据
        gt_bboxes_3d: list of LiDARInstance3DBoxes  # 每帧的3D框
        gt_labels_3d: list of Tensor      # 每帧的类别标签
        ...
    """
```

#### 处理步骤

**Step 1: 确定时序帧范围**

```python
final_index = index                    # 当前帧索引
first_index = index - queue_length + 1 # 第一帧索引

# 检查是否在同一场景
if self.data_infos[first_index]['scene_token'] != \
        self.data_infos[final_index]['scene_token']:
    return None  # 跨场景，返回None
```

**Step 2: 构建数据队列**

```python
data_queue = []
# 获取当前帧数据
input_dict = self.get_data_info(final_index)
self.pre_pipeline(input_dict)
example = self.pipeline(input_dict)
data_queue.insert(0, example)

# 倒序获取历史帧
prev_indexs_list = list(reversed(range(first_index, final_index)))
for i in prev_indexs_list:
    input_dict = self.get_data_info(i)
    self.pre_pipeline(input_dict)
    example = self.pipeline(input_dict)
    data_queue.insert(0, copy.deepcopy(example))
```

**Step 3: 合并队列**

```python
data_queue = self.union2one(data_queue)
return data_queue
```

---

### 3. 核心方法：`get_data_info()`

此方法是数据处理的核心，负责获取单帧的所有信息。

#### 3.1 基础信息获取

```python
def get_data_info(self, index):
    info = self.data_infos[index]  # 从预加载的数据信息中获取
    
    # 基础信息
    input_dict = dict(
        sample_idx=info['token'],
        pts_filename=info['lidar_path'],
        sweeps=info['sweeps'],
        ego2global_translation=info['ego2global_translation'],
        ego2global_rotation=info['ego2global_rotation'],
        scene_token=info['scene_token'],
        can_bus=info['can_bus'],
        frame_idx=info['frame_idx'],
        timestamp=info['timestamp'] / 1e6,
        ...
    )
```

#### 3.2 坐标变换矩阵计算

```python
# Lidar to Ego
l2e_r = info['lidar2ego_rotation']      # 四元数
l2e_t = info['lidar2ego_translation']   # 平移向量
l2e_r_mat = Quaternion(l2e_r).rotation_matrix

# Ego to Global
e2g_r = info['ego2global_rotation']
e2g_t = info['ego2global_translation']
e2g_r_mat = Quaternion(e2g_r).rotation_matrix

# Lidar to Global (组合变换)
l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T   # 旋转矩阵
l2g_t = l2e_t @ e2g_r_mat.T + e2g_t     # 平移向量

input_dict.update(dict(
    l2g_r_mat=l2g_r_mat.astype(np.float32),
    l2g_t=l2g_t.astype(np.float32)
))
```

#### 3.3 相机信息处理

```python
if self.modality['use_camera']:
    image_paths = []
    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
    
    for cam_type, cam_info in info['cams'].items():
        image_paths.append(cam_info['data_path'])
        
        # Lidar to Camera 变换
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        
        # 相机内参
        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        
        # Lidar to Image 投影矩阵
        lidar2img_rt = viewpad @ lidar2cam_rt.T
        lidar2img_rts.append(lidar2img_rt)
        cam_intrinsics.append(viewpad)
        lidar2cam_rts.append(lidar2cam_rt.T)
```

#### 3.4 获取标注信息

```python
# 调用 get_ann_info() 获取检测、运动、规划等标注
annos = self.get_ann_info(index)
input_dict['ann_info'] = annos
```

#### 3.5 地图信息处理（详见后文）

```python
# 获取地图向量化数据和栅格化标注
location = self.nusc.get('log', self.nusc.get(
    'scene', info['scene_token'])['log_token'])['location']
vectors = self.vector_map.gen_vectorized_samples(...)
semantic_masks, instance_masks, ... = preprocess_map(vectors, ...)
```

---

### 4. 时序数据合并：`union2one()`

此方法将多帧数据合并为一个batch可用的数据结构。

```python
def union2one(self, queue):
    """
    输入: queue (list of dict) - 包含queue_length个单帧数据
    输出: dict - 合并后的数据字典
    """
```

#### 4.1 收集各帧数据

```python
# 图像数据: List[Tensor] -> Tensor
imgs_list = [each['img'].data for each in queue]  # L x [6, 3, H, W]

# 检测标注: 保持list形式
gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]  # L个Tensor
gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]  # L个Box对象
gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]      # L个Tensor

# 时序信息
l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]  # L x [3, 3]
l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]          # L x [3]
timestamp_list = [torch.tensor([each["timestamp"]], dtype=torch.float64) 
                  for each in queue]  # L x [1]
```

#### 4.2 计算相对运动（CAN Bus处理）

```python
metas_map = {}
prev_pos = None
prev_angle = None

for i, each in enumerate(queue):
    metas_map[i] = each['img_metas'].data
    
    if i == 0:  # 第一帧作为参考
        metas_map[i]['prev_bev'] = False
        prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])   # 保存位置
        prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1]) # 保存航向角
        metas_map[i]['can_bus'][:3] = 0      # 第一帧位置设为0
        metas_map[i]['can_bus'][-1] = 0      # 第一帧角度设为0
    else:  # 后续帧计算相对运动
        metas_map[i]['prev_bev'] = True
        tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
        tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
        metas_map[i]['can_bus'][:3] -= prev_pos    # 相对位移
        metas_map[i]['can_bus'][-1] -= prev_angle  # 相对角度
        prev_pos = copy.deepcopy(tmp_pos)
        prev_angle = copy.deepcopy(tmp_angle)
```

**CAN Bus向量结构**：
```python
can_bus = [
    tx, ty, tz,           # [0:3]  ego车辆位置（相对或绝对）
    qw, qx, qy, qz,       # [3:7]  ego车辆姿态四元数
    vx, vy, vz,           # [7:10] ego车辆线速度
    ...                   # 其他车辆状态信息
    patch_angle_rad,      # [-2]   航向角（弧度）
    patch_angle_deg       # [-1]   航向角（角度）
]
```

#### 4.3 封装为DataContainer

```python
queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
# 形状: [queue_length, 6, 3, H, W] = [5, 6, 3, 900, 1600]

queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
# 字典: {0: meta_0, 1: meta_1, ..., 4: meta_4}

queue[-1]['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
# List: [boxes_0, boxes_1, ..., boxes_4]

queue[-1]['gt_labels_3d'] = DC(gt_labels_3d_list)
# List: [labels_0, labels_1, ..., labels_4]

queue[-1]['l2g_r_mat'] = DC(l2g_r_mat_list)
# List: [R_0, R_1, ..., R_4]，每个 [3, 3]

queue[-1]['timestamp'] = DC(timestamp_list)
# List: [t_0, t_1, ..., t_4]，每个 [1]
```

**DataContainer (DC)** 的作用：
- `cpu_only=False`: 数据可以移动到GPU
- `cpu_only=True`: 数据保持在CPU（如元数据、3D框对象）
- `stack=True`: 在collate时进行stack操作

---

## 数据增强Pipeline详解

### Pipeline组成

UniAD的数据处理pipeline包含多个阶段，按顺序执行：

#### 训练Pipeline

```python
train_pipeline = [
    # 1. 加载多视角图像
    dict(type="LoadMultiViewImageFromFilesInCeph", 
         to_float32=True, 
         file_client_args=file_client_args, 
         img_root=''),
    
    # 2. 光度失真（数据增强）
    dict(type="PhotoMetricDistortionMultiViewImage"),
    
    # 3. 加载3D标注
    dict(type="LoadAnnotations3D_E2E",
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False,
         with_future_anns=True,  # 加载未来帧标注（用于占用预测）
         with_ins_inds_3d=True,  # 加载实例ID
         ins_inds_add_1=True),   # 实例ID从1开始（0保留给背景）
    
    # 4. 生成占用和光流标注
    dict(type='GenerateOccFlowLabels', 
         grid_conf=occflow_grid_conf, 
         ignore_index=255, 
         only_vehicle=True,
         filter_invisible=False),
    
    # 5. 过滤超出范围的目标
    dict(type="ObjectRangeFilterTrack", 
         point_cloud_range=point_cloud_range),
    
    # 6. 过滤不需要的类别
    dict(type="ObjectNameFilterTrack", 
         classes=class_names),
    
    # 7. 图像归一化
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    
    # 8. 图像填充（对齐到32的倍数）
    dict(type="PadMultiViewImage", size_divisor=32),
    
    # 9. 格式化为训练格式
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    
    # 10. 收集需要的字段
    dict(type="CustomCollect3D", keys=[...]),
]
```

### 关键Pipeline组件详解

#### 1. LoadMultiViewImageFromFilesInCeph

**功能**：加载多视角相机图像（NuScenes有6个相机）

```python
class LoadMultiViewImageFromFilesInCeph(object):
    def __call__(self, results):
        images_multiView = []
        filename = results['img_filename']  # 6个相机的图像路径
        
        for img_path in filename:
            img_path = os.path.join(self.img_root, img_path)
            img = mmcv.imread(img_path, self.color_type)
            images_multiView.append(img)
        
        # 堆叠为 (H, W, C, N_views) 然后拆分为列表
        img = np.stack(images_multiView, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        
        # 拆分为列表，每个相机独立（方便后续单独处理）
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
```

**输出**：
- `results['img']`：List[ndarray]，长度为6，每个shape为(H, W, 3)
- `results['img_shape']`：(H, W, 3, 6)

**6个相机布局**：
```
              FRONT
               ↑
               |
FRONT_LEFT → EGO ← FRONT_RIGHT
               |
               ↓
             BACK
     BACK_LEFT  BACK_RIGHT
```

#### 2. PhotoMetricDistortionMultiViewImage

**功能**：对每个视角独立应用光度失真增强

**增强操作**（每个操作以0.5概率执行）：
1. **Random Brightness**：亮度调整，delta ∈ [-32, 32]
2. **Random Contrast**：对比度调整，alpha ∈ [0.5, 1.5]
3. **BGR → HSV**
4. **Random Saturation**：饱和度调整，∈ [0.5, 1.5]
5. **Random Hue**：色调调整，delta ∈ [-18, 18]
6. **HSV → BGR**
7. **Random Channel Swap**：随机交换RGB通道

```python
for img in imgs:
    # 亮度
    if random.randint(2):
        delta = random.uniform(-brightness_delta, brightness_delta)
        img += delta
    
    # 对比度（mode=0时在前，mode=1时在后）
    mode = random.randint(2)
    if mode == 1 and random.randint(2):
        img *= random.uniform(contrast_lower, contrast_upper)
    
    # HSV调整
    img = mmcv.bgr2hsv(img)
    if random.randint(2):  # 饱和度
        img[..., 1] *= random.uniform(saturation_lower, saturation_upper)
    if random.randint(2):  # 色调
        img[..., 0] += random.uniform(-hue_delta, hue_delta)
    img = mmcv.hsv2bgr(img)
    
    # 通道交换
    if random.randint(2):
        img = img[..., random.permutation(3)]
```

**注意**：所有6个相机使用**相同的随机参数**，保证多视角一致性。

#### 3. LoadAnnotations3D_E2E

**功能**：加载3D标注，包括当前帧和未来帧

```python
class LoadAnnotations3D_E2E(LoadAnnotations3D):
    def __call__(self, results):
        # 父类加载当前帧标注
        results = super().__call__(results)
        
        # 加载未来帧标注（用于占用预测）
        if self.with_future_anns:
            results = self._load_future_anns(results)
        
        # 加载实例ID
        if self.with_ins_inds_3d:
            results = self._load_ins_inds_3d(results)
        
        return results
    
    def _load_future_anns(self, results):
        """加载未来帧的3D框、标签、实例ID"""
        gt_bboxes_3d = []
        gt_labels_3d = []
        gt_inds_3d = []
        gt_vis_tokens = []
        
        for ann_info in results['occ_future_ann_infos']:
            if ann_info is not None:
                gt_bboxes_3d.append(ann_info['gt_bboxes_3d'])
                gt_labels_3d.append(ann_info['gt_labels_3d'])
                ann_gt_inds = ann_info['gt_inds']
                if self.ins_inds_add_1:
                    ann_gt_inds += 1  # 实例ID从1开始
                gt_inds_3d.append(ann_gt_inds)
                gt_vis_tokens.append(ann_info['gt_vis_tokens'])
            else:
                # 无效帧（例如，该帧不存在）
                gt_bboxes_3d.append(None)
                gt_labels_3d.append(None)
                gt_inds_3d.append(None)
                gt_vis_tokens.append(None)
        
        results['future_gt_bboxes_3d'] = gt_bboxes_3d  # List长度=occ_n_future
        results['future_gt_labels_3d'] = gt_labels_3d
        results['future_gt_inds'] = gt_inds_3d
        results['future_gt_vis_tokens'] = gt_vis_tokens
        return results
```

#### 4. ObjectRangeFilterTrack

**功能**：过滤超出point_cloud_range的目标，同时保持轨迹数据一致性

```python
class ObjectRangeFilterTrack(object):
    def __call__(self, input_dict):
        # 提取BEV范围 [x_min, y_min, x_max, y_max]
        bev_range = self.pcd_range[[0, 1, 3, 4]]
        
        # 获取所有相关数据
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_inds = input_dict['gt_inds']
        gt_fut_traj = input_dict['gt_fut_traj']
        gt_fut_traj_mask = input_dict['gt_fut_traj_mask']
        gt_past_traj = input_dict['gt_past_traj']
        gt_past_traj_mask = input_dict['gt_past_traj_mask']
        
        # 判断哪些框在范围内
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        mask = mask.numpy().astype(np.bool)
        
        # 过滤所有相关数据（保持一致性）
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]
        gt_inds = gt_inds[mask]
        gt_fut_traj = gt_fut_traj[mask]
        gt_fut_traj_mask = gt_fut_traj_mask[mask]
        gt_past_traj = gt_past_traj[mask]
        gt_past_traj_mask = gt_past_traj_mask[mask]
        
        # 限制yaw角到[-π, π]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        
        # 更新结果
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_inds'] = gt_inds
        input_dict['gt_fut_traj'] = gt_fut_traj
        input_dict['gt_fut_traj_mask'] = gt_fut_traj_mask
        input_dict['gt_past_traj'] = gt_past_traj
        input_dict['gt_past_traj_mask'] = gt_past_traj_mask
        return input_dict
```

**关键点**：
- 同时过滤bbox、label、instance_id、trajectory等所有相关字段
- 保证数据一致性，避免索引错位

#### 5. NormalizeMultiviewImage

**功能**：图像归一化（减均值，除标准差）

```python
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],  # BGR格式
    std=[1.0, 1.0, 1.0],
    to_rgb=False  # 保持BGR
)

class NormalizeMultiviewImage(object):
    def __call__(self, results):
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) 
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results
```

**注意**：UniAD使用预训练的ResNet101-DCN，因此使用ImageNet的BGR均值，但std=1.0（不做标准化）。

#### 6. PadMultiViewImage

**功能**：将图像填充到size_divisor的倍数（用于适配网络下采样）

```python
class PadMultiViewImage(object):
    def __init__(self, size_divisor=32, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val
    
    def __call__(self, results):
        # 每个视角独立填充到32的倍数
        padded_img = [
            mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
            for img in results['img']
        ]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        return results
```

**示例**：
- 输入：(900, 1600, 3) → 输出：(928, 1600, 3)  # 928 = 29×32

#### 7. CustomCollect3D

**功能**：收集需要的字段，构建img_metas

```python
class CustomCollect3D(object):
    def __init__(self, keys, meta_keys=(...)):
        self.keys = keys  # 数据字段
        self.meta_keys = meta_keys  # 元数据字段
    
    def __call__(self, results):
        data = {}
        img_metas = {}
        
        # 收集元数据
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        # 封装为DataContainer
        data['img_metas'] = DC(img_metas, cpu_only=True)
        
        # 收集数据字段
        for key in self.keys:
            data[key] = results[key]
        
        return data
```

**meta_keys**包含：
- `filename`, `ori_shape`, `img_shape`, `pad_shape`
- `lidar2img`, `cam2img`（相机参数）
- `l2g_r_mat`, `l2g_t`（坐标变换矩阵）
- `can_bus`（车辆状态信息）
- `scene_token`, `sample_idx`（场景和样本索引）

---

## 目标检测真值生成

### 1. 入口方法：`get_ann_info()`

```python
def get_ann_info(self, index):
    """
    返回值:
        dict{
            'gt_bboxes_3d': LiDARInstance3DBoxes,  # 3D边界框
            'gt_labels_3d': np.ndarray,            # 类别标签
            'gt_names': list[str],                 # 类别名称
            'gt_inds': np.ndarray,                 # 实例ID
            'gt_fut_traj': np.ndarray,             # 未来轨迹
            'gt_past_traj': np.ndarray,            # 历史轨迹
            ...
        }
    """
```

### 2. 3D边界框处理

#### 2.1 过滤有效框

```python
info = self.data_infos[index]

# 根据valid_flag或点云数量过滤
if self.use_valid_flag:
    mask = info['valid_flag']
else:
    mask = info['num_lidar_pts'] > 0

# 应用mask
gt_bboxes_3d = info['gt_boxes'][mask]      # [N, 7] or [N, 9]
gt_names_3d = info['gt_names'][mask]       # [N]
gt_inds = info['gt_inds'][mask]            # [N] 实例ID

# 获取对应的annotation tokens
sample = self.nusc.get('sample', info['token'])
ann_tokens = np.array(sample['anns'])[mask]
```

#### 2.2 类别映射

```python
gt_labels_3d = []
for cat in gt_names_3d:
    if cat in self.CLASSES:
        gt_labels_3d.append(self.CLASSES.index(cat))
    else:
        gt_labels_3d.append(-1)  # 未知类别
gt_labels_3d = np.array(gt_labels_3d)
```

**NuScenes类别**（10类）：
```python
self.CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
```

#### 2.3 添加速度信息

```python
if self.with_velocity:
    gt_velocity = info['gt_velocity'][mask]  # [N, 2]
    nan_mask = np.isnan(gt_velocity[:, 0])
    gt_velocity[nan_mask] = [0.0, 0.0]
    gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
    # 形状: [N, 7] -> [N, 9]
```

#### 2.4 边界框格式转换

```python
# NuScenes原始格式: center=[0.5, 0.5, 0.5] (box中心在几何中心)
# 转换为KITTI格式: origin=[0.5, 0.5, 0] (box中心在底面中心)
gt_bboxes_3d = LiDARInstance3DBoxes(
    gt_bboxes_3d,
    box_dim=gt_bboxes_3d.shape[-1],  # 7 or 9
    origin=(0.5, 0.5, 0.5)
).convert_to(self.box_mode_3d)
```

**边界框表示**：
```python
# 无速度: [x, y, z, w, l, h, yaw]  (7维)
# 有速度: [x, y, z, w, l, h, yaw, vx, vy]  (9维)
# 其中:
#   (x, y, z): 中心坐标（Lidar坐标系）
#   (w, l, h): 宽度、长度、高度
#   yaw: 航向角（绕z轴旋转）
#   (vx, vy): 速度（m/s）
```

### 3. 轨迹标注（由trajectory_api生成）

```python
# 通过NuScenes轨迹API生成
gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = \
    self.traj_api.get_traj_label(info['token'], ann_tokens)
```

**轨迹形状**：
```python
# gt_fut_traj: [N, predict_steps, 2]  # N个物体，12步预测，(x,y)坐标
# gt_fut_traj_mask: [N, predict_steps]  # 有效性mask
# gt_past_traj: [N, past_steps, 2]     # N个物体，4步历史，(x,y)坐标
# gt_past_traj_mask: [N, past_steps]   # 有效性mask
```

### 4. SDC（自车）标注

```python
# 生成自车边界框和标签
sdc_vel = self.traj_api.sdc_vel_info[info['token']]
gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)

# 自车未来轨迹
gt_sdc_fut_traj, gt_sdc_fut_traj_mask = \
    self.traj_api.get_sdc_traj_label(info['token'])

# 自车规划轨迹
sdc_planning, sdc_planning_mask, command = \
    self.traj_api.get_sdc_planning_label(info['token'])
```

**Command类型**：
```python
# command = 0: 右转 (Right)
# command = 1: 左转 (Left)
# command = 2: 直行 (Forward)
```

### 5. 返回标注字典

```python
anns_results = dict(
    gt_bboxes_3d=gt_bboxes_3d,              # LiDARInstance3DBoxes [N, 9]
    gt_labels_3d=gt_labels_3d,              # np.ndarray [N]
    gt_names=gt_names_3d,                   # list[str] [N]
    gt_inds=gt_inds,                        # np.ndarray [N]
    gt_fut_traj=gt_fut_traj,                # np.ndarray [N, 12, 2]
    gt_fut_traj_mask=gt_fut_traj_mask,      # np.ndarray [N, 12]
    gt_past_traj=gt_past_traj,              # np.ndarray [N, 4, 2]
    gt_past_traj_mask=gt_past_traj_mask,    # np.ndarray [N, 4]
    gt_sdc_bbox=gt_sdc_bbox,                # LiDARInstance3DBoxes [1, 9]
    gt_sdc_label=gt_sdc_label,              # np.ndarray [1]
    gt_sdc_fut_traj=gt_sdc_fut_traj,        # np.ndarray [planning_steps, 2]
    gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,  # np.ndarray [planning_steps]
    sdc_planning=sdc_planning,              # np.ndarray [planning_steps, 2]
    sdc_planning_mask=sdc_planning_mask,    # np.ndarray [planning_steps]
    command=command,                        # int: 0/1/2
)
```

---

## 地图分割真值生成

地图分割包含两个流程：
1. **向量地图生成**：从NuScenes HD地图提取向量化表示
2. **栅格化处理**：将向量转换为像素级分割标注

### 1. 向量地图生成

#### 1.1 VectorizedLocalMap初始化

```python
self.vector_map = VectorizedLocalMap(
    self.data_root,
    patch_size=self.patch_size,      # (102.4, 102.4) 米
    canvas_size=self.canvas_size     # (200, 200) 像素
)
```

#### 1.2 向量提取：`gen_vectorized_samples()`

```python
# 在 get_data_info() 中调用
location = self.nusc.get('log', self.nusc.get(
    'scene', info['scene_token'])['log_token'])['location']

vectors = self.vector_map.gen_vectorized_samples(
    location,
    info['ego2global_translation'],
    info['ego2global_rotation']
)
```

**功能**：从NuScenes地图中提取以ego车为中心的局部向量地图

**输出格式**：
```python
vectors = {
    'divider': List[LineString],      # 车道分隔线
    'ped_crossing': List[LineString], # 人行横道
    'boundary': List[LineString],     # 道路边界
    'centerline': List[LineString],   # 车道中心线
    ...
}
```

每个`LineString`是一系列有序的2D点：`[(x1, y1), (x2, y2), ...]`

### 2. 栅格化处理

#### 2.1 preprocess_map函数

```python
from projects.mmdet3d_plugin.datasets.data_utils.rasterize import preprocess_map

semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
    vectors,
    patch_size=self.patch_size,        # (102.4, 102.4)
    canvas_size=self.canvas_size,      # (200, 200)
    num_classes=self.map_num_classes,  # 3
    thickness=self.thickness,          # 2
    angle_class=self.angle_class       # 36
)
```

**输入**：
- `vectors`: 向量地图字典
- `patch_size`: 物理尺寸（米）
- `canvas_size`: 画布尺寸（像素）
- `num_classes`: 地图类别数（3类）
- `thickness`: 线条粗细（像素）
- `angle_class`: 角度分类数（36类，每类10度）

**输出形状**：
```python
semantic_masks:  # [3, 200, 200] 语义分割mask（二值：0/1）
instance_masks:  # [3, 200, 200] 实例分割mask（实例ID：0, 1, 2, ...）
forward_masks:   # [200, 200] 前向方向mask（角度类别：1-36）
backward_masks:  # [200, 200] 后向方向mask（角度类别：1-36）
```

**第一个维度为3的原因**：
- `num_classes=3` 表示有**3个地图类别**
- 在 `preprocess_map()` 中，对每个类别分别生成mask：
  ```python
  for i in range(num_classes):  # i = 0, 1, 2
      map_mask, idx = line_geom_to_mask(vector_num_list[i], ...)
      instance_masks.append(map_mask)  # 每个类别一个 [200, 200]
  
  instance_masks = np.stack(instance_masks)  # 堆叠为 [3, 200, 200]
  semantic_masks = instance_masks != 0       # 形状保持 [3, 200, 200]
  ```
- 这样设计可以**区分不同类别的地图元素**，每个通道对应一种地图类型

**注意**：
- `forward_masks` 和 `backward_masks` 初始也是 `[3, 200, 200]`
- 经过 `overlap_filter` 和 `sum(0)` 操作后，降维到 `[200, 200]`
- 每个像素值表示该位置地图元素的方向（1-36表示0-360度的离散化）

**三个地图类别**（来自`CLASS2LABEL`字典）：
```python
# Class 0: Divider (road_divider + lane_divider 合并)
# Class 1: Ped Crossing (人行横道)
# Class 2: Contours (road_segment + lane 的轮廓边界)
```

#### 2.2 栅格化流程

**Step 1: 坐标转换**

向量坐标从全局坐标系转换到ego车局部坐标系，再映射到像素坐标：

```python
# 物理坐标范围: [-51.2, 51.2] 米
# 像素坐标范围: [0, 200]
# 分辨率: 102.4 / 200 = 0.512 米/像素
```

**Step 2: 向量绘制**

使用OpenCV的`cv2.polylines()`绘制线段：

```python
for cls_id, line_list in enumerate(vectors.values()):
    for line in line_list:
        # 转换为像素坐标
        pixel_coords = world_to_pixel(line, patch_size, canvas_size)
        # 绘制到对应类别的mask上
        cv2.polylines(
            semantic_masks[cls_id], 
            [pixel_coords], 
            isClosed=False,
            color=1,
            thickness=thickness
        )
```

**Step 3: 实例分割mask生成**

为每条线段分配唯一的实例ID：

```python
instance_id = 1
for cls_id, line_list in enumerate(vectors.values()):
    for line in line_list:
        pixel_coords = world_to_pixel(line, patch_size, canvas_size)
        cv2.polylines(
            instance_masks[cls_id],
            [pixel_coords],
            isClosed=False,
            color=instance_id,  # 每条线段唯一ID
            thickness=thickness
        )
        instance_id += 1
```

**Step 4: 方向mask生成**

方向mask用于编码地图元素（如车道线、道路边界）的**方向信息**，这对于理解车道走向、规划路径等任务非常重要。

```python
def get_discrete_degree(vec, angle_class=36):
    """将方向向量转换为离散角度类别
    
    Args:
        vec: [dx, dy] 方向向量
        angle_class: 角度分类数，默认36（每类10度）
    
    Returns:
        deg: 1-36的整数，表示角度类别
    """
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)  # 0-360度
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1  # 1-36
    return deg

# Forward mask: 按原始顺序遍历线段点
for i in range(len(coords) - 1):
    direction_vec = coords[i+1] - coords[i]  # 前向方向
    angle_class = get_discrete_degree(direction_vec, angle_class=36)
    cv2.polylines(forward_mask, [coords[i:]], False, 
                  color=angle_class, thickness=thickness)

# Backward mask: 反转点序后遍历（相当于反向）
coords_reversed = np.flip(coords, 0)
for i in range(len(coords_reversed) - 1):
    direction_vec = coords_reversed[i+1] - coords_reversed[i]  # 后向方向
    angle_class = get_discrete_degree(direction_vec, angle_class=36)
    cv2.polylines(backward_mask, [coords_reversed[i:]], False,
                  color=angle_class, thickness=thickness)
```

**关键点**：
1. **角度量化**：360度被分为36个类别，每类10度，编码为1-36的整数
2. **Forward mask**：记录沿线段正向（起点→终点）的方向
3. **Backward mask**：记录沿线段反向（终点→起点）的方向，角度差约180度
4. **应用场景**：
   - 车道保持：理解车道方向
   - 路径规划：判断可行驶方向
   - 语义理解：区分单向/双向道路

### 3. 实例提取

从实例mask中提取单个实例的bbox和mask：

```python
instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
instance_masks = torch.tensor(instance_masks.copy())

gt_labels = []
gt_bboxes = []
gt_masks = []

for cls in range(self.map_num_classes):  # 遍历3个类别
    for i in np.unique(instance_masks[cls]):
        if i == 0:  # 跳过背景
            continue
        
        # 提取单个实例的mask
        gt_mask = (instance_masks[cls] == i).to(torch.uint8)  # [200, 200]
        
        # 计算bbox
        ys, xs = np.where(gt_mask)
        gt_bbox = [min(xs), min(ys), max(xs), max(ys)]  # [x1, y1, x2, y2]
        
        gt_labels.append(cls)
        gt_bboxes.append(gt_bbox)
        gt_masks.append(gt_mask)
```

### 4. 附加地图层（Lane Divider & Road Divider）

除了3类主要地图元素，还从`obtain_map_info()`获取额外的地图层：

```python
map_mask = obtain_map_info(
    self.nusc,
    self.nusc_maps,
    info,
    patch_size=self.patch_size,
    canvas_size=self.canvas_size,
    layer_names=['lane_divider', 'road_divider']
)
# 形状: [2, 200, 200] 或 [3, 200, 200]（可能包含背景层）
```

**处理流程**：
```python
# 翻转和旋转以匹配坐标系
map_mask = np.flip(map_mask, axis=1)
map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
map_mask = torch.tensor(map_mask.copy())

# 为每个额外层创建实例
for i, gt_mask in enumerate(map_mask[:-1]):  # 跳过最后的背景层
    ys, xs = np.where(gt_mask)
    gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
    gt_labels.append(i + self.map_num_classes)  # 类别ID: 3, 4, ...
    gt_bboxes.append(gt_bbox)
    gt_masks.append(gt_mask)
```

### 5. 封装地图标注

```python
gt_labels = torch.tensor(gt_labels)            # [M] M个实例
gt_bboxes = torch.tensor(np.stack(gt_bboxes))  # [M, 4]
gt_masks = torch.stack(gt_masks)               # [M, 200, 200]

input_dict.update({
    'gt_lane_labels': gt_labels,
    'gt_lane_bboxes': gt_bboxes,
    'gt_lane_masks': gt_masks,
})
```

**地图类别汇总**（5类）：
```python
# preprocess_map 生成的3个基础类别（来自VectorizedLocalMap）:
# 0: Divider (road_divider + lane_divider 合并)
# 1: Ped Crossing (人行横道 ped_crossing)
# 2: Contours (道路和车道的轮廓边界)

# obtain_map_info 额外添加的2个类别:
# 3: Lane Divider (车道分隔线)
# 4: Road Divider (道路分隔线)
```

---

## 运动预测真值生成

运动预测（Motion Prediction）任务负责预测场景中所有物体的未来轨迹，基于NuScenes的PredictHelper API实现。

### 1. 入口：`NuScenesTraj.get_traj_label()`

```python
def get_traj_label(self, sample_token, ann_tokens):
    """
    为所有检测到的物体生成轨迹标注
    
    Args:
        sample_token: 当前帧的sample token
        ann_tokens: 所有标注的annotation tokens
    
    Returns:
        fut_traj_all: [N, 12, 2] 未来轨迹
        fut_traj_valid_mask_all: [N, 12, 2] 未来轨迹有效性mask
        past_traj_all: [N, 8, 2] 历史轨迹 (4 past + 4 future)
        past_traj_valid_mask_all: [N, 8, 2] 历史轨迹有效性mask
    """
```

### 2. 轨迹提取流程

#### 2.1 获取物体信息

```python
sd_rec = self.nusc.get('sample', sample_token)
_, boxes, _ = self.nusc.get_sample_data(
    sd_rec['data']['LIDAR_TOP'], 
    selected_anntokens=ann_tokens
)

for i, ann_token in enumerate(ann_tokens):
    box = boxes[i]
    instance_token = self.nusc.get('sample_annotation', ann_token)['instance_token']
```

#### 2.2 使用PredictHelper API获取轨迹

```python
# 未来6秒轨迹 (12步，每步0.5秒)
fut_traj_local = self.predict_helper.get_future_for_agent(
    instance_token, 
    sample_token, 
    seconds=6,           # 6秒 = 12步
    in_agent_frame=True  # 物体局部坐标系
)

# 历史2秒轨迹 (4步)
past_traj_local = self.predict_helper.get_past_for_agent(
    instance_token, 
    sample_token, 
    seconds=2,           # 2秒 = 4步
    in_agent_frame=True
)
```

**PredictHelper返回格式**：
```python
# fut_traj_local: [T, 2] 物体局部坐标系下的未来位置
# past_traj_local: [T, 2] 物体局部坐标系下的历史位置
# T 可能小于期望步数（物体可能消失或新出现）
```

#### 2.3 坐标变换：局部→场景中心

```python
# 初始化固定长度数组
fut_traj = np.zeros((self.predict_steps, 2))  # [12, 2]
fut_traj_valid_mask = np.zeros((self.predict_steps, 2))

if fut_traj_local.shape[0] > 0:
    if self.use_nonlinear_optimizer:
        trans = box.center  # 使用box中心作为平移
    else:
        trans = np.array([0, 0, 0])  # 默认无平移
    
    rot = Quaternion(matrix=box.rotation_matrix)
    
    # 局部坐标 → 场景中心坐标
    fut_traj_scene_centric = convert_local_coords_to_global(
        fut_traj_local, trans, rot
    )
    
    # 填充有效轨迹点
    fut_traj[:fut_traj_scene_centric.shape[0], :] = fut_traj_scene_centric
    fut_traj_valid_mask[:fut_traj_scene_centric.shape[0], :] = 1
```

**坐标系说明**：
- **Agent Frame（物体局部）**: 以物体为中心，x轴为物体前方
- **Scene Centric（场景中心）**: 以当前lidar位置为原点的坐标系
- **变换公式**: `p_scene = R @ p_local + t`

#### 2.4 历史轨迹处理

```python
past_traj = np.zeros((self.past_steps + self.fut_steps, 2))  # [8, 2]
past_traj_valid_mask = np.zeros((self.past_steps + self.fut_steps, 2))

if past_traj_local.shape[0] > 0:
    trans = np.array([0, 0, 0])
    rot = Quaternion(matrix=box.rotation_matrix)
    past_traj_scene_centric = convert_local_coords_to_global(
        past_traj_local, trans, rot
    )
    
    # 填充历史部分 (前4步)
    past_traj[:past_traj_scene_centric.shape[0], :] = past_traj_scene_centric
    past_traj_valid_mask[:past_traj_scene_centric.shape[0], :] = 1
    
    # 填充未来部分 (后4步)
    if fut_traj_local.shape[0] > 0:
        fut_steps = min(self.fut_steps, fut_traj_scene_centric.shape[0])
        past_traj[self.past_steps:self.past_steps+fut_steps, :] = \
            fut_traj_scene_centric[:fut_steps]
        past_traj_valid_mask[self.past_steps:self.past_steps+fut_steps, :] = 1
```

**Past Traj结构**：
```python
# past_traj: [8, 2]
# [0:4] - 历史4步 (t-2s 到 t-0.5s)
# [4:8] - 未来4步 (t+0.5s 到 t+2s)
```

### 3. 批量处理

```python
# 堆叠所有物体的轨迹
if len(ann_tokens) > 0:
    fut_traj_all = np.stack(fut_traj_all, axis=0)              # [N, 12, 2]
    fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)  # [N, 12, 2]
    past_traj_all = np.stack(past_traj_all, axis=0)            # [N, 8, 2]
    past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)  # [N, 8, 2]
else:
    # 空场景
    fut_traj_all = np.zeros((0, self.predict_steps, 2))
    fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))
    past_traj_all = np.zeros((0, self.predict_steps, 2))
    past_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))
```

### 4. 时间轴说明

```python
# 时间步长：0.5秒/步
# predict_steps = 12 (未来6秒)
# past_steps = 4 (历史2秒)
# fut_steps = 4 (用于past_traj的未来部分)

时间轴示意：
    历史           当前          未来
    |<--2s-->|<-t->|<-------6s-------->|
    [      past     ]
    [-4 -3 -2 -1] 0 [1 2 3 4 5 6 ... 12]
    |<-past_traj->|
                    |<---fut_traj---->|
```

### 5. 数据结构总结

```python
# 单帧返回
{
    'gt_fut_traj': np.ndarray,           # [N, 12, 2] 未来轨迹
    'gt_fut_traj_mask': np.ndarray,      # [N, 12, 2] 有效性mask
    'gt_past_traj': np.ndarray,          # [N, 8, 2] 历史+近期未来
    'gt_past_traj_mask': np.ndarray,     # [N, 8, 2] 有效性mask
}

# 时序批次 (union2one后)
{
    'gt_fut_traj': DC(Tensor),           # [N4, 12, 2] 仅当前帧
    'gt_fut_traj_mask': DC(Tensor),      # [N4, 12, 2]
    'gt_past_traj': DC([                 # List of Tensor
        past_traj_0,                     # [N0, 8, 2] 第0帧
        ...
        past_traj_4                      # [N4, 8, 2] 第4帧
    ]),
    'gt_past_traj_mask': DC([...]),      # List of Tensor
}
```

---

## 规划真值生成

规划（Planning）任务生成自车（SDC）的未来轨迹作为真值，用于训练端到端规划模型。

### 1. SDC速度预处理：`prepare_sdc_vel_info()`

在初始化时，为所有样本预计算SDC速度信息。

```python
def prepare_sdc_vel_info(self):
    """预计算所有帧的SDC速度（在lidar坐标系下）"""
    self.sdc_vel_info = {}
    
    for scene in self.nusc.scene:
        sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        
        sample = self.nusc.get('sample', sample_token)
        xyz, time = self.get_vel_and_time(sample)
        
        while sample['token'] != last_sample_token:
            next_sample = self.nusc.get('sample', sample['next'])
            next_xyz, next_time = self.get_vel_and_time(next_sample)
            
            # 计算速度 (全局坐标系)
            dc = np.array(next_xyz) - np.array(xyz)
            dt = (next_time - time) / 1e6  # 微秒转秒
            vel = dc / dt
            
            # 全局坐标系 → Lidar坐标系
            l2e_r_mat, e2g_r_mat = self.get_vel_transform_mats(sample)
            vel = vel @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            vel = vel[:2]  # 只保留 (vx, vy)
            
            self.sdc_vel_info[sample['token']] = vel
            xyz, time = next_xyz, next_time
            sample = next_sample
```

**速度坐标变换**：
```python
# Global → Ego → Lidar
v_lidar = v_global @ inv(e2g_R).T @ inv(l2e_R).T
```

### 2. SDC边界框生成：`generate_sdc_info()`

```python
def generate_sdc_info(self, sdc_vel, as_lidar_instance3d_box=False):
    """生成伪SDC边界框
    
    Args:
        sdc_vel: [2] SDC速度 (vx, vy)
        as_lidar_instance3d_box: 是否返回Box对象
    
    Returns:
        gt_bboxes_3d: LiDARInstance3DBoxes [1, 9]
        gt_labels_3d: np.ndarray [1] (类别为'car')
    """
    # SDC尺寸 (根据NuScenes官方数据)
    psudo_sdc_bbox = np.array([
        0.0, 0.0, 0.0,    # x, y, z (lidar中心为原点)
        4.08,             # 长度 (m)
        1.73,             # 宽度 (m)
        1.56,             # 高度 (m)
        0.5 * np.pi       # 航向角 (yaw)
    ])
    
    if self.with_velocity:
        psudo_sdc_bbox = np.concatenate([psudo_sdc_bbox, sdc_vel], axis=-1)
        # 形状: [9] = [x, y, z, l, w, h, yaw, vx, vy]
    
    gt_bboxes_3d = LiDARInstance3DBoxes(
        np.array([psudo_sdc_bbox]).astype(np.float32),
        box_dim=psudo_sdc_bbox.shape[-1],
        origin=(0.5, 0.5, 0.5)
    ).convert_to(self.box_mode_3d)
```

### 3. 规划轨迹生成：`get_sdc_planning_label()`

生成SDC的未来规划轨迹，涉及复杂的多步坐标变换。

#### 3.1 初始化

```python
def get_sdc_planning_label(self, sample_token):
    sd_rec = self.nusc.get('sample', sample_token)
    
    # 获取初始帧的坐标变换矩阵
    l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = \
        self.get_l2g_transform(sd_rec)
    
    planning = []
```

#### 3.2 循环生成未来步

```python
for _ in range(self.planning_steps):  # planning_steps = 6
    next_annotation_token = sd_rec['next']
    if next_annotation_token == '':
        break  # 场景结束
    
    sd_rec = self.nusc.get('sample', next_annotation_token)
    
    # 获取当前帧（未来某一步）的坐标变换
    l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = \
        self.get_l2g_transform(sd_rec)
    
    # 生成当前帧的SDC bbox (在当前帧的lidar坐标系下)
    next_bbox3d = self.generate_sdc_info(
        self.sdc_vel_info[next_annotation_token], 
        as_lidar_instance3d_box=True
    )
```

#### 3.3 多步坐标变换

将未来帧的SDC bbox变换到初始帧的lidar坐标系：

```python
# Step 1: 当前lidar → 当前ego
next_bbox3d.rotate(l2e_r_mat_curr.T)
next_bbox3d.translate(l2e_t_curr)

# Step 2: 当前ego → 全局坐标系
next_bbox3d.rotate(e2g_r_mat_curr.T)
next_bbox3d.translate(e2g_t_curr)

# Step 3: 全局 → 初始ego (逆变换)
next_bbox3d.translate(-e2g_t_init)
m1 = np.linalg.inv(e2g_r_mat_init)
next_bbox3d.rotate(m1.T)

# Step 4: 初始ego → 初始lidar (逆变换)
next_bbox3d.translate(-l2e_t_init)
m2 = np.linalg.inv(l2e_r_mat_init)
next_bbox3d.rotate(m2.T)

planning.append(next_bbox3d)
```

**坐标变换流程图**：
```
当前帧lidar(t+i) → 当前帧ego(t+i) → Global → 初始帧ego(t) → 初始帧lidar(t)
     ↓                    ↓              ↓           ↓                  ↓
   原点                rotate           rotate      inv_rotate        inv_rotate
                      translate        translate   inv_translate     inv_translate
```

#### 3.4 提取规划点

```python
planning_all = np.zeros((1, self.planning_steps, 3))  # [1, 6, 3]
planning_mask_all = np.zeros((1, self.planning_steps, 2))

n_valid_timestep = len(planning)
if n_valid_timestep > 0:
    planning = [p.tensor.squeeze(0) for p in planning]
    planning = np.stack(planning, axis=0)  # [valid_t, 9]
    planning = planning[:, [0, 1, 6]]      # 提取 (x, y, yaw)
    
    planning_all[:, :n_valid_timestep, :] = planning
    planning_mask_all[:, :n_valid_timestep, :] = 1
```

#### 3.5 指令生成（Command）

根据最终规划点的横向偏移判断转向指令：

```python
mask = planning_mask_all[0].any(axis=1)
if mask.sum() == 0:
    command = 2  # FORWARD
elif planning_all[0, mask][-1][0] >= 2:   # 最后一步的x坐标
    command = 0  # RIGHT
elif planning_all[0, mask][-1][0] <= -2:
    command = 1  # LEFT
else:
    command = 2  # FORWARD
```

**Command定义**：
```python
# command = 0: 右转 (Right)  - 最终x >= 2米
# command = 1: 左转 (Left)   - 最终x <= -2米
# command = 2: 直行 (Forward) - 其他情况
```

### 4. 数据结构总结

```python
# 单帧返回
{
    'sdc_planning': np.ndarray,          # [1, 6, 3] (x, y, yaw)
    'sdc_planning_mask': np.ndarray,     # [1, 6, 2] 有效性mask
    'command': int,                      # 0/1/2 转向指令
}

# 时序批次 (union2one后)
{
    'sdc_planning': Tensor,              # [6, 2] 仅当前帧的(x,y)
    'sdc_planning_mask': Tensor,         # [6]
    'command': int,                      # 0/1/2
}
```

### 5. 规划时间轴

```python
# planning_steps = 6 (未来3秒)
# 时间步长：0.5秒/步

时间轴：
    当前帧                    未来
      t     t+0.5s  t+1.0s  t+1.5s  t+2.0s  t+2.5s  t+3.0s
      0       1       2       3       4       5       6
      |-------|-------|-------|-------|-------|-------|
            SDC规划轨迹 (相对于初始lidar坐标系)
```

---

## 占用预测真值生成

占用预测（Occupancy Prediction）任务预测未来时刻的BEV占用栅格图，基于3D检测框的时序信息生成。

### 1. 概述

**关键参数**：
```python
occ_receptive_field = 3    # 过去+当前帧数 (t-1, t, t+1)
occ_n_future = 4           # 未来帧数 (t+1, t+2, t+3, t+4)
occ_only_total_frames = 7  # 总帧数限制（用于评估）
```

**数据流**：
```
当前index → 获取时序索引 → 获取变换矩阵 → 获取检测标注 → 后处理生成占用GT
```

### 2. 时序索引获取：`occ_get_temporal_indices()`

```python
def occ_get_temporal_indices(self, index, receptive_field, n_future):
    """
    获取占用预测所需的时序帧索引
    
    Args:
        index: 当前帧索引
        receptive_field: 3 (过去+当前)
        n_future: 4 (未来)
    
    Returns:
        previous_indices: [t-2, t-1] 历史帧索引列表
        future_indices: [t+1, t+2, t+3, t+4] 未来帧索引列表
    """
    current_scene_token = self.data_infos[index]['scene_token']
    
    # 生成历史帧索引
    previous_indices = []
    for t in range(-receptive_field + 1, 0):  # t = -2, -1
        index_t = index + t
        if index_t >= 0 and \
           self.data_infos[index_t]['scene_token'] == current_scene_token:
            previous_indices.append(index_t)
        else:
            previous_indices.append(-1)  # 无效帧
    
    # 生成未来帧索引
    future_indices = []
    for t in range(1, n_future + 1):  # t = 1, 2, 3, 4
        index_t = index + t
        if index_t < len(self.data_infos) and \
           self.data_infos[index_t]['scene_token'] == current_scene_token:
            future_indices.append(index_t)
        else:
            future_indices.append(-1)  # 无效帧
    
    return previous_indices, future_indices
```

**时序帧示意**：
```python
previous_indices: [idx-2, idx-1]
current: idx
future_indices: [idx+1, idx+2, idx+3, idx+4]

all_frames = previous_indices + [index] + future_indices
# 总共7帧: [t-2, t-1, t, t+1, t+2, t+3, t+4]
```

### 3. 坐标变换获取：`occ_get_transforms()`

为每个有效帧获取Lidar→Ego→Global的变换矩阵。

```python
def occ_get_transforms(self, indices, data_type=torch.float32):
    """
    获取多帧的坐标变换矩阵
    
    Args:
        indices: 帧索引列表 (包含当前+未来)
    
    Returns:
        dict{
            'occ_l2e_r_mats': List[Tensor or None],  # Lidar→Ego 旋转
            'occ_l2e_t_vecs': List[Tensor or None],  # Lidar→Ego 平移
            'occ_e2g_r_mats': List[Tensor or None],  # Ego→Global 旋转
            'occ_e2g_t_vecs': List[Tensor or None],  # Ego→Global 平移
        }
    """
    l2e_r_mats = []
    l2e_t_vecs = []
    e2g_r_mats = []
    e2g_t_vecs = []
    
    for index in indices:
        if index == -1:
            # 无效帧，填充None
            l2e_r_mats.append(None)
            l2e_t_vecs.append(None)
            e2g_r_mats.append(None)
            e2g_t_vecs.append(None)
        else:
            info = self.data_infos[index]
            
            # Lidar → Ego
            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            l2e_r_mat = torch.from_numpy(
                Quaternion(l2e_r).rotation_matrix
            ).to(data_type)
            l2e_t_vec = torch.tensor(l2e_t, dtype=data_type)
            
            # Ego → Global
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            e2g_r_mat = torch.from_numpy(
                Quaternion(e2g_r).rotation_matrix
            ).to(data_type)
            e2g_t_vec = torch.tensor(e2g_t, dtype=data_type)
            
            l2e_r_mats.append(l2e_r_mat)  # [3, 3]
            l2e_t_vecs.append(l2e_t_vec)  # [3]
            e2g_r_mats.append(e2g_r_mat)  # [3, 3]
            e2g_t_vecs.append(e2g_t_vec)  # [3]
    
    return {
        'occ_l2e_r_mats': l2e_r_mats,
        'occ_l2e_t_vecs': l2e_t_vecs,
        'occ_e2g_r_mats': e2g_r_mats,
        'occ_e2g_t_vecs': e2g_t_vecs,
    }
```

### 4. 未来检测标注获取：`get_future_detection_infos()`

```python
def get_future_detection_infos(self, future_frames):
    """
    获取当前+未来帧的检测标注
    
    Args:
        future_frames: [index, index+1, ..., index+4]
    
    Returns:
        detection_ann_infos: List[dict or None]
    """
    detection_ann_infos = []
    for future_frame in future_frames:
        if future_frame >= 0:
            detection_ann_infos.append(
                self.occ_get_detection_ann_info(future_frame)
            )
        else:
            detection_ann_infos.append(None)
    
    return detection_ann_infos
```

### 5. 单帧检测标注：`occ_get_detection_ann_info()`

```python
def occ_get_detection_ann_info(self, index):
    """
    获取单帧的3D检测标注（用于占用预测）
    
    Returns:
        dict{
            'gt_bboxes_3d': LiDARInstance3DBoxes,  # [N, 9]
            'gt_labels_3d': np.ndarray,            # [N]
            'gt_inds': np.ndarray,                 # [N] 实例ID
            'gt_vis_tokens': np.ndarray,           # [N] 可见性token
        }
    """
    info = self.data_infos[index].copy()
    gt_bboxes_3d = info['gt_boxes'].copy()
    gt_names_3d = info['gt_names'].copy()
    gt_ins_inds = info['gt_inds'].copy()
    gt_vis_tokens = info.get('visibility_tokens', None)
    
    # 有效性过滤
    if self.use_valid_flag:
        gt_valid_flag = info['valid_flag']
    else:
        gt_valid_flag = info['num_lidar_pts'] > 0
    
    # 类别映射
    gt_labels_3d = []
    for cat in gt_names_3d:
        if cat in self.CLASSES:
            gt_labels_3d.append(self.CLASSES.index(cat))
        else:
            gt_labels_3d.append(-1)
    gt_labels_3d = np.array(gt_labels_3d)
    
    # 添加速度
    if self.with_velocity:
        gt_velocity = info['gt_velocity']
        nan_mask = np.isnan(gt_velocity[:, 0])
        gt_velocity[nan_mask] = [0.0, 0.0]
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
    
    # 转换为LiDARInstance3DBoxes
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d,
        box_dim=gt_bboxes_3d.shape[-1],
        origin=(0.5, 0.5, 0.5)
    ).convert_to(self.box_mode_3d)
    
    return dict(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        gt_inds=gt_ins_inds,
        gt_vis_tokens=gt_vis_tokens,
    )
```

### 6. 占用数据整合：`get_occ_data_infos()`

```python
def get_occ_data_infos(self, index):
    """
    整合所有占用预测所需的数据
    
    Returns:
        all_frames: List[int] 所有帧的索引 (长度7)
        has_invalid_frame: bool 前7帧是否有无效帧
        occ_transforms: dict 坐标变换矩阵
        occ_future_ann_infos: List[dict] 未来帧的检测标注
    """
    # 获取时序索引
    prev_indices, future_indices = self.occ_get_temporal_indices(
        index, self.occ_receptive_field, self.occ_n_future
    )
    
    # 组合所有帧
    all_frames = prev_indices + [index] + future_indices
    # all_frames = [t-2, t-1, t, t+1, t+2, t+3, t+4]
    
    # 检查前7帧是否有无效帧
    has_invalid_frame = -1 in all_frames[:self.occ_only_total_frames]
    
    # 获取当前+未来帧
    future_frames = [index] + future_indices
    # future_frames = [t, t+1, t+2, t+3, t+4]
    
    # 获取坐标变换
    occ_transforms = self.occ_get_transforms(future_frames)
    
    # 获取检测标注
    occ_future_ann_infos = self.get_future_detection_infos(future_frames)
    
    return all_frames, has_invalid_frame, occ_transforms, occ_future_ann_infos
```

### 7. 占用GT后处理（在训练pipeline中）

占用真值的最终生成在数据增强pipeline中完成，主要步骤：

#### 7.1 GenerateOccFlowLabels

```python
class GenerateOccFlowLabels:
    """
    从3D boxes生成BEV占用和流场标注
    
    输入:
        - occ_future_ann_infos: 未来帧的3D boxes
        - occ_transforms: 坐标变换矩阵
    
    输出:
        - gt_segmentation: [T, H, W] 语义分割
        - gt_instance: [T, H, W] 实例分割
        - gt_centerness: [T, H, W] 中心度
        - gt_offset: [T, 2, H, W] 偏移场
        - gt_flow: [T, 2, H, W] 前向流场
        - gt_backward_flow: [T, 2, H, W] 后向流场
    """
```

#### 7.2 3D Boxes → BEV投影

```python
# 对每个未来时刻
for t, ann_info in enumerate(occ_future_ann_infos):
    boxes_3d = ann_info['gt_bboxes_3d']  # [N, 9]
    labels = ann_info['gt_labels_3d']    # [N]
    ins_ids = ann_info['gt_inds']        # [N]
    
    # 将boxes投影到BEV网格
    for i, box in enumerate(boxes_3d):
        # 获取box的8个角点
        corners = box.corners  # [8, 3]
        
        # 投影到BEV平面 (只保留x, y)
        corners_bev = corners[:, :2]  # [8, 2]
        
        # 映射到BEV网格坐标
        grid_coords = world_to_grid(corners_bev, bev_range, bev_resolution)
        
        # 光栅化到BEV
        rasterize_polygon(
            gt_segmentation[t],
            gt_instance[t],
            grid_coords,
            labels[i],
            ins_ids[i]
        )
```

#### 7.3 流场计算

```python
# 根据实例ID匹配计算流场
for t in range(T-1):
    for ins_id in unique_instances:
        # 当前时刻的质心
        mask_curr = (gt_instance[t] == ins_id)
        centroid_curr = compute_centroid(mask_curr)
        
        # 下一时刻的质心
        mask_next = (gt_instance[t+1] == ins_id)
        centroid_next = compute_centroid(mask_next)
        
        # 流场 = 下一时刻位置 - 当前位置
        flow = centroid_next - centroid_curr
        
        # 填充到流场图
        gt_flow[t, :, mask_curr] = flow[:, None]
```

### 8. 数据结构总结

```python
# 在 get_data_info() 中添加
{
    'occ_has_invalid_frame': bool,           # 是否有无效帧
    'occ_img_is_valid': np.ndarray,         # [7] 每帧是否有效
    'occ_l2e_r_mats': List[Tensor or None], # [5] 变换矩阵
    'occ_l2e_t_vecs': List[Tensor or None], # [5]
    'occ_e2g_r_mats': List[Tensor or None], # [5]
    'occ_e2g_t_vecs': List[Tensor or None], # [5]
    'occ_future_ann_infos': List[dict],     # [5] 检测标注
}

# 经过pipeline后生成
{
    'gt_segmentation': Tensor,       # [T, H, W] 语义分割
    'gt_instance': Tensor,           # [T, H, W] 实例ID
    'gt_centerness': Tensor,         # [T, H, W] 中心度
    'gt_offset': Tensor,             # [T, 2, H, W] 到质心的偏移
    'gt_flow': Tensor,               # [T, 2, H, W] 前向光流
    'gt_backward_flow': Tensor,      # [T, 2, H, W] 后向光流
}
```

### 9. 占用预测时间轴

```python
# 总共7帧用于占用预测
all_frames = [t-2, t-1, t, t+1, t+2, t+3, t+4]
             |<-past->| |<----future---->|
             
# 但只为当前+未来生成GT
future_frames = [t, t+1, t+2, t+3, t+4]
                |<--预测目标(5帧)-->|

时间轴：
    t-2    t-1     t     t+1   t+2   t+3   t+4
    |------|------|------|-----|-----|-----|
    [  历史观测   ] [     预测目标       ]
                   |<--生成occupancy GT-->|
```

---

## 多帧融合机制：union2one

### 功能概述

`union2one`函数将时序队列中的多帧数据（queue_length帧）融合为一个batch，同时计算帧间相对运动。

### 核心代码

```python
def union2one(self, queue):
    """
    将queue中的多帧数据融合为单个样本
    queue: List[Dict]，长度为queue_length（例如5）
    """
    # 1. 收集各帧的数据
    imgs_list = [each['img'].data for each in queue]  # List[Tensor], 长度=5
    gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
    gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
    gt_inds_list = [to_tensor(each['gt_inds']) for each in queue]
    
    # 轨迹数据（历史）
    gt_past_traj_list = [to_tensor(each['gt_past_traj']) for each in queue]
    gt_past_traj_mask_list = [to_tensor(each['gt_past_traj_mask']) for each in queue]
    
    # 坐标变换矩阵
    l2g_r_mat_list = [to_tensor(each['l2g_r_mat']) for each in queue]
    l2g_t_list = [to_tensor(each['l2g_t']) for each in queue]
    
    # 时间戳
    timestamp_list = [
        torch.tensor([each["timestamp"]], dtype=torch.float64) 
        for each in queue
    ]  # List[Tensor(1,)], 长度=5
    
    # 2. 未来轨迹只取最后一帧（当前帧）
    gt_fut_traj = to_tensor(queue[-1]['gt_fut_traj'])  # [N, 12, 2]
    gt_fut_traj_mask = to_tensor(queue[-1]['gt_fut_traj_mask'])  # [N, 12]
    gt_sdc_fut_traj = to_tensor(queue[-1]['gt_sdc_fut_traj'])  # [6, 2]
    gt_sdc_fut_traj_mask = to_tensor(queue[-1]['gt_sdc_fut_traj_mask'])  # [6]
    
    # 未来帧检测框（用于规划）
    gt_future_boxes_list = queue[-1]['gt_future_boxes']
    gt_future_labels_list = [
        to_tensor(each) for each in queue[-1]['gt_future_labels']
    ]
    
    # 3. 构建img_metas，计算相对运动
    metas_map = {}
    prev_pos = None
    prev_angle = None
    
    for i, each in enumerate(queue):
        metas_map[i] = each['img_metas'].data
        
        if i == 0:  # 第一帧（最早的历史帧）
            metas_map[i]['prev_bev'] = False
            # 保存初始位置和角度
            prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])  # [x, y, z]
            prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])  # yaw
            # 第一帧的相对运动为0
            metas_map[i]['can_bus'][:3] = 0
            metas_map[i]['can_bus'][-1] = 0
        else:  # 后续帧
            metas_map[i]['prev_bev'] = True
            # 计算相对于前一帧的运动
            tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
            tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
            metas_map[i]['can_bus'][:3] -= prev_pos  # 相对位移
            metas_map[i]['can_bus'][-1] -= prev_angle  # 相对角度变化
            # 更新为当前帧的绝对位置（供下一次迭代使用）
            prev_pos = copy.deepcopy(tmp_pos)
            prev_angle = copy.deepcopy(tmp_angle)
    
    # 4. 封装为DataContainer
    queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
    # imgs: Tensor [queue_length, N_cams, C, H, W] = [5, 6, 3, H, W]
    
    queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
    # img_metas: Dict {0: meta_frame0, 1: meta_frame1, ..., 4: meta_frame4}
    
    queue = queue[-1]  # 使用最后一帧作为容器
    
    # 5. 封装各字段为DataContainer
    queue['gt_labels_3d'] = DC(gt_labels_3d_list)  # List[Tensor], 长度=5
    queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
    queue['gt_inds'] = DC(gt_inds_list)  # List[Tensor], 长度=5
    
    queue['l2g_r_mat'] = DC(l2g_r_mat_list)  # List[Tensor], 长度=5
    queue['l2g_t'] = DC(l2g_t_list)  # List[Tensor], 长度=5
    queue['timestamp'] = DC(timestamp_list)  # List[Tensor], 长度=5
    
    # 历史轨迹（每帧都有）
    queue['gt_past_traj'] = DC(gt_past_traj_list)  # List[Tensor [N, 4, 2]], 长度=5
    queue['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
    
    # 未来轨迹（只有当前帧）
    queue['gt_fut_traj'] = DC(gt_fut_traj)  # Tensor [N, 12, 2]
    queue['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)  # Tensor [N, 12]
    queue['gt_sdc_fut_traj'] = DC(gt_sdc_fut_traj)  # Tensor [6, 2]
    queue['gt_sdc_fut_traj_mask'] = DC(gt_sdc_fut_traj_mask)  # Tensor [6]
    
    # 未来帧检测框（用于规划碰撞检测）
    queue['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
    queue['gt_future_labels'] = DC(gt_future_labels_list)
    
    return queue
```

### CAN Bus相对运动计算

**CAN Bus数据格式**：
```python
can_bus = [x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw]
# 长度=12，包含位置、速度、加速度、姿态
```

**相对运动计算逻辑**：

```
初始状态：
  Frame 0 (t=t0): pos0=(x0, y0, z0), yaw0
  → 相对运动 = (0, 0, 0), Δyaw=0  （参考帧）

  Frame 1 (t=t1): pos1=(x1, y1, z1), yaw1
  → 相对运动 = pos1 - pos0, Δyaw = yaw1 - yaw0

  Frame 2 (t=t2): pos2=(x2, y2, z2), yaw2
  → 相对运动 = pos2 - pos1, Δyaw = yaw2 - yaw1
  
  ...
```

**作用**：
- **BEV特征对齐**：通过相对运动将历史BEV特征warp到当前帧坐标系
- **Temporal Self-Attention**：利用相对位姿信息对齐不同时刻的特征

**可视化示例**：
```
时间轴：t-2        t-1        t (当前)
       Frame0     Frame1     Frame2
位置：  (0,0)      (2,0)      (5,1)
角度：   0°         5°         10°

CAN Bus相对运动：
Frame0: pos=(0,0,0), yaw=0     # 参考帧
Frame1: pos=(2,0,0), yaw=5°    # 相对Frame0
Frame2: pos=(3,1,0), yaw=5°    # 相对Frame1
```

---

## DataContainer机制

### 概述

**DataContainer (DC)**是MMDetection的核心数据结构，用于灵活处理batch数据。

```python
from mmcv.parallel import DataContainer as DC

# 参数说明
DC(data, 
   cpu_only=False,  # True: 数据保持在CPU（如3D框对象）
   stack=False,     # True: collate时自动堆叠成Tensor
   padding_value=0, # padding时的填充值
   pad_dims=None)   # 指定padding的维度
```

### 使用场景

#### 1. cpu_only=True：无法直接转为Tensor的数据

```python
queue['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
# gt_bboxes_3d_list: List[LiDARInstance3DBoxes对象]
```

**原因**：
- `LiDARInstance3DBoxes`是自定义类，包含方法和属性
- 无法直接转换为Tensor
- 需要保持在CPU，使用时再提取数据

#### 2. stack=True：需要堆叠的Tensor

```python
queue['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
# imgs: Tensor [5, 6, 3, H, W]
```

**作用**：
- `stack=True`告诉collate函数，这个Tensor已经是堆叠好的
- Batch时直接再次堆叠：`[B, 5, 6, 3, H, W]`

#### 3. 默认（cpu_only=False, stack=False）：保持List结构

```python
queue['gt_labels_3d'] = DC(gt_labels_3d_list)
# gt_labels_3d_list: List[Tensor], 长度=5
```

**作用**：
- 保持List结构不变
- Batch时变成List[List]

### Collate时的行为

```python
# Batch size = 2
batch = [
    {'img': DC(tensor1, stack=True), 'gt_labels': DC([label1_frame0, label1_frame1])},
    {'img': DC(tensor2, stack=True), 'gt_labels': DC([label2_frame0, label2_frame1])}
]

# Collate后：
{
    'img': Tensor [2, 5, 6, 3, H, W],  # stack=True，自动堆叠
    'gt_labels': [
        [label1_frame0, label1_frame1],  # sample 1
        [label2_frame0, label2_frame1]   # sample 2
    ]  # stack=False，保持List[List]结构
}
```

### 为什么需要DataContainer？

**问题**：不同字段有不同的数据组织需求
- 图像：固定shape，可以stack成大Tensor
- 检测框：每帧数量不同，无法直接stack
- 实例ID：需要保持原始索引关系

**解决方案**：DataContainer提供统一接口，支持灵活的collate策略

---

## 数据结构总结

### 1. 单帧数据结构（get_data_info返回）

```python
{
    # ===== 基础信息 =====
    'sample_idx': str,              # 样本token
    'pts_filename': str,            # 点云文件路径
    'sweeps': list,                 # sweep信息
    'timestamp': float,             # 时间戳（秒）
    'scene_token': str,             # 场景token
    'frame_idx': int,               # 帧索引
    'can_bus': np.ndarray,          # [18] CAN总线数据
    
    # ===== 坐标变换 =====
    'ego2global_translation': np.ndarray,  # [3]
    'ego2global_rotation': np.ndarray,     # [4] 四元数
    'l2g_r_mat': np.ndarray,        # [3, 3] Lidar到Global旋转
    'l2g_t': np.ndarray,            # [3] Lidar到Global平移
    
    # ===== 相机信息 =====
    'img_filename': list[str],      # 6个相机图像路径
    'lidar2img': list[np.ndarray],  # 6个 [4, 4] 投影矩阵
    'lidar2cam': list[np.ndarray],  # 6个 [4, 4] 变换矩阵
    'cam_intrinsic': list[np.ndarray],  # 6个 [4, 4] 内参矩阵
    
    # ===== 检测标注 =====
    'ann_info': {
        'gt_bboxes_3d': LiDARInstance3DBoxes,  # [N, 9]
        'gt_labels_3d': np.ndarray,            # [N]
        'gt_names': list[str],                 # [N]
        'gt_inds': np.ndarray,                 # [N] 实例ID
        'gt_fut_traj': np.ndarray,             # [N, 12, 2] 未来轨迹
        'gt_fut_traj_mask': np.ndarray,        # [N, 12, 2] 未来轨迹mask
        'gt_past_traj': np.ndarray,            # [N, 8, 2] 历史+近期未来
        'gt_past_traj_mask': np.ndarray,       # [N, 8, 2] 历史轨迹mask
        'gt_sdc_bbox': LiDARInstance3DBoxes,   # [1, 9] 自车bbox
        'gt_sdc_label': np.ndarray,            # [1] 自车标签
        'gt_sdc_fut_traj': np.ndarray,         # [1, 12, 2] 自车未来轨迹
        'gt_sdc_fut_traj_mask': np.ndarray,    # [1, 12, 2] 自车轨迹mask
        'sdc_planning': np.ndarray,            # [1, 6, 3] 规划轨迹(x,y,yaw)
        'sdc_planning_mask': np.ndarray,       # [1, 6, 2] 规划mask
        'command': int,                        # 0/1/2 转向指令
    },
    
    # ===== 地图标注 =====
    'gt_lane_labels': torch.Tensor,   # [M] 实例类别
    'gt_lane_bboxes': torch.Tensor,   # [M, 4] 实例bbox
    'gt_lane_masks': torch.Tensor,    # [M, 200, 200] 实例mask
    
    # ===== 占用预测相关 =====
    'occ_has_invalid_frame': bool,             # 是否有无效帧
    'occ_img_is_valid': np.ndarray,           # [7] 每帧有效性
    'occ_l2e_r_mats': List[Tensor or None],   # [5] Lidar→Ego旋转
    'occ_l2e_t_vecs': List[Tensor or None],   # [5] Lidar→Ego平移
    'occ_e2g_r_mats': List[Tensor or None],   # [5] Ego→Global旋转
    'occ_e2g_t_vecs': List[Tensor or None],   # [5] Ego→Global平移
    'occ_future_ann_infos': List[dict],       # [5] 未来帧检测标注
}
```

### 2. 时序批次数据结构（union2one返回）

```python
{
    # ===== 图像数据 =====
    'img': DC(Tensor),              # [queue_length, 6, 3, H, W] = [5, 6, 3, 900, 1600]
    
    # ===== 元数据 =====
    'img_metas': DC({
        0: {                        # 第0帧（最早）
            'prev_bev': False,
            'can_bus': [...],       # 相对运动为0
            'lidar2img': [...],
            'timestamp': float,
            ...
        },
        1: {                        # 第1帧
            'prev_bev': True,
            'can_bus': [...],       # 相对于第0帧的运动
            ...
        },
        ...
        4: {                        # 第4帧（当前）
            'prev_bev': True,
            'can_bus': [...],       # 相对于第3帧的运动
            ...
        }
    }),
    
    # ===== 检测标注（多帧） =====
    'gt_bboxes_3d': DC([            # List of LiDARInstance3DBoxes
        boxes_0,                    # 第0帧的boxes
        boxes_1,
        ...
        boxes_4                     # 第4帧的boxes
    ]),
    'gt_labels_3d': DC([            # List of Tensor
        labels_0,                   # [N0]
        labels_1,                   # [N1]
        ...
        labels_4                    # [N4]
    ]),
    'gt_inds': DC([                 # List of Tensor
        inds_0,                     # [N0] 实例ID
        inds_1,
        ...
        inds_4
    ]),
    
    # ===== 轨迹标注（仅当前帧） =====
    'gt_fut_traj': DC(Tensor),              # [N4, 12, 2] 物体未来轨迹
    'gt_fut_traj_mask': DC(Tensor),         # [N4, 12, 2] 未来轨迹mask
    'gt_past_traj': DC([                    # List of Tensor
        past_traj_0,                        # [N0, 8, 2] 第0帧
        ...
        past_traj_4                         # [N4, 8, 2] 第4帧
    ]),
    'gt_past_traj_mask': DC([               # List of Tensor
        past_mask_0,                        # [N0, 8, 2]
        ...
        past_mask_4                         # [N4, 8, 2]
    ]),
    
    # ===== SDC轨迹和规划 =====
    'gt_sdc_fut_traj': DC(Tensor),          # [1, 12, 2] SDC未来轨迹
    'gt_sdc_fut_traj_mask': DC(Tensor),     # [1, 12, 2] SDC轨迹mask
    'sdc_planning': Tensor,                 # [6, 2] 规划点(x,y)
    'sdc_planning_mask': Tensor,            # [6] 规划mask
    'command': int,                         # 0/1/2 转向指令
    
    # ===== 坐标变换（多帧） =====
    'l2g_r_mat': DC([               # List of Tensor
        R_0,                        # [3, 3]
        R_1,
        ...
        R_4
    ]),
    'l2g_t': DC([                   # List of Tensor
        t_0,                        # [3]
        t_1,
        ...
        t_4
    ]),
    'timestamp': DC([               # List of Tensor
        ts_0,                       # [1]
        ts_1,
        ...
        ts_4
    ]),
    
    # ===== 地图标注（当前帧） =====
    'gt_lane_labels': Tensor,       # [M] 地图实例类别
    'gt_lane_bboxes': Tensor,       # [M, 4] 地图实例bbox
    'gt_lane_masks': Tensor,        # [M, 200, 200] 地图实例mask
    
    # ===== 占用预测相关 =====
    'occ_has_invalid_frame': bool,          # 是否有无效帧
    'occ_img_is_valid': np.ndarray,        # [7] 每帧有效性
    'occ_l2e_r_mats': List[Tensor],        # [5] Lidar→Ego旋转
    'occ_l2e_t_vecs': List[Tensor],        # [5] Lidar→Ego平移
    'occ_e2g_r_mats': List[Tensor],        # [5] Ego→Global旋转
    'occ_e2g_t_vecs': List[Tensor],        # [5] Ego→Global平移
    'occ_future_ann_infos': List[dict],    # [5] 未来帧检测标注
    
    # ===== 占用GT（pipeline生成） =====
    'gt_segmentation': Tensor,      # [5, H, W] BEV语义分割
    'gt_instance': Tensor,          # [5, H, W] BEV实例分割
    'gt_centerness': Tensor,        # [5, H, W] 中心度图
    'gt_offset': Tensor,            # [5, 2, H, W] 偏移场
    'gt_flow': Tensor,              # [5, 2, H, W] 前向光流
    'gt_backward_flow': Tensor,     # [5, 2, H, W] 后向光流
}
```

### 3. 数据流图

```
数据加载入口
    │
    ├─> __getitem__(idx)
    │       │
    │       ├─> prepare_train_data(index)  [训练模式]
    │       │       │
    │       │       ├─> 确定时序范围 (queue_length=5)
    │       │       │   ├─> first_index = index - 4
    │       │       │   └─> final_index = index
    │       │       │
    │       │       ├─> 循环获取5帧数据
    │       │       │   └─> get_data_info(i) × 5
    │       │       │           │
    │       │       │           ├─> 加载基础信息
    │       │       │           ├─> 计算坐标变换矩阵
    │       │       │           ├─> 处理相机信息
    │       │       │           ├─> get_ann_info(i) [检测标注]
    │       │       │           │       │
    │       │       │           │       ├─> 过滤有效框
    │       │       │           │       ├─> 类别映射
    │       │       │           │       ├─> 添加速度
    │       │       │           │       ├─> 格式转换
    │       │       │           │       ├─> trajectory_api.get_traj_label()
    │       │       │           │       │       └─> 生成物体轨迹 (Motion)
    │       │       │           │       ├─> trajectory_api.get_sdc_traj_label()
    │       │       │           │       │       └─> 生成SDC轨迹
    │       │       │           │       └─> trajectory_api.get_sdc_planning_label()
    │       │       │           │               └─> 生成规划轨迹 (Planning)
    │       │       │           │
    │       │       │           ├─> 地图标注处理
    │       │       │           │       │
    │       │       │           │       ├─> vector_map.gen_vectorized_samples()
    │       │       │           │       │       └─> 提取向量地图
    │       │       │           │       │
    │       │       │           │       ├─> preprocess_map()
    │       │       │           │       │       ├─> 向量→栅格转换
    │       │       │           │       │       ├─> 语义分割mask
    │       │       │           │       │       ├─> 实例分割mask
    │       │       │           │       │       └─> 方向mask
    │       │       │           │       │
    │       │       │           │       ├─> 实例提取
    │       │       │           │       │       ├─> 遍历实例ID
    │       │       │           │       │       ├─> 提取bbox
    │       │       │           │       │       └─> 提取mask
    │       │       │           │       │
    │       │       │           │       └─> obtain_map_info()
    │       │       │           │               └─> 额外地图层
    │       │       │           │
    │       │       │           └─> 占用预测标注处理 (Occupancy)
    │       │       │                   │
    │       │       │                   ├─> occ_get_temporal_indices()
    │       │       │                   │       └─> 获取时序帧索引[t-2...t+4]
    │       │       │                   │
    │       │       │                   ├─> occ_get_transforms()
    │       │       │                   │       └─> 获取坐标变换矩阵
    │       │       │                   │
    │       │       │                   └─> get_future_detection_infos()
    │       │       │                           └─> 获取未来5帧检测标注
    │       │       │
    │       │       ├─> pipeline(input_dict) × 5
    │       │       │       ├─> 数据增强和预处理
    │       │       │       └─> GenerateOccFlowLabels (生成占用GT)
    │       │       │               ├─> 3D boxes → BEV投影
    │       │       │               ├─> 光栅化语义/实例分割
    │       │       │               └─> 计算光流场
    │       │       │
    │       │       └─> union2one(data_queue)
    │       │               │
    │       │               ├─> 收集各帧数据
    │       │               ├─> 计算相对运动（CAN Bus）
    │       │               ├─> 封装为DataContainer
    │       │               └─> 返回batch字典
    │       │
    │       └─> prepare_test_data(index)  [测试模式]
    │               └─> get_data_info(index)
    │                   └─> 单帧处理（同上）
    │
    └─> 返回数据字典
```

---

## 关键坐标系统说明

### 1. 坐标系定义

**Lidar坐标系**：
- 原点：Lidar传感器位置
- X轴：前方
- Y轴：左方
- Z轴：上方

**Ego坐标系**：
- 原点：车辆中心
- X轴：前方
- Y轴：左方
- Z轴：上方

**Global坐标系**：
- 原点：地图原点
- X轴：东
- Y轴：北
- Z轴：上

### 2. 变换矩阵计算

```python
# Lidar → Ego
l2e_R = Quaternion(lidar2ego_rotation).rotation_matrix  # [3, 3]
l2e_t = lidar2ego_translation                           # [3]

# Ego → Global
e2g_R = Quaternion(ego2global_rotation).rotation_matrix  # [3, 3]
e2g_t = ego2global_translation                           # [3]

# Lidar → Global (组合)
l2g_R = l2e_R.T @ e2g_R.T  # 旋转组合
l2g_t = l2e_t @ e2g_R.T + e2g_t  # 平移组合

# 点变换公式
p_global = p_lidar @ l2g_R.T + l2g_t
```

### 3. BEV空间定义

```python
# 物理空间
x_range = [-51.2, 51.2]  # 米
y_range = [-51.2, 51.2]  # 米

# 像素空间
H = W = 200  # 像素

# 分辨率
resolution = 102.4 / 200 = 0.512  # 米/像素

# 坐标转换
pixel_x = (world_x - x_min) / resolution
pixel_y = (world_y - y_min) / resolution
```

---

## 附录：重要API说明

### 1. VectorizedLocalMap

**功能**：从NuScenes HD地图提取局部向量地图

**方法**：
```python
def gen_vectorized_samples(location, ego_translation, ego_rotation):
    """
    Args:
        location: 地图名称 (e.g., 'boston-seaport')
        ego_translation: [x, y, z] 全局坐标
        ego_rotation: [w, x, y, z] 四元数
    
    Returns:
        vectors: {
            'divider': List[LineString],
            'ped_crossing': List[LineString],
            'boundary': List[LineString],
        }
    """
```

### 2. preprocess_map

**功能**：向量地图栅格化

**方法**：
```python
def preprocess_map(vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    """
    Args:
        vectors: 向量地图字典
        patch_size: (102.4, 102.4) 物理尺寸
        canvas_size: (200, 200) 像素尺寸
        num_classes: 3 类别数
        thickness: 2 线条粗细
        angle_class: 36 角度分类数
    
    Returns:
        semantic_masks: [3, 200, 200]
        instance_masks: [3, 200, 200]
        forward_masks: [3, 200, 200]
        backward_masks: [3, 200, 200]
    """
```

### 3. NuScenesTraj

**功能**：生成轨迹和规划标注

**关键方法**：
```python
def get_traj_label(sample_token, ann_tokens):
    """生成物体轨迹标注"""
    
def get_sdc_traj_label(sample_token):
    """生成自车轨迹标注"""
    
def get_sdc_planning_label(sample_token):
    """生成自车规划标注"""
```

### 4. LiDARInstance3DBoxes

**功能**：3D边界框容器

**属性**：
```python
box.tensor       # [N, 9] 原始tensor
box.corners      # [N, 8, 3] 8个角点坐标
box.center       # [N, 3] 中心坐标
box.wlh          # [N, 3] 宽长高
box.yaw          # [N] 航向角
box.velocity     # [N, 2] 速度
```

---

## 总结

UniAD的数据集处理流程特点：

### 1. **多任务统一处理**
在一个dataset类中完成5个任务的真值生成：
- **Detection**（检测）：3D bounding boxes + 实例ID
- **Tracking**（跟踪）：通过gt_inds关联多帧实例
- **Motion**（运动预测）：基于PredictHelper的未来轨迹
- **Planning**（规划）：SDC的未来轨迹和转向指令
- **Occupancy**（占用预测）：BEV占用栅格图 + 光流场
- **Map**（地图分割）：向量+栅格混合表示

### 2. **时序建模策略**
- **检测/地图**：连续5帧 (queue_length=5)
- **运动预测**：未来12步 (6秒)，历史4步 (2秒)
- **规划**：未来6步 (3秒)
- **占用预测**：历史2帧 + 当前 + 未来4帧 (共7帧)

### 3. **坐标系统层次**
```
Lidar坐标系 (数据原始坐标)
    ↓ l2e_r_mat, l2e_t
Ego坐标系 (车辆中心)
    ↓ e2g_r_mat, e2g_t
Global坐标系 (世界坐标)
    ↓ 各任务的坐标变换
Scene Centric / Agent Frame (任务特定坐标)
```

### 4. **高效的数据组织**
- **DataContainer (DC)**：灵活的batch处理机制
  - `cpu_only=True`：元数据、3D框对象
  - `cpu_only=False`：可GPU加速的tensor
  - `stack=True`：collate时自动堆叠
- **List结构**：多帧数据保持独立性
- **嵌套字典**：img_metas按帧索引组织

### 5. **真值生成特点**

#### Detection
- 直接从NuScenes标注转换
- 过滤低质量框（use_valid_flag）
- KITTI格式兼容性处理

#### Map
- 向量地图 → 栅格化
- 多通道分类（3类基础 + 2类额外）
- 方向信息编码（36个角度类别）

#### Motion
- PredictHelper API调用
- 局部坐标 → 场景中心坐标变换
- 双向轨迹（过去+未来）

#### Planning
- 复杂的多步坐标变换
- 未来帧Lidar → 当前帧Lidar
- 基于轨迹终点的指令生成

#### Occupancy
- 时序检测框 → BEV占用图
- 实例匹配 → 光流计算
- 7帧观测窗口

### 6. **数据pipeline流程**

```
__getitem__(idx)
    ↓
prepare_train_data(index)
    ↓
for i in [index-4, ..., index]:
    get_data_info(i)
        ├─> 加载基础信息
        ├─> 计算坐标变换
        ├─> get_ann_info(i)
        │   ├─> Detection GT
        │   ├─> Motion GT (trajectory_api)
        │   └─> Planning GT (trajectory_api)
        ├─> 地图标注处理
        │   ├─> vector_map.gen_vectorized_samples()
        │   └─> preprocess_map()
        └─> 占用标注准备
            ├─> occ_get_temporal_indices()
            ├─> occ_get_transforms()
            └─> get_future_detection_infos()
    ↓
    pipeline(input_dict)
        ├─> 数据增强
        └─> GenerateOccFlowLabels (占用GT)
    ↓
union2one(data_queue)
    ├─> 收集各帧数据
    ├─> 计算相对运动（CAN Bus）
    └─> 封装为DataContainer
    ↓
返回batch数据
```

### 7. **关键设计理念**

1. **一次加载，多任务复用**：同一份检测框生成多个任务的GT
2. **时序信息保留**：通过list结构保持帧间独立性
3. **灵活的坐标变换**：支持多层级坐标系转换
4. **向量+栅格结合**：地图既有向量表示也有栅格表示
5. **实例ID追踪**：gt_inds实现跨帧实例关联

整个pipeline从`__getitem__`开始，经过`prepare_train_data` → `get_data_info` → 多任务GT生成 → `pipeline` → `union2one`，最终生成包含5个任务标注的完整训练数据。

---

## 常见问题FAQ

### 1. 为什么gt_inds要加1？

```python
ins_inds_add_1=True  # 实例ID从1开始
```

**原因**：
- NuScenes原始数据中，实例ID从0开始编号
- 在跟踪任务中，需要区分"无效实例"（-1或0）和"有效实例"（≥1）
- 通过`ins_inds_add_1=True`，将实例ID整体+1：
  - 原始: [0, 1, 2, ...] → 处理后: [1, 2, 3, ...]
  - 0可以保留给背景或无效实例
  - SDC（自车）的query ID从-10变为-9

### 2. 为什么有些字段用List，有些用Tensor？

**设计原则**：
- **List结构**：多帧数据保持独立性，方便按帧索引
  - 例如：`gt_labels_3d_list = [labels_frame0, labels_frame1, ...]`
  - 每帧的目标数量可能不同，无法直接堆叠
  
- **Tensor结构**：统一形状，可批处理
  - 例如：`gt_fut_traj: Tensor [N, 12, 2]`
  - 所有实例的未来轨迹长度固定（12步）

### 3. queue_length到底是多少？

**配置文件中的定义**：
```python
queue_length = 3  # 历史帧数（不含当前帧）
```

**实际时序长度**：
```python
总帧数 = queue_length + 1 = 3 + 1 = 4帧
# 包含：3个历史帧 + 1个当前帧
```

**时间轴**：
```
t-3    t-2    t-1    t (当前)
 ↓      ↓      ↓      ↓
Frame0 Frame1 Frame2 Frame3
```

**代码中的索引**：
```python
final_index = index  # 当前帧
first_index = index - queue_length  # 最早的历史帧
# range(first_index, final_index+1) = [index-3, index-2, index-1, index]
```

### 4. 占用预测的时序窗口是多少？

**配置**：
```python
occ_receptive_field = 3  # 过去+当前
occ_n_future = 4  # 未来
```

**时序窗口**：
```
总共 = occ_receptive_field + occ_n_future = 3 + 4 = 7帧

过去        当前  未来
t-2  t-1    t    t+1  t+2  t+3  t+4
 ↓    ↓     ↓     ↓    ↓    ↓    ↓
obs  obs   obs   fut  fut  fut  fut
```

**与检测/地图的queue_length的关系**：
- 检测/地图：queue_length=3，总共4帧 (t-3 ~ t)
- 占用：感受野3帧 (t-2 ~ t)，未来4帧 (t+1 ~ t+4)
- **不完全重叠**：占用的历史窗口更短

### 5. 为什么地图分割有3个类别，但semantic_masks有5个通道？

**基础类别（3类）**：
```python
CLASS2LABEL = {
    'road_divider': 0,
    'lane_divider': 0,  # 合并到divider
    'ped_crossing': 1,
    'contours': 2,
}
```

**扩展类别（额外2类）**：
```python
line_classes = ['road_divider', 'lane_divider']  # 2类
contour_classes = ['road_segment', 'lane']  # 2类
```

**实际输出**：
- `semantic_masks`: [3, 200, 200]  # 3个基础类别
- 在某些实现中，可能扩展到5通道：
  - Channel 0: divider (road + lane)
  - Channel 1: ped_crossing
  - Channel 2: contours
  - Channel 3: road_divider (单独)
  - Channel 4: lane_divider (单独)

### 6. Pipeline中的pre_pipeline做了什么？

```python
def pre_pipeline(self, results):
    """在pipeline之前的预处理"""
    results['img_prefix'] = self.data_root
    results['seg_prefix'] = None
    results['proposal_file'] = None
    results['bbox3d_fields'] = []
    results['mask_fields'] = []
    results['seg_fields'] = []
```

**作用**：初始化必要的字段，确保pipeline中的transform能正常访问。

### 7. 为什么训练和测试的pipeline不同？

**训练Pipeline**：
- 包含数据增强：`PhotoMetricDistortionMultiViewImage`
- 包含过滤器：`ObjectRangeFilterTrack`, `ObjectNameFilterTrack`
- 加载完整标注：`with_bbox_3d=True`, `with_label_3d=True`

**测试Pipeline**：
- 无数据增强（保证可复现性）
- 无过滤器（评估所有目标）
- 标注加载最小化：`with_bbox_3d=False`, `with_label_3d=False`
  - 仅加载用于评估的GT（如占用、规划）

**示例**：
```python
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', ...),
    # 无PhotoMetricDistortion
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,  # 推理时不需要检测GT
         with_label_3d=False,
         with_future_anns=True),  # 但需要占用/规划GT用于评估
    dict(type='GenerateOccFlowLabels', ...),
    dict(type="MultiScaleFlipAug3D", ...),
]
```

### 8. 坐标系变换的优先级是什么？

**变换链**：
```
Sensor坐标 (Lidar/Camera)
    ↓ sensor2lidar (s2l)
Lidar坐标
    ↓ lidar2ego (l2e_r_mat, l2e_t)
Ego坐标 (车辆中心)
    ↓ ego2global (e2g_r_mat, e2g_t)
Global坐标 (世界坐标系)
```

**每个任务使用的坐标系**：
- **Detection**：Lidar坐标（预测框直接在Lidar坐标系）
- **Map**：Ego坐标（BEV以ego为中心）
- **Motion**：Global坐标（轨迹需要跨帧关联，用全局坐标）
  - 预测时转换为Scene Centric（以当前ego为中心的全局坐标）
- **Planning**：Lidar坐标（ego vehicle的规划轨迹在lidar系）
- **Occupancy**：Ego坐标（BEV栅格以ego为中心）

### 9. 如何调试Pipeline？

**方法1：逐步打印**
```python
from projects.mmdet3d_plugin.datasets import NuScenesE2EDataset

# 创建dataset
dataset = NuScenesE2EDataset(...)

# 获取单个样本
data = dataset[0]

# 打印各字段shape
for key, value in data.items():
    if hasattr(value, 'data'):
        print(f"{key}: {type(value.data)}")
        if isinstance(value.data, torch.Tensor):
            print(f"  shape: {value.data.shape}")
        elif isinstance(value.data, list):
            print(f"  len: {len(value.data)}")
```

**方法2：禁用某些Pipeline组件**
```python
# 临时移除数据增强
train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", ...),
    # dict(type="PhotoMetricDistortionMultiViewImage"),  # 注释掉
    ...
]
```

**方法3：使用debugger**
```python
# 在nuscenes_e2e_dataset.py的prepare_train_data中设置断点
def prepare_train_data(self, index):
    import pdb; pdb.set_trace()  # 断点
    ...
```

### 10. 如何验证数据正确性？

**检查清单**：

1. **Shape一致性**
```python
assert len(gt_labels_3d) == len(gt_bboxes_3d)
assert len(gt_inds) == len(gt_fut_traj)
```

2. **坐标范围**
```python
assert gt_bboxes_3d.in_range_bev(bev_range).all()
```

3. **实例ID连续性**
```python
# 跨帧检查同一实例的ID是否一致
for i in range(len(queue)):
    print(f"Frame {i} instance IDs: {queue[i]['gt_inds']}")
```

4. **可视化**
```python
# 可视化BEV地图
import matplotlib.pyplot as plt
plt.imshow(semantic_masks[0])  # divider
plt.show()
```
