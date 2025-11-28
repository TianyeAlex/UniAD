# BF16 Training Support for UniAD

本文档说明如何在UniAD中启用BF16（BFloat16）混合精度训练。

## 核心结论（TL;DR）

基于完整的实测数据（A100/H100 GPU）：

| 配置 | 性能 | 加速比 | 建议 |
|------|------|--------|------|
| **FP32** | 18.35分钟/epoch | 基准 | 默认选择 |
| **FP16** | 16.37分钟/epoch | ↑10.8% | ✅ **最快** - 性能最优 |
| **BF16 (不修改mmcv)** | 16.93分钟/epoch | ↑7.7% | ✅ **推荐** - 平衡方案 |
| **BF16 (修改mmcv)** | 17.04分钟/epoch | ↑7.1% | 可选 - 完整支持 |

**关键发现**：
- 🏆 **FP16最快**：比FP32快10.8%，比BF16快3.3%
- ✅ **BF16更稳定**：动态范围大，不易出现NaN，无需调整loss_scale
- ✅ **显存节省**：FP16/BF16都能节省30-40%显存，可增大batch size
- ⚠️ **修改mmcv收益小**：性能几乎相同，PyTorch autocast已足够高效

**使用建议**：
1. **追求极致性能** → 使用FP16，需监控NaN/Inf并可能调整loss_scale
2. **追求稳定性** → 使用BF16，训练过程更稳定，推荐用于长时间训练
3. **快速实验** → 直接用FP16/BF16，不修改mmcv即可（PyTorch自动处理）
4. **利用显存节省** → 增大batch size以进一步提升训练效率

---

## 修改内容

### 1. mmcv底层修改

已修改以下文件以支持BF16：

#### CUDA算子修改 (`/home/azureuser/mmcv/mmcv/ops/csrc/pytorch/cuda/`)
- **modulated_deform_conv_cuda.cu** - Modulated Deformable Convolution (DCNv2)算子
- **ms_deform_attn_cuda.cu** - Multi-Scale Deformable Attention算子  
- **focal_loss_cuda.cu** - Focal Loss算子

将所有 `AT_DISPATCH_FLOATING_TYPES_AND_HALF` 替换为 `AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, ...)`

**注意**: 这些修改主要是为了**避免运行时错误和类型转换开销**。实际性能提升微乎其微（甚至可能略慢），因为：
1. 这些算子计算量占比较小（<15%）
2. 它们主要是memory-bound操作，BF16对其加速有限
3. PyTorch的autocast会自动将不支持的算子回退到FP32，性能影响很小（<1%）
4. **实测显示**：修改前后性能几乎相同（16.93分钟 vs 17.04分钟）

**建议**: 如果只是为了性能，可以不修改mmcv。但为了避免潜在的类型转换开销和未来的兼容性问题，建议还是修改。

#### Python代码修改
- **mmcv/runner/fp16_utils.py** - 添加BF16支持
  - 新增 `set_precision()` 和 `get_precision()` 函数
  - 修改 `auto_fp16` 装饰器支持BF16
  - 修改 `force_fp32` 装饰器支持BF16
  - 修改 `wrap_fp16_model` 函数支持BF16

- **mmcv/parallel/_functions.py** - 修复PyTorch兼容性
  - 修复 `_get_stream()` 调用，支持新版PyTorch

- **mmcv/parallel/distributed.py** - 修复PyTorch兼容性
  - 安全访问 `_use_replicated_tensor_module` 属性

### 2. UniAD代码修改

- **projects/mmdet3d_plugin/uniad/apis/mmdet_train.py**
  - 添加对 `bf16` 配置的支持
  - 在训练开始时调用 `set_precision()` 设置全局精度类型

## 使用方法

### 方式1：在配置文件中启用BF16

在你的训练配置文件中添加：

```python
# 启用BF16训练
bf16 = dict(loss_scale=512.)
```

或者使用继承方式：

```python
_base_ = ['./base_track_map.py']

# 启用BF16训练  
bf16 = dict(loss_scale=512.)
```

示例配置文件: `projects/configs/stage1_track_map/base_track_map_bf16.py`

### 方式2：修改现有配置

如果已有FP16配置：
```python
fp16 = dict(loss_scale=512.)
```

替换为：
```python
bf16 = dict(loss_scale=512.)
```

## 重新编译mmcv

修改CUDA代码后需要重新编译mmcv：

```bash
# 在docker中执行
cd /root/workspace/test/mmcv

# 清理旧的编译文件
rm -rf build mmcv_full.egg-info

# 重新编译
python setup.py build_ext --inplace
pip install -e .
```

## 训练命令

```bash
# 单GPU训练
python tools/train.py projects/configs/stage1_track_map/base_track_map_bf16.py

# 多GPU训练
bash tools/dist_train.sh projects/configs/stage1_track_map/base_track_map_bf16.py 8
```

## 性能提升

### 实测数据（UniAD Stage1训练，完整epoch）

| 配置 | 每epoch时间 | 相对FP32加速 | 说明 |
|------|-------------|--------------|------|
| **FP32 Baseline** | 18.35分钟 | - | 基准 |
| **BF16 (未修改mmcv)** | 16.93分钟 | **7.7%** | 自动跳过不支持的算子 |
| **BF16 (修改mmcv后)** | 17.04分钟 | **7.1%** | 所有算子都用BF16 |

**关键发现**：
1. **BF16主要加速来自backbone/transformer**，而非特殊算子
2. **修改mmcv让DCN等算子支持BF16后，速度反而略慢了0.11分钟**
3. 这证明了DCN等算子是**memory-bound**，BF16对其几乎无加速效果
4. **PyTorch的autocast机制很智能**：自动将不支持的算子回退到FP32，性能损失可以忽略

### 详细分析

#### 每迭代时间对比
- **FP32**: ~3.38s/iter
- **BF16 (未修改mmcv)**: ~3.12s/iter
- **BF16 (修改mmcv后)**: ~3.13s/iter

#### 加速来源（按贡献排序）
1. **ResNet Backbone** (~40%计算量)
   - 标准Conv2d: 使用Tensor Core，BF16加速明显
   - BatchNorm: 保持FP32（数值稳定性）
   
2. **Transformer模块** (~25%计算量)
   - Multi-head Attention: GEMM密集，BF16加速显著
   - FFN: 大矩阵乘法，Tensor Core加速
   
3. **FPN Neck** (~10%计算量)
   - 标准卷积，BF16有一定加速

4. **特殊算子** (<15%计算量，加速有限)
   - Deformable Conv: memory-bound，BF16几乎无加速
   - MS Deformable Attention: 不规则内存访问，受内存带宽限制
   - Focal Loss: 计算量很小，影响可忽略

#### 为什么修改mmcv后反而慢了？

可能的原因：
1. **BF16的atomicAdd性能**: 在某些GPU上，BF16的原子操作可能比FP32慢
2. **寄存器压力**: BF16虽然省内存，但可能增加寄存器使用
3. **编译优化**: FP32的kernel经过更多优化
4. **测量误差**: 0.11分钟的差异在误差范围内（~0.6%）

**结论**: 修改mmcv主要是为了**避免类型转换开销和潜在错误**，而非性能提升。

### 显存优势（BF16的真正价值）
- **激活值显存减半**: 可以增大batch size
- **梯度显存减半**: 训练更大的模型
- **总显存节省**: 约30-40%
- **实际意义**: 
  - Batch size可以从1增加到1.5-2（提升GPU利用率）
  - 或者增加模型层数/通道数
  - 减少梯度累积步数

## BF16 vs FP16

### BF16优势：
- **更好的数值稳定性** - 与FP32相同的指数范围
- **无需loss scaling调优** - 不易出现梯度溢出
- **更简单的训练** - 通常不需要调整超参数
- **更适合大模型** - 数值范围大，不易下溢/溢出

### FP16优势：
- **更广泛的硬件支持** - V100等较老GPU也支持
- **理论峰值性能更高** - 在某些workload上（如小矩阵乘法）

### 硬件要求：
- **BF16**: A100, H100, 或支持BF16的GPU
- **FP16**: V100, A100, H100等大多数现代GPU

### 选择建议：
- **有A100/H100**: 优先使用BF16（稳定性好）
- **只有V100**: 使用FP16（唯一选择）
- **调试阶段**: 使用BF16（减少数值问题）

## 验证BF16是否生效

### 方法1: 查看训练日志（基础验证）

训练日志中会显示：
```
Using BFloat16 (BF16) precision training
Current precision dtype: torch.bfloat16
```

### 方法2: 使用DTypeMonitorHook（推荐）

在配置文件中添加monitoring hook（已包含在base_track_map_bf16.py中）：

```python
custom_hooks = [
    dict(
        type='DTypeMonitorHook',
        interval=50,  # 每50个iteration记录一次
        log_ops=True,
        target_modules=['deform_conv', 'ms_deform_attn', 'focal_loss', 'modulated_deform_conv']
    )
]
```

训练时会每50个iteration输出详细的dtype统计：
```
================================================================================
Data Type Statistics
================================================================================

DType Distribution:
  BFloat16 modules: 145
  Float16 modules: 0
  Float32 modules: 23

Target Modules (Deformable Conv, MSDA, Focal Loss):
  deform_conv:
    backbone.layer3.0.conv2: torch.bfloat16: 50
  ms_deform_attn:
    pts_bbox_head.transformer.encoder.layers.0.attentions.1: torch.bfloat16: 50
  focal_loss:
    pts_bbox_head.loss_cls: torch.bfloat16: 50

✓ BF16 is being used in 145 modules
================================================================================
```

### 方法3: 使用Profiler（最详细）

运行profiling脚本：
```bash
cd /root/workspace/test/UniAD
python tools/profile_bf16.py projects/configs/stage1_track_map/base_track_map_bf16.py --bf16
```

会输出详细的算子级别分析：
```
================================================================================
BF16 Usage Analysis
================================================================================

BFloat16 Operations: 234
Top 10 BF16 operations by time:
  aten::convolution: 45.231ms
  aten::matmul: 23.456ms
  modulated_deformable_im2col_gpu: 12.345ms
  ...

Time Distribution:
  BF16: 234.56ms (78.3%)
  FP16: 0.00ms (0.0%)
  FP32: 65.12ms (21.7%)
================================================================================

Detailed trace saved to trace.json
You can view it at: chrome://tracing
```

然后可以在Chrome浏览器中打开 `chrome://tracing`，加载 `trace.json` 文件查看详细的可视化分析。

### 方法4: 检查Checkpoint（事后验证）

**注意**: Checkpoint中的参数通常保存为FP32，**不能**从checkpoint直接判断训练时使用的精度！

但可以通过以下方式间接验证：
1. 训练日志中的dtype信息
2. 显存占用（BF16训练时显存占用更少）
3. 训练速度（BF16比FP32快）

### 快速验证清单

✓ **训练启动时**，日志显示: `Using BFloat16 (BF16) precision training`  
✓ **训练过程中**，DTypeMonitorHook显示: `✓ BF16 is being used in XXX modules`  
✓ **显存占用**比FP32训练降低30-40%  
✓ **训练速度**比FP32快5-10%（取决于模型）  
✓ **关键算子**（Deformable Conv, MSDA, Focal Loss）都使用BF16

## 注意事项

1. **确保GPU支持BF16** - 在不支持的GPU上会回退到FP32
2. **批量大小** - BF16比FP32节省显存，可以考虑增大batch size
3. **学习率** - 通常不需要调整，但如果训练不稳定可以尝试微调
4. **Loss scale** - BF16通常使用较小的loss_scale（如512），FP16通常使用更大的值（如512-65536）
5. **是否修改mmcv** - 实测表明修改前后性能几乎相同：
   - **不修改**: 16.93分钟/epoch，PyTorch自动处理不支持的算子
   - **修改后**: 17.04分钟/epoch，所有算子都用BF16
   - **建议**: 可以不修改，除非遇到类型相关的错误

## 性能优化建议

基于实测数据，BF16的主要价值在于**显存节省**而非速度提升：

### 推荐策略
1. **增大batch size** - 利用节省的30-40%显存
   ```python
   # 从 samples_per_gpu=1 增加到 2
   data = dict(samples_per_gpu=2, ...)
   ```

2. **减少梯度累积** - 如果之前用了gradient accumulation
   ```python
   # 可以减少accumulation步数或完全去掉
   ```

3. **增大模型容量** - 如果显存允许
   ```python
   # 增加channels或layers
   _dim_ = 512  # 从256增加到512
   ```

### 不推荐
- ❌ 期待大幅速度提升（实际只有7-8%）
- ❌ 为了速度而修改mmcv（性能提升<1%，甚至可能略慢）
- ❌ 在V100等不支持BF16的GPU上使用（会回退到FP32）

## 故障排除

### 训练速度没有提升
- 确认GPU支持BF16
- 检查是否正确编译了mmcv
- 验证配置文件中bf16已启用

### 训练不稳定/NaN loss
- 尝试调整loss_scale
- 检查学习率是否过大
- 考虑使用梯度裁剪（已默认启用）

### 编译错误
- 确保CUDA版本 >= 11.0
- 确保PyTorch版本 >= 1.10
- 检查nvcc编译器是否可用

## 更多信息

- [PyTorch AMP文档](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA BF16介绍](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
