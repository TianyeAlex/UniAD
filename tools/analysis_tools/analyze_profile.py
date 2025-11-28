#!/usr/bin/env python3
"""
完整的训练性能分析 - 针对最新 profiler 数据
分析墙上时间与 profiler 统计时间的差异
配置: wait=10, warmup=1, active=3, repeat=1
捕获迭代: 第 12、13、14 次（稳态性能）
"""
import json
from collections import defaultdict
import sys
import os
import glob

# 自动查找最新的 trace 文件
trace_pattern = "projects/work_dirs/stage1_track_map/base_track_map/profiler_logs/plugins/profile/*/*.pt.trace.json"
trace_files = glob.glob(trace_pattern)

if not trace_files:
    print(f"❌ 未找到 trace 文件，搜索路径: {trace_pattern}")
    sys.exit(1)

# 选择最新的文件
trace_file = max(trace_files, key=os.path.getmtime)

print("="*120)
print("UniAD 训练性能深度分析 (稳态性能)")
print("="*120)
print(f"\n正在加载 trace 文件:")
print(f"  {trace_file}")

file_size_mb = os.path.getsize(trace_file) / (1024 * 1024)
print(f"  文件大小: {file_size_mb:.1f} MB")

with open(trace_file, 'r') as f:
    trace_data = json.load(f)

trace_events = trace_data.get('traceEvents', [])
print(f"✓ 加载完成！总事件数: {len(trace_events):,}\n")

# ============= 数据结构 =============
cpu_ops = defaultdict(lambda: {'count': 0, 'total_time': 0, 'min_time': float('inf'), 'max_time': 0})
gpu_kernels = defaultdict(lambda: {'count': 0, 'total_time': 0, 'min_time': float('inf'), 'max_time': 0})
cuda_api = defaultdict(lambda: {'count': 0, 'total_time': 0})

# 训练阶段分类
stage_times = {
    'Forward Pass': 0,
    'Backward Pass': 0,
    'Optimizer Step': 0,
    'Loss Computation': 0,
    'Memory Operations': 0,
    'Data Loading': 0,
    'Synchronization': 0,
    'Other': 0
}

# 按时间轴分析（用于推算实际迭代时间）
event_timeline = []

print("正在分析事件...")
progress_interval = max(1, len(trace_events) // 20)
for i, event in enumerate(trace_events):
    if i % progress_interval == 0:
        print(f"  进度: {i:,}/{len(trace_events):,} ({i/len(trace_events)*100:.0f}%)")
    
    if event.get('ph') != 'X':  # 只处理完整事件
        continue
    
    name = event.get('name', 'unknown')
    dur = event.get('dur', 0)  # 微秒
    cat = event.get('cat', '')
    ts = event.get('ts', 0)  # 时间戳
    tid = event.get('tid', 0)
    
    # 记录时间轴（用于后续分析）
    if cat in ['cpu_op', 'kernel', 'cuda_runtime']:
        event_timeline.append({
            'ts': ts,
            'dur': dur,
            'cat': cat,
            'name': name,
            'tid': tid
        })
    
    # CPU 操作
    if cat == 'cpu_op':
        cpu_ops[name]['count'] += 1
        cpu_ops[name]['total_time'] += dur
        cpu_ops[name]['min_time'] = min(cpu_ops[name]['min_time'], dur)
        cpu_ops[name]['max_time'] = max(cpu_ops[name]['max_time'], dur)
        
        # 分类到训练阶段
        name_lower = name.lower()
        if any(x in name_lower for x in ['backward', 'grad', 'accumulategrad']):
            stage_times['Backward Pass'] += dur
        elif any(x in name_lower for x in ['loss', 'crossentropy', 'bceloss', 'mseloss']):
            stage_times['Loss Computation'] += dur
        elif any(x in name_lower for x in ['optimizer', 'adam', 'sgd', 'step']):
            stage_times['Optimizer Step'] += dur
        elif any(x in name_lower for x in ['copy_', 'to', 'clone', 'contiguous', '_copy']):
            stage_times['Memory Operations'] += dur
        elif any(x in name_lower for x in ['dataloader', 'collate', 'getitem']):
            stage_times['Data Loading'] += dur
        elif 'forward' not in name_lower:
            stage_times['Forward Pass'] += dur
        else:
            stage_times['Other'] += dur
    
    # GPU Kernel
    elif cat in ['kernel', 'gpu_memcpy', 'gpu_memset']:
        gpu_kernels[name]['count'] += 1
        gpu_kernels[name]['total_time'] += dur
        gpu_kernels[name]['min_time'] = min(gpu_kernels[name]['min_time'], dur)
        gpu_kernels[name]['max_time'] = max(gpu_kernels[name]['max_time'], dur)
    
    # CUDA API 调用
    elif cat in ['cuda_runtime', 'cuda_driver', 'Runtime']:
        cuda_api[name]['count'] += 1
        cuda_api[name]['total_time'] += dur
        
        if 'sync' in name.lower():
            stage_times['Synchronization'] += dur

print("\n✓ 分析完成！\n")

# ============= 时间轴分析 =============
print("="*120)
print("【关键指标】训练时间对比")
print("="*120)

# 计算总时间
cpu_total_us = sum(stats['total_time'] for stats in cpu_ops.values())
gpu_total_us = sum(stats['total_time'] for stats in gpu_kernels.values())
cuda_total_us = sum(stats['total_time'] for stats in cuda_api.values())

cpu_total_ms = cpu_total_us / 1000
gpu_total_ms = gpu_total_us / 1000
cuda_total_ms = cuda_total_us / 1000

# 从时间轴推算实际运行时间（墙上时间）
if event_timeline:
    event_timeline.sort(key=lambda x: x['ts'])
    start_ts = event_timeline[0]['ts']
    end_ts = max(e['ts'] + e['dur'] for e in event_timeline)
    wall_time_us = end_ts - start_ts
    wall_time_ms = wall_time_us / 1000
else:
    wall_time_ms = 0

print(f"""
根据训练日志: 每次迭代实际耗时约 3.6 秒
Profiler 配置: wait=10, warmup=1, active=3, repeat=1
因此捕获了: 第 12、13、14 次迭代（共 3 次迭代）
           ^^^^^^^^^^^^^^^^^^^^^^^^
           跳过前 10 次迭代，避免冷启动影响，数据更接近稳态性能

预期墙上时间: 3 × 3.6s = 10.8 秒

Profiler 统计时间分析:
┌────────────────────────────────────────────────────────────┐
│ 类型              │  总时间(秒)  │  平均/迭代  │  说明      │
├────────────────────────────────────────────────────────────┤
│ CPU 操作累计时间  │  {cpu_total_ms/1000:>10.2f}  │  {cpu_total_ms/3000:>9.2f}  │  所有CPU算子│
│ GPU Kernel 时间   │  {gpu_total_ms/1000:>10.2f}  │  {gpu_total_ms/3000:>9.2f}  │  GPU计算   │
│ CUDA API 开销     │  {cuda_total_ms/1000:>10.2f}  │  {cuda_total_ms/3000:>9.2f}  │  同步/API  │
│ 时间轴跨度        │  {wall_time_ms/1000:>10.2f}  │  {wall_time_ms/3000:>9.2f}  │  推算值    │
└────────────────────────────────────────────────────────────┘

关键发现:
• CPU 累计时间({cpu_total_ms/1000:.1f}s) >> 实际迭代时间(3.6s)
  → 说明大量 CPU 操作与 GPU 并行执行
  
• 并行效率 = 1 - (实际时间 / CPU时间)
             = 1 - (3.6 / {cpu_total_ms/3000:.2f})
             ≈ {(1 - 3.6/(cpu_total_ms/3000))*100:.1f}%
  → CPU-GPU 流水线已有不错的重叠

• GPU 实际计算时间: {gpu_total_ms/3000:.2f}s/迭代
  → GPU 利用率 = {gpu_total_ms/3000/3.6*100:.1f}% (基于3.6s墙上时间)
  → 仍有优化空间，理想情况可接近 2-2.5s/迭代
""")

# ============= 1. 训练阶段分析 =============
print("\n" + "="*120)
print("【1】训练各阶段耗时分布")
print("="*120)

stage_total_us = sum(stage_times.values())
stage_total_ms = stage_total_us / 1000
sorted_stages = sorted(stage_times.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'阶段':<25} {'总耗时(ms)':>15} {'平均/迭代(ms)':>18} {'占比':>10} {'累计':>10}")
print("-"*120)
cumulative = 0
for stage, time_us in sorted_stages:
    time_ms = time_us / 1000
    avg_ms = time_ms / 3
    percentage = (time_us / stage_total_us * 100) if stage_total_us > 0 else 0
    cumulative += percentage
    print(f"{stage:<25} {time_ms:>15.2f} {avg_ms:>18.2f} {percentage:>9.1f}% {cumulative:>9.1f}%")

print("-"*120)
print(f"{'总计':<25} {stage_total_ms:>15.2f} {stage_total_ms/3:>18.2f} {'100.0%':>10}")

# ============= 2. CPU 热点 TOP 30 =============
print("\n" + "="*120)
print("【2】CPU 端最耗时操作 (TOP 30)")
print("="*120)

sorted_cpu = sorted(cpu_ops.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'操作名称':<65} {'调用次数':>10} {'总耗时(ms)':>13} {'平均(μs)':>11} {'占比':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_cpu[:30], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / cpu_total_us * 100) if cpu_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'CPU 总计':<65} {sum(s['count'] for s in cpu_ops.values()):>10,} {cpu_total_ms:>13.2f}")

# ============= 3. GPU Kernel TOP 30 =============
print("\n" + "="*120)
print("【3】GPU Kernel 耗时排行 (TOP 30)")
print("="*120)

sorted_gpu = sorted(gpu_kernels.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'Kernel 名称':<65} {'调用次数':>10} {'总耗时(ms)':>13} {'平均(μs)':>11} {'占比':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_gpu[:30], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / gpu_total_us * 100) if gpu_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'GPU Kernel 总计':<65} {sum(s['count'] for s in gpu_kernels.values()):>10,} {gpu_total_ms:>13.2f}")

# ============= 4. CUDA API 开销 =============
print("\n" + "="*120)
print("【4】CUDA API 调用开销分析")
print("="*120)

sorted_cuda = sorted(cuda_api.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'CUDA API':<65} {'调用次数':>10} {'总耗时(ms)':>13} {'平均(μs)':>11} {'占比':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_cuda[:20], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / cuda_total_us * 100) if cuda_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'CUDA API 总开销':<65} {sum(s['count'] for s in cuda_api.values()):>10,} {cuda_total_ms:>13.2f}")

# 分析同步开销
sync_calls = {name: stats for name, stats in cuda_api.items() if 'sync' in name.lower()}
if sync_calls:
    sync_total_ms = sum(s['total_time'] for s in sync_calls.values()) / 1000
    sync_count = sum(s['count'] for s in sync_calls.values())
    print(f"\n⚠️  同步操作: {sync_count:,} 次调用, 总耗时 {sync_total_ms:.2f} ms ({sync_total_ms/3:.2f} ms/迭代)")

# ============= 5. GPU Kernel 类型分类 =============
print("\n" + "="*120)
print("【5】GPU Kernel 类型分布")
print("="*120)

kernel_categories = {
    'GEMM/矩阵乘法': [],
    'Deformable Conv/Attn': [],
    'Convolution': [],
    'Elementwise': [],
    'Reduction': [],
    'Normalization': [],
    'Memory Ops': [],
    'Attention': [],
    '其他': []
}

for name, stats in gpu_kernels.items():
    time_ms = stats['total_time'] / 1000
    
    if any(x in name for x in ['gemm', 'GEMM', 'matmul', 'mm_', 'xmma']):
        kernel_categories['GEMM/矩阵乘法'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['deformable', 'Deformable', 'ms_deform']):
        kernel_categories['Deformable Conv/Attn'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['conv', 'Conv', 'cudnn_convolution']):
        kernel_categories['Convolution'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['elementwise', 'Elementwise', 'pointwise']):
        kernel_categories['Elementwise'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['reduce', 'Reduce', 'sum_', 'Sum', 'max_', 'min_']):
        kernel_categories['Reduction'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['norm', 'Norm', 'bn_', 'layer_norm']):
        kernel_categories['Normalization'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['copy', 'Copy', 'Memcpy', 'memcpy', 'Memset', 'memset']):
        kernel_categories['Memory Ops'].append((name, time_ms, stats['count']))
    elif any(x in name for x in ['attention', 'Attention', 'softmax', 'Softmax']):
        kernel_categories['Attention'].append((name, time_ms, stats['count']))
    else:
        kernel_categories['其他'].append((name, time_ms, stats['count']))

print(f"\n{'类型':<28} {'Kernel数':>10} {'调用次数':>12} {'总耗时(ms)':>15} {'占GPU时间':>12}")
print("-"*120)
for category, kernels in sorted(kernel_categories.items(), key=lambda x: sum(t for _, t, _ in x[1]), reverse=True):
    total_time = sum(t for _, t, _ in kernels)
    total_count = sum(c for _, _, c in kernels)
    num_kernels = len(kernels)
    pct = (total_time / gpu_total_ms * 100) if gpu_total_ms > 0 else 0
    print(f"{category:<28} {num_kernels:>10} {total_count:>12,} {total_time:>15.2f} {pct:>11.1f}%")
    
    # 显示该类型的 TOP 3
    top3 = sorted(kernels, key=lambda x: x[1], reverse=True)[:3]
    for name, time, count in top3:
        short_name = name[:55] if len(name) > 55 else name
        print(f"  └─ {short_name:<55} {time:>10.2f} ms ({count:,} calls)")

# ============= 6. 性能瓶颈与优化建议 =============
print("\n" + "="*120)
print("【6】性能瓶颈分析与优化建议")
print("="*120)

# 找出最大瓶颈
top_cpu_op = sorted_cpu[0] if sorted_cpu else (None, {'total_time': 0, 'count': 0})
top_gpu_kernel = sorted_gpu[0] if sorted_gpu else (None, {'total_time': 0, 'count': 0})
top_cuda_api = sorted_cuda[0] if sorted_cuda else (None, {'total_time': 0, 'count': 0})

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOP 性能热点                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. CPU 最慢操作:                                                            │
│    {top_cpu_op[0][:72] if top_cpu_op[0] else 'N/A':<72} │
│    耗时: {top_cpu_op[1]['total_time']/1000:.2f} ms, 调用: {top_cpu_op[1]['count']:,} 次{' '*30}│
│                                                                             │
│ 2. GPU 最慢 Kernel:                                                         │
│    {top_gpu_kernel[0][:72] if top_gpu_kernel[0] else 'N/A':<72} │
│    耗时: {top_gpu_kernel[1]['total_time']/1000:.2f} ms, 调用: {top_gpu_kernel[1]['count']:,} 次{' '*30}│
│                                                                             │
│ 3. CUDA API 最大开销:                                                       │
│    {top_cuda_api[0][:72] if top_cuda_api[0] else 'N/A':<72} │
│    耗时: {top_cuda_api[1]['total_time']/1000:.2f} ms, 调用: {top_cuda_api[1]['count']:,} 次{' '*30}│
└─────────────────────────────────────────────────────────────────────────────┘

优化策略（按优先级排序）:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 高优先级 - 内存操作优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   当前: Memory Operations 占用 {stage_times['Memory Operations']/1000/3:.2f} ms/迭代
   
   建议:
   • 检查模型中不必要的 .to() 和 .copy_() 调用
   • 确保所有张量在训练开始前已在正确设备上
   • 使用 inplace 操作（如 relu_、add_）减少内存分配
   • 避免频繁的 CPU-GPU 数据传输
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 高优先级 - GPU 同步优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   当前: Synchronization 占用 {stage_times['Synchronization']/1000/3:.2f} ms/迭代
   
   建议:
   • 使用 torch.cuda.stream() 创建多个 CUDA 流
   • 避免在训练循环中使用 .item()、.cpu() 等同步操作
   • 使用异步数据传输: to(device, non_blocking=True)
   • 减少不必要的 torch.cuda.synchronize() 调用
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 中优先级 - Deformable 操作优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   建议:
   • Deformable Conv/Attn 是自定义 CUDA kernel
   • 确保已使用 FP16/BF16 混合精度训练
   • 考虑优化 CUDA kernel 实现（使用 TensorCore）
   • 评估是否可以减少 deformable 层的数量
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 中优先级 - 提升 GPU 利用率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   当前: GPU 计算时间 {gpu_total_ms/3000:.2f}s, 墙上时间 3.6s
         GPU 利用率 ≈ {gpu_total_ms/3000/3.6*100:.1f}%
   
   建议:
   • 增大 batch size（如果显存允许）
   • 使用梯度累积模拟更大 batch size
   • 启用 torch.backends.cudnn.benchmark = True
   • 检查是否有 CPU-bound 的数据增强操作
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 低优先级 - 数据加载优化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   建议:
   • 增加 dataloader num_workers
   • 启用 pin_memory=True
   • 使用预取: prefetch_factor=2
   • 考虑使用 DALI 等加速库
""")

print("\n" + "="*120)
print("【性能目标】")
print("="*120)
print(f"""
当前状态: 3.6 秒/迭代 (基于稳态性能，第 12-14 次迭代)
优化目标: 2.5-3.0 秒/迭代 (提升 20-30%)

关键路径:
  1. 减少内存操作开销: 节省约 {stage_times['Memory Operations']/1000/3*0.5:.2f}s
  2. 优化同步操作:     节省约 {stage_times['Synchronization']/1000/3*0.5:.2f}s
  3. 提升GPU利用率:    节省约 0.3-0.5s
  ────────────────────────────────────────
  预期总提升:          0.6-1.1 秒/迭代

建议优先处理 TOP 10 CPU 操作和 TOP 10 GPU Kernel！
""")

print("="*120)
print(f"✓ 分析完成！")
print(f"  使用 Perfetto 查看详细时间轴: https://ui.perfetto.dev")
print(f"  上传文件: {trace_file}")
print("="*120)
