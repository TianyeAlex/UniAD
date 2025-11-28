#!/usr/bin/env python3
"""
Comprehensive Training Performance Analysis - For Latest Profiler Data
Analyzes the discrepancy between wall-clock time and profiler statistics
Configuration: wait=10, warmup=1, active=3, repeat=1
Captured iterations: 12th, 13th, 14th (steady-state performance)
"""
import json
from collections import defaultdict
import sys
import os
import glob

# Auto-find the latest trace file
trace_pattern = "projects/work_dirs/stage1_track_map/base_track_map/profiler_logs/plugins/profile/*/*.pt.trace.json"
trace_files = glob.glob(trace_pattern)

if not trace_files:
    print(f"❌ No trace file found, search path: {trace_pattern}")
    sys.exit(1)

# Select the newest file
trace_file = max(trace_files, key=os.path.getmtime)

print("="*120)
print("UniAD Training Performance Deep Analysis (Steady-State Performance)")
print("="*120)
print(f"\nLoading trace file:")
print(f"  {trace_file}")

file_size_mb = os.path.getsize(trace_file) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")

with open(trace_file, 'r') as f:
    trace_data = json.load(f)

trace_events = trace_data.get('traceEvents', [])
print(f"✓ Loaded successfully! Total events: {len(trace_events):,}\n")

# ============= Data Structures =============
cpu_ops = defaultdict(lambda: {'count': 0, 'total_time': 0, 'min_time': float('inf'), 'max_time': 0})
gpu_kernels = defaultdict(lambda: {'count': 0, 'total_time': 0, 'min_time': float('inf'), 'max_time': 0})
cuda_api = defaultdict(lambda: {'count': 0, 'total_time': 0})

# Training stage classification
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

# Timeline analysis (for estimating actual iteration time)
event_timeline = []

print("Analyzing events...")
progress_interval = max(1, len(trace_events) // 20)
for i, event in enumerate(trace_events):
    if i % progress_interval == 0:
        print(f"  Progress: {i:,}/{len(trace_events):,} ({i/len(trace_events)*100:.0f}%)")
    
    if event.get('ph') != 'X':  # Only process complete events
        continue
    
    name = event.get('name', 'unknown')
    dur = event.get('dur', 0)  # microseconds
    cat = event.get('cat', '')
    ts = event.get('ts', 0)  # timestamp
    tid = event.get('tid', 0)
    
    # Record timeline (for subsequent analysis)
    if cat in ['cpu_op', 'kernel', 'cuda_runtime']:
        event_timeline.append({
            'ts': ts,
            'dur': dur,
            'cat': cat,
            'name': name,
            'tid': tid
        })
    
    # CPU operations
    if cat == 'cpu_op':
        cpu_ops[name]['count'] += 1
        cpu_ops[name]['total_time'] += dur
        cpu_ops[name]['min_time'] = min(cpu_ops[name]['min_time'], dur)
        cpu_ops[name]['max_time'] = max(cpu_ops[name]['max_time'], dur)
        
        # Categorize by training stage
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
    
    # CUDA API calls
    elif cat in ['cuda_runtime', 'cuda_driver', 'Runtime']:
        cuda_api[name]['count'] += 1
        cuda_api[name]['total_time'] += dur
        
        if 'sync' in name.lower():
            stage_times['Synchronization'] += dur

print("\n✓ Analysis complete!\n")

# ============= Timeline Analysis =============
print("="*120)
print("【Key Metrics】Training Time Comparison")
print("="*120)

# Calculate total time
cpu_total_us = sum(stats['total_time'] for stats in cpu_ops.values())
gpu_total_us = sum(stats['total_time'] for stats in gpu_kernels.values())
cuda_total_us = sum(stats['total_time'] for stats in cuda_api.values())

cpu_total_ms = cpu_total_us / 1000
gpu_total_ms = gpu_total_us / 1000
cuda_total_ms = cuda_total_us / 1000

# Estimate actual runtime from timeline (wall-clock time)
if event_timeline:
    event_timeline.sort(key=lambda x: x['ts'])
    start_ts = event_timeline[0]['ts']
    end_ts = max(e['ts'] + e['dur'] for e in event_timeline)
    wall_time_us = end_ts - start_ts
    wall_time_ms = wall_time_us / 1000
else:
    wall_time_ms = 0

print(f"""
From training logs: Each iteration takes approximately 3.6 seconds
Profiler config: wait=10, warmup=1, active=3, repeat=1
Therefore captured: 12th, 13th, 14th iterations (total 3 iterations)
                    ^^^^^^^^^^^^^^^^^^^^^^^^
                    Skipped first 10 iterations to avoid cold-start, data closer to steady-state

Expected wall-clock time: 3 × 3.6s = 10.8 seconds

Profiler Statistics Analysis:
┌────────────────────────────────────────────────────────────┐
│ Type              │  Total(s)    │  Avg/Iter   │  Notes     │
├────────────────────────────────────────────────────────────┤
│ CPU Ops Cumulative│  {cpu_total_ms/1000:>10.2f}  │  {cpu_total_ms/3000:>9.2f}  │  All CPU ops│
│ GPU Kernel Time   │  {gpu_total_ms/1000:>10.2f}  │  {gpu_total_ms/3000:>9.2f}  │  GPU compute│
│ CUDA API Overhead │  {cuda_total_ms/1000:>10.2f}  │  {cuda_total_ms/3000:>9.2f}  │  Sync/API  │
│ Timeline Span     │  {wall_time_ms/1000:>10.2f}  │  {wall_time_ms/3000:>9.2f}  │  Estimated │
└────────────────────────────────────────────────────────────┘

Key Findings:
• CPU cumulative time ({cpu_total_ms/1000:.1f}s) >> Actual iteration time (3.6s)
  → Indicates significant CPU-GPU parallelism
  
• Parallel Efficiency = 1 - (Actual / CPU Time)
                      = 1 - (3.6 / {cpu_total_ms/3000:.2f})
                      ≈ {(1 - 3.6/(cpu_total_ms/3000))*100:.1f}%
  → CPU-GPU pipeline already has good overlap

• GPU actual compute time: {gpu_total_ms/3000:.2f}s/iteration
  → GPU utilization = {gpu_total_ms/3000/3.6*100:.1f}% (based on 3.6s wall-clock)
  → Still room for optimization, ideal case ~2-2.5s/iteration
""")

# ============= 1. Training Stage Analysis =============
print("\n" + "="*120)
print("【1】Training Stage Time Distribution")
print("="*120)

stage_total_us = sum(stage_times.values())
stage_total_ms = stage_total_us / 1000
sorted_stages = sorted(stage_times.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Stage':<25} {'Total Time(ms)':>15} {'Avg/Iter(ms)':>18} {'Percent':>10} {'Cumul.':>10}")
print("-"*120)
cumulative = 0
for stage, time_us in sorted_stages:
    time_ms = time_us / 1000
    avg_ms = time_ms / 3
    percentage = (time_us / stage_total_us * 100) if stage_total_us > 0 else 0
    cumulative += percentage
    print(f"{stage:<25} {time_ms:>15.2f} {avg_ms:>18.2f} {percentage:>9.1f}% {cumulative:>9.1f}%")

print("-"*120)
print(f"{'Total':<25} {stage_total_ms:>15.2f} {stage_total_ms/3:>18.2f} {'100.0%':>10}")

# ============= 2. CPU Hotspots TOP 30 =============
print("\n" + "="*120)
print("【2】Most Time-Consuming CPU Operations (TOP 30)")
print("="*120)

sorted_cpu = sorted(cpu_ops.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'Operation Name':<65} {'Call Count':>10} {'Total(ms)':>13} {'Avg(μs)':>11} {'Pct':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_cpu[:30], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / cpu_total_us * 100) if cpu_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'CPU Total':<65} {sum(s['count'] for s in cpu_ops.values()):>10,} {cpu_total_ms:>13.2f}")

# ============= 3. GPU Kernel TOP 30 =============
print("\n" + "="*120)
print("【3】GPU Kernel Time Ranking (TOP 30)")
print("="*120)

sorted_gpu = sorted(gpu_kernels.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'Kernel Name':<65} {'Call Count':>10} {'Total(ms)':>13} {'Avg(μs)':>11} {'Pct':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_gpu[:30], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / gpu_total_us * 100) if gpu_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'GPU Kernel Total':<65} {sum(s['count'] for s in gpu_kernels.values()):>10,} {gpu_total_ms:>13.2f}")

# ============= 4. CUDA API Overhead =============
print("\n" + "="*120)
print("【4】CUDA API Call Overhead Analysis")
print("="*120)

sorted_cuda = sorted(cuda_api.items(), key=lambda x: x[1]['total_time'], reverse=True)

print(f"\n{'CUDA API':<65} {'Call Count':>10} {'Total(ms)':>13} {'Avg(μs)':>11} {'Pct':>8}")
print("-"*120)
for i, (name, stats) in enumerate(sorted_cuda[:20], 1):
    time_ms = stats['total_time'] / 1000
    avg_us = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
    pct = (stats['total_time'] / cuda_total_us * 100) if cuda_total_us > 0 else 0
    short_name = name[:62] if len(name) > 62 else name
    print(f"{i:2d}. {short_name:<62} {stats['count']:>10,} {time_ms:>13.2f} {avg_us:>11.1f} {pct:>7.1f}%")

print(f"\n{'CUDA API Total Overhead':<65} {sum(s['count'] for s in cuda_api.values()):>10,} {cuda_total_ms:>13.2f}")

# Analyze synchronization overhead
sync_calls = {name: stats for name, stats in cuda_api.items() if 'sync' in name.lower()}
if sync_calls:
    sync_total_ms = sum(s['total_time'] for s in sync_calls.values()) / 1000
    sync_count = sum(s['count'] for s in sync_calls.values())
    print(f"\n⚠️  Synchronization operations: {sync_count:,} calls, total {sync_total_ms:.2f} ms ({sync_total_ms/3:.2f} ms/iter)")

# ============= 5. GPU Kernel Category Distribution =============
print("\n" + "="*120)
print("【5】GPU Kernel Type Distribution")
print("="*120)

kernel_categories = {
    'GEMM/Matrix Multiply': [],
    'Deformable Conv/Attn': [],
    'Convolution': [],
    'Elementwise': [],
    'Reduction': [],
    'Normalization': [],
    'Memory Ops': [],
    'Attention': [],
    'Other': []
}

for name, stats in gpu_kernels.items():
    time_ms = stats['total_time'] / 1000
    
    if any(x in name for x in ['gemm', 'GEMM', 'matmul', 'mm_', 'xmma']):
        kernel_categories['GEMM/Matrix Multiply'].append((name, time_ms, stats['count']))
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
        kernel_categories['Other'].append((name, time_ms, stats['count']))

print(f"\n{'Category':<28} {'#Kernels':>10} {'Call Count':>12} {'Total Time(ms)':>15} {'GPU Time %':>12}")
print("-"*120)
for category, kernels in sorted(kernel_categories.items(), key=lambda x: sum(t for _, t, _ in x[1]), reverse=True):
    total_time = sum(t for _, t, _ in kernels)
    total_count = sum(c for _, _, c in kernels)
    num_kernels = len(kernels)
    pct = (total_time / gpu_total_ms * 100) if gpu_total_ms > 0 else 0
    print(f"{category:<28} {num_kernels:>10} {total_count:>12,} {total_time:>15.2f} {pct:>11.1f}%")
    
    # Show TOP 3 of this category
    top3 = sorted(kernels, key=lambda x: x[1], reverse=True)[:3]
    for name, time, count in top3:
        short_name = name[:55] if len(name) > 55 else name
        print(f"  └─ {short_name:<55} {time:>10.2f} ms ({count:,} calls)")

# ============= 6. Performance Bottlenecks & Optimization Recommendations =============
print("\n" + "="*120)
print("【6】Performance Bottleneck Analysis & Optimization Recommendations")
print("="*120)

# Find biggest bottlenecks
top_cpu_op = sorted_cpu[0] if sorted_cpu else (None, {'total_time': 0, 'count': 0})
top_gpu_kernel = sorted_gpu[0] if sorted_gpu else (None, {'total_time': 0, 'count': 0})
top_cuda_api = sorted_cuda[0] if sorted_cuda else (None, {'total_time': 0, 'count': 0})

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOP Performance Hotspots                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Slowest CPU Operation:                                                   │
│    {top_cpu_op[0][:72] if top_cpu_op[0] else 'N/A':<72} │
│    Time: {top_cpu_op[1]['total_time']/1000:.2f} ms, Calls: {top_cpu_op[1]['count']:,}{' '*30}│
│                                                                             │
│ 2. Slowest GPU Kernel:                                                      │
│    {top_gpu_kernel[0][:72] if top_gpu_kernel[0] else 'N/A':<72} │
│    Time: {top_gpu_kernel[1]['total_time']/1000:.2f} ms, Calls: {top_gpu_kernel[1]['count']:,}{' '*30}│
│                                                                             │
│ 3. Highest CUDA API Overhead:                                               │
│    {top_cuda_api[0][:72] if top_cuda_api[0] else 'N/A':<72} │
│    Time: {top_cuda_api[1]['total_time']/1000:.2f} ms, Calls: {top_cuda_api[1]['count']:,}{' '*30}│
└─────────────────────────────────────────────────────────────────────────────┘

Optimization Strategies (Prioritized):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 P0 - Memory Operations Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Current: Memory Operations take {stage_times['Memory Operations']/1000/3:.2f} ms/iteration
   
   Recommendations:
   • Eliminate unnecessary .to() and .copy_() calls in model
   • Ensure all tensors on correct device before training starts
   • Use in-place operations (e.g., relu_, add_) to reduce memory allocation
   • Avoid frequent CPU-GPU data transfers
   • Remove .cpu() calls in training loop (defer to inference time)
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ P0 - GPU Synchronization Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Current: Synchronization takes {stage_times['Synchronization']/1000/3:.2f} ms/iteration
   
   Recommendations:
   • Use torch.cuda.stream() to create multiple CUDA streams
   • Avoid .item(), .cpu() sync operations in training loop
   • Use async data transfer: to(device, non_blocking=True)
   • Reduce unnecessary torch.cuda.synchronize() calls
   • Batch process items instead of one-by-one synchronization
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 P1 - Deformable Operations Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Recommendations:
   • Deformable Conv/Attn uses custom CUDA kernels
   • Ensure FP16/BF16 mixed precision training is enabled
   • Consider optimizing CUDA kernel implementation (use TensorCores)
   • Evaluate if number of deformable layers can be reduced
   • Check if kernels properly utilize GPU memory hierarchy
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 P1 - Improve GPU Utilization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Current: GPU compute time {gpu_total_ms/3000:.2f}s, wall-clock 3.6s
            GPU utilization ≈ {gpu_total_ms/3000/3.6*100:.1f}%
   
   Recommendations:
   • Increase batch size (if memory allows)
   • Use gradient accumulation to simulate larger batch
   • Enable torch.backends.cudnn.benchmark = True
   • Check for CPU-bound data augmentation operations
   • Profile data loading to ensure it's not the bottleneck
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 P2 - Data Loading Optimization
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Recommendations:
   • Increase dataloader num_workers
   • Enable pin_memory=True
   • Use prefetching: prefetch_factor=2
   • Consider using NVIDIA DALI or other acceleration libraries
   • Move data preprocessing to GPU when possible
""")

print("\n" + "="*120)
print("【Performance Target】")
print("="*120)
print(f"""
Current State: 3.6 seconds/iteration (steady-state, iterations 12-14)
Target:        2.5-3.0 seconds/iteration (20-30% improvement)

Critical Path Optimizations:
  1. Reduce memory operation overhead:  Save ~{stage_times['Memory Operations']/1000/3*0.5:.2f}s
  2. Optimize synchronization:          Save ~{stage_times['Synchronization']/1000/3*0.5:.2f}s
  3. Improve GPU utilization:           Save ~0.3-0.5s
  ────────────────────────────────────────────────────────
  Expected Total Improvement:           0.6-1.1 s/iteration

Recommended to prioritize TOP 10 CPU operations and TOP 10 GPU kernels!
""")

print("="*120)
print(f"✓ Analysis complete!")
print(f"  View detailed timeline with Perfetto: https://ui.perfetto.dev")
print(f"  Upload file: {trace_file}")
print("="*120)
