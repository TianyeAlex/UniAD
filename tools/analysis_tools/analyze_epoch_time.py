#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Analyze epoch wall-clock time from training logs.
Calculate total time for each epoch based on timestamps.
"""
import argparse
import re
import json
from datetime import datetime
from collections import defaultdict


def parse_log_file(log_path):
    """Parse log file and extract epoch timestamps."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    workflow_time = None
    checkpoint_times = {}
    max_epochs = 0
    
    for line in lines:
        # Match workflow start
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*workflow:.*max: (\d+) epochs', line)
        if match:
            workflow_time = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            max_epochs = int(match.group(2))
            continue
        
        # Match checkpoint saving
        match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Saving checkpoint at (\d+) epochs', line)
        if match:
            timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            epoch = int(match.group(2))
            checkpoint_times[epoch] = timestamp
    
    return workflow_time, checkpoint_times, max_epochs


def parse_json_log(json_log_path):
    """Parse JSON log to extract time and data_time per epoch."""
    epoch_stats = defaultdict(lambda: {'time': [], 'data_time': []})
    
    try:
        with open(json_log_path, 'r') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    if 'epoch' not in log:
                        continue
                    
                    epoch = log['epoch']
                    if 'time' in log:
                        epoch_stats[epoch]['time'].append(log['time'])
                    if 'data_time' in log:
                        epoch_stats[epoch]['data_time'].append(log['data_time'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None
    
    # Calculate averages (excluding first iter as warmup)
    epoch_averages = {}
    for epoch, stats in epoch_stats.items():
        time_values = stats['time'][1:] if len(stats['time']) > 1 else stats['time']
        data_time_values = stats['data_time'][1:] if len(stats['data_time']) > 1 else stats['data_time']
        
        epoch_averages[epoch] = {
            'avg_time': sum(time_values) / len(time_values) if time_values else 0,
            'avg_data_time': sum(data_time_values) / len(data_time_values) if data_time_values else 0,
            'num_iters': len(stats['time'])
        }
    
    return epoch_averages


def calculate_epoch_times(workflow_time, checkpoint_times, max_epochs, epoch_averages=None):
    """Calculate wall-clock time for each epoch."""
    if workflow_time is None:
        print("Error: Could not find workflow start time")
        return
    
    print(f'\n{"="*100}')
    print(f'Epoch Wall-Clock Time Analysis')
    print(f'{"="*100}')
    
    if epoch_averages:
        print(f'\n{"Epoch":<8} {"Start Time":<22} {"End Time":<22} {"Duration":<12} {"Avg Iter":<12} {"Avg Data":<12}')
        print('-' * 100)
    else:
        print(f'\n{"Epoch":<8} {"Start Time":<22} {"End Time":<22} {"Duration":<12}')
        print('-' * 100)
    
    total_time = 0
    prev_time = workflow_time
    
    for epoch in range(1, max_epochs + 1):
        if epoch not in checkpoint_times:
            if epoch_averages:
                print(f'{epoch:<8} {"N/A":<22} {"N/A":<22} {"Incomplete":<12} {"N/A":<12} {"N/A":<12}')
            else:
                print(f'{epoch:<8} {"N/A":<22} {"N/A":<22} {"Incomplete":<12}')
            continue
        
        end_time = checkpoint_times[epoch]
        duration = (end_time - prev_time).total_seconds()
        total_time += duration
        
        base_str = (f'{epoch:<8} {prev_time.strftime("%Y-%m-%d %H:%M:%S"):<22} '
                   f'{end_time.strftime("%Y-%m-%d %H:%M:%S"):<22} '
                   f'{duration/60:.2f} min')
        
        if epoch_averages and epoch in epoch_averages:
            stats = epoch_averages[epoch]
            base_str += (f'    {stats["avg_time"]:.4f}s'
                        f'    {stats["avg_data_time"]:.4f}s')
        
        print(base_str)
        prev_time = end_time
    
    print('=' * 100)
    print(f'\nSummary:')
    print(f'  Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)')
    print(f'  Average time per epoch: {total_time/max_epochs/60:.2f} minutes')
    print(f'  Completed epochs: {len(checkpoint_times)}/{max_epochs}')
    
    if epoch_averages:
        all_times = [stats['avg_time'] for stats in epoch_averages.values()]
        all_data_times = [stats['avg_data_time'] for stats in epoch_averages.values()]
        if all_times:
            avg_iter_time = sum(all_times) / len(all_times)
            avg_data_time = sum(all_data_times) / len(all_data_times)
            print(f'  Overall avg iter time: {avg_iter_time:.4f} s/iter')
            print(f'  Overall avg data time: {avg_data_time:.4f} s/iter')
            print(f'  Compute time (iter - data): {avg_iter_time - avg_data_time:.4f} s/iter')
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze epoch wall-clock time from training logs')
    parser.add_argument('log_file', type=str, help='Path to training log file (.log format)')
    parser.add_argument('--json-log', type=str, default=None, 
                       help='Path to JSON log file for iter time stats (optional, auto-detected if not provided)')
    args = parser.parse_args()
    
    workflow_time, checkpoint_times, max_epochs = parse_log_file(args.log_file)
    
    # Try to find JSON log automatically if not provided
    json_log_path = args.json_log
    if json_log_path is None:
        # Try .log.json extension
        json_log_path = args.log_file + '.json'
    
    # Parse JSON log for iter time statistics
    epoch_averages = parse_json_log(json_log_path)
    if epoch_averages is None:
        print(f"Note: JSON log not found at {json_log_path}, skipping iter time stats")
    
    calculate_epoch_times(workflow_time, checkpoint_times, max_epochs, epoch_averages)


if __name__ == '__main__':
    main()
