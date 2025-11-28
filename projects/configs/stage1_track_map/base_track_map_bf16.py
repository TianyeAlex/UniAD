_base_ = ['./base_track_map.py']

# Enable BF16 training
# BF16 (bfloat16) has better numerical stability than FP16
# and is supported on modern GPUs (A100, H100, etc.)
bf16 = dict(loss_scale=512.)

# Alternatively, for FP16:
# fp16 = dict(loss_scale=512.)
