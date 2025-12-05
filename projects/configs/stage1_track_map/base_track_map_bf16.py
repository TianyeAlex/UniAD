_base_ = ['./base_track_map.py']

# Enable BF16 training
# BF16 (bfloat16) has better numerical stability than FP16
# and is supported on modern GPUs (A100, H100, etc.)
# Note: BF16 does NOT need loss_scale (unlike FP16)
bf16 = dict()

# For FP16 (needs gradient scaling):
# fp16 = dict(loss_scale=512.)
