_base_ = ['./base_track_map.py']

# Enable FP16 training
# FP16 has wider hardware support (V100, A100, etc.)
# but may require more careful tuning of loss_scale
fp16 = dict(loss_scale='dynamic')

# Note: You can try different loss_scale values if training is unstable:
# fp16 = dict(loss_scale='dynamic')  # Auto-adjust loss scale
# fp16 = dict(loss_scale=1024.)      # Higher scale for more stability
# fp16 = dict(loss_scale=256.)       # Lower scale for faster training
