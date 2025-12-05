# Copyright (c) OpenMMLab. All rights reserved.
"""
BFloat16 Optimizer Hook for mixed precision training.

This hook maintains FP32 optimizer states while using BF16 for model parameters,
ensuring numerical stability during training.

Key differences from using plain OptimizerHook with BF16 model:
- OptimizerHook: Optimizer states are BF16 (precision loss in momentum/variance)
- Bf16OptimizerHook: Optimizer states are FP32 (numerically stable)

Key differences from Fp16OptimizerHook:
- Fp16OptimizerHook: Uses GradScaler (not compatible with BF16)
- Bf16OptimizerHook: No GradScaler (BF16 has same dynamic range as FP32)
"""

import copy
from collections import defaultdict
from itertools import chain
from typing import Optional

import torch
import torch.nn as nn
from mmcv.runner import HOOKS, OptimizerHook
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.utils import _BatchNorm


@HOOKS.register_module()
class Bf16OptimizerHook(OptimizerHook):
    """BFloat16 optimizer hook for mixed precision training.
    
    The BF16 training workflow:
    1. Model parameters are stored in BF16 (saves memory)
    2. Forward/backward passes use BF16 (faster computation)
    3. Gradients are computed in BF16
    4. Gradients are copied to FP32 optimizer parameters
    5. Optimizer states (momentum, variance) are maintained in FP32
    6. Parameters are updated in FP32
    7. Updated FP32 parameters are copied back to BF16 model
    
    This ensures numerical stability while benefiting from BF16 speed and memory savings.
    
    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        coalesce (bool): Whether allreduce parameters as a whole.
            Default: True.
        bucket_size_mb (int): Size of bucket, the unit is MB. Default: -1.
        distributed (bool): Whether to use distributed training. Default: True.
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 coalesce: bool = True,
                 bucket_size_mb: int = -1,
                 distributed: bool = True):
        super().__init__(grad_clip=grad_clip)
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed

    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training.
        
        1. Make a master copy of FP32 weights for optimization.
        2. Convert the main model from FP32 to BF16.
        """
        # Keep a copy of FP32 weights for the optimizer
        old_groups = runner.optimizer.param_groups
        runner.optimizer.param_groups = copy.deepcopy(
            runner.optimizer.param_groups)
        
        # Map old parameters to new parameters
        state = defaultdict(dict)
        p_map = {
            old_p: p
            for old_p, p in zip(
                chain(*(g['params'] for g in old_groups)),
                chain(*(g['params'] for g in runner.optimizer.param_groups)))
        }
        for k, v in runner.optimizer.state.items():
            state[p_map[k]] = v
        runner.optimizer.state = state
        
        # Convert model to BF16
        wrap_fp16_model(runner.model)
        runner.logger.info('BF16 training: Model converted to BF16, '
                          'optimizer states kept in FP32')

    def copy_grads_to_fp32(self, bf16_net: nn.Module, fp32_weights) -> None:
        """Copy gradients from BF16 model to FP32 weight copy."""
        for fp32_param, bf16_param in zip(fp32_weights,
                                          bf16_net.parameters()):
            if bf16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                # Convert BF16 gradients to FP32
                fp32_param.grad.copy_(bf16_param.grad.float())

    def copy_params_to_bf16(self, bf16_net: nn.Module, fp32_weights) -> None:
        """Copy updated params from FP32 weight copy to BF16 model."""
        for bf16_param, fp32_param in zip(bf16_net.parameters(),
                                          fp32_weights):
            # Convert FP32 parameters back to BF16
            bf16_param.data.copy_(fp32_param.data.bfloat16())

    def after_train_iter(self, runner) -> None:
        """Backward optimization steps for BF16 Mixed Precision Training.
        
        1. Backward the loss to obtain the gradients (BF16).
        2. Copy gradients from the model to the FP32 weight copy.
        3. Allreduce FP32 gradients (if distributed).
        4. Clip FP32 gradients (if configured).
        5. Update the FP32 weight copy using optimizer.
        6. Copy back the params from FP32 weight copy to the BF16 model.
        
        Note: Unlike FP16, BF16 does NOT need gradient scaling because it has
        the same dynamic range as FP32 (no underflow issues).
        """
        # Clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        
        # Backward pass (gradients computed in BF16)
        runner.outputs['loss'].backward()
        
        # Get FP32 parameters from optimizer
        fp32_weights = []
        for param_group in runner.optimizer.param_groups:
            fp32_weights += param_group['params']
        
        # Copy BF16 grads from model to FP32 params in optimizer
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        
        # Allreduce grads (on FP32 gradients for numerical stability)
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce,
                           self.bucket_size_mb)
        
        # Clip gradients (on FP32 for better precision)
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(fp32_weights)
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(
                    {'grad_norm': float(grad_norm)},
                    runner.outputs['num_samples'])
        
        # Update FP32 params (optimizer states in FP32)
        runner.optimizer.step()
        
        # Copy FP32 params back to the BF16 model
        self.copy_params_to_bf16(runner.model, fp32_weights)
