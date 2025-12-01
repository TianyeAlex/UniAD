"""FlashAttention wrapper for replacing nn.MultiheadAttention in UniAD."""
import warnings
import torch
import torch.nn as nn
from typing import Optional
from mmcv.cnn.bricks.registry import ATTENTION

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("FlashAttention not available, using standard MultiheadAttention")


@ATTENTION.register_module(name='FlashMultiheadAttention')
class FlashMultiheadAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention using FlashAttention.
    
    This module provides the same interface as torch.nn.MultiheadAttention but
    uses FlashAttention for better performance when possible.
    
    Args:
        embed_dim (int): Total dimension of the model (also accepts embed_dims for mmcv compatibility).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability on attn_output_weights. Default: 0.0
        bias (bool): If specified, adds bias to input/output projection layers. Default: True
        add_bias_kv (bool): If specified, adds bias to the key and value sequences. Default: False
        add_zero_attn (bool): If specified, adds a new batch of zeros to key and value. Default: False
        kdim (int): Total number of features for keys. Default: None (uses embed_dim)
        vdim (int): Total number of features for values. Default: None (uses embed_dim)
        batch_first (bool): If True, input/output tensors are (batch, seq, feature). Default: False
    """
    
    def __init__(
        self,
        embed_dim=None,
        num_heads=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        # mmcv-style parameter names
        embed_dims=None,
        attn_drop=None,
        proj_drop=None,
        **kwargs  # Catch any other unused parameters
    ):
        super().__init__()
        
        # Support both embed_dim (PyTorch) and embed_dims (mmcv) parameter names
        if embed_dims is not None:
            embed_dim = embed_dims
        if embed_dim is None:
            raise ValueError("Must provide either embed_dim or embed_dims")
        
        # Support both dropout and attn_drop parameter names
        if attn_drop is not None:
            dropout = attn_drop
        
        self.embed_dim = embed_dim
        self.embed_dims = embed_dim  # Add embed_dims attribute for mmcv compatibility
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Use FlashAttention if available, otherwise fallback
        self.use_flash_attn = FLASH_ATTN_AVAILABLE and (kdim is None) and (vdim is None) and (not add_bias_kv) and (not add_zero_attn)
        
        # Always create fallback attention for cross-attention and masked attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn,
            kdim, vdim, batch_first, device=device, dtype=dtype
        )
        
        if self.use_flash_attn:
            # FlashAttention path - we need Q, K, V projections for self-attention
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
            
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        identity: torch.Tensor = None,  # mmcv-style residual connection
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        # mmcv-style parameters
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        **kwargs  # Catch any other unused parameters
    ):
        """Forward pass.
        
        Args:
            query: Query embeddings of shape (L, N, E) or (N, L, E) if batch_first
            key: Key embeddings of shape (S, N, E) or (N, S, E) if batch_first
            value: Value embeddings of shape (S, N, E) or (N, S, E) if batch_first
            identity: Identity tensor for residual connection (mmcv compatibility)
            key_padding_mask: Mask of shape (N, S) where True positions are ignored
            need_weights: If True, return attention weights
            attn_mask: Attention mask of shape (L, S) or (N*num_heads, L, S)
            average_attn_weights: If True, average attention weights over heads
            query_pos: Positional encoding for query (mmcv compatibility)
            key_pos: Positional encoding for key (mmcv compatibility)
            
        Returns:
            attn_output: Output of shape (L, N, E) or (N, L, E) if batch_first
            attn_weights: Attention weights (None if use_flash_attn and not need_weights)
        """
        # Handle mmcv-style positional encoding
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            if key is None:
                key = query
            key = key + key_pos
        
        # Handle default values for key and value (for self-attention)
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Use FlashAttention for self-attention without masks
        if self.use_flash_attn and attn_mask is None and key_padding_mask is None:
            # Check if this is self-attention
            is_self_attn = (query is key) and (key is value)
            
            if is_self_attn:
                # Convert to batch_first if needed
                if not self.batch_first:
                    query = query.transpose(0, 1)  # (L, N, E) -> (N, L, E)
                
                batch_size, seq_len, _ = query.shape
                
                # FlashAttention only supports fp16 and bf16
                input_dtype = query.dtype
                target_dtype = None
                
                if input_dtype not in [torch.float16, torch.bfloat16]:
                    # Prefer bf16 if available (better numerical stability), otherwise use fp16
                    if torch.cuda.is_bf16_supported():
                        target_dtype = torch.bfloat16
                    else:
                        target_dtype = torch.float16
                    query = query.to(target_dtype)
                else:
                    target_dtype = input_dtype
                
                # Always convert weights to match target dtype
                if self.qkv_proj.weight.dtype != target_dtype:
                    self.qkv_proj = self.qkv_proj.to(target_dtype)
                    self.out_proj = self.out_proj.to(target_dtype)
                
                # Project to Q, K, V
                qkv = self.qkv_proj(query)
                qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
                
                # FlashAttention expects (batch, seqlen, 3, nheads, headdim)
                out = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=False,
                    softmax_scale=None,  # Will use default 1/sqrt(d)
                )
                
                # Reshape and project output
                out = out.reshape(batch_size, seq_len, self.embed_dim)
                out = self.out_proj(out)
                
                # Convert back to original dtype if needed
                if input_dtype not in [torch.float16, torch.bfloat16]:
                    out = out.to(input_dtype)
                
                # Convert back to original format
                if not self.batch_first:
                    out = out.transpose(0, 1)  # (N, L, E) -> (L, N, E)
                
                # Return only output by default (mmcv compatibility)
                # FlashAttention doesn't return weights anyway
                return out if not need_weights else (out, None)
        
        # Fallback to standard attention for cross-attention or with masks
        result = self.attn(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )
        
        # Return format: output only (mmcv style) or (output, weights) (PyTorch style)
        if need_weights:
            return result  # (output, weights) tuple
        else:
            return result[0] if isinstance(result, tuple) else result
