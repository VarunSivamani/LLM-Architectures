import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import RMSNorm as FusedRMSNorm 

"""
          Output
            │
       ┌────────┐
       │ Linear │
       └────────┘
            │
       ┌────────┐        ┌──────┐
       │ Linear │        │ SiLU |
       └────────┘        └──────┘
             │              │
             └─────x────────┘
                   │
                   |
             ┌──────────┐
             │  Linear  │
             └──────────┘
                   │
                 Input
"""

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc2 = nn.Linear(config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False)
        self.fc3 = nn.Linear(config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x) * self.fc2(x)
        return self.fc3(x)
    
"""
               Output (y)
                   │
       ┌────────────────────────┐
       │ Unscale:               │
       │    x_norm = y / γ      │  ← inverse of scaling
       └────────────────────────┘
                   │
       ┌────────────────────────┐
       │ Multiply by RMS        │
       │     x = x_norm * rms   │  ← reapply RMS
       └────────────────────────┘
                   │
              Reconstructed x

"""

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
    
"""
                Input (q or k)
                     │
          ┌─────────────────────────┐
          │ Split into pairs        │  ← [x₀, x₁, x₂, x₃, ..., xₙ₋₂, xₙ₋₁]
          └─────────────────────────┘
                     │
          ┌─────────────────────────┐
          │ Apply rotation matrix:  │
          │  [x₀', x₁'] = RoPE(x₀, x₁, θ)  ← uses sin/cos
          └─────────────────────────┘
                     │
                Output (rotated q/k)
"""

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):

    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # compute inverse frequency
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float() / head_dim))

    # generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # expand the angles to match the head dim
    angles = torch.cat([angles, angles], dim=1) # Shape: (context_length, head_dim)

    # pre-compute sine and cosine angles
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):

    # x: (batch_size, n_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # split x into first half, second half
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2: ]

    # adjust sine and cosine shapes
    # Shape: (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # aply the rotary transformations
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # return the rotated matrix
    return x_rotated.to(dtype=x.dtype)

"""
             Input (hidden states)
                     │
      ┌────────────────────────────────┐
      │ Project into:                  │
      │  - Hq query heads (Wq)         │
      │  - Hkv key heads   (Wk shared) │
      │  - Hkv value heads (Wv shared) │
      └────────────────────────────────┘
                     │
      ┌────────────────────────────────┐
      │ Attention:                     │
      │  Each query head attends over  │
      │  its corresponding key/value   │
      │  group (from Hkv)              │
      └────────────────────────────────┘
                     │
      ┌────────────────────────────────┐
      │ Concatenate attended vectors   │
      │ and apply final linear layer   │
      └────────────────────────────────┘
                     │
                 Output

"""

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()

        assert num_heads % num_kv_groups == 0, "num_head must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            
            assert d_in % num_heads == 0, "input dimensions must be divisible by num_heads"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):

        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        # (batch_size, num_heads, seq_len, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Flash Attention: fast and memory-efficient
        context = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=mask,  # This can be None or a boolean mask
            dropout_p=0.0,   # or nonzero if training
            is_causal=False  # or True, depending on your attention type
        )
        return self.out_proj(context)
    
"""
                    Input: x
                       │
              ┌─────────────────┐
              │   RMSNorm (1)   │
              └─────────────────┘
                       │
              ┌─────────────────┐
              │ GroupedQueryAtt │ ◄── mask, cos, sin
              └─────────────────┘
                       │
              ┌─────────────────┐
              │ Residual Add    │  ← adds original input (shortcut)
              └─────────────────┘
                       │
              ┌─────────────────┐
              │   RMSNorm (2)   │
              └─────────────────┘
                       │
              ┌─────────────────┐
              │ FeedForward (FF)│
              └─────────────────┘
                       │
              ┌─────────────────┐
              │ Residual Add    │  ← adds original input (shortcut)
              └─────────────────┘
                       │
                    Output: x
"""

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=config["emb_dim"],
            num_heads=config["n_heads"],
            head_dim=config["head_dim"],
            num_kv_groups=config["n_kv_groups"],
            qk_norm=config["qk_norm"],
            dtype=config["dtype"]
        )
        self.ff = FeedForward(config)
        self.norm1 = FusedRMSNorm(config["emb_dim"], eps=1e-6)
        self.norm2 = FusedRMSNorm(config["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x
    
class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"], dtype=config["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.final_norm = FusedRMSNorm(config["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False, dtype=config["dtype"])

        # implementing weight sharing
        self.out_head.weight = self.tok_emb.weight 

        # Reusuable utilities
        if config["head_dim"] is None:
            head_dim = config["emb_dim"] // config["n_heads"]
        else:
            head_dim = config["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=config["rope_base"],
            context_length=config["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.config = config


    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.config["dtype"]))
        return logits
    
# Qwen3 0.6B

QWEN3_CONFIG = {
    "vocab_size": 151_936,           # Vocabulary size
    "context_length": 40_960,        # Context length that was used to train the model
    "emb_dim": 1024,                 # Embedding dimension
    "n_heads": 16,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
    "head_dim": 128,                 # Size of the heads in GQA
    "qk_norm": True,                 # Whether to normalize queries and values in GQA
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

torch.manual_seed(123)
model = Qwen3Model(QWEN3_CONFIG)

print('=' * 50)
print(model)
print('=' * 50)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Account for weight tying
total_params_normalized = total_params - model.tok_emb.weight.numel()
print(f"\nTotal number of unique parameters: {total_params_normalized:,}")
print('=' * 50)

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
print('=' * 50)