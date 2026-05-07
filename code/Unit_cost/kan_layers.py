import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EfficientKANLayer(nn.Module):
    """
    Efficient B-Spline KAN Layer for Graph Message Passing.
    Reformulates B-spline basis evaluation to avoid OOM errors on large graphs.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(EfficientKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1).contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the base linear weights using standard Kaiming uniform
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            # MATCH THE DIMENSIONS: We need exactly (grid_size + spline_order) weights
            num_spline_params = self.grid_size + self.spline_order
            
            # Generate scaled noise for the B-spline control points
            noise = (
                torch.rand(num_spline_params, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            
            # Permute the noise to match the shape of self.spline_weight: 
            # (out_features, in_features, grid_size + spline_order)
            self.spline_weight.data.copy_(noise.permute(2, 1, 0).contiguous())

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                + (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        # Flatten graph feature tensor if needed
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)
            
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        out = base_output + spline_output
        
        if len(original_shape) > 2:
            out = out.view(*original_shape[:-1], self.out_features)
        return out
    
class MultiHeadKANAttention(nn.Module):
    """
    Theoretical Foundation: Transformer-style Multi-Head Attention where Q, K, V projections 
    are parameterized by non-linear Kolmogorov-Arnold Networks (B-Splines) rather than linear layers.
    This enables mapping node embeddings onto non-convex manifolds for superior similarity scaling.
    """
    def __init__(self, embed_dim, num_heads, grid_size=5, spline_order=3):
        super(MultiHeadKANAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # KAN-based mapping for Q, K, V spaces
        self.kan_q = EfficientKANLayer(embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order)
        self.kan_k = EfficientKANLayer(embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order)
        self.kan_v = EfficientKANLayer(embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order)
        
        # KAN-based Output projection for the concatenated attention heads
        self.kan_o = EfficientKANLayer(embed_dim, embed_dim, grid_size=grid_size, spline_order=spline_order)

    def forward(self, query, key, value):
        # Expected shape: [Batch_Size (Num Nodes), Sequence_Length (Num Layers), Features]
        batch_size = query.size(0)

        # 1. Non-linear B-Spline Projections
        Q = self.kan_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.kan_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.kan_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: Softmax(QK^T / sqrt(d)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)

        # 3. Contextual Feature Extraction
        context = torch.matmul(attention_weights, V)
        
        # 4. Concatenation and Final Non-linear Projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.kan_o(context)
        
        return output, attention_weights