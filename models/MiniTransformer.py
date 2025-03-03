import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embeddings import SinusoidalPositionalEmbedding

###############################################################################
# 1) BitEmbedding:  from {0,1} -> R^(d_model)
###############################################################################
class BitEmbedding(nn.Module):
    """
    Learns an embedding for each possible bit (0 or 1).
    We store 2 embeddings in an nn.Embedding(2, d_model).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # We have 2 possible bit values: 0 or 1
        self.embed = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

    def forward(self, x_bits: torch.Tensor) -> torch.Tensor:
        """
        x_bits: shape (batch_size, seq_len), each entry is 0 or 1.
        returns: shape (batch_size, seq_len, d_model).
        """
        return self.embed(x_bits)

###############################################################################
# 2) TimestepMLP:  from scalar t -> R^(d_model)
###############################################################################
class TimestepMLP(nn.Module):
    """
    A small MLP that maps a scalar time t in R to a d_model-dimensional embedding.
    This is more flexible than an embedding table if t can be large or continuous.
    """
    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (batch_size,) of scalar timesteps (float or int).
           We'll reshape to (batch_size, 1) and pass through the MLP.
        returns: shape (batch_size, d_model).
        """
        if t.dim() == 1:
            # shape => (batch_size, 1)
            t = t.unsqueeze(-1).float()
        return self.net(t)

###############################################################################
# 3) A minimal Transformer encoder layer (standard architecture)
###############################################################################
class TransformerEncoderLayer(nn.Module):
    """
    One layer of multi-head self-attention + feedforward network
    with residual connections + LayerNorm, akin to "TransformerEncoderLayer".

    We set `batch_first=True` so the input shape is (batch, seq_len, d_model).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, seq_len, d_model)
        Returns the same shape.
        """
        # 1) Self-attention
        x_normed = self.norm1(x)
        attn_out, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = x + self.dropout1(attn_out)

        # 2) Feedforward
        x_normed = self.norm2(x)
        ff_out = self.linear2(self.dropout(F.silu(self.linear1(x_normed))))
        x = x + self.dropout2(ff_out)

        return x

###############################################################################
# 4) The main Transformer-based model for bit vectors w/ time conditioning
###############################################################################
class TransformerForBits(nn.Module):
    """
    Processes a batch of bit sequences (batch_size, d) plus integer timesteps,
    via a stack of TransformerEncoderLayers.  Output is either (batch_size, d, 2)
    for classification of each bit, or any other shape you prefer.

    Key steps:
      - Use BitEmbedding to get (batch_size, d, d_model).
      - Use TimestepEmbedding to get (batch_size, d_model), then add it in.
      - Pass through N Transformer layers.
      - Final linear -> (batch_size, d, out_dim).

    If you want to incorporate continuous timesteps, you could:
      - replace TimestepEmbedding w/ an MLP or sinusoidal embed function.
      - e.g. a small MLP that maps t in R to R^(d_model).

    If you don't need time t at all, you can omit that part.
    """
    def __init__(
        self,
        bit_dim: int,        # length of the sequence
        time_hidden_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        device: torch.device = 'cpu'
    ):
        super().__init__()

        self.bit_dim = bit_dim
        self.d_model = d_model

        # 4.1) Bit embedding
        self.bit_embed = BitEmbedding(d_model)

        # Create the sinusoidal embedding for positions [0..seq_len-1]
        self.pos_embed = SinusoidalPositionalEmbedding(
            max_time_steps=bit_dim,
            embedding_size=d_model,
            device=device
        )

        # 4.2) Timestep embedding
        self.time_embed = TimestepMLP(d_model, time_hidden_dim)

        # 4.3) Stack of TransformerEncoderLayers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) 
            for _ in range(num_layers)
        ])
        
        self.time_embed_layers = nn.ModuleList([nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model)) for _ in range(num_layers)])
        
        # 4.4) Final output projection: (d_model*seq_len) -> (seq_len) at each bit position
        self.last_layer = nn.Linear(d_model*bit_dim, bit_dim)

    def forward(
        self, 
        x_bits: torch.Tensor,     # (batch_size, seq_len) of 0/1
        t: torch.Tensor           # (batch_size,) of integer timesteps in [0, max_time-1]
    ) -> torch.Tensor:
        """
        Returns: shape (batch_size, seq_len, out_dim)
                 e.g. out_dim=2 => logits for (bit=0, bit=1).
        """
        batch_size, seq_len = x_bits.size()
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x_emb = self.bit_embed(x_bits.int())
        
        positions = torch.arange(seq_len, device=x_bits.device).unsqueeze(0)
        # positions shape => (1, seq_len)
        # Expand to (batch_size, seq_len)
        positions = positions.expand(batch_size, seq_len)
        
        pos_emb = self.pos_embed(positions)
        x_emb = x_emb + pos_emb

        # Timestep embedding => shape (batch_size, d_model)
        t_emb = self.time_embed(t)  # (batch_size, d_model)

        # Pass through multiple Transformer layers
        h = x_emb
        for layer, layer_time_emb in zip(self.layers, self.time_embed_layers):
            # Add time embedding to each bit token
            # We'll unsqueeze(1) to broadcast across seq_len
            # => shape (batch_size, 1, d_model) => add to (batch_size, seq_len, d_model)
            t_emb_layer = layer_time_emb(t_emb).unsqueeze(1)
            h = h + t_emb_layer
            h = layer(h)  # shape remains (batch_size, seq_len, d_model)

        # Final projection
        # expand h to (batch_size, seq_len*d_model)
        h = h.view(batch_size, -1)
        logits = self.last_layer(h)  # (batch_size, seq_len)
        # # # squeeze last layer to get (batch_size, seq_len)
        # alpha = torch.exp(-2 * (t)).unsqueeze(1).expand_as(logits)
        # kappa = (1 - alpha) / 2
        return torch.sigmoid(logits)