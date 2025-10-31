import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, num_layers, prompt_length, hidden_size):
        super(FeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.prompt_length = prompt_length
        self.hidden_size = hidden_size

    def forward(self, ent_out_tensor, rel_out_tensor):
        batch_size, L_K, D = ent_out_tensor.shape
        K = self.prompt_length
        L = self.num_layers - 1
        assert L * K == L_K, "Invalid dimensions for ent_out_tensor"

        # Reshape to (batch_size, num_layers, prompt_length, hidden_size)
        ent_out_reshaped = ent_out_tensor.view(batch_size, L, K, D)
        rel_out_reshaped = rel_out_tensor.view(batch_size, L, K, D)

        # Average pooling along the num_layers dimension
        ent_out_pooled = ent_out_reshaped.mean(dim=1)  # (batch_size, prompt_length, hidden_size)
        rel_out_pooled = rel_out_reshaped.mean(dim=1)  # (batch_size, prompt_length, hidden_size)

        # Alternative: max pooling
        # ent_out_pooled, _ = ent_out_reshaped.max(dim=1)
        # rel_out_pooled, _ = rel_out_reshaped.max(dim=1)

        return ent_out_pooled, rel_out_pooled



# class FeatureExtractor(nn.Module):
#     def __init__(self, num_layers, prompt_length, hidden_size):
#         super().__init__()
#         # Initialize key parameters
#         self.num_layers = num_layers
#         self.prompt_length = prompt_length
#         self.hidden_size = hidden_size
#
#         # Attention weight calculation layer
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_size, 128),  # Feature mapping to intermediate dimension
#             nn.Tanh(),  # Non-linear activation function
#             nn.Linear(128, 1)  # Output attention score scalar
#         )
#
#     def forward(self, ent_out_tensor, rel_out_tensor):
#         batch_size, L_K, D = ent_out_tensor.shape
#
#         # Calculate effective number of layers and prompt length
#         L = self.num_layers - 1  # Actual number of layers used (excluding input layer)
#         K = self.prompt_length  # Number of prompt tokens per layer
#
#         # Dimension validation
#         assert L * K == L_K, f"Dimension mismatch: {L}*{K} != {L_K}"
#
#         # Reshape tensor: (batch, L*K, D) -> (batch, L, K, D)
#         ent_out_reshaped = ent_out_tensor.view(batch_size, L, K, D)
#         rel_out_reshaped = rel_out_tensor.view(batch_size, L, K, D)
#
#         # Calculate attention weights (shape: batch, L, K, 1)
#         ent_attn = self.attention(ent_out_reshaped).softmax(dim=1)
#         rel_attn = self.attention(rel_out_reshaped).softmax(dim=1)
#
#         # Weighted sum (aggregate along layer dimension)
#         ent_out_pooled = (ent_out_reshaped * ent_attn).sum(dim=1)  # (batch, K, D)
#         rel_out_pooled = (rel_out_reshaped * rel_attn).sum(dim=1)  # (batch, K, D)
#
#         return ent_out_pooled, rel_out_pooled