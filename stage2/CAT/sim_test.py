import torch
import torch.nn as nn
import torch.nn.functional as F

# Additive Attention
class AdditiveAttention(nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.query_vector_dim = query_vector_dim
        self.candidate_vector_dim = candidate_vector_dim
        
        # Linear transformation to project candidate vectors to query vector space
        self.dense = nn.Linear(candidate_vector_dim, query_vector_dim)
        
        # Learnable query vector
        self.attention_query_vector = nn.Parameter(torch.randn(query_vector_dim, 1) * 0.1)

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: Tensor of shape (batch_size, candidate_size, candidate_vector_dim)
        Returns:
            A Tensor of shape (batch_size, candidate_vector_dim)
        """
        # Apply dense layer and tanh activation
        dense_output = torch.tanh(self.dense(candidate_vector))  # Shape: (batch_size, candidate_size, query_vector_dim)

        # Compute the attention scores by multiplying with the query vector
        candidate_weights = torch.matmul(dense_output, self.attention_query_vector).squeeze(-1)  # Shape: (batch_size, candidate_size)

        # Apply softmax to normalize the attention scores
        candidate_weights = F.softmax(candidate_weights, dim=1)  # Shape: (batch_size, candidate_size)

        # Compute the weighted sum of candidate vectors
        candidate_weights = candidate_weights.unsqueeze(1)  # Shape: (batch_size, 1, candidate_size)
        target = torch.bmm(candidate_weights, candidate_vector).squeeze(1)  # Shape: (batch_size, candidate_vector_dim)

        return target

# 测试模型
batch_size = 32
candidate_size = 10
candidate_vector_dim = 128
query_vector_dim = 64

# 模拟输入 (batch_size, candidate_size, candidate_vector_dim)
candidate_vector = torch.randn(batch_size, candidate_size, candidate_vector_dim)

# 初始化Additive Attention
attention = AdditiveAttention(query_vector_dim, candidate_vector_dim)

# 前向传播
output = attention(candidate_vector)

print(output.shape)  # 期待输出形状: (batch_size, candidate_vector_dim)
