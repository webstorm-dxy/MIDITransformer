import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import math

class MIDIEmbedding(nn.Module):
    def __init__(self, pitch_range:int = 128, embed_dim:int = 256):
        super(MIDIEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.pitch_embedding_dim = embed_dim // 3
        self.duration_embedding_dim = embed_dim // 3
        self.velocity_embedding_dim = embed_dim - self.pitch_embedding_dim - self.duration_embedding_dim  # 处理不能整除的情况
        
        self.pit_embedding = nn.Embedding(
            num_embeddings=pitch_range,
            embedding_dim=self.pitch_embedding_dim
        )

        self.dur_embedding = nn.Sequential(
            nn.Linear(1, self.duration_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.duration_embedding_dim, self.duration_embedding_dim)
        )

        self.vel_embbeding = nn.Sequential(
            nn.Linear(1, self.velocity_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.velocity_embedding_dim, self.velocity_embedding_dim)
        )
        
        # 初始化位置编码
        self.register_buffer('positional_encoding', self.position_encoding(embed_dim, 2048))

    def position_encoding(self, d_model: int, max_len: int):
        """创建音乐感知的位置编码"""
        position = torch.arange(max_len).unsqueeze(1)
        # 使用更安全的计算方式避免数值问题
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_pit: torch.Tensor, input_dur: torch.Tensor, input_vel: torch.Tensor):
        # input_pit: (batch_size, seq_len)
        # input_dur: (batch_size, seq_len, 1)
        # input_vel: (batch_size, seq_len, 1)
        
        # 检查输入数据中是否有NaN或inf值
        if torch.isnan(input_pit).any() or torch.isnan(input_dur).any() or torch.isnan(input_vel).any():
            print("Warning: NaN values found in embedding input")
            # 替换NaN值为0
            input_pit = torch.nan_to_num(input_pit, nan=0.0)
            input_dur = torch.nan_to_num(input_dur, nan=0.0)
            input_vel = torch.nan_to_num(input_vel, nan=0.0)
        
        if torch.isinf(input_pit).any() or torch.isinf(input_dur).any() or torch.isinf(input_vel).any():
            print("Warning: Inf values found in embedding input")
            # 替换inf值为合理的数值
            input_pit = torch.clamp(input_pit, min=-1e4, max=1e4)
            input_dur = torch.clamp(input_dur, min=-1e4, max=1e4)
            input_vel = torch.clamp(input_vel, min=-1e4, max=1e4)
        
        # 确保输入在合理范围内
        # 音高应该在[0, pitch_range-1]范围内
        input_pit = torch.clamp(input_pit, min=0, max=127).long()
        
        # 持续时间和力度应该为正数，添加一个小的epsilon以避免0值
        epsilon = 1e-8
        input_dur = torch.clamp(input_dur, min=epsilon, max=1e4)
        input_vel = torch.clamp(input_vel, min=epsilon, max=127)
        
        # 检查嵌入层参数是否有问题
        if torch.isnan(self.pit_embedding.weight).any():
            print("Warning: NaN values found in pitch embedding weights")
            with torch.no_grad():
                self.pit_embedding.weight.copy_(torch.nan_to_num(self.pit_embedding.weight, nan=0.0))
        
        pitch_embedding = self.pit_embedding(input_pit)  # (batch_size, seq_len, pitch_embedding_dim)
        
        # 检查音高嵌入输出
        if torch.isnan(pitch_embedding).any() or torch.isinf(pitch_embedding).any():
            print("Warning: NaN or Inf values found in pitch embedding output")
            pitch_embedding = torch.nan_to_num(pitch_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        duration_embedding = self.dur_embedding(input_dur)  # (batch_size, seq_len, duration_embedding_dim)
        
        # 检查持续时间嵌入输出
        if torch.isnan(duration_embedding).any() or torch.isinf(duration_embedding).any():
            print("Warning: NaN or Inf values found in duration embedding output")
            duration_embedding = torch.nan_to_num(duration_embedding, nan=0.0, posinf=1e4, neginf=-1e4)
        
        velocity_embedding = self.vel_embbeding(input_vel)  # (batch_size, seq_len, velocity_embedding_dim)
        
        # 检查力度嵌入输出
        if torch.isnan(velocity_embedding).any() or torch.isinf(velocity_embedding).any():
            print("Warning: NaN or Inf values found in velocity embedding output")
            velocity_embedding = torch.nan_to_num(velocity_embedding, nan=0.0, posinf=1e4, neginf=-1e4)

        # 拼接所有嵌入
        combine = torch.cat([pitch_embedding, duration_embedding, velocity_embedding], dim=-1)
        
        # 检查拼接后的嵌入是否有NaN或inf值
        if torch.isnan(combine).any():
            print("Warning: NaN values found in combined embedding")
            combine = torch.nan_to_num(combine, nan=0.0)
        
        if torch.isinf(combine).any():
            print("Warning: Inf values found in combined embedding")
            combine = torch.clamp(combine, min=-1e4, max=1e4)

        # 添加位置编码
        batch_size, seq_len, embed_dim = combine.shape
        pos_emb = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        # 检查位置编码是否有NaN或inf值
        if torch.isnan(pos_emb).any():
            print("Warning: NaN values found in positional encoding")
            pos_emb = torch.nan_to_num(pos_emb, nan=0.0)
        
        if torch.isinf(pos_emb).any():
            print("Warning: Inf values found in positional encoding")
            pos_emb = torch.clamp(pos_emb, min=-1e4, max=1e4)

        output = combine + pos_emb
        
        # 最终检查输出是否有NaN或inf值
        if torch.isnan(output).any():
            print("Warning: NaN values found in embedding output")
            output = torch.nan_to_num(output, nan=0.0)
        
        if torch.isinf(output).any():
            print("Warning: Inf values found in embedding output")
            output = torch.clamp(output, min=-1e4, max=1e4)
        
        return output