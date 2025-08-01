import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.w_Q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_K(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_V(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.d_model ** -0.5)
        
        # 检查注意力分数是否有NaN或inf
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            print("Warning: NaN or Inf values found in attention scores")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
        
        if mask is not None:
            # 使用更适合半精度的值
            attention_scores = attention_scores.masked_fill(mask == 0, float(torch.finfo(torch.float16).min))
        
        # 使用数值稳定的softmax
        attention_scores = attention_scores - torch.max(attention_scores, dim=-1, keepdim=True)[0]
        attention = F.softmax(attention_scores, dim=-1)
        
        # 检查注意力权重是否有NaN或inf
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            print("Warning: NaN or Inf values found in attention weights")
            attention = torch.nan_to_num(attention, nan=0.0)
        
        out = torch.matmul(attention, V)
        
        # 检查输出是否有NaN或inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Warning: NaN or Inf values found in attention output")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc_out(out)
        
        return out

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MaskedMultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.w_Q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_K(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_V(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.d_model ** -0.5)
        
        # 检查注意力分数是否有NaN或inf
        if torch.isnan(attention_scores).any() or torch.isinf(attention_scores).any():
            print("Warning: NaN or Inf values found in masked attention scores")
            attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=1e4, neginf=-1e4)
        
        if mask is not None:
            # 使用更适合半精度的值
            attention_scores = attention_scores.masked_fill(mask == 0, float(torch.finfo(torch.float16).min))
        
        # 使用数值稳定的softmax
        attention_scores = attention_scores - torch.max(attention_scores, dim=-1, keepdim=True)[0]
        attention = F.softmax(attention_scores, dim=-1)
        
        # 检查注意力权重是否有NaN或inf
        if torch.isnan(attention).any() or torch.isinf(attention).any():
            print("Warning: NaN or Inf values found in masked attention weights")
            attention = torch.nan_to_num(attention, nan=0.0)
        
        out = torch.matmul(attention, V)
        
        # 检查输出是否有NaN或inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("Warning: NaN or Inf values found in masked attention output")
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.fc_out(out)
        
        return out

class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # 可选：添加Dropout

    def forward(self, x, sublayer_output):
        # X 是Attention的输入
        # Step 1: 残差连接 (Add)
        residual = x + sublayer_output
        
        # 检查残差连接后是否有NaN或inf
        if torch.isnan(residual).any() or torch.isinf(residual).any():
            print("Warning: NaN or Inf values found in residual connection")
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)

        # Step 2: 层归一化 (Norm)
        output = self.norm(residual)
        
        # 检查层归一化后是否有NaN或inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf values found after layer normalization")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return self.dropout(output)  # 若需Dropout


class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 创建解码器层列表
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        x: 目标序列输入 (batch_size, tgt_seq_len, d_model)
        encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
        src_mask: 源序列掩码 (batch_size, 1, 1, src_seq_len)
        tgt_mask: 目标序列掩码 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        # 检查输入是否有NaN或inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf values found in decoder input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 通过所有解码器层
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
            # 检查每层输出是否有NaN或inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN or Inf values found in decoder layer {i} output")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 最终层归一化
        output = self.norm(x)
        
        # 检查最终输出是否有NaN或inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf values found in final decoder output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        
        # 自注意力机制（带掩码）
        self.masked_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        
        # 编码器-解码器注意力机制
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model, dropout)
        
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.add_norm3 = AddNorm(d_model, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        x: 目标序列输入 (batch_size, tgt_seq_len, d_model)
        encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
        src_mask: 源序列掩码 (batch_size, 1, 1, src_seq_len)
        tgt_mask: 目标序列掩码 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        # 第一层：带掩码的自注意力 + 残差连接和层归一化
        masked_attn_output = self.masked_attention(x, x, x, tgt_mask)
        x = self.add_norm1(x, masked_attn_output)
        
        # 第二层：编码器-解码器注意力 + 残差连接和层归一化
        enc_dec_attn_output = self.enc_dec_attention(x, encoder_output, encoder_output, src_mask)
        x = self.add_norm2(x, enc_dec_attn_output)
        
        # 第三层：前馈神经网络 + 残差连接和层归一化
        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)
        
        return x