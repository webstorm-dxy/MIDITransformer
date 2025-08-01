import torch
import torch.nn as nn
import sys
import os
import json
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embedding import MIDIEmbedding
from model.transformer import Decoder, DecoderLayer


class MIDITransformer(nn.Module):
    """
    MIDI Transformer模型，包含嵌入层和解码器
    """
    def __init__(self, pitch_range=128, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(MIDITransformer, self).__init__()
        self.embedding = MIDIEmbedding(pitch_range, embed_dim)
        self.decoder = Decoder(embed_dim, num_heads, num_layers, dropout)
        self.pitch_head = nn.Linear(embed_dim, pitch_range)  # 音高预测头
        self.duration_head = nn.Linear(embed_dim, 1)  # 持续时间预测头
        self.velocity_head = nn.Linear(embed_dim, 1)  # 力度预测头

    def forward(self, input_pitches, input_durations, input_velocities, target_pitches=None):
        """
        前向传播
        
        Args:
            input_pitches: 输入音高 (batch_size, seq_len)
            input_durations: 输入持续时间 (batch_size, seq_len, 1)
            input_velocities: 输入力度 (batch_size, seq_len, 1)
            target_pitches: 目标音高，用于创建目标序列掩码 (batch_size, seq_len)
            
        Returns:
            预测的音高、持续时间和力度
        """
        # 嵌入输入
        embedded = self.embedding(input_pitches, input_durations, input_velocities)
        
        # 创建目标序列掩码（用于解码器的自注意力）
        if target_pitches is not None:
            tgt_seq_len = target_pitches.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=embedded.device)).bool()
        else:
            tgt_seq_len = input_pitches.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=embedded.device)).bool()
        
        # 解码器处理
        decoder_output = self.decoder(embedded, embedded, tgt_mask=tgt_mask)
        
        # 预测输出
        pitch_output = self.pitch_head(decoder_output)
        duration_output = self.duration_head(decoder_output)
        velocity_output = self.velocity_head(decoder_output)
        
        return pitch_output, duration_output, velocity_output


def load_model(model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备 (CPU 或 GPU)
        
    Returns:
        加载好的模型
    """
    # 创建模型实例
    model = MIDITransformer()
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    return model


def generate_midi_sequence(model, seed_pitches, seed_durations, seed_velocities, sequence_length, device):
    """
    使用训练好的模型生成MIDI序列
    
    Args:
        model: 训练好的模型
        seed_pitches: 种子音高序列
        seed_durations: 种子持续时间序列
        seed_velocities: 种子力度序列
        sequence_length: 要生成的序列长度
        device: 设备 (CPU 或 GPU)
        
    Returns:
        生成的音高、持续时间和力度序列
    """
    with torch.no_grad():
        # 初始化生成序列
        generated_pitches = seed_pitches.clone()
        generated_durations = seed_durations.clone()
        generated_velocities = seed_velocities.clone()
        
        # 逐个生成音符
        for _ in range(sequence_length):
            # 获取当前序列长度
            seq_len = generated_pitches.size(1)
            
            # 创建目标序列掩码
            tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
            
            # 模型前向传播
            pitch_output, duration_output, velocity_output = model(
                generated_pitches, generated_durations, generated_velocities
            )
            
            # 获取最后一个时间步的预测结果
            next_pitch_logits = pitch_output[:, -1, :]  # (batch_size, pitch_range)
            next_duration = duration_output[:, -1, :]   # (batch_size, 1)
            next_velocity = velocity_output[:, -1, :]   # (batch_size, 1)
            
            # 从音高分布中采样
            next_pitch_probs = torch.softmax(next_pitch_logits, dim=-1)
            next_pitch = torch.multinomial(next_pitch_probs, 1)  # (batch_size, 1)
            
            # 将新生成的音符添加到序列中
            generated_pitches = torch.cat([generated_pitches, next_pitch], dim=1)
            generated_durations = torch.cat([generated_durations, next_duration.unsqueeze(1)], dim=1)
            generated_velocities = torch.cat([generated_velocities, next_velocity.unsqueeze(1)], dim=1)
        
        return generated_pitches, generated_durations, generated_velocities


def test_model(model_path="midi_transformer_final.pth"):
    """
    测试模型功能
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    try:
        model = load_model(model_path, device)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 创建种子序列进行测试
    batch_size = 1
    seed_length = 10
    
    # 创建种子输入
    seed_pitches = torch.randint(40, 90, (batch_size, seed_length)).to(device)  # 随机音高 (40-89)
    seed_durations = torch.rand(batch_size, seed_length, 1).to(device) * 100    # 随机持续时间 (0-100)
    seed_velocities = torch.randint(60, 120, (batch_size, seed_length, 1)).to(device).float()  # 随机力度 (60-119)
    
    print("Seed sequence:")
    print(f"  Pitches: {seed_pitches.cpu().numpy()[0]}")
    print(f"  Durations: {seed_durations.cpu().numpy()[0].flatten()}")
    print(f"  Velocities: {seed_velocities.cpu().numpy()[0].flatten()}")
    
    # 生成新的MIDI序列
    generated_length = 20
    print(f"\nGenerating {generated_length} new notes...")
    
    generated_pitches, generated_durations, generated_velocities = generate_midi_sequence(
        model, seed_pitches, seed_durations, seed_velocities, generated_length, device
    )
    
    # 打印生成结果
    print("\nGenerated sequence:")
    print(f"  Pitches: {generated_pitches.cpu().numpy()[0]}")
    print(f"  Durations: {generated_durations.cpu().numpy()[0].flatten()}")
    print(f"  Velocities: {generated_velocities.cpu().numpy()[0].flatten()}")
    
    # 保存生成的MIDI数据到JSON文件
    save_generated_midi(generated_pitches, generated_durations, generated_velocities, "generated_midi.json")
    print("\nGenerated MIDI data saved to 'generated_midi.json'")


def save_generated_midi(pitches, durations, velocities, filename):
    """
    将生成的MIDI数据保存为JSON格式
    
    Args:
        pitches: 音高张量
        durations: 持续时间张量
        velocities: 力度张量
        filename: 保存的文件名
    """
    # 转换为numpy数组
    pitches_np = pitches.cpu().numpy()[0]
    durations_np = durations.cpu().numpy()[0].flatten()
    velocities_np = velocities.cpu().numpy()[0].flatten()
    
    # 创建MIDI数据列表
    midi_data = []
    start_time = 0
    
    for i in range(len(pitches_np)):
        note = {
            "track": 1,
            "pitch": int(pitches_np[i]),
            "velocity": int(velocities_np[i]),
            "start_time": int(start_time),
            "end_time": int(start_time + durations_np[i]),
            "duration": int(durations_np[i])
        }
        midi_data.append(note)
        start_time += durations_np[i]
    
    # 保存到JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(midi_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    test_model()