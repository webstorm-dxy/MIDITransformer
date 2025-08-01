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

# 尝试导入midiutil库用于生成MIDI文件
try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False
    print("Warning: midiutil not installed. MIDI file generation will not be available.")
    print("To install midiutil, run: pip install midiutil")


class MIDITransformerResoner(nn.Module):
    """
    MIDI Transformer模型，包含嵌入层和解码器
    """
    def __init__(self, pitch_range=128, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1):
        super(MIDITransformerResoner, self).__init__()
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
    model = MIDITransformerResoner()
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    return model


def load_input_json(json_path):
    """
    加载输入的JSON文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        包含音高、持续时间、力度和开始时间的numpy数组
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取特征
    pitches = np.array([note['pitch'] for note in data])
    durations = np.array([note['duration'] for note in data])
    velocities = np.array([note['velocity'] for note in data])
    start_times = np.array([note['start_time'] for note in data])
    
    return pitches, durations, velocities, start_times


def load_context_from_midi(json_path, num_notes=None):
    """
    从MIDI文件中加载指定数量的音符作为上下文
    
    Args:
        json_path: JSON文件路径
        num_notes: 要用作上下文的音符数量，如果为None则使用全部音符
        
    Returns:
        包含音高、持续时间、力度和开始时间的numpy数组
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果指定了音符数量，则只取前num_notes个音符
    if num_notes is not None:
        data = data[:num_notes]
    
    # 提取特征
    pitches = np.array([note['pitch'] for note in data])
    durations = np.array([note['duration'] for note in data])
    velocities = np.array([note['velocity'] for note in data])
    start_times = np.array([note['start_time'] for note in data])
    
    return pitches, durations, velocities, start_times


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
            
            # 对持续时间添加一些随机性，避免所有音符持续时间相同
            # 使用指数函数将负值转换为正值，并添加一些随机扰动
            # 增加持续时间的平均值，使音符更长
            next_duration = torch.abs(next_duration) * 10 + 50  # 缩放并增加基础值
            duration_noise = torch.randn_like(next_duration) * 0.3 * next_duration
            next_duration = torch.abs(next_duration + duration_noise)
            
            # 对力度也添加一些随机性
            velocity_noise = torch.randn_like(next_velocity) * 5
            next_velocity = torch.clamp(next_velocity + velocity_noise, 0, 127)
            
            # 将新生成的音符添加到序列中
            generated_pitches = torch.cat([generated_pitches, next_pitch], dim=1)
            generated_durations = torch.cat([generated_durations, next_duration.unsqueeze(1)], dim=1)
            generated_velocities = torch.cat([generated_velocities, next_velocity.unsqueeze(1)], dim=1)
        
        return generated_pitches, generated_durations, generated_velocities


def save_as_midi_json(pitches, durations, velocities, output_path, start_time=0):
    """
    将生成的音符序列保存为MIDI格式的JSON文件
    
    Args:
        pitches: 音高序列
        durations: 持续时间序列
        velocities: 力度序列
        output_path: 输出文件路径
        start_time: 起始时间（用于延续输入序列的时间）
    """
    midi_data = []
    
    # 添加BPM信息作为第一个元素
    bpm_info = {
        "track": 0,
        "type": "tempo",
        "bpm": 120,
        "start_time": 0
    }
    midi_data.append(bpm_info)
    
    current_time = start_time
    for i in range(len(pitches)):
        # 增加持续时间的最小值，使音符更长一些
        duration = max(50, int(durations[i]))  # 增加最小持续时间为50
        note = {
            "track": 1,
            "pitch": int(pitches[i]),
            "velocity": int(velocities[i]),
            "start_time": int(current_time),
            "end_time": int(current_time + duration),
            "duration": duration
        }
        midi_data.append(note)
        current_time += duration
    
    # 保存到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(midi_data, f, ensure_ascii=False, indent=2)


def save_as_standard_midi(pitches, durations, velocities, start_times, output_path, tempo=120):
    """
    将音符序列保存为标准MIDI文件(.mid)
    
    Args:
        pitches: 音高序列
        durations: 持续时间序列
        velocities: 力度序列
        start_times: 开始时间序列
        output_path: 输出文件路径
        tempo: 曲速(BPM)
    """
    if not MIDIUTIL_AVAILABLE:
        print("Error: midiutil not installed. Cannot generate standard MIDI file.")
        print("Please install midiutil with: pip install midiutil")
        return False
    
    # 创建MIDI文件对象
    midi_file = MIDIFile(1)  # 1个轨道
    track = 0
    time = 0
    midi_file.addTrackName(track, time, "Generated Track")
    midi_file.addTempo(track, time, tempo)
    
    # 添加音符
    channel = 0
    for i in range(len(pitches)):
        pitch = int(pitches[i])
        # 确保持续时间是正数，并增加最小持续时间使音符更长
        duration = max(0.05, durations[i] / 1000.0)  # 转换为秒，最小值为0.05秒（原来是0.001）
        start_time = start_times[i] / 1000.0  # 转换为秒
        velocity = int(velocities[i])
        
        # 确保音高在有效范围内
        if 0 <= pitch <= 127:
            midi_file.addNote(track, channel, pitch, start_time, duration, velocity)
    
    # 保存MIDI文件
    with open(output_path, 'wb') as outf:
        midi_file.writeFile(outf)
    
    return True


def inference(input_json_path, model_path="./pth/midi_transformer_final.pth", output_midi_path="output.mid", context_length=None):
    """
    执行推理过程
    
    Args:
        input_json_path: 输入JSON文件路径
        model_path: 模型文件路径
        output_midi_path: 输出MIDI文件路径
        context_length: 用作上下文的音符数量，如果为None则使用全部音符
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
    
    # 加载输入JSON数据
    try:
        if context_length is not None:
            pitches, durations, velocities, start_times = load_context_from_midi(input_json_path, context_length)
            print(f"Loaded {context_length} notes as context from {input_json_path}")
        else:
            pitches, durations, velocities, start_times = load_input_json(input_json_path)
            print(f"Input JSON loaded successfully from {input_json_path}")
        print(f"Input sequence length: {len(pitches)}")
    except FileNotFoundError:
        print(f"Input JSON file {input_json_path} not found.")
        return
    except Exception as e:
        print(f"Error loading input JSON: {e}")
        return
    
    # 将输入数据转换为张量并移动到设备
    seed_pitches = torch.tensor(pitches, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
    seed_durations = torch.tensor(durations, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
    seed_velocities = torch.tensor(velocities, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
    
    # 生成新的MIDI序列
    generated_length = 50  # 生成50个新音符
    print(f"Generating {generated_length} new notes...")
    
    generated_pitches, generated_durations, generated_velocities = generate_midi_sequence(
        model, seed_pitches, seed_durations, seed_velocities, generated_length, device
    )
    
    # 提取生成的部分（去掉原始输入部分）
    new_pitches = generated_pitches[0, len(pitches):].cpu().numpy()
    new_durations = generated_durations[0, len(durations):, 0].cpu().numpy()
    new_velocities = generated_velocities[0, len(velocities):, 0].cpu().numpy()
    
    # 计算输入序列的结束时间
    input_end_time = start_times[-1] + durations[-1]
    
    # 如果输出文件是.mid格式，则生成标准MIDI文件
    if output_midi_path.endswith('.mid'):
        # 生成新的开始时间（基于输入序列的结束时间）
        new_start_times = []
        current_time = input_end_time
        for duration in new_durations:
            new_start_times.append(current_time)
            # 确保持续时间是正数，并增加最小持续时间
            current_time += max(50, int(duration))
        
        # 保存为标准MIDI文件
        success = save_as_standard_midi(
            new_pitches, new_durations, new_velocities, new_start_times, output_midi_path
        )
        if success:
            print(f"Generated MIDI saved to {output_midi_path}")
        else:
            print("Failed to generate standard MIDI file.")
    else:
        # 保存为JSON格式
        save_as_midi_json(new_pitches, new_durations, new_velocities, output_midi_path, input_end_time)
        print(f"Generated MIDI saved to {output_midi_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python inference.py <input_json_path> <output_midi_path> [context_length]")
        print("Example: python inference.py data/MIDI.json output.mid 20")
        print("If context_length is not specified, the entire input will be used as context")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 如果提供了上下文长度参数，则解析它
    context_length = None
    if len(sys.argv) == 4:
        try:
            context_length = int(sys.argv[3])
            if context_length <= 0:
                print("Context length must be a positive integer")
                sys.exit(1)
        except ValueError:
            print("Context length must be a valid integer")
            sys.exit(1)
    
    inference(input_path, output_midi_path=output_path, context_length=context_length)
