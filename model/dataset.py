import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict, Tuple
import math
import numpy as np

class MIDIDataset(Dataset):
    """
    MIDI数据集类，用于读取和处理MIDI JSON文件
    """
    
    def __init__(self, json_path: str, sequence_length: int = 64):
        """
        初始化数据集
        
        Args:
            json_path: MIDI JSON文件路径
            sequence_length: 序列长度，用于将MIDI数据分割成固定长度的序列
        """
        self.json_path = json_path
        self.sequence_length = sequence_length
        self.data = self._load_data(json_path)
        self.sequences = self._create_sequences()
        
    def _load_data(self, json_path: str) -> List[Dict]:
        """
        从JSON文件加载MIDI数据
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            MIDI数据列表
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 过滤掉BPM信息等非音符数据（如果存在）
            notes_data = [note for note in data if 'pitch' in note]
            
            # 清理数据，确保没有NaN或无穷大值
            cleaned_data = []
            for note in notes_data:
                # 检查并清理音高数据
                if 'pitch' in note and not (np.isnan(note['pitch']) or np.isinf(note['pitch'])):
                    note['pitch'] = int(max(0, min(127, note['pitch'])))  # 限制在MIDI有效范围内
                else:
                    continue  # 跳过无效数据点
                
                # 检查并清理持续时间数据
                if 'duration' in note and not (np.isnan(note['duration']) or np.isinf(note['duration'])):
                    note['duration'] = max(0.001, note['duration'])  # 确保为正数
                else:
                    continue  # 跳过无效数据点
                
                # 检查并清理力度数据
                if 'velocity' in note and not (np.isnan(note['velocity']) or np.isinf(note['velocity'])):
                    note['velocity'] = max(0, min(127, note['velocity']))  # 限制在MIDI有效范围内
                else:
                    continue  # 跳过无效数据点
                
                # 检查开始时间
                if 'start_time' in note and not (np.isnan(note['start_time']) or np.isinf(note['start_time'])):
                    note['start_time'] = max(0, note['start_time'])  # 确保为非负数
                else:
                    continue  # 跳过无效数据点
                
                cleaned_data.append(note)
            
            print(f"Loaded {len(data)} total items, {len(notes_data)} notes, {len(cleaned_data)} cleaned notes from {json_path}")
            return cleaned_data
        except Exception as e:
            print(f"Error loading data from {json_path}: {e}")
            return []
    
    def _create_sequences(self) -> List[List[Dict]]:
        """
        将MIDI数据分割成固定长度的序列
        
        Returns:
            序列列表，每个序列包含sequence_length个音符
        """
        sequences = []
        # 按开始时间排序
        sorted_data = sorted(self.data, key=lambda x: x['start_time'])
        
        # 检查是否有足够的数据
        if len(sorted_data) < 2:  # 至少需要2个音符才能形成输入-目标对
            print(f"Warning: Not enough data in {self.json_path}. Need at least 2 notes, got {len(sorted_data)}")
            return sequences
        
        # 如果数据少于序列长度，使用填充方式处理
        if len(sorted_data) < self.sequence_length:
            print(f"Warning: Not enough data for full sequence length in {self.json_path}. "
                  f"Got {len(sorted_data)} notes, need {self.sequence_length}. "
                  f"Will repeat data to fill sequence.")
            # 重复数据直到达到序列长度
            repeats = math.ceil(self.sequence_length / len(sorted_data))
            extended_data = (sorted_data * repeats)[:self.sequence_length]
            sequences.append(extended_data)
        else:
            # 将数据分割成固定长度的序列
            for i in range(0, len(sorted_data) - self.sequence_length + 1, self.sequence_length // 2):
                sequence = sorted_data[i:i + self.sequence_length]
                if len(sequence) == self.sequence_length:
                    # 再次检查序列中的数据有效性
                    valid_sequence = True
                    for note in sequence:
                        # 检查所有必要字段是否存在且有效
                        if ('pitch' not in note or 'duration' not in note or 'velocity' not in note or 'start_time' not in note):
                            valid_sequence = False
                            break
                        if (np.isnan(note['pitch']) or np.isinf(note['pitch']) or 
                            np.isnan(note['duration']) or np.isinf(note['duration']) or
                            np.isnan(note['velocity']) or np.isinf(note['velocity']) or
                            np.isnan(note['start_time']) or np.isinf(note['start_time'])):
                            valid_sequence = False
                            break
                    
                    if valid_sequence:
                        sequences.append(sequence)
                
        print(f"Created {len(sequences)} sequences from {self.json_path}")
        return sequences
    
    def __len__(self) -> int:
        """
        返回数据集大小
        
        Returns:
            数据集中的序列数量
        """
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含输入和目标序列的元组 (input_seq, target_seq)
        """
        sequence = self.sequences[idx]
        
        # 提取特征
        pitches = [max(0, min(127, int(note['pitch']))) for note in sequence]  # 确保在有效范围内
        durations = [max(0.001, note['duration']) for note in sequence]  # 确保为正数
        velocities = [max(0, min(127, note['velocity'])) for note in sequence]  # 确保在有效范围内
        
        # 转换为张量
        # 输入序列（除了最后一个元素）
        input_pitches = torch.tensor(pitches[:-1], dtype=torch.long)
        input_durations = torch.tensor(durations[:-1], dtype=torch.float32).unsqueeze(1)
        input_velocities = torch.tensor(velocities[:-1], dtype=torch.float32).unsqueeze(1)
        
        # 目标序列（除了第一个元素）
        target_pitches = torch.tensor(pitches[1:], dtype=torch.long)
        target_durations = torch.tensor(durations[1:], dtype=torch.float32).unsqueeze(1)
        target_velocities = torch.tensor(velocities[1:], dtype=torch.float32).unsqueeze(1)
        
        # 检查并处理可能的NaN或无穷大值
        if torch.isnan(input_durations).any() or torch.isinf(input_durations).any():
            input_durations = torch.clamp(input_durations, min=0.001, max=1e6)
            
        if torch.isnan(input_velocities).any() or torch.isinf(input_velocities).any():
            input_velocities = torch.clamp(input_velocities, min=0, max=127)
            
        if torch.isnan(target_durations).any() or torch.isinf(target_durations).any():
            target_durations = torch.clamp(target_durations, min=0.001, max=1e6)
            
        if torch.isnan(target_velocities).any() or torch.isinf(target_velocities).any():
            target_velocities = torch.clamp(target_velocities, min=0, max=127)
        
        # 组合特征
        input_seq = (input_pitches, input_durations, input_velocities)
        target_seq = (target_pitches, target_durations, target_velocities)
        
        return input_seq, target_seq

def collate_fn(batch):
    """
    自定义批处理函数
    
    Args:
        batch: 批次数据
        
    Returns:
        批次张量
    """
    input_batch, target_batch = zip(*batch)
    
    # 解包输入和目标
    input_pitches, input_durations, input_velocities = zip(*input_batch)
    target_pitches, target_durations, target_velocities = zip(*target_batch)
    
    # 打印调试信息
    # print(f"Batch sizes - Input pitches: {[t.size(0) for t in input_pitches]}, "
    #       f"Target pitches: {[t.size(0) for t in target_pitches]}")
    
    # 检查并确保所有张量具有相同的大小
    # 获取最大长度
    max_input_len = max(tensor.size(0) for tensor in input_pitches)
    max_target_len = max(tensor.size(0) for tensor in target_pitches)
    
    # 如果所有张量已经具有相同的大小，则直接堆叠
    if all(tensor.size(0) == max_input_len for tensor in input_pitches) and \
       all(tensor.size(0) == max_target_len for tensor in target_pitches):
        input_pitches = torch.stack(input_pitches)
        input_durations = torch.stack(input_durations)
        input_velocities = torch.stack(input_velocities)
        
        target_pitches = torch.stack(target_pitches)
        target_durations = torch.stack(target_durations)
        target_velocities = torch.stack(target_velocities)
    else:
        # 如果张量大小不同，则进行填充
        def pad_tensors(tensor_list, max_len):
            padded_tensors = []
            for tensor in tensor_list:
                if tensor.size(0) < max_len:
                    # 计算需要填充的数量
                    pad_len = max_len - tensor.size(0)
                    # 对于不同维度的张量采用不同的填充方式
                    if tensor.dim() == 1:
                        padded_tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=tensor.dtype)])
                    else:  # tensor.dim() == 2
                        padding = torch.zeros(pad_len, tensor.size(1), dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, padding])
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            return torch.stack(padded_tensors)
        
        input_pitches = pad_tensors(input_pitches, max_input_len)
        input_durations = pad_tensors(input_durations, max_input_len)
        input_velocities = pad_tensors(input_velocities, max_input_len)
        
        target_pitches = pad_tensors(target_pitches, max_target_len)
        target_durations = pad_tensors(target_durations, max_target_len)
        target_velocities = pad_tensors(target_velocities, max_target_len)
    
    # 最终检查并清理可能的NaN或无穷大值
    def clean_tensor(tensor, min_val=None, max_val=None):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=max_val or 1e6, neginf=min_val or 0.0)
        if min_val is not None or max_val is not None:
            tensor = torch.clamp(tensor, min=min_val, max=max_val)
        return tensor
    
    # 清理各个张量
    input_pitches = clean_tensor(input_pitches, min_val=0, max_val=127)
    input_durations = clean_tensor(input_durations, min_val=0.001)
    input_velocities = clean_tensor(input_velocities, min_val=0, max_val=127)
    
    target_pitches = clean_tensor(target_pitches, min_val=0, max_val=127)
    target_durations = clean_tensor(target_durations, min_val=0.001)
    target_velocities = clean_tensor(target_velocities, min_val=0, max_val=127)
    
    return (input_pitches, input_durations, input_velocities), (target_pitches, target_durations, target_velocities)

# 测试代码
if __name__ == "__main__":
    import glob
    # 获取所有JSON数据文件
    data_files = glob.glob("../data/*.json")
    if not data_files:
        print("No JSON files found in ../data/ directory")
    else:
        print(f"Found {len(data_files)} JSON files:")
        for file in data_files:
            print(f"  - {file}")
        
        # 创建数据集实例（使用第一个文件作为示例）
        dataset = MIDIDataset(data_files[0])
        print(f"Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
            
            # 测试获取一个批次
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_pitches, input_durations, input_velocities = input_seq
                target_pitches, target_durations, target_velocities = target_seq
                
                print(f"Batch {batch_idx}:")
                print(f"Input pitches shape: {input_pitches.shape}")
                print(f"Input durations shape: {input_durations.shape}")
                print(f"Input velocities shape: {input_velocities.shape}")
                print(f"Target pitches shape: {target_pitches.shape}")
                print(f"Target durations shape: {target_durations.shape}")
                print(f"Target velocities shape: {target_velocities.shape}")
                
                # 检查是否有NaN或无穷大值
                if (torch.isnan(input_pitches).any() or torch.isnan(input_durations).any() or 
                    torch.isnan(input_velocities).any() or torch.isnan(target_pitches).any() or
                    torch.isnan(target_durations).any() or torch.isnan(target_velocities).any()):
                    print("Warning: NaN values found in batch data")
                
                if (torch.isinf(input_pitches).any() or torch.isinf(input_durations).any() or 
                    torch.isinf(input_velocities).any() or torch.isinf(target_pitches).any() or
                    torch.isinf(target_durations).any() or torch.isinf(target_velocities).any()):
                    print("Warning: Inf values found in batch data")
                
                break