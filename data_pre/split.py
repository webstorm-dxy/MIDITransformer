import json
import os
import glob
from typing import List, Dict

# 尝试导入midiutil库用于生成MIDI文件
try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False
    print("Warning: midiutil not installed. MIDI file generation will not be available.")
    print("To install midiutil, run: pip install midiutil")


def load_midi_json(file_path: str) -> List[Dict]:
    """
    加载MIDI JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        MIDI数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_midi_json(data: List[Dict], file_path: str) -> None:
    """
    保存MIDI数据到JSON文件
    
    Args:
        data: MIDI数据列表
        file_path: 保存文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_as_standard_midi(midi_data: List[Dict], output_path: str, tempo: int = 120) -> bool:
    """
    将MIDI数据保存为标准MIDI文件(.mid)
    
    Args:
        midi_data: MIDI数据列表
        output_path: 输出文件路径
        tempo: 曲速(BPM)
        
    Returns:
        保存是否成功
    """
    if not MIDIUTIL_AVAILABLE:
        print("Error: midiutil not installed. Cannot generate standard MIDI file.")
        print("Please install midiutil with: pip install midiutil")
        return False
    
    # 查找BPM信息
    bpm_info = None
    for item in midi_data:
        if item.get('type') == 'tempo':
            bpm_info = item
            tempo = item.get('bpm', tempo)
            break
    
    # 提取音符信息（具有track和pitch字段的元素）
    notes_data = [item for item in midi_data if 'pitch' in item and 'start_time' in item]
    
    # 如果没有找到明确的音符数据，尝试使用所有包含pitch的项
    if not notes_data:
        notes_data = [item for item in midi_data if 'pitch' in item]
    
    # 提取音符参数
    if notes_data:
        pitches = [note['pitch'] for note in notes_data]
        durations = [note['duration'] for note in notes_data]
        velocities = [note.get('velocity', 64) for note in notes_data]  # 默认力度为64
        start_times = [note['start_time'] for note in notes_data]
    else:
        # 没有音符数据
        print("警告: 没有找到音符数据")
        pitches, durations, velocities, start_times = [], [], [], []
    
    # 创建MIDI文件对象
    midi_file = MIDIFile(1)  # 1个轨道
    track = 0
    time = 0
    
    # 添加轨道名称
    midi_file.addTrackName(track, time, "Split MIDI Track")
    
    # 添加BPM信息
    midi_file.addTempo(track, time, tempo)
    
    # 添加音符
    channel = 0
    for i in range(len(pitches)):
        pitch = pitches[i]
        # 转换为秒
        duration = durations[i] / 1000.0
        start_time = start_times[i] / 1000.0
        velocity = velocities[i]
        
        # 确保音高在有效范围内
        if 0 <= pitch <= 127:
            midi_file.addNote(track, channel, pitch, start_time, duration, velocity)
    
    # 保存MIDI文件
    try:
        with open(output_path, 'wb') as outf:
            midi_file.writeFile(outf)
        return True
    except Exception as e:
        print(f"保存MIDI文件时出错: {e}")
        return False


def split_midi_by_measures(midi_data: List[Dict], notes_per_segment: int = 32) -> List[List[Dict]]:
    """
    按照固定音符数量将MIDI数据切分成多个片段，保留所有原始数据
    
    Args:
        midi_data: MIDI数据列表
        notes_per_segment: 每个片段包含的音符数量
        
    Returns:
        切分后的MIDI片段列表
    """
    segments = []
    
    # 查找非音符数据（元数据），如BPM、歌词等
    # 这里我们假设非音符数据是没有start_time字段或者type不为note的数据
    meta_data = [item for item in midi_data if 'start_time' not in item or item.get('type') != 'note']
    
    # 音符数据是具有start_time字段的数据
    notes_data = [item for item in midi_data if 'start_time' in item and item.get('type') == 'note']
    
    # 如果没有明确标记的音符数据，则假设所有具有pitch和start_time的数据都是音符
    if not notes_data:
        notes_data = [item for item in midi_data if 'pitch' in item and 'start_time' in item]
    
    # 按开始时间排序音符数据
    sorted_notes = sorted(notes_data, key=lambda x: x['start_time'])
    
    # 按固定音符数量切分
    for i in range(0, len(sorted_notes), notes_per_segment):
        segment_notes = sorted_notes[i:i + notes_per_segment]
        if len(segment_notes) >= notes_per_segment // 2:  # 至少要有半数音符
            # 为每个片段添加元数据和当前片段的音符数据
            segment = meta_data.copy() + segment_notes
            segments.append(segment)
    
    return segments


def split_midi_by_time(midi_data: List[Dict], time_segment_length: int = 2000) -> List[List[Dict]]:
    """
    按照时间长度将MIDI数据切分成多个片段，保留所有原始数据
    
    Args:
        midi_data: MIDI数据列表
        time_segment_length: 每个时间片段的长度（毫秒）
        
    Returns:
        切分后的MIDI片段列表
    """
    segments = []
    
    # 查找非音符数据（元数据），如BPM、歌词等
    meta_data = [item for item in midi_data if 'start_time' not in item or item.get('type') != 'note']
    
    # 音符数据是具有start_time字段的数据
    notes_data = [item for item in midi_data if 'start_time' in item and item.get('type') == 'note']
    
    # 如果没有明确标记的音符数据，则假设所有具有pitch和start_time的数据都是音符
    if not notes_data:
        notes_data = [item for item in midi_data if 'pitch' in item and 'start_time' in item]
    
    # 按开始时间排序音符数据
    sorted_notes = sorted(notes_data, key=lambda x: x['start_time'])
    
    if not sorted_notes:
        return segments
    
    # 获取时间范围
    min_time = sorted_notes[0]['start_time']
    max_time = max(note['start_time'] + note['duration'] for note in sorted_notes)
    
    # 按时间切分
    current_time = min_time
    while current_time < max_time:
        next_time = current_time + time_segment_length
        segment_notes = [note for note in sorted_notes 
                        if note['start_time'] >= current_time and note['start_time'] < next_time]
        
        # 只有当片段不为空且包含足够的音符时才添加
        if segment_notes and len(segment_notes) >= 3:  # 至少3个音符
            # 为每个片段添加元数据和当前片段的音符数据
            segment = meta_data.copy() + segment_notes
            segments.append(segment)
        
        current_time = next_time
    
    return segments


def process_midi_files(input_dir: str = "../data", output_dir: str = "../splited_midi", 
                      split_method: str = "time", output_format: str = "midi", **kwargs) -> None:
    """
    处理目录中的所有MIDI JSON文件，将它们切分成小片段
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        split_method: 切分方法 ("time" 或 "notes")
        output_format: 输出格式 ("midi" 或 "json")
        **kwargs: 其他参数
            - time_segment_length: 时间切分时每段的长度（毫秒）
            - notes_per_segment: 音符切分时每段的音符数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"在目录 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    total_segments = 0
    
    for file_path in json_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"处理文件: {file_name}")
        
        try:
            # 加载MIDI数据（包含所有信息）
            midi_data = load_midi_json(file_path)
            
            print(f"  原始数据项数量: {len(midi_data)}")
            
            # 切分MIDI数据
            if split_method == "time":
                time_segment_length = kwargs.get("time_segment_length", 2000)
                segments = split_midi_by_time(midi_data, time_segment_length)
                print(f"  按时间切分 ({time_segment_length}ms): {len(segments)} 个片段")
            elif split_method == "notes":
                notes_per_segment = kwargs.get("notes_per_segment", 32)
                segments = split_midi_by_measures(midi_data, notes_per_segment)
                print(f"  按音符数切分 ({notes_per_segment} 个音符): {len(segments)} 个片段")
            else:
                raise ValueError(f"不支持的切分方法: {split_method}")
            
            # 保存切分后的片段
            for i, segment in enumerate(segments):
                if output_format == "midi":
                    output_file = os.path.join(output_dir, f"{file_name}_part_{i+1:03d}.mid")
                    success = save_as_standard_midi(segment, output_file)
                    if not success:
                        print(f"  保存MIDI文件失败: {output_file}")
                else:
                    output_file = os.path.join(output_dir, f"{file_name}_part_{i+1:03d}.json")
                    save_midi_json(segment, output_file)
            
            total_segments += len(segments)
            
        except Exception as e:
            print(f"  处理文件 {file_name} 时出错: {e}")
    
    format_name = "MIDI" if output_format == "midi" else "JSON"
    print(f"总共生成 {total_segments} 个片段，保存为 {format_name} 格式在 {output_dir} 目录中")


def main():
    """
    主函数
    """
    print("MIDI文件切分工具")
    print("=" * 30)
    
    # 检查midiutil是否可用
    if not MIDIUTIL_AVAILABLE:
        print("警告: 未安装midiutil库，将输出JSON格式文件")
        output_format = "json"
    else:
        print("检测到midiutil库，将输出MIDI格式文件")
        output_format = "midi"
    
    print()
    
    # 按时间切分（默认）
    print("按时间切分 (2000ms 每段):")
    process_midi_files(split_method="time", time_segment_length=2000, output_format=output_format)
    
    print("\n" + "=" * 30)
    
    # 按音符数切分
    print("按音符数切分 (32 个音符每段):")
    process_midi_files(split_method="notes", notes_per_segment=32, output_format=output_format)


if __name__ == "__main__":
    main()