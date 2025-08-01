import os
import glob
from typing import List, Tuple

# 尝试导入midiutil库用于读取和生成MIDI文件
try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False
    print("Warning: midiutil not installed.")
    print("To install midiutil, run: pip install midiutil")

# 尝试导入music21库用于读取MIDI文件
try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("Warning: music21 not installed.")
    print("To install music21, run: pip install music21")

def load_midi_file(file_path: str):
    """
    加载MIDI文件
    
    Args:
        file_path: MIDI文件路径
        
    Returns:
        music21.stream.Score对象
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required to load MIDI files. Please install it with: pip install music21")
    
    try:
        # 使用music21加载MIDI文件
        score = music21.converter.parse(file_path)
        return score
    except Exception as e:
        print(f"Error loading MIDI file {file_path}: {e}")
        return None

def extract_notes_and_meta(score) -> Tuple[List, List]:
    """
    从music21 Score对象中提取音符和元数据
    
    Args:
        score: music21.stream.Score对象
        
    Returns:
        (notes_data, meta_data) 元组
    """
    notes_data = []
    meta_data = []
    
    # 提取元数据
    # BPM信息
    metronome_marks = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
    for tempo_mark in metronome_marks:
        meta_data.append({
            'type': 'tempo',
            'bpm': tempo_mark.number,
            'offset': float(tempo_mark.offset) if tempo_mark.offset is not None else 0.0
        })
    
    # 拍号信息
    time_signatures = score.flatten().getElementsByClass(music21.meter.TimeSignature)
    for ts in time_signatures:
        meta_data.append({
            'type': 'time_signature',
            'numerator': ts.numerator,
            'denominator': ts.denominator,
            'offset': float(ts.offset) if ts.offset is not None else 0.0
        })
    
    # 调号信息
    key_signatures = score.flatten().getElementsByClass(music21.key.KeySignature)
    for key in key_signatures:
        meta_data.append({
            'type': 'key_signature',
            'tonic': str(key.tonic) if key.tonic else '',
            'mode': key.mode,
            'offset': float(key.offset) if key.offset is not None else 0.0
        })
    
    # 提取音符
    notes = score.flatten().notes
    for element in notes:
        if hasattr(element, 'pitch'):  # 单音符
            notes_data.append({
                'type': 'note',
                'pitch': element.pitch.midi,
                'start_time': float(element.offset),
                'duration': float(element.duration.quarterLength),
                'velocity': 64,  # 默认力度
                'track': 0
            })
        elif hasattr(element, 'pitches'):  # 和弦
            for pitch in element.pitches:
                notes_data.append({
                    'type': 'note',
                    'pitch': pitch.midi,
                    'start_time': float(element.offset),
                    'duration': float(element.duration.quarterLength),
                    'velocity': 64,  # 默认力度
                    'track': 0
                })
    
    return notes_data, meta_data

def remove_leading_and_trailing_gaps(notes_data: List) -> List:
    """
    移除音符片段前后的空白时间段
    
    Args:
        notes_data: 音符数据列表
        
    Returns:
        移除前后空白时间段后的音符数据列表
    """
    if not notes_data:
        return notes_data
    
    # 按开始时间排序
    sorted_notes = sorted(notes_data, key=lambda x: x['start_time'])
    
    # 找到第一个音符的开始时间和最后一个音符的结束时间
    first_start_time = sorted_notes[0]['start_time']
    last_end_time = max(note['start_time'] + note['duration'] for note in sorted_notes)
    
    # 计算需要移除的前置时间（第一个音符开始前的空白）
    offset = first_start_time
    
    # 创建新的音符列表，移除前后空白
    trimmed_notes = []
    for note in sorted_notes:
        # 将音符向前移动offset时间单位，移除前置空白
        new_note = note.copy()
        new_note['start_time'] = note['start_time'] - offset
        trimmed_notes.append(new_note)
    
    return trimmed_notes

def save_as_standard_midi(notes_data: List, meta_data: List, output_path: str) -> bool:
    """
    将音符数据和元数据保存为标准MIDI文件
    
    Args:
        notes_data: 音符数据列表
        meta_data: 元数据列表
        output_path: 输出文件路径
        
    Returns:
        保存是否成功
    """
    if not MIDIUTIL_AVAILABLE:
        print("Error: midiutil not installed. Cannot generate standard MIDI file.")
        print("Please install midiutil with: pip install midiutil")
        return False
    
    # 移除片段前后的空白时间段
    trimmed_notes = remove_leading_and_trailing_gaps(notes_data)
    
    # 创建MIDI文件对象
    midi_file = MIDIFile(1)  # 1个轨道
    track = 0
    time = 0
    
    # 添加轨道名称
    midi_file.addTrackName(track, time, "Split MIDI Track")
    
    # 添加BPM信息（默认120，如果有元数据则使用元数据中的BPM）
    tempo = 120
    for meta in meta_data:
        if meta.get('type') == 'tempo':
            tempo = meta.get('bpm', tempo)
            break
    midi_file.addTempo(track, time, tempo)
    
    # 添加拍号信息
    for meta in meta_data:
        if meta.get('type') == 'time_signature':
            # midiutil不直接支持添加拍号，但可以通过其他方式处理
            pass
    
    # 添加音符
    channel = 0
    for note in trimmed_notes:
        pitch = note['pitch']
        start_time = note['start_time']
        duration = note['duration']
        velocity = note.get('velocity', 64)
        
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

def split_midi_by_time(notes_data: List, meta_data: List, time_segment_length: float = 8.0) -> List[Tuple[List, List]]:
    """
    按照时间长度将MIDI数据切分成多个片段
    
    Args:
        notes_data: 音符数据列表
        meta_data: 元数据列表
        time_segment_length: 每个时间片段的长度（四分音符为单位）
        
    Returns:
        切分后的MIDI片段列表，每个元素是(音符数据, 元数据)的元组
    """
    segments = []
    
    if not notes_data:
        return segments
    
    # 按开始时间排序音符数据
    sorted_notes = sorted(notes_data, key=lambda x: x['start_time'])
    
    # 获取时间范围
    min_time = sorted_notes[0]['start_time']
    max_time = max(note['start_time'] + note['duration'] for note in sorted_notes)
    
    # 按时间切分
    current_time = min_time
    while current_time < max_time:
        next_time = current_time + time_segment_length
        segment_notes = [note for note in sorted_notes 
                        if note['start_time'] >= current_time and note['start_time'] < next_time]
        
        # 只有当片段不为空时才添加
        if segment_notes:
            # 为每个片段添加元数据
            segment = (segment_notes, meta_data)
            segments.append(segment)
        
        current_time = next_time
    
    return segments

def process_midi_files(input_dir: str = "../midi", output_dir: str = "../splited_midi", 
                      time_segment_length: float = 8.0) -> None:
    """
    处理目录中的所有MIDI文件，将它们切分成小片段
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        time_segment_length: 时间切分时每段的长度（四分音符为单位）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MIDI文件
    midi_files = glob.glob(os.path.join(input_dir, "*.mid"))
    
    if not midi_files:
        print(f"在目录 {input_dir} 中未找到MIDI文件")
        return
    
    print(f"找到 {len(midi_files)} 个MIDI文件")
    
    total_segments = 0
    
    for file_path in midi_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"处理文件: {file_name}")
        
        try:
            # 加载MIDI文件
            score = load_midi_file(file_path)
            if score is None:
                print(f"  无法加载文件: {file_path}")
                continue
            
            # 提取音符和元数据
            notes_data, meta_data = extract_notes_and_meta(score)
            
            print(f"  原始音符数量: {len(notes_data)}")
            print(f"  元数据数量: {len(meta_data)}")
            
            # 切分MIDI数据
            segments = split_midi_by_time(notes_data, meta_data, time_segment_length)
            print(f"  按时间切分 ({time_segment_length}四分音符每段): {len(segments)} 个片段")
            
            # 保存切分后的片段
            for i, (segment_notes, segment_meta) in enumerate(segments):
                output_file = os.path.join(output_dir, f"{file_name}_part_{i+1:03d}.mid")
                success = save_as_standard_midi(segment_notes, segment_meta, output_file)
                if not success:
                    print(f"  保存MIDI文件失败: {output_file}")
            
            total_segments += len(segments)
            
        except Exception as e:
            print(f"  处理文件 {file_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"总共生成 {total_segments} 个片段，保存在 {output_dir} 目录中")

def main():
    """
    主函数
    """
    print("MIDI文件切分工具")
    print("=" * 30)
    
    # 检查依赖
    if not MIDIUTIL_AVAILABLE:
        print("错误: 需要安装midiutil库")
        print("请运行: pip install midiutil")
        return
    
    if not MUSIC21_AVAILABLE:
        print("错误: 需要安装music21库")
        print("请运行: pip install music21")
        return
    
    # 按时间切分（默认8个四分音符每段，相当于2小节4/4拍）
    print("按时间切分 (8个四分音符每段):")
    process_midi_files(time_segment_length=8.0)

if __name__ == "__main__":
    main()