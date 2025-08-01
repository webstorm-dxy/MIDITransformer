import os
import json
import mido


def extract_pitch_velocity(midi_file_path):
    """
    从MIDI文件中提取音高、力度和长度信息
    
    Args:
        midi_file_path (str): MIDI文件路径
        
    Returns:
        list: 包含音高、力度和长度信息的列表
    """
    # 使用mido读取MIDI文件
    midi_file = mido.MidiFile(midi_file_path)
    
    # 存储音符信息
    notes_data = []
    
    # 用于跟踪当前活动的音符
    active_notes = {}
    
    # 遍历所有track
    for track_idx, track in enumerate(midi_file.tracks):
        # 当前时间（以ticks为单位）
        current_time = 0
        
        # 初始化该轨道的活动音符字典
        active_notes[track_idx] = {}
        
        # 遍历track中的所有消息
        for msg in track:
            # 更新当前时间
            current_time += msg.time
            
            # 检查是否为音符开启消息
            if msg.type == 'note_on' and msg.velocity > 0:
                # 使用(轨道, 音高)作为键存储音符开始时间
                note_key = (track_idx, msg.note)
                active_notes[track_idx][note_key] = {
                    'start_time': current_time,
                    'velocity': msg.velocity
                }
            
            # 检查是否为音符关闭消息
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # 查找对应的音符开启消息
                note_key = (track_idx, msg.note)
                if note_key in active_notes[track_idx]:
                    # 获取音符开始信息
                    note_info = active_notes[track_idx].pop(note_key)
                    start_time = note_info['start_time']
                    velocity = note_info['velocity']
                    
                    # 计算持续时间
                    duration = current_time - start_time
                    
                    # 添加音符信息到结果列表
                    notes_data.append({
                        'track': track_idx,
                        'pitch': msg.note,
                        'velocity': velocity,
                        'start_time': start_time,
                        'end_time': current_time,
                        'duration': duration
                    })
    
    return notes_data


def save_to_json(data, output_path):
    """
    将数据保存为JSON文件
    
    Args:
        data (list): 要保存的数据
        output_path (str): 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_midi_files(input_dir="../splited_midi", output_dir="../data"):
    """
    处理目录中的所有MIDI文件
    
    Args:
        input_dir (str): MIDI文件输入目录
        output_dir (str): JSON文件输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果输入目录不存在，创建示例
    os.makedirs(input_dir, exist_ok=True)
    
    # 遍历输入目录中的所有MIDI文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mid', '.midi')):
            midi_path = os.path.join(input_dir, filename)
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(output_dir, json_filename)
            
            # 提取音高、力度和长度信息
            try:
                notes_data = extract_pitch_velocity(midi_path)
                
                # 保存为JSON文件
                save_to_json(notes_data, json_path)
                print(f"已处理 {filename} 并保存到 {json_path}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")


if __name__ == "__main__":
    # 处理MIDI文件
    process_midi_files()
    
    print("MIDI文件处理完成！")