import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import glob
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np  # 添加numpy用于数值检查
import math  # 添加math模块用于数学运算检查
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dataset import MIDIDataset, collate_fn
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
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重以提高数值稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用较小的标准差初始化线性层权重
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                # 使用较小的标准差初始化嵌入层权重
                nn.init.normal_(m.weight, mean=0, std=0.01)

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
        # 检查输入数据中是否有NaN或inf值
        if torch.isnan(input_pitches).any() or torch.isnan(input_durations).any() or torch.isnan(input_velocities).any():
            print("Warning: NaN values found in model input")
            # 替换NaN值为0
            input_pitches = torch.nan_to_num(input_pitches, nan=0.0)
            input_durations = torch.nan_to_num(input_durations, nan=0.0)
            input_velocities = torch.nan_to_num(input_velocities, nan=0.0)
        
        if torch.isinf(input_pitches).any() or torch.isinf(input_durations).any() or torch.isinf(input_velocities).any():
            print("Warning: Inf values found in model input")
            # 替换inf值为合理的数值
            input_pitches = torch.clamp(input_pitches, min=-1e4, max=1e4)
            input_durations = torch.clamp(input_durations, min=-1e4, max=1e4)
            input_velocities = torch.clamp(input_velocities, min=-1e4, max=1e4)
        
        # 确保输入在合理范围内
        # 音高应该在[0, pitch_range-1]范围内
        input_pitches = torch.clamp(input_pitches, min=0, max=127).long()
        
        # 持续时间和力度应该为正数
        epsilon = 1e-8
        input_durations = torch.clamp(input_durations, min=epsilon, max=1e4)
        input_velocities = torch.clamp(input_velocities, min=epsilon, max=127)
        
        # 嵌入输入
        embedded = self.embedding(input_pitches, input_durations, input_velocities)
        
        # 检查嵌入输出是否有NaN或inf
        if torch.isnan(embedded).any() or torch.isinf(embedded).any():
            print("Warning: NaN or Inf values found in embedding output")
            embedded = torch.nan_to_num(embedded, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 创建目标序列掩码（用于解码器的自注意力）
        if target_pitches is not None:
            tgt_seq_len = target_pitches.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=embedded.device)).bool()
        else:
            tgt_seq_len = input_pitches.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=embedded.device)).bool()
        
        # 解码器处理
        decoder_output = self.decoder(embedded, embedded, tgt_mask=tgt_mask)
        
        # 检查解码器输出是否有NaN或inf
        if torch.isnan(decoder_output).any() or torch.isinf(decoder_output).any():
            print("Warning: NaN or Inf values found in decoder output")
            decoder_output = torch.nan_to_num(decoder_output, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 预测输出
        pitch_output = self.pitch_head(decoder_output)
        duration_output = self.duration_head(decoder_output)
        velocity_output = self.velocity_head(decoder_output)
        
        # 检查预测输出是否有NaN或inf
        if torch.isnan(pitch_output).any() or torch.isinf(pitch_output).any():
            print("Warning: NaN or Inf values found in pitch output")
            pitch_output = torch.nan_to_num(pitch_output, nan=0.0, posinf=1e4, neginf=-1e4)
            
        if torch.isnan(duration_output).any() or torch.isinf(duration_output).any():
            print("Warning: NaN or Inf values found in duration output")
            duration_output = torch.nan_to_num(duration_output, nan=0.0, posinf=1e4, neginf=-1e4)
            
        if torch.isnan(velocity_output).any() or torch.isinf(velocity_output).any():
            print("Warning: NaN or Inf values found in velocity output")
            velocity_output = torch.nan_to_num(velocity_output, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return pitch_output, duration_output, velocity_output


def main():
    """
    主函数，用于启动训练
    """
    # 检查是否可以使用分布式训练
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using distributed training with {torch.cuda.device_count()} GPUs")
        # 使用多GPU分布式训练
        mp.spawn(train_distributed, nprocs=torch.cuda.device_count(), join=True)
    else:
        # 使用单GPU或CPU训练
        train_model()


def setup_distributed(rank, world_size):
    """
    设置分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()


def train_distributed(rank, world_size):
    """
    分布式训练函数
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
    """
    # 设置分布式训练环境
    setup_distributed(rank, world_size)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    print(f"Process {rank} using device: {device}")
    
    # 创建TensorBoard日志写入器 (仅主进程)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir='runs/midi_transformer')
    
    # 超参数
    pitch_range = 128  # MIDI音符范围(0-127)，表示模型需要预测的音高类别数
    embed_dim = 256    # 嵌入维度，控制模型中各层的特征表示大小
    num_heads = 16     # 多头注意力机制中的头数，用于并行处理不同方面的信息
    num_layers = 8     # Transformer编码器层的数量，影响模型的深度
    batch_size = 32    # 每个训练批次的样本数量，影响内存使用和训练稳定性
    learning_rate = 0.0001  # 降低学习率以提高训练稳定性
    num_epochs = 100000    # 训练轮数，控制模型训练的总时长
    sequence_length = 256   # 序列长度，用作上下文长度，表示模型一次处理的音符数量
    num_workers = min(4, os.cpu_count())  # 数据加载时使用的线程数，根据CPU核心数动态调整
    
    # 获取所有JSON数据文件
    data_files = glob.glob("data/*.json")
    if not data_files:
        raise ValueError("No JSON files found in data/ directory")
    
    # 创建数据集和数据加载器
    datasets = []
    total_sequences = 0
    for file in data_files:
        try:
            dataset = MIDIDataset(file, sequence_length)
            dataset_size = len(dataset)
            if rank == 0:
                print(f"Loaded {dataset_size} sequences from {file}")
            if dataset_size > 0:
                datasets.append(dataset)
                total_sequences += dataset_size
            else:
                if rank == 0:
                    print(f"Warning: No sequences loaded from {file}")
        except Exception as e:
            if rank == 0:
                print(f"Error loading {file}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets loaded. Check your data files.")
    
    if total_sequences == 0:
        raise ValueError("No sequences found in any dataset. Check your data files have enough notes.")
    
    if rank == 0:
        print(f"Total sequences across all datasets: {total_sequences}")
    
    # 合并所有数据集
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # 创建分布式数据采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        combined_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # 优化数据加载器设置以提高GPU利用率
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 使用sampler时必须设置为False
        collate_fn=collate_fn, 
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,  # 保持工作进程活跃以提高效率
        prefetch_factor=2  # 每个工作进程预取2个批次
    )
    
    # 创建模型
    model = MIDITransformer(pitch_range, embed_dim, num_heads, num_layers).to(device)
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank])
    
    # 损失函数和优化器
    pitch_criterion = nn.CrossEntropyLoss()
    duration_criterion = nn.MSELoss()
    velocity_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 添加梯度裁剪以防止梯度爆炸
    gradient_clip = 1.0
    
    # 使用自动混合精度训练以提高训练速度
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练循环
    model.train()
    total_steps = 0
    
    try:
        for epoch in range(num_epochs):
            # 设置sampler的epoch以确保每个epoch的shuffle不同
            sampler.set_epoch(epoch)
            
            total_loss = 0
            num_batches = 0
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                # 解包输入和目标
                input_pitches, input_durations, input_velocities = input_seq
                target_pitches, target_durations, target_velocities = target_seq
                
                # 检查输入数据中是否有NaN或inf值
                if torch.isnan(input_pitches).any() or torch.isnan(input_durations).any() or torch.isnan(input_velocities).any():
                    if rank == 0:
                        print(f"Warning: NaN values found in input data at epoch {epoch}, batch {batch_idx}")
                    continue  # 跳过这个批次
                
                if torch.isinf(input_pitches).any() or torch.isinf(input_durations).any() or torch.isinf(input_velocities).any():
                    if rank == 0:
                        print(f"Warning: Inf values found in input data at epoch {epoch}, batch {batch_idx}")
                    continue  # 跳过这个批次
                
                # 移动到设备
                input_pitches = input_pitches.to(device, non_blocking=True)
                input_durations = input_durations.to(device, non_blocking=True)
                input_velocities = input_velocities.to(device, non_blocking=True)
                target_pitches = target_pitches.to(device, non_blocking=True)
                target_durations = target_durations.to(device, non_blocking=True)
                target_velocities = target_velocities.to(device, non_blocking=True)
                
                # 前向传播和反向传播使用自动混合精度
                optimizer.zero_grad()
                
                # 使用自动混合精度训练
                with torch.amp.autocast('cuda'):
                    try:
                        pitch_output, duration_output, velocity_output = model(
                            input_pitches, input_durations, input_velocities, target_pitches
                        )
                    except Exception as e:
                        if rank == 0:
                            print(f"Error in forward pass for batch {batch_idx}: {e}")
                        raise
                    
                    # 检查模型输出中是否有NaN或inf值
                    if torch.isnan(pitch_output).any() or torch.isnan(duration_output).any() or torch.isnan(velocity_output).any():
                        if rank == 0:
                            print(f"Warning: NaN values found in model output at epoch {epoch}, batch {batch_idx}")
                        continue  # 跳过这个批次
                    
                    if torch.isinf(pitch_output).any() or torch.isinf(duration_output).any() or torch.isinf(velocity_output).any():
                        if rank == 0:
                            print(f"Warning: Inf values found in model output at epoch {epoch}, batch {batch_idx}")
                        continue  # 跳过这个批次
                    
                    # 计算损失
                    # 音高损失 (分类任务)
                    try:
                        # 确保输入张量维度正确
                        pitch_output_reshaped = pitch_output.reshape(-1, pitch_range)
                        target_pitches_reshaped = target_pitches.reshape(-1)
                        
                        # 检查目标值是否在有效范围内
                        if torch.any(target_pitches_reshaped < 0) or torch.any(target_pitches_reshaped >= pitch_range):
                            if rank == 0:
                                print(f"Warning: Target pitch values out of range [0, {pitch_range-1}] at epoch {epoch}, batch {batch_idx}")
                                print(f"Target pitch min: {target_pitches_reshaped.min()}, max: {target_pitches_reshaped.max()}")
                            continue  # 跳过这个批次
                        
                        pitch_loss = pitch_criterion(pitch_output_reshaped, target_pitches_reshaped)
                    except Exception as e:
                        if rank == 0:
                            print(f"Error calculating pitch loss for batch {batch_idx}: {e}")
                            print(f"Pitch output shape: {pitch_output.shape}")
                            print(f"Target pitches shape: {target_pitches.shape}")
                        raise
                    
                    # 持续时间和力度损失 (回归任务)
                    try:
                        duration_loss = duration_criterion(duration_output.squeeze(), target_durations.squeeze())
                        velocity_loss = velocity_criterion(velocity_output.squeeze(), target_velocities.squeeze())
                    except Exception as e:
                        if rank == 0:
                            print(f"Error calculating regression loss for batch {batch_idx}: {e}")
                        raise
                    
                    # 总损失
                    loss = pitch_loss + duration_loss + velocity_loss
                
                # 检查损失是否为NaN或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        print(f"Warning: Invalid loss (NaN or Inf) at epoch {epoch}, batch {batch_idx}")
                        print(f"Loss: {loss}, Pitch Loss: {pitch_loss}, Duration Loss: {duration_loss}, Velocity Loss: {velocity_loss}")
                    continue  # 跳过这个批次
                
                # 反向传播和优化
                scaler.scale(loss).backward()
                
                # 梯度裁剪防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 记录到TensorBoard (仅主进程)
                if rank == 0 and total_steps % 1000 == 0 and total_steps > 1000:
                    # 打印到控制台
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], '
                          f'Loss: {loss.item():.4f}, Pitch Loss: {pitch_loss.item():.4f}, '
                          f'Duration Loss: {duration_loss.item():.4f}, Velocity Loss: {velocity_loss.item():.4f}')
                    
                    # 写入TensorBoard
                    if writer is not None:
                        writer.add_scalar('Training/Loss', loss.item(), total_steps)
                        writer.add_scalar('Training/Pitch_Loss', pitch_loss.item(), total_steps)
                        writer.add_scalar('Training/Duration_Loss', duration_loss.item(), total_steps)
                        writer.add_scalar('Training/Velocity_Loss', velocity_loss.item(), total_steps)
                        writer.add_scalar('Training/Learning_Rate', learning_rate, total_steps)
                
                total_steps += 1
            
            # 打印每轮平均损失 (仅主进程)
            if rank == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                # print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
                
                # 记录每个epoch的平均损失到TensorBoard
                if writer is not None:
                    writer.add_scalar('Training/Epoch_Average_Loss', avg_loss, epoch)
                
                # 每10轮保存一次模型
                if (epoch + 1) % 1000 == 0 and epoch > 1000:
                    torch.save(model.module.state_dict(), f"./pth/midi_transformer_epoch_{epoch+1}.pth")
                    print(f"Model saved at epoch {epoch+1}")
                
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user. Saving final model...")
    
    # 保存最终模型 (仅主进程)
    if rank == 0:
        os.makedirs("./pth", exist_ok=True)  # 确保目录存在
        torch.save(model.module.state_dict(), "./pth/midi_transformer_final.pth")
        print("Training completed. Final model saved.")
        
        # 关闭TensorBoard写入器
        if writer is not None:
            writer.close()
    
    # 清理分布式训练环境
    cleanup_distributed()


def train_model():
    """
    训练模型 (单GPU/多线程版本)
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建TensorBoard日志写入器
    writer = SummaryWriter(log_dir='runs/midi_transformer')
    
    # 超参数
    pitch_range = 128  # MIDI音符范围(0-127)，表示模型需要预测的音高类别数
    embed_dim = 256    # 嵌入维度，控制模型中各层的特征表示大小
    num_heads = 16      # 多头注意力机制中的头数，用于并行处理不同方面的信息
    num_layers = 8     # Transformer编码器层的数量，影响模型的深度
    batch_size = 16    # 降低批次大小以提高稳定性
    learning_rate = 0.00005  # 进一步降低学习率以提高训练稳定性
    num_epochs = 100000    # 训练轮数，控制模型训练的总时长
    sequence_length = 256   # 序列长度，用作上下文长度，表示模型一次处理的音符数量
    num_workers = min(4, os.cpu_count())  # 数据加载时使用的线程数，根据CPU核心数动态调整
    
    # 获取所有JSON数据文件
    data_files = glob.glob("data/*.json")
    if not data_files:
        raise ValueError("No JSON files found in data/ directory")
    
    # 创建数据集和数据加载器
    datasets = []
    total_sequences = 0
    for file in data_files:
        try:
            dataset = MIDIDataset(file, sequence_length)
            dataset_size = len(dataset)
            print(f"Loaded {dataset_size} sequences from {file}")
            if dataset_size > 0:
                datasets.append(dataset)
                total_sequences += dataset_size
            else:
                print(f"Warning: No sequences loaded from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets loaded. Check your data files.")
    
    if total_sequences == 0:
        raise ValueError("No sequences found in any dataset. Check your data files have enough notes.")
    
    print(f"Total sequences across all datasets: {total_sequences}")
    
    # 合并所有数据集
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # 优化数据加载器设置以提高GPU利用率
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,  # 保持工作进程活跃以提高效率
        prefetch_factor=2 if num_workers > 0 else None  # 每个工作进程预取2个批次
    )
    
    # 创建模型
    model = MIDITransformer(pitch_range, embed_dim, num_heads, num_layers).to(device)
    
    # 损失函数和优化器
    pitch_criterion = nn.CrossEntropyLoss()
    duration_criterion = nn.MSELoss()
    velocity_criterion = nn.MSELoss()
    # 使用带权重衰减的优化器提高稳定性
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 添加梯度裁剪以防止梯度爆炸
    gradient_clip = 0.5
    
    # 使用自动混合精度训练以提高训练速度
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练循环
    model.train()
    total_steps = 0
    
    try:
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                # 解包输入和目标
                input_pitches, input_durations, input_velocities = input_seq
                target_pitches, target_durations, target_velocities = target_seq
                
                # 检查输入数据中是否有NaN或inf值
                if torch.isnan(input_pitches).any() or torch.isnan(input_durations).any() or torch.isnan(input_velocities).any():
                    print(f"Warning: NaN values found in input data at epoch {epoch}, batch {batch_idx}")
                    continue  # 跳过这个批次
                
                if torch.isinf(input_pitches).any() or torch.isinf(input_durations).any() or torch.isinf(input_velocities).any():
                    print(f"Warning: Inf values found in input data at epoch {epoch}, batch {batch_idx}")
                    continue  # 跳过这个批次
                
                # 移动到设备
                input_pitches = input_pitches.to(device, non_blocking=True)
                input_durations = input_durations.to(device, non_blocking=True)
                input_velocities = input_velocities.to(device, non_blocking=True)
                target_pitches = target_pitches.to(device, non_blocking=True)
                target_durations = target_durations.to(device, non_blocking=True)
                target_velocities = target_velocities.to(device, non_blocking=True)
                
                # 前向传播和反向传播使用自动混合精度
                optimizer.zero_grad()
                
                # 使用自动混合精度训练
                with torch.amp.autocast('cuda'):
                    try:
                        pitch_output, duration_output, velocity_output = model(
                            input_pitches, input_durations, input_velocities, target_pitches
                        )
                    except Exception as e:
                        print(f"Error in forward pass for batch {batch_idx}: {e}")
                        raise
                    
                    # 检查模型输出中是否有NaN或inf值
                    if torch.isnan(pitch_output).any() or torch.isnan(duration_output).any() or torch.isnan(velocity_output).any():
                        print(f"Warning: NaN values found in model output at epoch {epoch}, batch {batch_idx}")
                        continue  # 跳过这个批次
                    
                    if torch.isinf(pitch_output).any() or torch.isinf(duration_output).any() or torch.isinf(velocity_output).any():
                        print(f"Warning: Inf values found in model output at epoch {epoch}, batch {batch_idx}")
                        continue  # 跳过这个批次
                    
                    # 计算损失
                    # 音高损失 (分类任务)
                    try:
                        # 确保输入张量维度正确
                        pitch_output_reshaped = pitch_output.reshape(-1, pitch_range)
                        target_pitches_reshaped = target_pitches.reshape(-1)
                        
                        # 检查目标值是否在有效范围内
                        if torch.any(target_pitches_reshaped < 0) or torch.any(target_pitches_reshaped >= pitch_range):
                            print(f"Warning: Target pitch values out of range [0, {pitch_range-1}] at epoch {epoch}, batch {batch_idx}")
                            print(f"Target pitch min: {target_pitches_reshaped.min()}, max: {target_pitches_reshaped.max()}")
                            continue  # 跳过这个批次
                        
                        pitch_loss = pitch_criterion(pitch_output_reshaped, target_pitches_reshaped)
                    except Exception as e:
                        print(f"Error calculating pitch loss for batch {batch_idx}: {e}")
                        print(f"Pitch output shape: {pitch_output.shape}")
                        print(f"Target pitches shape: {target_pitches.shape}")
                        raise
                    
                    # 持续时间和力度损失 (回归任务)
                    try:
                        duration_loss = duration_criterion(duration_output.squeeze(), target_durations.squeeze())
                        velocity_loss = velocity_criterion(velocity_output.squeeze(), target_velocities.squeeze())
                        
                        # 检查回归损失是否为NaN或inf
                        if torch.isnan(duration_loss) or torch.isinf(duration_loss):
                            print(f"Warning: Invalid duration loss at epoch {epoch}, batch {batch_idx}")
                            continue  # 跳过这个批次
                            
                        if torch.isnan(velocity_loss) or torch.isinf(velocity_loss):
                            print(f"Warning: Invalid velocity loss at epoch {epoch}, batch {batch_idx}")
                            continue  # 跳过这个批次
                    except Exception as e:
                        print(f"Error calculating regression loss for batch {batch_idx}: {e}")
                        raise
                    
                    # 总损失
                    loss = pitch_loss + duration_loss + velocity_loss
                
                # 检查损失是否为NaN或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss (NaN or Inf) at epoch {epoch}, batch {batch_idx}")
                    print(f"Loss: {loss}, Pitch Loss: {pitch_loss}, Duration Loss: {duration_loss}, Velocity Loss: {velocity_loss}")
                    continue  # 跳过这个批次
                
                # 反向传播和优化
                scaler.scale(loss).backward()
                
                # 梯度裁剪防止梯度爆炸
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                
                # 记录到TensorBoard
                # 记录到TensorBoard和控制台
                if total_steps % 1000 == 0 and total_steps > 1000:
                    # 打印到控制台
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], '
                          f'Loss: {loss.item():.4f}, Pitch Loss: {pitch_loss.item():.4f}, '
                          f'Duration Loss: {duration_loss.item():.4f}, Velocity Loss: {velocity_loss.item():.4f}')
                    
                    # 写入TensorBoard
                    writer.add_scalar('Training/Loss', loss.item(), total_steps)
                    writer.add_scalar('Training/Pitch_Loss', pitch_loss.item(), total_steps)
                    writer.add_scalar('Training/Duration_Loss', duration_loss.item(), total_steps)
                    writer.add_scalar('Training/Velocity_Loss', velocity_loss.item(), total_steps)
                    writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], total_steps)
                
                total_steps += 1
            
            # 打印每轮平均损失
            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
            
            # 更新学习率调度器
            scheduler.step(avg_loss)
            
            # 记录每个epoch的平均损失到TensorBoard
            writer.add_scalar('Training/Epoch_Average_Loss', avg_loss, epoch)
            
            # 每10轮保存一次模型
            if (epoch + 1) % 1000 == 0 and epoch > 1000:
                torch.save(model.state_dict(), f"./pth/midi_transformer_epoch_{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
    
    # 保存最终模型
    os.makedirs("./pth", exist_ok=True)  # 确保目录存在
    torch.save(model.state_dict(), "./pth/midi_transformer_final.pth")
    print("Training completed. Final model saved.")
    
    # 关闭TensorBoard写入器
    writer.close()


if __name__ == "__main__":
    main()