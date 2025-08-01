import torch
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embedding import MIDIEmbedding

class TestMIDIEmbedding(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.pitch_range = 128
        self.embed_dim = 258  # 修改为能被3整除的数
        self.embedding = MIDIEmbedding(pitch_range=self.pitch_range, embed_dim=self.embed_dim)
        
    def test_embedding_initialization(self):
        """测试MIDIEmbedding类的初始化"""
        # 检查嵌入层是否正确初始化
        self.assertIsInstance(self.embedding.pit_embedding, torch.nn.Embedding)
        self.assertEqual(self.embedding.pit_embedding.num_embeddings, self.pitch_range)
        self.assertEqual(self.embedding.pit_embedding.embedding_dim, self.embed_dim // 3)
        
        # 检查持续时间嵌入层是否正确初始化
        self.assertIsInstance(self.embedding.dur_embedding, torch.nn.Sequential)
        self.assertEqual(len(self.embedding.dur_embedding), 3)  # Linear -> ReLU -> Linear
        
        # 检查速度嵌入层是否正确初始化
        self.assertIsInstance(self.embedding.vel_embbeding, torch.nn.Sequential)
        self.assertEqual(len(self.embedding.vel_embbeding), 3)  # Linear -> ReLU -> Linear
        
        # 检查位置编码是否正确初始化
        self.assertIsInstance(self.embedding.positional_encoding, torch.Tensor)
        
    def test_positional_encoding(self):
        """测试位置编码函数"""
        d_model = 258
        max_len = 100
        pe = self.embedding.position_encoding(d_model, max_len)
        
        # 检查输出形状
        self.assertEqual(pe.shape, (max_len, d_model))
        
        # 检查输出类型
        self.assertIsInstance(pe, torch.Tensor)
        
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 32
        seq_len = 64
        
        # 创建测试输入
        input_pit = torch.randint(0, self.pitch_range, (batch_size, seq_len))
        input_dur = torch.rand(batch_size, seq_len, 1)
        input_vel = torch.rand(batch_size, seq_len, 1)
        
        # 执行前向传播
        output = self.embedding(input_pit, input_dur, input_vel)
        
        # 检查输出形状
        expected_dim = self.embed_dim  # (embed_dim//3) * 3 = embed_dim
        self.assertEqual(output.shape, (batch_size, seq_len, expected_dim))
        
    def test_embedding_output_range(self):
        """测试嵌入输出值在合理范围内"""
        batch_size = 16
        seq_len = 32
        
        # 创建测试输入
        input_pit = torch.randint(0, self.pitch_range, (batch_size, seq_len))
        input_dur = torch.rand(batch_size, seq_len, 1) * 10  # 持续时间在0-10之间
        input_vel = torch.rand(batch_size, seq_len, 1) * 127  # 速度在0-127之间
        
        # 执行前向传播
        output = self.embedding(input_pit, input_dur, input_vel)
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_embedding_with_edge_cases(self):
        """测试边界情况"""
        # 测试最小输入
        batch_size = 1
        seq_len = 1
        
        input_pit = torch.zeros(batch_size, seq_len, dtype=torch.long)  # 最小音高
        input_dur = torch.zeros(batch_size, seq_len, 1)  # 最小持续时间
        input_vel = torch.zeros(batch_size, seq_len, 1)  # 最小速度
        
        output = self.embedding(input_pit, input_dur, input_vel)
        self.assertEqual(output.shape, (batch_size, seq_len, self.embed_dim))
        
        # 测试最大音高输入
        input_pit = torch.full((batch_size, seq_len), self.pitch_range - 1, dtype=torch.long)  # 最大音高
        input_dur = torch.ones(batch_size, seq_len, 1) * 100  # 较大持续时间
        input_vel = torch.ones(batch_size, seq_len, 1) * 127  # 最大速度
        
        output = self.embedding(input_pit, input_dur, input_vel)
        self.assertEqual(output.shape, (batch_size, seq_len, self.embed_dim))

if __name__ == '__main__':
    unittest.main()