import torch
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import MaskedMultiHeadAttention

class TestMaskedMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.d_model = 256
        self.num_heads = 8
        self.head_dim = self.d_model // self.num_heads
        self.batch_size = 4
        self.seq_len = 16
        
        # 初始化MaskedMultiHeadAttention模块
        self.attention = MaskedMultiHeadAttention(self.d_model, self.num_heads)
        
        # 创建测试输入
        self.query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_attention_initialization(self):
        """测试MaskedMultiHeadAttention类的初始化"""
        # 检查属性是否正确初始化
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.head_dim, self.head_dim)
        
        # 检查线性变换层是否正确初始化
        self.assertIsInstance(self.attention.w_Q, torch.nn.Linear)
        self.assertIsInstance(self.attention.w_K, torch.nn.Linear)
        self.assertIsInstance(self.attention.w_V, torch.nn.Linear)
        self.assertIsInstance(self.attention.fc_out, torch.nn.Linear)
        
        # 检查线性层的输入输出维度
        self.assertEqual(self.attention.w_Q.in_features, self.d_model)
        self.assertEqual(self.attention.w_Q.out_features, self.d_model)
        self.assertEqual(self.attention.w_K.in_features, self.d_model)
        self.assertEqual(self.attention.w_K.out_features, self.d_model)
        self.assertEqual(self.attention.w_V.in_features, self.d_model)
        self.assertEqual(self.attention.w_V.out_features, self.d_model)
        self.assertEqual(self.attention.fc_out.in_features, self.d_model)
        self.assertEqual(self.attention.fc_out.out_features, self.d_model)
        
    def test_forward_pass_without_mask(self):
        """测试不带掩码的前向传播"""
        # 执行前向传播
        output = self.attention.forward(self.query, self.key, self.value)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_forward_pass_with_mask(self):
        """测试带掩码的前向传播"""
        # 创建注意力掩码 (batch_size, 1, 1, seq_len)
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len)
        # 将部分位置设为0，模拟掩码效果
        mask[:, :, :, self.seq_len//2:] = 0
        
        # 执行前向传播
        output = self.attention.forward(self.query, self.key, self.value, mask)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_different_sequence_lengths(self):
        """测试不同序列长度的输入"""
        # 测试较短序列
        short_seq_len = 8
        short_query = torch.randn(self.batch_size, short_seq_len, self.d_model)
        short_key = torch.randn(self.batch_size, short_seq_len, self.d_model)
        short_value = torch.randn(self.batch_size, short_seq_len, self.d_model)
        
        output = self.attention.forward(short_query, short_key, short_value)
        self.assertEqual(output.shape, (self.batch_size, short_seq_len, self.d_model))
        
        # 测试较长序列
        long_seq_len = 32
        long_query = torch.randn(self.batch_size, long_seq_len, self.d_model)
        long_key = torch.randn(self.batch_size, long_seq_len, self.d_model)
        long_value = torch.randn(self.batch_size, long_seq_len, self.d_model)
        
        output = self.attention.forward(long_query, long_key, long_value)
        self.assertEqual(output.shape, (self.batch_size, long_seq_len, self.d_model))
        
    def test_multihead_attention_properties(self):
        """测试多头注意力的特性"""
        # 确保查询、键、值相同时的行为正确（自注意力）
        self_query = self.query
        self_key = self.query
        self_value = self.query
        
        output = self.attention.forward(self_query, self_key, self_value)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_head_dim_calculation(self):
        """测试头维度计算的正确性"""
        # 测试d_model能被num_heads整除的情况
        attention = MaskedMultiHeadAttention(d_model=256, num_heads=8)
        self.assertEqual(attention.head_dim, 32)
        
        attention = MaskedMultiHeadAttention(d_model=128, num_heads=4)
        self.assertEqual(attention.head_dim, 32)
        
    def test_assertion_error_on_invalid_dimensions(self):
        """测试在d_model不能被num_heads整除时是否抛出断言错误"""
        # 这个测试应该触发断言错误
        with self.assertRaises(AssertionError):
            MaskedMultiHeadAttention(d_model=100, num_heads=3)  # 100不能被3整除

if __name__ == '__main__':
    unittest.main()