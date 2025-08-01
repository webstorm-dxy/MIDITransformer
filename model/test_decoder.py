import torch
import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import Decoder, DecoderLayer

class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.d_model = 256
        self.num_heads = 8
        self.dropout = 0.1
        self.batch_size = 4
        self.src_seq_len = 16
        self.tgt_seq_len = 12
        
        # 初始化DecoderLayer模块
        self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.dropout)
        
        # 创建测试输入
        self.tgt_input = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        self.encoder_output = torch.randn(self.batch_size, self.src_seq_len, self.d_model)
        
        # 创建掩码
        self.src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_len)
        self.tgt_mask = torch.ones(self.batch_size, 1, self.tgt_seq_len, self.tgt_seq_len)
        
    def test_decoder_layer_initialization(self):
        """测试DecoderLayer类的初始化"""
        # 检查属性是否正确初始化
        self.assertIsInstance(self.decoder_layer.masked_attention, torch.nn.Module)
        self.assertIsInstance(self.decoder_layer.enc_dec_attention, torch.nn.Module)
        self.assertIsInstance(self.decoder_layer.feed_forward, torch.nn.Sequential)
        self.assertIsInstance(self.decoder_layer.add_norm1, torch.nn.Module)
        self.assertIsInstance(self.decoder_layer.add_norm2, torch.nn.Module)
        self.assertIsInstance(self.decoder_layer.add_norm3, torch.nn.Module)
        
    def test_forward_pass_without_mask(self):
        """测试不带掩码的前向传播"""
        # 执行前向传播
        output = self.decoder_layer.forward(self.tgt_input, self.encoder_output)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_forward_pass_with_mask(self):
        """测试带掩码的前向传播"""
        # 执行前向传播
        output = self.decoder_layer.forward(
            self.tgt_input, 
            self.encoder_output, 
            self.src_mask, 
            self.tgt_mask
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_different_sequence_lengths(self):
        """测试不同序列长度的输入"""
        # 测试较短的目标序列
        short_tgt_seq_len = 8
        short_tgt_input = torch.randn(self.batch_size, short_tgt_seq_len, self.d_model)
        
        output = self.decoder_layer.forward(short_tgt_input, self.encoder_output)
        self.assertEqual(output.shape, (self.batch_size, short_tgt_seq_len, self.d_model))
        
        # 测试较长的目标序列
        long_tgt_seq_len = 20
        long_tgt_input = torch.randn(self.batch_size, long_tgt_seq_len, self.d_model)
        
        output = self.decoder_layer.forward(long_tgt_input, self.encoder_output)
        self.assertEqual(output.shape, (self.batch_size, long_tgt_seq_len, self.d_model))

class TestDecoder(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 6
        self.dropout = 0.1
        self.batch_size = 4
        self.src_seq_len = 16
        self.tgt_seq_len = 12
        
        # 初始化Decoder模块
        self.decoder = Decoder(self.d_model, self.num_heads, self.num_layers, self.dropout)
        
        # 创建测试输入
        self.tgt_input = torch.randn(self.batch_size, self.tgt_seq_len, self.d_model)
        self.encoder_output = torch.randn(self.batch_size, self.src_seq_len, self.d_model)
        
        # 创建掩码
        self.src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_len)
        self.tgt_mask = torch.ones(self.batch_size, 1, self.tgt_seq_len, self.tgt_seq_len)
        
    def test_decoder_initialization(self):
        """测试Decoder类的初始化"""
        # 检查属性是否正确初始化
        self.assertEqual(self.decoder.d_model, self.d_model)
        self.assertEqual(self.decoder.num_heads, self.num_heads)
        self.assertEqual(self.decoder.num_layers, self.num_layers)
        
        # 检查decoder_layers是否正确初始化
        self.assertIsInstance(self.decoder.decoder_layers, torch.nn.ModuleList)
        self.assertEqual(len(self.decoder.decoder_layers), self.num_layers)
        
        # 检查每个decoder_layer是否正确初始化
        for layer in self.decoder.decoder_layers:
            self.assertIsInstance(layer, torch.nn.Module)
            
        # 检查norm层是否正确初始化
        self.assertIsInstance(self.decoder.norm, torch.nn.LayerNorm)
        
    def test_forward_pass_without_mask(self):
        """测试不带掩码的前向传播"""
        # 执行前向传播
        output = self.decoder.forward(self.tgt_input, self.encoder_output)
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_forward_pass_with_mask(self):
        """测试带掩码的前向传播"""
        # 执行前向传播
        output = self.decoder.forward(
            self.tgt_input, 
            self.encoder_output, 
            self.src_mask, 
            self.tgt_mask
        )
        
        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        
        # 检查输出不包含NaN或无穷大值
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_different_number_of_layers(self):
        """测试不同层数的解码器"""
        # 测试较少层数
        few_layers = 2
        decoder_few = Decoder(self.d_model, self.num_heads, few_layers, self.dropout)
        
        output = decoder_few.forward(self.tgt_input, self.encoder_output)
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))
        
        # 测试较多层数
        many_layers = 12
        decoder_many = Decoder(self.d_model, self.num_heads, many_layers, self.dropout)
        
        output = decoder_many.forward(self.tgt_input, self.encoder_output)
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.d_model))

if __name__ == '__main__':
    unittest.main()