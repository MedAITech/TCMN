"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.network import Decoder,Decoder2,MyLinear
from model.mymodel import SpatialOCRNet
from model.att_model import GuideDecoderLayer
from thop import profile


class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder() 

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()
        # self.new_decoder = SpatialOCRNet()
        self.my_linear = MyLinear()
        # self.decoder2 = Decoder2()
        # self.att1 = GuideDecoderLayer(512,2,2)
        # self.att2 = GuideDecoderLayer(512,2,2)
        # self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)

    def encode_value(self, frame, kf16, masks): 
        k, _, h, w = masks.shape
        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask
        if k != 1:
            others = torch.cat([
                torch.sum(
                    masks[[j for j in range(k) if i!=j]]
                , dim=0, keepdim=True)
            for i in range(k)], 0)
        else:
            others = torch.zeros_like(masks)

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks, others)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16):
        k = mem_bank.num_objects

        # readout_mem = mem_bank.match_memory(qk16)
        # qv16 = self.new_decoder(qv16,torch.cat([readout_mem, qv16], 1))

        # readout_mem = mem_bank.match_memory3(qk16, qv16, self.my_linear)
        # flops, params = profile(mem_bank,inputs=(qk16, qv16, self.my_linear))
        readout_mem = mem_bank(qk16, qv16, self.my_linear)

        # readout_mem = mem_bank.match_memory5(qk16, qv16, self.my_linear)

        # readout_mem = self.my_linear(qv16,readout_mem)

        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)

        # qv16_change = qv16.flatten(2).transpose(2,1)
        # readout_mem_change = readout_mem.flatten(2).transpose(2,1)
        # q1 = self.att1(qv16_change,readout_mem_change)
        # r1 = self.att2(readout_mem_change,qv16_change)
        # att_out1 = torch.cat([r1, q1], dim=2)
        # att_out1 = att_out1.transpose(2,1).view(att_out1.shape[0],att_out1.shape[-1],readout_mem.shape[-2],readout_mem.shape[-1])
        # qv16 = self.scale*att_out1 + torch.cat([readout_mem, qv16], 1)
        return torch.sigmoid(self.decoder(qv16, qf8, qf4))


class STCNSO(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoderSO()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.decoder = Decoder()

    def encode_value(self, frame, kf16, masks):
        k, _, h, w = masks.shape

        # Extract memory key/value for a frame with multiple masks
        frame = frame.view(1, 3, h, w).repeat(k, 1, 1, 1)
        # Compute the "others" mask

        f16 = self.value_encoder(frame, kf16.repeat(k,1,1,1), masks)

        return f16.unsqueeze(2)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16):
        k = mem_bank.num_objects

        readout_mem = mem_bank.match_memory(qk16)
        qv16 = qv16.expand(k, -1, -1, -1)
        qv16 = torch.cat([readout_mem, qv16], 1)

        return torch.sigmoid(self.decoder(qv16, qf8, qf4))
