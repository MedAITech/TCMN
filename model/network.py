"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *
from model.mymodel import SpatialOCRNet
from model.att_model import GuideDecoderLayer


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = FeatureFusionBlock(1024,1024)
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4
        # self.up_4_2 = UpsampleBlock(64,256,256)

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4, f16_thin=None):
        # f16 = self.fuse(f16[:,:512],f16[:,512:])
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        # x = self.up_4_2(f2, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        if f16_thin is not None:
            x2 = self.up_16_8(f8, f16_thin)
            x2 = self.up_8_4(f4, x2)
            # x2 = self.up_4_2(f2, x2)
            x2 = self.pred(F.relu(x2))
            x2 = F.interpolate(x2, scale_factor=4, mode='bilinear', align_corners=False)
            return x, x2

        return x

class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4, f16_thin=None):
        # x = self.compress(f16)
        x = self.up_16_8(f8, f16)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        if f16_thin is not None:
            x2 = self.up_16_8(f8, f16_thin)
            x2 = self.up_8_4(f4, x2)
            x2 = self.pred(F.relu(x2))
            x2 = F.interpolate(x2, scale_factor=4, mode='bilinear', align_corners=False)
            return x, x2

        return x

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.layer1 = nn.Linear(64, 512)
        self.layer2 = nn.Linear(512, 1024)
        # self.layer2 = nn.Linear(1024, 1024)
        # self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, k, v):
        H, W = v.shape[-2],v.shape[-1]
        x = self.pool(k)
        x = x.squeeze(-1).squeeze(-1)
        # x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))

        # Train
        # x = x.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
        # x = x.unsqueeze(1).unsqueeze(-3)
        # x = torch.cat((x,x),dim=1)
        # a = x[:,:,:512]
        # b = x[:,:,512:]

        # Train 后线性变化
        # x = x.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
        # a = x[:,:512]
        # b = x[:,512:]

        # Valid
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,H,W)
        a = x[:,:512]
        b = x[:,512:]


        # v = self.scale*(a*v + b)+v  # 残差连接
        v = a*v +b

        return v




class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = self.pool1(mk)
        qk = self.pool2(qk)
        mk = mk.flatten(start_dim=2)  # B*CK*(THW)
        qk = qk.flatten(start_dim=2)  # B*CK*(HW)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)  # B*(THW)*1
        ab = mk.transpose(1, 2) @ qk  # B*(THW)*(HW)

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def get_affinity2(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)  # B*CK*(THW)
        qk = qk.flatten(start_dim=2)  # B*CK*(HW)

        affinity = mk.transpose(1, 2) @ qk   # B, THW, HW

        affinity /= H

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def get_affinity3(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)  # B*CK*(THW)
        qk = qk.flatten(start_dim=2)  # B*CK*(HW)

        affinity = F.cosine_similarity(mk.unsqueeze(3),qk.unsqueeze(2),dim=1)   # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def get_affinity4(self, mk, qk, mv, qv):
        B, CK, T, H, W = mk.shape
        B, CV, T, H, W = mv.shape
        mk = mk.flatten(start_dim=2)  # B*CK*(THW)
        mv = mv.flatten(start_dim=2)  # B*CK*(THW)
        qk = qk.flatten(start_dim=2)  # B*CK*(HW)
        new_mk = torch.zeros_like(mk)
        new_mv = torch.zeros_like(mv)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)  # B*(THW)*1
        ab = mk.transpose(1, 2) @ qk  # B*(THW)*(HW)

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW

        # softmax operation; aligned the evaluation style
        # maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        # x_exp = torch.exp(affinity - maxes)
        # x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        # affinity = x_exp / x_exp_sum

        for i in range(T):
            temp_affinity = affinity[:,i*H*W:(i+1)*H*W]
            temp_mk = mk[:,:,i*H*W:(i+1)*H*W]
            temp_mv = mv[:,:,i*H*W:(i+1)*H*W]
            maxes = torch.max(temp_affinity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(temp_affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            temp_affinity = x_exp / x_exp_sum
            temp_mk = temp_mk @ temp_affinity
            temp_mv = temp_mv @ temp_affinity
            new_mk[:,:,i*H*W:(i+1)*H*W] = temp_mk
            new_mv[:,:,i*H*W:(i+1)*H*W] = temp_mv

        a_sq = new_mk.pow(2).sum(1).unsqueeze(2)  # B*(THW)*1
        ab = new_mk.transpose(1, 2) @ qk  # B*(THW)*(HW)
        new_affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW
        maxes = torch.max(new_affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(new_affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        new_affinity = x_exp / x_exp_sum

        mem = torch.bmm(new_mv, new_affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem, mem_out


        # mk_ = mk

        # return affinity

    def get_affinity5(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = self.pool1(mk)
        qk = self.pool2(qk)
        mk = mk.flatten(start_dim=2)  # B*CK*(THW)
        qk = qk.flatten(start_dim=2)  # B*CK*(HW)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)  # B*(THW)*1
        ab = mk.transpose(1, 2) @ qk  # B*(THW)*(HW)

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        affinity = affinity.squeeze(-1)

        return affinity


    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W)
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem, mem_out


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.my_linear = MyLinear()
        # self.decoder2 = Decoder2()
        # self.new_decoder = SpatialOCRNet()
        # self.att1 = GuideDecoderLayer(512,2,2)
        # self.att2 = GuideDecoderLayer(512,2,2)
        # self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def encode_key(self, frame):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])
        # f2 = f2.view(b, t, *f2.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2) # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, qf16=None, selector=None):
        # q - query, m - memory
        # qv16 is f16_thin above
        # affinity = self.memory.get_affinity5(mk16, qk16)  # 放射变换时计算亲和力
        # affinity = self.memory.get_affinity(mk16, qk16)  # 正常计算亲和力
        mv16 = self.my_linear(qv16,mv16) # mv16 BNCTHW 利用放射变换提取memory

        if self.single_object:
            # logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            # prob = torch.sigmoid(logits)
            print("????????????????????????????single????????????????????????????")
        else:
            # 全部替换film
            # affinity = affinity.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5)
            # mv16 = torch.sum(affinity*mv16,dim=3)
            mv16 = torch.mean(mv16,dim=3) # BNCTHW -> BNCHW
            mem1 = mv16[:,0]
            mem2 = mv16[:,1]
            mem_out1 = torch.cat([mem1, qv16], dim=1)
            mem_out2 = torch.cat([mem2, qv16], dim=1)

            # ------------------------------------------------
            # mem1, mem_out1 = self.memory.readout(affinity, mv16[:,0], qv16)
            # mem2, mem_out2 = self.memory.readout(affinity, mv16[:,1], qv16)

            # Linear Hou
            # mem1, mem_out1 = self.memory.readout(affinity, mv16[:,0], qv16)
            # mem2, mem_out2 = self.memory.readout(affinity, mv16[:,1], qv16)
            # mem_out1 = torch.cat([self.my_linear(qv16,mem1), qv16], dim=1)
            # mem_out2 = torch.cat([self.my_linear(qv16,mem2), qv16], dim=1)

            # att ---------------------------------------------------
            # qv16_change = qv16.flatten(2).transpose(2,1)
            # mem1_change = mem1.flatten(2).transpose(2,1)
            # mem2_change = mem2.flatten(2).transpose(2,1)
            # q1 = self.att1(qv16_change,mem1_change)
            # r1 = self.att2(mem1_change,qv16_change)
            # q2 = self.att1(qv16_change,mem2_change)
            # r2 = self.att2(mem2_change,qv16_change)
            # att_out1 = torch.cat([r1, q1], dim=2)
            # att_out2 = torch.cat([r2, q2], dim=2)
            # att_out1 = att_out1.transpose(2,1).view(att_out1.shape[0],att_out1.shape[-1],24,24)
            # att_out2 = att_out2.transpose(2,1).view(att_out2.shape[0],att_out2.shape[-1],24,24)

            # mem_out1 = self.scale*att_out1 + mem_out1
            # mem_out2 = self.scale*att_out2 + mem_out2
            # mem_out1 = att_out1
            # mem_out2 = att_out2

            # ocr ----------------------------------------------------
            # mem_out1 = self.new_decoder(qv16,mem_out1)
            # mem_out2 = self.new_decoder(qv16,mem_out2)
            # mem1, mem_out1 = self.memory.get_affinity4(mk16, qk16, mv16[:,0], qv16)
            # mem2, mem_out2 = self.memory.get_affinity4(mk16, qk16, mv16[:,1], qv16)

            final_pred1, coarse_pred1 = self.decoder(mem_out1, qf8, qf4, qv16)
            final_pred2, coarse_pred2 = self.decoder(mem_out2, qf8, qf4, qv16)
            # final_pred1 = self.decoder(mem_out1, qf8, qf4)
            # final_pred2 = self.decoder(mem_out2, qf8, qf4)
            # coarse_pred1 = self.decoder(qf16, qf8, qf4)
            # coarse_pred2 = self.decoder(qf16, qf8, qf4)

            # coarse_pred1, final_pred1 = self.decoder(qv16,mem_out1)
            # coarse_pred2, final_pred2 = self.decoder(qv16,mem_out2)


            # final_pred1 = self.decoder(mem1, qf8, qf4)
            # coarse_pred1 = self.decoder2(qv16, qf8, qf4)
            # final_pred2 = self.decoder(mem2, qf8, qf4)
            # coarse_pred2 = self.decoder2(qv16, qf8, qf4)
            final_logits = torch.cat([
                final_pred1,
                final_pred2,
            ], 1)

            coarse_logits = torch.cat([
                coarse_pred1,
                coarse_pred2,
            ], 1)

            final_prob = torch.sigmoid(final_logits)
            coarse_prob = torch.sigmoid(coarse_logits)
            final_prob = final_prob * selector.unsqueeze(2).unsqueeze(2)
            coarse_prob = coarse_prob * selector.unsqueeze(2).unsqueeze(2)

        final_logits = self.aggregate(final_prob)
        coarse_logits = self.aggregate(coarse_prob)
        final_prob = F.softmax(final_logits, dim=1)[:, 1:]
        coarse_prob = F.softmax(coarse_logits, dim=1)[:, 1:]

        return final_logits, final_prob, coarse_logits, coarse_prob

    # def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):
    #     # q - query, m - memory
    #     # qv16 is f16_thin above
    #     affinity = self.memory.get_affinity(mk16, qk16)
    #
    #     if self.single_object:
    #         logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
    #         prob = torch.sigmoid(logits)
    #     else:
    #         logits = torch.cat([
    #             self.decoder(self.memory.readout(affinity, mv16[:,0], qv16), qf8, qf4),
    #             self.decoder(self.memory.readout(affinity, mv16[:,1], qv16), qf8, qf4),
    #         ], 1)
    #
    #         prob = torch.sigmoid(logits)
    #         prob = prob * selector.unsqueeze(2).unsqueeze(2)
    #
    #     logits = self.aggregate(prob)
    #     prob = F.softmax(logits, dim=1)[:, 1:]
    #
    #     return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError


