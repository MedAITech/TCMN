import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
# from monai.networks.blocks.unetr_block import UnetrUpBlock


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels:int, output_masks_len:int, mask_embed_dim:int ,input_mask_len:int=20, embed_dim:int=768):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.vis_pos = PositionalEncoding(in_channels)
        self.masks_pos = PositionalEncoding(in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm_masks = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)


    def forward(self,x,sam_masks):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        # txt = self.text_project(txt)

        # sam_masks2 = self.mask_project(sam_masks)

        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention
        sam_masks2 = self.norm_masks(sam_masks)
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                 key=self.masks_pos(sam_masks2),
                                 value=sam_masks2)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis