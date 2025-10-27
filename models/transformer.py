import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
from typing import List, Tuple, Union

import math

#Thay thế LayerNorm => RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        #self.norm = nn.LayerNorm(latent_dim)
        self.norm = RMSNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        # B, T, D
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


# Thay thế class FFN để sử dụng SwigLU thay vì GELU

#class FFN(nn.Module):

    #def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        #super().__init__()
        #self.linear1 = nn.Linear(latent_dim, ffn_dim)
        #self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        #self.activation = nn.SwiGLU()
        #self.dropout = nn.Dropout(dropout)
        #self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    #def forward(self, x, emb):
        """
             x: B, T, D (D=latent_dim)
        """
        #y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        #y = x + self.proj_out(y, emb)
        #return y

        
class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        
        # SwiGLU cần 3 lớp tuyến tính
        hidden_features = int(ffn_dim * 2 / 3) # Kích thước ẩn theo khuyến nghị của paper
        
        self.w1 = nn.Linear(latent_dim, hidden_features, bias=False)
        self.w2 = nn.Linear(latent_dim, hidden_features, bias=False)
        self.activation = nn.SiLU() # SiLU là hàm kích hoạt của Swish
        self.w3 = zero_module(nn.Linear(hidden_features, latent_dim, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        """
            x: B, T, D (D=latent_dim)
        """
        # Tính toán SwiGLU
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.activation(x1) * x2
        
        y = self.w3(self.dropout(hidden))
        y = x + self.proj_out(y, emb)
        return y


class TemporalSelfAttention(nn.Module):

    def __init__(self, n_frames, latent_dim, num_head, dropout, time_embed_dim, output_attention = True):
        super().__init__()
        self.num_head = num_head
        self.output_attention = output_attention
        #self.norm = nn.LayerNorm(latent_dim)
        self.norm = RMSNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.sigma_projection = nn.Linear(latent_dim, num_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        n_frames = n_frames
        self.distances = torch.zeros((n_frames, n_frames)).cuda(0)

        for i in range(n_frames):
            for j in range(n_frames):
                self.distances[i][j] = abs(i - j)

    def forward(self, x, emb):
        """
        x: B, T, D (D=latent_dim)
        """
        B, T, D = x.shape
        H = self.num_head

        ## series-association
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        # B, T, H, D/H
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        scale = 1. / math.sqrt(D/H)
        # B, H, T, T
        scores = torch.einsum('bnhd,bmhd->bhnm', query, key) / math.sqrt(D // H)
        attention = scale * scores
        # B, H, T, T
        series = self.dropout(F.softmax(attention, dim=-1))

        ## prior-association
        sigma = self.sigma_projection(x).view(B, T, H)  # B, T, H
        sigma = sigma.transpose(1, 2)  # B T H ->  B H T
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, T)  # B, H, T, T
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1) # B, H, T, T
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2)).cuda(0) # B, H, T, T

        # B, T, H, D/H
        value = self.value(self.norm(x)).view(B, T, H, -1)

        # Kết hợp series và prior attention
        # alpha 0.3 với epoch = 20 -> 0.85
        # alpha 0.15 với epoch = 10 -> 0.75
        alpha = 0.3 # Đây là một siêu tham số cần tinh chỉnh
        combined_attention = (1 - alpha) * series + alpha * prior

        # B, T, D
        y = torch.einsum('bhnm,bmhd->bnhd', combined_attention, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)

        if self.output_attention:
            return y.contiguous(), series, prior, sigma
        else:
            return y.contiguous(), None

# Thêm lớp này vào đầu file transformer.py
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, context_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        
        #self.norm = nn.LayerNorm(latent_dim)
        self.norm = RMSNorm(latent_dim)
        #self.context_norm = nn.LayerNorm(context_dim)
        self.context_norm = RMSNorm(context_dim)
        
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(context_dim, latent_dim, bias=False)
        self.value = nn.Linear(context_dim, latent_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, context):
        """
        x: B, T, D_latent (Tín hiệu chuyển động)
        context: B, T_context, D_context (Tín hiệu điều kiện)
        """
        B, T, D = x.shape
        H = self.num_head
        
        query = self.query(self.norm(x))  # B, T, D
        key = self.key(self.context_norm(context))      # B, T_context, D
        value = self.value(self.context_norm(context))    # B, T_context, D

        # Reshape for multi-head attention
        query = query.view(B, T, H, D // H).transpose(1, 2)  # B, H, T, D/H
        key = key.view(B, -1, H, D // H).transpose(1, 2)   # B, H, T_context, D/H
        value = value.view(B, -1, H, D // H).transpose(1, 2) # B, H, T_context, D/H

        # Attention scores
        scores = torch.einsum('bhtd,bhsd->bhts', query, key) / math.sqrt(D // H)
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to value
        y = torch.einsum('bhts,bhsd->bhtd', attention_probs, value)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        
        # Output projection
        y = self.proj_out(y)
        return y

class TemporalDiffusionTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 n_frames=7,
                 latent_dim=16,
                 time_embed_dim=16,
                 ffn_dim=32,
                 num_head=4,
                 dropout=0.5,
                 context_dim=256  # Thêm context_dim
                 ):
        super().__init__()
        # 1. Self-Attention Block (không đổi)
        self.sa_block = TemporalSelfAttention(
            n_frames, latent_dim, num_head, dropout, time_embed_dim)
        
        # 2. Cross-Attention Block (MỚI)
        self.ca_block = CrossAttention(
            latent_dim, context_dim, num_head, dropout)
            
        # 3. FFN Block (không đổi)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb, context): # Thêm context vào forward
        # Self-Attention
        x_sa, series, prior, sigma = self.sa_block(x, emb)
        x = x + x_sa # Kết nối residual

        # Cross-Attention
        x_ca = self.ca_block(x, context)
        x = x + x_ca # Kết nối residual

        # FFN
        x_ffn = self.ffn(x, emb)
        x = x + x_ffn # Kết nối residual

        return x, series, prior, sigma


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=7,
                 # --- ĐỀ XUẤT 2: TĂNG SIÊU THAM SỐ ---
                 latent_dim=32,  # Tăng từ 16
                 ff_size=128,    # Tăng từ 32 (thường là 4*latent_dim)
                 num_layers=10,  # Tăng từ 8
                 num_heads=8,    # Giữ nguyên hoặc tăng lên 16
                 # -----------------------------------
                 dropout=0.2,
                 activation="gelu",
                 output_attention = True,
                 device: Union[str, torch.DeviceObjType] = 'cpu',
                 inject_condition: bool = True, # Giữ True để bật cơ chế điều kiện
                 **kargs):
        super().__init__()

        self.input_feats = input_feats
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.device = device
        self.time_embed_dim = latent_dim
        self.inject_condition = inject_condition
        
        # --- MỚI: Định nghĩa chiều của vector điều kiện ---
        self.context_dim = 256
        # -----------------------------------------------

        self.build_model()

    def build_model(self):
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, self.latent_dim))

        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        # Sửa tên cond_embed để rõ ràng hơn
        self.condition_embed = nn.Linear(self.context_dim, self.context_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    n_frames=self.num_frames,
                    latent_dim=self.latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=self.ff_size,
                    num_head=self.num_heads,
                    dropout=self.dropout,
                    context_dim=self.context_dim # Truyền context_dim vào
                )
            )
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, condition_data:torch.Tensor=None):
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        # --- THAY ĐỔI CÁCH XỬ LÝ ĐIỀU KIỆN ---
        context = None
        if self.inject_condition and condition_data is not None:
            # Nhúng và chuẩn bị context cho Cross-Attention
            context = self.condition_embed(condition_data)
            context = context.unsqueeze(1) # Đổi shape từ (B, D_c) -> (B, 1, D_c)
        else:
            # Tạo context giả nếu không có
            context = torch.zeros(B, 1, self.context_dim, device=x.device)
        # Không cộng context vào emb nữa
        # --------------------------------------

        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        
        series_list, prior_list, sigma_list = [], [], []
        # Bỏ kiến trúc skip-connection phức tạp để làm gọn
        for module in self.temporal_decoder_blocks:
            # Truyền context vào mỗi block
            h, series, prior, sigmas = module(h, emb, context)
            if self.output_attention:
                series_list.append(series)
                prior_list.append(prior)
                sigma_list.append(sigmas)

        output = self.out(h).view(B, T, -1).contiguous()
        
        if self.output_attention:
            return output, series_list, prior_list, sigma_list
        return output
