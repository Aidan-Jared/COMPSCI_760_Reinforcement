import torch
from torch import nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding = 0):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernal_size, stride=stride, padding=padding, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    
    def forward(self, x):
        channel_attn_sum = 0
        for i in self.pool_types:
            if i == 'avg':
                pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            else:  
                pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            pool = pool.view(pool.size(0), -1)
            channel_attn = self.mlp(pool)
            channel_attn_sum = channel_attn_sum + channel_attn
        scale = F.sigmoid(channel_attn_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
    
class SpatialGate(nn.Module):
    def __init__(self, in_planes, out_planes, kernal_size, stride, padding = 0):
        super().__init__()
        self.spatial = BasicConv(in_planes // 2, out_planes, kernal_size, stride,  padding=(kernal_size-stride) // 2)

    def forward(self, x):
        x_compress = torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, kernal,stride, padding):
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels,  reduction_ratio)
        self.spatial_gate = SpatialGate(gate_channels, gate_channels, kernal, stride, padding)
    
    def forward(self, x):
        x_out = self.channel_gate(x)
        return self.spatial_gate(x_out)