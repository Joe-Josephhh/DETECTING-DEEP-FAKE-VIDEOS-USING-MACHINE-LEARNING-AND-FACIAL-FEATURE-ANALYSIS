import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, stride=1, drop_prob=0.2):
        super().__init__()
        hidden_dim = in_ch * expansion
        self.use_res = (in_ch == out_ch and stride == 1)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = self.dropout(out)
            return x + out
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, drop_prob=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_attn = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.dropout(x_attn)
        x_flat = x_flat + self.dropout(self.mlp(self.norm2(x_flat)))
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return x


class CoAtNet(nn.Module):
    """
    Hybrid Convolution + Self-Attention Neural Network for Deepfake Detection.
    Combines spatial feature extraction from MBConv with global context modeling
    via Multi-Head Self-Attention.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stage1 = ConvBlock(3, 64, stride=2)
        self.stage2 = nn.Sequential(MBConv(64, 128, stride=2), MBConv(128, 128))
        self.stage3 = nn.Sequential(MBConv(128, 256, stride=2), MBConv(256, 256))
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=1),
            TransformerBlock(512, heads=8),
            TransformerBlock(512, heads=8)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


def load_coatnet(weights_path=None, device="cpu"):
    """
    Instantiates CoAtNet and optionally loads weights from file.
    """
    model = CoAtNet(num_classes=2)
    has_weights = False
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            has_weights = True
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
    model.to(device)
    model.eval()
    return model, has_weights
