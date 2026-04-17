import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    """U-Netの基本ブロック：2つの畳み込み層"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ダウンサンプリングブロック：MaxPool + DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """アップサンプリングブロック：UpConv + Concatenation + DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # パディングを調整してサイズを合わせる
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """最終出力層"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Netモデル：2チャネル入力（Q値 + フィルター）→ 1チャネル出力（T値）"""
    
    def __init__(self, n_channels: int = 2, n_classes: int = 1, bilinear: bool = False):
        """
        Args:
            n_channels: 入力チャネル数（Q値 + フィルター = 2）
            n_classes: 出力チャネル数（T値 = 1）
            bilinear: アップサンプリングに双線形補間を使用するか
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # エンコーダー（ダウンサンプリング） - 2回固定
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        
        # デコーダー（アップサンプリング） - 2回固定
        self.up1 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 入力形状: (batch_size, 2, height, width)
        x1 = self.inc(x)      # (batch_size, 64, height, width)
        x2 = self.down1(x1)   # (batch_size, 128, height/2, width/2)
        x3 = self.down2(x2)   # (batch_size, 256, height/4, width/4)
        
        x = self.up1(x3, x2)  # (batch_size, 128, height/2, width/2)
        x = self.up2(x, x1)   # (batch_size, 64, height, width)
        
        logits = self.outc(x) # (batch_size, 1, height, width)
        
        return logits

    def get_model_info(self) -> dict:
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNet',
            'input_channels': self.n_channels,
            'output_channels': self.n_classes,
            'bilinear_upsampling': self.bilinear,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # float32の場合
        }


def create_unet_model(input_channels: int = 2, output_channels: int = 1, bilinear: bool = False) -> UNet:
    """U-Netモデルを作成（2回ダウンサンプリング固定）"""
    model = UNet(n_channels=input_channels, n_classes=output_channels, bilinear=bilinear)
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """モデルのパラメータ数をカウント"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def test_model():
    """モデルのテスト実行"""
    print("U-Netモデルのテスト実行...")
    
    # モデル作成（3回ダウンサンプリング固定）
    model = create_unet_model(input_channels=2, output_channels=1, bilinear=False)
    
    # モデル情報を表示
    info = model.get_model_info()
    print(f"モデル名: {info['model_name']}")
    print(f"入力チャネル数: {info['input_channels']}")
    print(f"出力チャネル数: {info['output_channels']}")
    print(f"ダウンサンプリング回数: 2")
    print(f"双線形アップサンプリング: {info['bilinear_upsampling']}")
    print(f"総パラメータ数: {info['total_parameters']:,}")
    print(f"学習可能パラメータ数: {info['trainable_parameters']:,}")
    print(f"モデルサイズ: {info['model_size_mb']:.2f} MB")
    
    # テストデータでフォワードパスを実行
    batch_size = 2
    height, width = 20, 20  # パッチサイズ
    
    # テスト入力データ（2チャネル）
    x = torch.randn(batch_size, 2, height, width)
    print(f"\n入力形状: {x.shape}")
    
    # フォワードパス
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"出力形状: {output.shape}")
    print(f"出力範囲: min={output.min():.4f}, max={output.max():.4f}")
    
    # パラメータ数の詳細
    total, trainable = count_parameters(model)
    print(f"\n詳細パラメータ数:")
    print(f"総パラメータ数: {total:,}")
    print(f"学習可能パラメータ数: {trainable:,}")
    
    return model


if __name__ == "__main__":
    # テスト実行
    model = test_model()
