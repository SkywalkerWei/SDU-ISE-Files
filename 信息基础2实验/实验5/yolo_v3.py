# models/yolo_v3.py
import torch
import torch.nn as nn
import sys
# 确保可以从项目根目录导入 config
sys.path.append('..')
import config

# PDF中的DBL模块: Conv -> BatchNorm -> LeakyReLU
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

# Res_unit模块: 两个CNNBlock和一个残差连接
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

# 预测头，用于输出最终的检测结果
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # 输出形状: BATCH_SIZE, 3, Grid_y, Grid_x, 5 + num_classes
        # 5 -> (obj_prob, x, y, w, h)
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        
# 完整的YOLOv3模型
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Darknet-53 结构
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8), # -> route_1 (for medium objects)
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8), # -> route_2 (for small objects)
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
            
            # --- Neck and Prediction Heads ---
            # 大目标预测
            ScalePrediction(1024, num_classes), # P5 -> y1
            
            # 上采样并与route_2拼接，用于中目标预测
            nn.Sequential(
                CNNBlock(1024, 512, kernel_size=1),
                nn.Upsample(scale_factor=2)
            ),
            ScalePrediction(1024, num_classes), # P4 -> y2

            # 上采样并与route_1拼接，用于小目标预测
            nn.Sequential(
                CNNBlock(1024, 256, kernel_size=1),
                nn.Upsample(scale_factor=2)
            ),
            ScalePrediction(512, num_classes), # P3 -> y3
        ])

    def forward(self, x):
        outputs = []  # [y1, y2, y3]
        route_connections = []

        for i, layer in enumerate(self.layers):
            if i in [11, 14, 17]: # 对应 ScalePrediction 的索引
                 # P5/y1, P4/y2, P3/y3
                if i > 11:
                    x = torch.cat([x, route_connections[-1]], dim=1)
                outputs.append(layer(x))
                continue
            
            x = layer(x)

            if i in [6, 8]: # 对应 ResidualBlock 输出的索引
                route_connections.append(x)
        
        # YOLO的输出顺序是从大到小物体，即 [P5, P4, P3]，我们将其反转以匹配ANCHORS的顺序
        return outputs[::-1]