import torch
import torch.nn as nn
import math

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view(1, self.na, 1, 1, 2).expand(1, self.na, ny, nx, 2).float()
        return grid, anchor_grid

class Model(nn.Module):
    def __init__(self, nc=80):  # model, input channels, number of classes
        super().__init__()
        self.nc = nc
        # Anchors
        self.anchors = [
            [10,13, 16,30, 33,23],  # P3/8
            [30,61, 62,45, 59,119],  # P4/16
            [116,90, 156,198, 373,326]  # P5/32
        ]

        # Simplified YOLOv5s backbone
        self.backbone = nn.Sequential(
            Conv(3, 32, 6, 2, 2),  # 0-P1/2
            Conv(32, 64, 3, 2),    # 1-P2/4
            C3(64, 64, 1),
            Conv(64, 128, 3, 2),   # 3-P3/8
            C3(128, 128, 2),
            Conv(128, 256, 3, 2),  # 5-P4/16
            C3(256, 256, 3),
            Conv(256, 512, 3, 2),  # 7-P5/32
            C3(512, 512, 1),
            SPPF(512, 512)
        )

        # Simplified YOLOv5s neck
        self.neck = nn.ModuleList([
            Conv(512, 256, 1, 1), # 10, upsample from P5
            nn.Upsample(scale_factor=2, mode='nearest'),
            C3(512, 256, 1, shortcut=False), # 12, from P4 and upsampled P5
            Conv(256, 128, 1, 1), # 13, upsample from P4'
            nn.Upsample(scale_factor=2, mode='nearest'),
            C3(256, 128, 1, shortcut=False), # 15, from P3 and upsampled P4'
            Conv(128, 128, 3, 2), # 16, downsample to P4''
            C3(128 + 256, 256, 1, shortcut=False), # 17, from P4' and downsampled P3'
            Conv(256, 256, 3, 2), # 18, downsample to P5''
            C3(256 + 512, 512, 1, shortcut=False) # 19, from P5 and downsampled P4''
        ])

        # Detection Head
        ch = [128, 256, 512]
        self.detect = Detect(nc, self.anchors, ch)
        self.detect.stride = torch.tensor([8., 16., 32.])
        self._initialize_biases()

    def forward(self, x):
        features = []
        # Backbone
        p3, p4, p5 = None, None, None
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i == 4: p3 = x
            if i == 6: p4 = x
            if i == 9: p5 = x
        
        # Neck
        up5 = self.neck[1](self.neck[0](p5)) # Upsample P5
        cat4 = torch.cat([up5, p4], 1)
        p4_prime = self.neck[2](cat4)

        up4 = self.neck[4](self.neck[3](p4_prime)) # Upsample P4'
        cat3 = torch.cat([up4, p3], 1)
        p3_out = self.neck[5](cat3)

        down3 = self.neck[6](p3_out)
        cat4_2 = torch.cat([down3, p4_prime], 1)
        p4_out = self.neck[7](cat4_2)

        down4 = self.neck[8](p4_out)
        cat5_2 = torch.cat([down4, p5], 1)
        p5_out = self.neck[9](cat5_2)

        return self.detect([p3_out, p4_out, p5_out])

    def _initialize_biases(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for mi, s in zip(self.detect.m, self.detect.stride):  
            b = mi.bias.view(self.detect.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99999))
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)