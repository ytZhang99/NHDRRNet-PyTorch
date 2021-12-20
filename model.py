<<<<<<< HEAD
import torch
import numpy as np
import torch.nn as nn
from option import args
import torch.nn.functional as F
from utils.utils import *


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride

        # Padding
        pad_length = self.kernel_size - self.stride
        pad_top = pad_length // 2
        pad_left = pad_length // 2
        pad_bottom = pad_length - pad_top
        pad_right = pad_length - pad_left

        self.conv = nn.Sequential(
            nn.ZeroPad2d([pad_left, pad_right, pad_top, pad_bottom]),
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, stride=self.stride, bias=True),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class DeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(DeconvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch, self.out_ch, self.kernel_size, self.stride, padding=1, bias=True),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.deconv(x)


class TriplePass(nn.Module):
    def __init__(self, tp_feats):
        super(TriplePass, self).__init__()
        self.tp_feats = tp_feats
        self.conv_small = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 1, 1, 0, bias=True),
            nn.ReLU()
        )
        self.conv_middle = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 3, 1, 1, bias=True),
            nn.ReLU()
        )
        self.conv_large = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 5, 1, 2, bias=True),
            nn.ReLU()
        )
        self.conv_compress = nn.Sequential(
            nn.Conv2d(3 * self.tp_feats, self.tp_feats, 3, 1, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x_small = self.conv_small(x)
        x_middle = self.conv_middle(x)
        x_large = self.conv_large(x)

        x_cat = torch.cat((x_small, x_middle, x_large), dim=1)
        x_cat = self.conv_compress(x_cat)
        return x + x_cat


class NonLocal(nn.Module):
    def __init__(self, in_ch, num_feats):
        super(NonLocal, self).__init__()
        self.in_ch = in_ch
        self.num_feats = num_feats
        self.theta = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.g = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.W = nn.Conv2d(self.num_feats, self.in_ch, 1, 1, 0)

    def forward(self, x):
        b = x.size(0)
        x_theta = self.theta(x).view(b, self.num_feats, -1).permute(0, 2, 1).contiguous()  # [B, HW, C/2]
        x_phi = self.phi(x).view(b, self.num_feats, -1)  # [B, C/2, HW]
        x_g = self.g(x).view(b, self.num_feats, -1).permute(0, 2, 1).contiguous()  # [B, HW, C/2]

        x_theta_phi = F.softmax(torch.matmul(x_theta, x_phi), dim=-1)  # [B, HW, HW]
        y = torch.matmul(x_theta_phi, x_g)  # [B, HW, C/2]
        y = y.permute(0, 2, 1).contiguous()  # [B, C/2, HW]
        y = y.view(b, self.num_feats, *x.size()[2:])  # [B, C/2, H, W]
        y = self.W(y)

        return x + y


class NHDRRNet(nn.Module):
    def __init__(self):
        super(NHDRRNet, self).__init__()
        self.num_channels = args.num_channels
        self.num_feats = args.num_features
        self.tp_feats = 256

        # Encoder
        self.encoder_u1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_u2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_u3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_u4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        self.encoder_m1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_m2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_m3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_m4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        self.encoder_o1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_o2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_o3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_o4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        # Merger
        self.cat_conv = nn.Sequential(
            nn.Conv2d(8 * 3 * self.num_feats, self.tp_feats, 3, 1, 1),
            nn.BatchNorm2d(self.tp_feats),
            nn.LeakyReLU()
        )

        # Triple pass module
        self.triple_pass_module = nn.Sequential()
        for i in range(10):
            self.triple_pass_module.add_module(str(i) + 'triple_pass_layer', TriplePass(self.tp_feats))

        # Non-local module
        self.non_local_module = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            NonLocal(self.tp_feats, self.tp_feats // 2)
        )

        # Decoder
        self.decoder_1 = DeconvLayer(2 * self.tp_feats, 4 * self.num_feats, 4, 2)
        self.decoder_2 = DeconvLayer(4 * 4 * self.num_feats, 2 * self.num_feats, 4, 2)
        self.decoder_3 = DeconvLayer(4 * 2 * self.num_feats, self.num_feats, 4, 2)
        self.decoder_4 = DeconvLayer(4 * self.num_feats, self.num_feats, 4, 2)

        # Output
        self.reconstruction = nn.Sequential(
            nn.Conv2d(self.num_feats, self.num_channels, 3, 1, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        x1 = check_image_size(x1, 16)
        x2 = check_image_size(x2, 16)
        x3 = check_image_size(x3, 16)
        feat_u1 = self.encoder_u1(x1)
        feat_u2 = self.encoder_u2(feat_u1)
        feat_u3 = self.encoder_u3(feat_u2)
        feat_u4 = self.encoder_u4(feat_u3)

        feat_m1 = self.encoder_m1(x2)
        feat_m2 = self.encoder_m2(feat_m1)
        feat_m3 = self.encoder_m3(feat_m2)
        feat_m4 = self.encoder_m4(feat_m3)

        feat_o1 = self.encoder_o1(x3)
        feat_o2 = self.encoder_o2(feat_o1)
        feat_o3 = self.encoder_o3(feat_o2)
        feat_o4 = self.encoder_o4(feat_o3)

        feat_cat = torch.cat((feat_u4, feat_m4, feat_o4), dim=1)
        feat_cat = self.cat_conv(feat_cat)

        feat_tp = self.triple_pass_module(feat_cat)
        feat_nl = self.non_local_module(feat_cat)
        feat_nl = F.interpolate(feat_nl, [feat_cat.size(2), feat_cat.size(3)])

        feat_merger = torch.cat((feat_tp, feat_nl), dim=1)
        feat_1 = self.decoder_1(feat_merger)
        feat_2 = torch.cat((feat_u3, feat_m3, feat_o3, feat_1), dim=1)
        feat_2 = self.decoder_2(feat_2)
        feat_3 = torch.cat((feat_u2, feat_m2, feat_o2, feat_2), dim=1)
        feat_3 = self.decoder_3(feat_3)
        feat_4 = torch.cat((feat_u1, feat_m1, feat_o1, feat_3), dim=1)
        feat_4 = self.decoder_4(feat_4)

        ans = self.reconstruction(feat_4)

        return ans


# if __name__ == '__main__':
#     a = torch.ones([1, 6, 500, 500])
#     b = torch.ones([1, 6, 500, 500])
#     c = torch.ones([1, 6, 500, 500])
#     net = NHDRRNet()
#     output = net(a, b, c)
#     print(output.shape)
=======
import torch
import numpy as np
import torch.nn as nn
from option import args
import torch.nn.functional as F
from utils.utils import *


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride

        # Padding
        pad_length = self.kernel_size - self.stride
        pad_top = pad_length // 2
        pad_left = pad_length // 2
        pad_bottom = pad_length - pad_top
        pad_right = pad_length - pad_left

        self.conv = nn.Sequential(
            nn.ZeroPad2d([pad_left, pad_right, pad_top, pad_bottom]),
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, stride=self.stride, bias=True),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class DeconvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(DeconvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch, self.out_ch, self.kernel_size, self.stride, padding=1, bias=True),
            nn.BatchNorm2d(self.out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.deconv(x)


class TriplePass(nn.Module):
    def __init__(self, tp_feats):
        super(TriplePass, self).__init__()
        self.tp_feats = tp_feats
        self.conv_small = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 1, 1, 0, bias=True),
            nn.ReLU()
        )
        self.conv_middle = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 3, 1, 1, bias=True),
            nn.ReLU()
        )
        self.conv_large = nn.Sequential(
            nn.Conv2d(self.tp_feats, self.tp_feats, 5, 1, 2, bias=True),
            nn.ReLU()
        )
        self.conv_compress = nn.Sequential(
            nn.Conv2d(3 * self.tp_feats, self.tp_feats, 3, 1, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x_small = self.conv_small(x)
        x_middle = self.conv_middle(x)
        x_large = self.conv_large(x)

        x_cat = torch.cat((x_small, x_middle, x_large), dim=1)
        x_cat = self.conv_compress(x_cat)
        return x + x_cat


class NonLocal(nn.Module):
    def __init__(self, in_ch, num_feats):
        super(NonLocal, self).__init__()
        self.in_ch = in_ch
        self.num_feats = num_feats
        self.theta = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.g = nn.Conv2d(self.in_ch, self.num_feats, 1, 1, 0)
        self.W = nn.Conv2d(self.num_feats, self.in_ch, 1, 1, 0)

    def forward(self, x):
        b = x.size(0)
        x_theta = self.theta(x).view(b, self.num_feats, -1).permute(0, 2, 1).contiguous()  # [B, HW, C/2]
        x_phi = self.phi(x).view(b, self.num_feats, -1)  # [B, C/2, HW]
        x_g = self.g(x).view(b, self.num_feats, -1).permute(0, 2, 1).contiguous()  # [B, HW, C/2]

        x_theta_phi = F.softmax(torch.matmul(x_theta, x_phi), dim=-1)  # [B, HW, HW]
        y = torch.matmul(x_theta_phi, x_g)  # [B, HW, C/2]
        y = y.permute(0, 2, 1).contiguous()  # [B, C/2, HW]
        y = y.view(b, self.num_feats, *x.size()[2:])  # [B, C/2, H, W]
        y = self.W(y)

        return x + y


class NHDRRNet(nn.Module):
    def __init__(self):
        super(NHDRRNet, self).__init__()
        self.num_channels = args.num_channels
        self.num_feats = args.num_features
        self.tp_feats = 256

        # Encoder
        self.encoder_u1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_u2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_u3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_u4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        self.encoder_m1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_m2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_m3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_m4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        self.encoder_o1 = ConvLayer(2 * self.num_channels, self.num_feats, 3, 2)
        self.encoder_o2 = ConvLayer(self.num_feats, 2 * self.num_feats, 3, 2)
        self.encoder_o3 = ConvLayer(2 * self.num_feats, 4 * self.num_feats, 3, 2)
        self.encoder_o4 = ConvLayer(4 * self.num_feats, 8 * self.num_feats, 3, 2)

        # Merger
        self.cat_conv = nn.Sequential(
            nn.Conv2d(8 * 3 * self.num_feats, self.tp_feats, 3, 1, 1),
            nn.BatchNorm2d(self.tp_feats),
            nn.LeakyReLU()
        )

        # Triple pass module
        self.triple_pass_module = nn.Sequential()
        for i in range(10):
            self.triple_pass_module.add_module(str(i) + 'triple_pass_layer', TriplePass(self.tp_feats))

        # Non-local module
        self.non_local_module = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            NonLocal(self.tp_feats, self.tp_feats // 2)
        )

        # Decoder
        self.decoder_1 = DeconvLayer(2 * self.tp_feats, 4 * self.num_feats, 4, 2)
        self.decoder_2 = DeconvLayer(4 * 4 * self.num_feats, 2 * self.num_feats, 4, 2)
        self.decoder_3 = DeconvLayer(4 * 2 * self.num_feats, self.num_feats, 4, 2)
        self.decoder_4 = DeconvLayer(4 * self.num_feats, self.num_feats, 4, 2)

        # Output
        self.reconstruction = nn.Sequential(
            nn.Conv2d(self.num_feats, self.num_channels, 3, 1, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        x1 = check_image_size(x1, 16)
        x2 = check_image_size(x2, 16)
        x3 = check_image_size(x3, 16)
        feat_u1 = self.encoder_u1(x1)
        feat_u2 = self.encoder_u2(feat_u1)
        feat_u3 = self.encoder_u3(feat_u2)
        feat_u4 = self.encoder_u4(feat_u3)

        feat_m1 = self.encoder_m1(x2)
        feat_m2 = self.encoder_m2(feat_m1)
        feat_m3 = self.encoder_m3(feat_m2)
        feat_m4 = self.encoder_m4(feat_m3)

        feat_o1 = self.encoder_o1(x3)
        feat_o2 = self.encoder_o2(feat_o1)
        feat_o3 = self.encoder_o3(feat_o2)
        feat_o4 = self.encoder_o4(feat_o3)

        feat_cat = torch.cat((feat_u4, feat_m4, feat_o4), dim=1)
        feat_cat = self.cat_conv(feat_cat)

        feat_tp = self.triple_pass_module(feat_cat)
        feat_nl = self.non_local_module(feat_cat)
        feat_nl = F.interpolate(feat_nl, [feat_cat.size(2), feat_cat.size(3)])

        feat_merger = torch.cat((feat_tp, feat_nl), dim=1)
        feat_1 = self.decoder_1(feat_merger)
        feat_2 = torch.cat((feat_u3, feat_m3, feat_o3, feat_1), dim=1)
        feat_2 = self.decoder_2(feat_2)
        feat_3 = torch.cat((feat_u2, feat_m2, feat_o2, feat_2), dim=1)
        feat_3 = self.decoder_3(feat_3)
        feat_4 = torch.cat((feat_u1, feat_m1, feat_o1, feat_3), dim=1)
        feat_4 = self.decoder_4(feat_4)

        ans = self.reconstruction(feat_4)

        return ans


# if __name__ == '__main__':
#     a = torch.ones([1, 6, 500, 500])
#     b = torch.ones([1, 6, 500, 500])
#     c = torch.ones([1, 6, 500, 500])
#     net = NHDRRNet()
#     output = net(a, b, c)
#     print(output.shape)
>>>>>>> 320cf2305e588f0eba31d550eec35c4282346f15
