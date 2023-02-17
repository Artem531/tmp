import numpy as np
from torch import nn
import torch
import time

class Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, input_channel=256):
        super(Head, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(input_channel, channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        self.wh_head = self.ConvReluConv(input_channel, 2)
        self.reg_head = self.ConvReluConv(input_channel, 2)

        self.rect_head = self.ConvReluConv(input_channel, 1)
        self.noise_head = nn.Sequential(
                                nn.Conv2d(input_channel, channel,
                                          kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel, 1,
                                          kernel_size=1, stride=1, padding=0))
        self.small_lines_head = nn.Sequential(
                                nn.Conv2d(input_channel, channel,
                                          kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel, 1,
                                          kernel_size=1, stride=1, padding=0))
        self.shift_head = nn.Sequential(
                                nn.Conv2d(input_channel, channel,
                                          kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel, 1,
                                          kernel_size=1, stride=1, padding=0))

        self.big_lines_head = nn.Sequential(
                                nn.Conv2d(input_channel, channel,
                                          kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(channel, 1,
                                          kernel_size=1, stride=1, padding=0))

    def ConvReluConv(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        relu = nn.ReLU()
        out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            out_conv.bias.data.fill_(bias_value)
        return nn.Sequential(feat_conv, relu, out_conv)

    def forward(self, x):
        st_time = time.time()
        hm = self.cls_head(x).sigmoid()
        wh = self.wh_head(x).relu()
        offset = self.reg_head(x)
        #print("base", time.time() - st_time)
        num_rect = self.rect_head(x)
        noise_key = self.noise_head(x)
        small_lines_key = self.small_lines_head(x)
        shift_key = self.shift_head(x)
        big_lines_key = self.big_lines_head(x)
        #print("mode", time.time() - st_time)
        #print("_______________")
        features = torch.cat((num_rect, noise_key, small_lines_key, shift_key, big_lines_key), axis=1)

        return hm, wh, offset, features.detach()


class HeadJointPoints(nn.Module):
    def __init__(self, num_classes=80, channel=64):
        super(HeadJointPoints, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        self.k_head = self.ConvReluConv(256, 8)
        self.reg_head = self.ConvReluConv(256, 2)

    def ConvReluConv(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        relu = nn.ReLU()
        out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            out_conv.bias.data.fill_(bias_value)
        return nn.Sequential(feat_conv, relu, out_conv)

    def forward(self, x):
        hm = self.cls_head(x).sigmoid()
        return hm, 0, 0