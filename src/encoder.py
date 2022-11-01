from typing import List

import torch
from torch import nn


def _get_act_dropout(act='ReLU', dropout=0.0):
    activation = getattr(torch.nn, act)()
    layers = [activation, nn.Dropout(p=dropout)]
    return layers

class QuartzNetBlock(torch.nn.Module):
    def __init__(
        self,
        feat_in: int,
        filters: int,
        repeat: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        residual: bool,
        separable: bool,
        dropout: float,
    ):

        super().__init__()

        if not residual:
            self.res = None
        else:
            self.res = self._build_one_convolution_block(
                kernel_size=1,
                stride=1,
                dilation=1,
                separable=False,
                in_channels=feat_in,
                out_channels=filters,
                norm=True,
                activation=False,
                return_as_nn_module=True,
            )

        # In this implementation stride should be 1 if repeat > 1
        self.conv = nn.ModuleList()
        self._build_conv_blocks(kernel_size, feat_in, filters, stride, dilation, separable, dropout, repeat)

        self.out = nn.Sequential(*_get_act_dropout(dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            residual = self.res(x)
        for layer in self.conv:
            x = layer(x)
        if self.res:
            x += residual
        return self.out(x)
    
    def _build_conv_blocks(self, kernel_size, in_channels, out_channels, stride, dilation, separable, dropout, repeat):
        if repeat > 1:
            assert stride == 1
            common_conv_params = {
                'kernel_size': kernel_size,
                'stride': stride,
                'dilation': dilation,
                'separable': separable,
                'out_channels': out_channels,
                'norm': True,
                'dropout': dropout,
            }
            for block_num in range(repeat):
                if block_num == 0:
                    self.conv.extend(self._build_one_convolution_block(in_channels=in_channels, activation=True, **common_conv_params))
                elif block_num == repeat - 1:
                    self.conv.extend(self._build_one_convolution_block(in_channels=out_channels, activation=False, **common_conv_params))
                else:
                    self.conv.extend(self._build_one_convolution_block(in_channels=out_channels, activation=True, **common_conv_params))
        else:
            self.conv.extend(self._build_one_convolution_block(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                separable=separable,
                in_channels=in_channels,
                out_channels=out_channels,
                norm=True,
                activation=True,
                dropout=dropout,
            ))

    @staticmethod
    def _build_one_convolution_block(kernel_size, in_channels, out_channels, stride, dilation, separable=True, norm=True, activation=True, return_as_nn_module=False, dropout=0):
        padding = (dilation * (kernel_size - 1)) // 2
        layers = []
        if not separable:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                )
            )
        else:
            layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    kernel_size=kernel_size,
                    groups=in_channels
                ),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                )
            ]
            )

        if norm:
            norm_layer = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
            layers.append(norm_layer)
        
        if activation:
            layers.extend(_get_act_dropout(dropout=dropout))

        if not return_as_nn_module:
            return layers
        else:
            return nn.Sequential(*layers)
        



class QuartzNet(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.stride_val = 1

        layers = []
        feat_in = conf.feat_in
        for block in conf.blocks:
            layers.append(QuartzNetBlock(feat_in, **block))
            self.stride_val *= block.stride**block.repeat
            feat_in = block.filters

        self.layers = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, features_length: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.layers(features)
        encoded_len = (
            torch.div(features_length - 1, self.stride_val, rounding_mode="trunc") + 1
        )

        return encoded, encoded_len
