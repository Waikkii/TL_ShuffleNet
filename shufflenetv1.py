#! /usr/bin/python
# -*- coding: utf-8 -*-
"""ShuffleNet for CsiNet."""
import tensorflow as tf
from tensorlayer.layers import Layer
from tensorlayer.layers import BatchNormLayer
from tensorlayer.layers import Conv2d
from tensorlayer.layers import ConcatLayer
from tensorlayer.layers import DepthwiseConv2d
from tensorlayer.layers import MaxPool2d
from tensorlayer.layers import GlobalMaxPool2d
from tensorlayer.layers import GlobalMeanPool2d
from tensorlayer.layers import InputLayer
from tensorlayer.layers import ElementwiseLayer
from tensorlayer.layers import DenseLayer
from tensorlayer.layers import ReshapeLayer
from tensorlayer.layers import TransposeLayer
from tensorlayer.layers import LambdaLayer
from tensorlayer.layers import MeanPool2d
from tensorlayer.layers import PadLayer
__all__ = [
    'ShuffleNetV1',
]

class ShuffleNetV1(Layer):
    def __init__(self, x, name):

        self.net = self.ShuffleNetV1(x, name)
        self.outputs = self.net.outputs
        self.all_params = list(self.net.all_params)
        self.all_layers = list(self.net.all_layers)
        self.all_drop = dict(self.net.all_drop)
        self.print_layers = self.net.print_layers
        self.print_params = self.net.print_params

    def ShuffleNetV1(self, inputlayer, name):
        inputlayer = InputLayer(inputlayer, name='input')#32*32*2
        #print(inputlayer.outputs.get_shape())
        x = Conv2d(inputlayer, 24, (3, 3), strides=(2, 2), padding='SAME', act=tf.nn.relu, name=name+'_Con2d')###24
        x = MaxPool2d(x, filter_size=(3, 3), strides=(2, 2), padding='SAME', name=name+'_MaxPool')
        x = self.stage(x, n_filter=384, filter_size=(3, 3), groups=8, repeat=4, stage=2, name=name+'_stage1')
        #print("stage1 finished!!!!!!!!!!!!!!!!")
        x = self.stage(x, n_filter=768, filter_size=(3, 3), groups=8, repeat=8, stage=3, name=name+'_stage2')
        #print("stage2 finished!!!!!!!!!!!!!!!!")
        x = self.stage(x, n_filter=1536, filter_size=(3, 3), groups=8, repeat=4, stage=4, name=name+'_stage3')
        #print("stage3 finished!!!!!!!!!!!!!!!!")
        print("stage3", x.outputs.get_shape())
        print(x.count_params())
        #x = GlobalMaxPool2d(x, name=name+'_GlobalMaxPool')
        #print("GMP", x.outputs.get_shape())
        #print(x.count_params())
        x = GlobalMeanPool2d(x, name=name+'_GlobalMaxPool')
        print("GAP", x.outputs.get_shape())
        print(x.count_params())
        x = DenseLayer(x, name=name+'_Dense')
        print("DENSE", x.outputs.get_shape())
        print(x.count_params())

        return x

    @classmethod
    def group_conv(cls, x, groups, n_filter, filter_size=(3, 3), strides=(1, 1), name='_groupconv'):
        with tf.variable_scope(name):
            in_channels = x.outputs.get_shape()[3]
            #print(in_channels)
            gc_list = []
            assert n_filter % groups == 0,'groups数必须可以整除组数！'
            for i in range(groups):
                x_group = LambdaLayer(prev_layer=x, fn=lambda z: z[:, :, :, i * in_channels: (i + 1) * in_channels])
                #print("xgroup"+str(i), x_group.outputs.get_shape())
                gc_list.append(Conv2d(x_group, n_filter=n_filter//groups, filter_size=filter_size, strides=strides,
                                      padding='SAME', name=name+'_Conv2d'+str(i)))
                #print(gc_list[i].outputs.get_shape())
            return ConcatLayer(gc_list)

    @classmethod
    def channel_shuffle(cls, x, num_groups, name='_shuffle'):
        with tf.variable_scope(name):
            n, h, w, c = x.outputs.get_shape()
            x_reshaped = ReshapeLayer(x, (-1, h, w, num_groups, c // num_groups))# 先合并重组
            x_transposed = TransposeLayer(x_reshaped, [0, 1, 2, 4, 3])# 转置
            output = ReshapeLayer(x_transposed, (-1, h, w, c))# 摊平
            return output

    def stage(self, x, n_filter, filter_size, groups, repeat, stage, name):
        x = self.shufflenet_unit(x, n_filter, filter_size, (2, 2), groups, stage, name=name+'_shufflenetunit0')
        for i in range(1, repeat):
            x = self.shufflenet_unit(x, n_filter, filter_size, (1, 1), groups, stage, name=name+'_shufflenetunit'+str(i))
        return x

    def shufflenet_unit(self, inputs, n_filter, filter_size, strides, groups, stage, bottleneck_ratio=0.25, name='_shufflenetunit'):
        in_channels = inputs.outputs.get_shape()[3]
        #print("input", inputs.outputs.get_shape())
        bottleneck_channels = int(n_filter * bottleneck_ratio)
        if stage == 2:
            x = Conv2d(inputs, n_filter=bottleneck_channels, filter_size=filter_size, strides=(1, 1),
                       padding='SAME', name=name+'_Conv2d1')
            #print("conv", x.outputs.get_shape())
        else:
            x = self.group_conv(inputs, groups, bottleneck_channels, (1, 1), (1, 1), name=name+'_groupconv1')
        x = BatchNormLayer(x, act=tf.nn.leaky_relu, name=name+'_Batch1')
        #print("batch", x.outputs.get_shape())
        x = self.channel_shuffle(x, groups, name=name+'_channelshuffle')
        #print("shuffle", x.outputs.get_shape())
        #x = PadLayer(x, [[0, 0], [4, 4], [4, 4], [0, 0]], "CONSTANT", name=name+'_pad')
        #print("pad", x.outputs.get_shape())
        x = DepthwiseConv2d(x, shape=filter_size, strides=strides, depth_multiplier=1,
                            padding='SAME', name=name+'_DepthwiseConv2d')
        #print("deep", x.outputs.get_shape())
        #x = Conv2d(x, n_filter=in_channels, filter_size=filter_size, strides=(1, 1),padding='SAME', name=name+'_Conv2d2')
        #print("conv", x.outputs.get_shape())
        x = BatchNormLayer(x, name=name+'_Batch2')
        #print("deep_batch", x.outputs.get_shape())
        if strides == (2, 2):
            x = self.group_conv(x, groups, n_filter - in_channels, (1, 1), (1, 1), name=name+'_groupconv2')#n_filter - in_channels ??????????
            #print("gonv", x.outputs.get_shape())
            x = BatchNormLayer(x, name=name+'_Batch3')
            #print("batch", x.outputs.get_shape())
            avg = MeanPool2d(inputs, filter_size=(3, 3), strides=(2, 2), padding='SAME', name=name+'_AvePool')
            #print("avg", avg.outputs.get_shape())
            x = ConcatLayer([x, avg], concat_dim=-1, name=name+'_Concat')
            #print("x1out", x.outputs.get_shape())
        else:
            x = self.group_conv(x, groups, n_filter, (1, 1), (1, 1), name=name+'_groupconv3')
            #print("x", x.outputs.get_shape())
            x = BatchNormLayer(x, name=name+'_Batch4')
            if x.outputs.get_shape()[3] != inputs.outputs.get_shape()[3]:
                x = Conv2d(x, n_filter=in_channels, filter_size=filter_size, strides=(1, 1),
                           padding='SAME', name=name+'_Conv2d2')
            x = ElementwiseLayer([x, inputs], combine_fn=tf.add, name=name+'_Elementwise')
        return x

