# -*- coding: utf-8 -*-

from conv_tasnet.layers.normalizations import CumulativeLayerNorm as cLN
from conv_tasnet.layers.normalizations import GlobalLayerNorm as gLN

import tensorflow as tf
import tensorflow.keras as keras
from typing import Dict, List, Tuple, Union


__all__ = ["ConvBlock", "Separator", "MaskApplier"]


class ConvBlock(keras.layers.Layer):
    def __init__(
        self,
        depthwise_channel: int,
        depthwise_kernel: int,
        depthwise_dilation: int,
        residual_channel: int,
        skipconn_channel: int,
        causal: bool = False,
        last: bool = False,
    ):
        super(ConvBlock, self).__init__()
        self.depthwise_channel = depthwise_channel
        self.depthwise_kernel = depthwise_kernel
        self.depthwise_dilation = depthwise_dilation
        self.residual_channel = residual_channel
        self.skipconn_channel = skipconn_channel
        self.causal = causal
        self.last = last

    def build(self, input_shape: tf.TensorShape):
        self.input_conv = keras.layers.Conv1D(
            filters=self.depthwise_channel,
            kernel_size=1,
            padding="same",
            input_shape=input_shape[1:],
        )
        self.prelu1 = keras.layers.PReLU(shared_axes=[1, 2])

        if self.causal:
            self.norm1 = cLN()
            self.norm2 = cLN()
            depthwise_padding = "causal"
        else:
            self.norm1 = gLN()
            self.norm2 = gLN()
            depthwise_padding = "same"

        self.depthwise_conv = keras.layers.Conv1D(
            filters=self.depthwise_channel,
            kernel_size=self.depthwise_kernel,
            padding=depthwise_padding,
            dilation_rate=self.depthwise_dilation,
            groups=self.depthwise_channel,
        )
        self.prelu2 = keras.layers.PReLU(shared_axes=[1, 2])

        self.skipconn_conv = keras.layers.Conv1D(
            filters=self.skipconn_channel, kernel_size=1, padding="same"
        )
        if not self.last:
            self.residual_conv = keras.layers.Conv1D(
                filters=self.residual_channel, kernel_size=1, padding="same"
            )
            self.residual_link = keras.layers.Add()

    def call(self, block_input: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        block_output = self.input_conv(block_input)
        block_output = self.prelu1(block_output)
        block_output = self.norm1(block_output)
        block_output = self.depthwise_conv(block_output)
        block_output = self.prelu2(block_output)
        block_output = self.norm2(block_output)
        skipconn_output = self.skipconn_conv(block_output)
        if not self.last:
            block_output = self.residual_conv(block_output)
            block_output = self.residual_link([block_input, block_output])
        return block_output, skipconn_output

    def get_config(self) -> Dict[str, Union[int, bool]]:
        return {
            "depthwise_channel": self.depthwise_channel,
            "depthwise_kernel": self.depthwise_kernel,
            "depthwise_dilation": self.depthwise_dilation,
            "residual_channel": self.residual_channel,
            "skipconn_channel": self.skipconn_channel,
            "causal": self.causal,
            "last": self.last,
        }


class Separator(keras.layers.Layer):
    def __init__(
        self,
        num_sources: int,
        num_basis: int,
        num_blocks: int,
        num_repeats: int,
        convblock_channel: int,
        convblock_kernel: int,
        bottleneck_channel: int,
        skipconn_channel: int,
        causal: bool,
    ):
        super(Separator, self).__init__()
        self.num_sources = num_sources
        self.num_basis = num_basis
        self.num_blocks = num_blocks
        self.num_repeats = num_repeats
        self.convblock_channel = convblock_channel
        self.convblock_kernel = convblock_kernel
        self.bottleneck_channel = bottleneck_channel
        self.skipconn_channel = skipconn_channel
        self.causal = causal

    def build(self, input_shape: tf.TensorShape):
        self.input_norm = keras.layers.LayerNormalization(input_shape=input_shape[1:])
        self.input_conv = keras.layers.Conv1D(
            filters=self.bottleneck_channel, kernel_size=1, padding="same"
        )

        self.temporal_conv_net: List[ConvBlock] = []
        for _ in range(self.num_repeats):
            for x in range(self.num_blocks):
                self.temporal_conv_net.append(
                    ConvBlock(
                        depthwise_channel=self.convblock_channel,
                        depthwise_kernel=self.convblock_kernel,
                        depthwise_dilation=2 ** x,
                        residual_channel=self.bottleneck_channel,
                        skipconn_channel=self.skipconn_channel,
                        causal=self.causal,
                    )
                )
        else:
            self.temporal_conv_net[-1].last = True
        self.merge = keras.layers.Add()
        self.prelu = keras.layers.PReLU(shared_axes=[1, 2])

        self.output_conv = keras.layers.Conv1D(
            filters=self.num_basis * self.num_sources,
            kernel_size=1,
            padding="same",
            activation=tf.nn.sigmoid,
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=[-1, self.num_sources, self.num_basis]
        )
        self.output_permute = keras.layers.Permute(dims=[2, 1, 3])

    def call(self, mixture_weights: tf.Tensor) -> tf.Tensor:
        block_input = self.input_norm(mixture_weights)
        block_input = self.input_conv(block_input)

        skipconn_outputs = []
        for conv_block in self.temporal_conv_net:
            block_input, skipconn_output = conv_block(block_input)
            skipconn_outputs.append(skipconn_output)
        tcn_output = self.merge(skipconn_outputs)
        tcn_output = self.prelu(tcn_output)

        source_masks = self.output_conv(tcn_output)
        source_masks = self.output_reshape(source_masks)
        source_masks = self.output_permute(source_masks)
        return source_masks

    def get_config(self) -> Dict[str, Union[int, bool]]:
        return {
            "num_sources": self.num_sources,
            "num_basis": self.num_basis,
            "num_blocks": self.num_blocks,
            "num_repeats": self.num_repeats,
            "convblock_channel": self.convblock_channel,
            "convblock_kernel": self.convblock_kernel,
            "bottleneck_channel": self.bottleneck_channel,
            "skipconn_channel": self.skipconn_channel,
            "causal": self.causal,
        }


class MaskApplier(keras.layers.Layer):
    def __init__(self, num_sources: int, num_basis: int):
        super(MaskApplier, self).__init__()
        self.num_sources = num_sources
        self.num_basis = num_basis

    def build(self, input_shapes: Tuple[tf.TensorShape, tf.TensorShape]):
        weight_shape, _ = input_shapes
        if self.num_sources > 1:
            self.concat = keras.layers.Concatenate(input_shape=weight_shape[1:])
        self.reshape = keras.layers.Reshape(
            target_shape=[-1, self.num_sources, self.num_basis]
        )
        self.permute = keras.layers.Permute(dims=[2, 1, 3])
        self.multply = keras.layers.Multiply()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        mixture_weights, source_masks = inputs
        if self.num_sources > 1:
            mixture_weights = self.concat([mixture_weights] * self.num_sources)
        mixture_weights = self.reshape(mixture_weights)
        mixture_weights = self.permute(mixture_weights)
        source_weights = self.multply([mixture_weights, source_masks])
        return source_weights

    def get_config(self) -> Dict[str, int]:
        return {"num_sources": self.num_sources, "num_basis": self.num_basis}
