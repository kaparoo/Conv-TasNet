# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras


__all__ = ["CumulativeLayerNorm", "GlobalLayerNorm"]


_EPS = 1e-7


class _LayerNormBase(keras.layers.Layer):
    def __init__(self):
        super(_LayerNormBase, self).__init__()

    def build(self, input_shape: tf.TensorShape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1],),
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, _: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()


class GlobalLayerNorm(_LayerNormBase):
    def __init__(self):
        super(GlobalLayerNorm, self).__init__()

    def build(self, input_shape: tf.TensorShape):
        super(GlobalLayerNorm, self).build(input_shape)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        mean = tf.math.reduce_mean(input, axis=[1, 2], keepdims=True)
        var = tf.math.reduce_variance(input, axis=[1, 2], keepdims=True)
        return ((input - mean) / tf.sqrt(var + _EPS)) * self.gamma + self.beta


class CumulativeLayerNorm(_LayerNormBase):
    def __init__(self):
        super(CumulativeLayerNorm, self).__init__()

    def build(self, input_shape: tf.TensorShape):
        super(CumulativeLayerNorm, self).build(input_shape)
        num_features, num_channels = input_shape[-2:]
        self.denominators = tf.reshape(
            num_channels * tf.range(1.0, num_features + 1.0), shape=[1, -1, 1]
        )

    def call(self, input: tf.Tensor) -> tf.Tensor:
        cum_sum = tf.cumsum(tf.reduce_sum(input, axis=-1, keepdims=True), axis=-2)
        cum_mean = cum_sum / self.denominators
        cum_square_sum = tf.cumsum(
            tf.reduce_sum(tf.square(input), axis=-1, keepdims=True), axis=-2
        )
        cum_var = cum_square_sum / self.denominators - tf.square(cum_mean)
        return ((input - cum_mean) / tf.sqrt(cum_var + _EPS)) * self.gamma + self.beta
