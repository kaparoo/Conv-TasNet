# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from typing import Dict


__all__ = ["Decoder", "Encoder"]


class Encoder(keras.layers.Layer):
    def __init__(self, num_filters: int, filter_len: int):
        super(Encoder, self).__init__()
        self.num_filters = num_filters
        self.filter_len = filter_len

    def build(self, input_shape: tf.TensorShape):
        self.reshape = keras.layers.Reshape(
            target_shape=(-1, 1), input_shape=input_shape[1:]
        )
        self.encode = keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.filter_len,
            strides=self.filter_len // 2,
        )

    def call(self, input_mixture: tf.Tensor) -> tf.Tensor:
        reshaped_mixture = self.reshape(input_mixture)
        mixture_weights = self.encode(reshaped_mixture)
        return mixture_weights

    def get_config(self) -> Dict[str, int]:
        return {"num_filters": self.num_filters, "filter_len": self.filter_len}


class Decoder(keras.layers.Layer):
    def __init__(self, num_filters: int, filter_len: int):
        super(Decoder, self).__init__()
        self.num_filters = num_filters
        self.filter_len = filter_len

    def build(self, input_shape: tf.TensorShape):
        self.decode = keras.layers.Conv2DTranspose(
            filters=self.num_filters,
            kernel_size=(1, self.filter_len),
            strides=(1, self.filter_len // 2),
            input_shape=input_shape[1:],
        )

    def call(self, source_weights: tf.Tensor) -> tf.Tensor:
        separated_sources = self.decode(source_weights)
        if self.num_filters == 1:
            separated_sources = tf.squeeze(separated_sources, axis=-1)
        return separated_sources

    def get_config(self) -> Dict[str, int]:
        return {"num_filters": self.num_filters, "filter_len": self.filter_len}
