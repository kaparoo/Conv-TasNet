# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import tensorflow.keras as keras
from typing import Dict


__all__ = ["SDRLoss"]


_COEF = 10 / math.log(10)
_EPS = 1e-7


def _inner_product(u: tf.Tensor, v: tf.Tensor, keepdims: bool = False) -> tf.Tensor:
    return tf.reduce_sum(tf.multiply(u, v), axis=-1, keepdims=keepdims)


def _calc_sdr(
    s_true: tf.Tensor, s_pred: tf.Tensor, scale_invariant: bool = False
) -> tf.Tensor:
    scale = 1.0
    if scale_invariant:
        scale = _inner_product(s_pred, s_true, keepdims=True) / (
            _inner_product(s_true, s_true, keepdims=True) + _EPS
        )
    s_target = scale * s_true
    e_noise = s_pred - s_target
    return _COEF * tf.math.log(
        _inner_product(s_target, s_target) / (_inner_product(e_noise, e_noise) + _EPS)
        + _EPS
    )


class SDRLoss(keras.losses.Loss):
    def __init__(self, scale_invariant: bool = False):
        super(SDRLoss, self).__init__(name="sdr")
        self.scale_invariant = scale_invariant

    def call(self, s_true: tf.Tensor, s_pred: tf.Tensor) -> tf.Tensor:
        return -1.0 * _calc_sdr(s_true, s_pred, self.scale_invariant)

    def get_config(self) -> Dict[str, bool]:
        return {"scale_invariant": self.scale_invariant}
