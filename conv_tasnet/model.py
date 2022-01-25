# -*- coding: utf-8 -*-

from beartype import beartype

from conv_tasnet.param import ConvTasNetParam
from conv_tasnet.layers import Decoder, Encoder, MaskApplier, Separator
from conv_tasnet.evaluation import SDRLoss, SDRMetric

import tensorflow.keras as keras
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Adam, Optimizer

from typing import List, Union


__all__ = ["make_model"]


OptimizerType = Union[str, Optimizer]
LossType = Union[str, Loss]
MetricType = Union[str, Metric, List[Union[str, Metric]]]


@beartype
def make_model(
    param: ConvTasNetParam,
    name: str = "Conv-TasNet",
    optimizer: OptimizerType = Adam(),
    loss: LossType = SDRLoss("sisdr", True),
    metrics: MetricType = ["accuracy", SDRMetric("sisdr", True)],
    **kwargs
) -> keras.Model:
    input_mixture = keras.Input(
        shape=(param.T,),
        name="input_mixture",
    )
    mixture_weights = Encoder(num_filters=param.N, filter_len=param.L)(input_mixture)
    source_masks = Separator(
        num_sources=param.C,
        num_basis=param.N,
        num_blocks=param.X,
        num_repeats=param.R,
        convblock_channel=param.H,
        convblock_kernel=param.P,
        bottleneck_channel=param.B,
        skipconn_channel=param.S,
        causal=param.causal,
    )(mixture_weights)
    source_weights = MaskApplier(num_sources=param.C, num_basis=param.N)(
        [mixture_weights, source_masks]
    )
    estimated_sources = Decoder(num_filters=1, filter_len=param.L)(source_weights)
    model = keras.Model(inputs=input_mixture, outputs=estimated_sources, name=name)
    model.compile(optimizer, loss, metrics, **kwargs)
    return model
