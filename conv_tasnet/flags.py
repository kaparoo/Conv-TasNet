# -*- coding: utf-8 -*-

from absl import flags


def define_flags():
    flags.DEFINE_integer(
        "num_filters",
        512,
        "Number of filters in autoencoder",
        lower_bound=1,
        short_name="N",
    )
    flags.DEFINE_integer(
        "filter_length",
        88,
        "Length of the filters (in samples)",
        lower_bound=1,
        short_name="L",
    )
    flags.DEFINE_integer(
        "residual_channels",
        128,
        "Number of channels in bottleneck and the residual paths' 1x1-conv blocks",
        lower_bound=1,
        short_name="B",
    )
    flags.DEFINE_integer(
        "skipconn_channels",
        128,
        "Number of channels in skip-connection paths' 1x1-conv blocks",
        lower_bound=1,
        short_name="S",
    )
    flags.DEFINE_integer(
        "convblock_channels",
        512,
        "Number of channels in convolutional blocks",
        lower_bound=1,
        short_name="H",
    )
    flags.DEFINE_integer(
        "dconv_kernel",
        3,
        "Kernel size in convolutional blocks",
        lower_bound=1,
        short_name="P",
    )
    flags.DEFINE_integer(
        "num_blocks",
        8,
        "Number of convolutional blocks in each repeat",
        lower_bound=1,
        short_name="X",
    )
    flags.DEFINE_integer(
        "num_repeats", 3, "Number of repeats", lower_bound=1, short_name="R"
    )
