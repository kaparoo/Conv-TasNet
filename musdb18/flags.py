# -*- coding: utf-8 -*-

from absl import flags


__all__ = ["define_flags"]


def define_flags():
    flags.DEFINE_string("dataset_root", None, "Path of dataset", required=True)
    flags.DEFINE_float(
        "validation_split",
        0.0,
        "Ratio for validation",
        lower_bound=0.0,
        upper_bound=0.5,
    )
    flags.DEFINE_integer("max_cached", 50, "", lower_bound=1)
    flags.DEFINE_integer("num_songs", 5, "", lower_bound=1)
    flags.DEFINE_integer("num_batches", 400, "", lower_bound=1)
    flags.DEFINE_integer("batch_size", 100, "", lower_bound=1)
