# -*- coding: utf-8 -*-

from absl import flags


__all__ = ["define_flags"]


def define_flags():
    flags.DEFINE_string("dataset_root", None, "Path of dataset", required=True)
    flags.DEFINE_integer("num_batches", 400, "", lower_bound=1)
    flags.DEFINE_integer("batch_size", 100, "", lower_bound=1)
    flags.DEFINE_float("validation_split", 0.0, "Flag for validation")
