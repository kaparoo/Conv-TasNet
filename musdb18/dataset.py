# -*- coding: utf-8 -*-

from beartype import beartype

import musdb
from musdb18.param import MUSDB18Param

import numpy as np
import random
import sys
import tensorflow as tf
from typing import List, Literal, Tuple, Union

if sys.version_info >= (3, 9):
    List = list
    Tuple = tuple


__all__ = ["make_dataset"]


class MUSDB18Provider(object):

    LABELS = ["vocals", "drums", "bass", "other"]

    @beartype
    def __init__(self, param: MUSDB18Param, subset: Literal["train", "test"] = "train"):
        self.param = param

        root = param.dataset_root
        if subset == "train" and param.validation_split > 0.0:
            self.tracks = list(musdb.DB(root, subsets="train", splits="train"))
            self.valids = list(musdb.DB(root, subsets="train", splits="valid"))
        else:
            self.tracks = list(musdb.DB(root, subsets=subset))
            self.valids = list()

        self.num_tracks = len(self.tracks)
        self.num_valids = len(self.valids)

        if self.num_tracks == 0:
            raise ValueError("no tracks: %s (subsets: %s)" % (root, subset))

    @beartype
    def generate(self, num_batches: int, is_valid: bool = False):
        tracks: List[musdb.MultiTrack] = self.tracks if not is_valid else self.valids

        batch_size = self.param.batch_size
        sample_length = self.param.sample_length
        for _ in range(num_batches):
            x_batch = np.zeros([2 * batch_size, sample_length], dtype=np.float32)
            y_batch = np.zeros([2 * batch_size, 4, sample_length], dtype=np.float32)

            for idx in range(batch_size):
                idx_l, idx_r = 2 * idx, 2 * idx + 1

                track = random.choice(tracks)
                track.chunk_duration = sample_length
                track.chunk_start = random.uniform(
                    0, track.duration - track.chunk_duration
                )

                mixture = track.targets["linear_mixture"].audio
                sources = {label: track.targets[label].audio for label in self.LABELS}

                x_batch[idx_l] = mixture[:, 0]
                x_batch[idx_r] = mixture[:, 1]
                for c, label in enumerate(self.LABELS):
                    y_batch[idx_l][c] = sources[label][:, 0]
                    y_batch[idx_r][c] = sources[label][:, 1]

            yield x_batch, y_batch


@beartype
def make_dataset(
    param: MUSDB18Param, mode: Literal["train", "test"] = "train"
) -> Tuple[tf.data.Dataset, Union[tf.data.Dataset, None]]:
    provider = MUSDB18Provider(param, mode)

    output_types = (tf.float32, tf.float32)
    output_shapes = (
        tf.TensorShape([2 * param.batch_size, param.sample_length]),
        tf.TensorShape([2 * param.batch_size, 4, param.sample_length]),
    )

    if provider.num_valids != 0:
        valid_batches = int(param.validation_split * param.num_batches)
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: provider.generate(valid_batches, is_valid=True),
            output_types=output_types,
            output_shapes=output_shapes,
        )
    else:
        valid_batches = 0
        valid_dataset = None

    track_batches = param.num_batches - valid_batches
    track_dataset = tf.data.Dataset.from_generator(
        lambda: provider.generate(track_batches, is_valid=False),
        output_types=output_types,
        output_shapes=output_shapes,
    )

    return track_dataset, valid_dataset
