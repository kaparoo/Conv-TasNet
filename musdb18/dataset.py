# -*- coding: utf-8 -*-

from turtle import begin_fill
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


class Track(object):
    __slots__ = ["length", "mixture", "sources"]

    def __init__(self, track: musdb.MultiTrack):
        mixture = track.targets["linear_mixture"]
        self.mixture = (mixture.audio[:, 0], mixture.audio[:, 1])
        self.length = self.mixture[0].shape[-1]
        self.sources = (track.targets[label] for label in MUSDB18.LABELS)
        self.sources = {
            label: (source.audio[:, 0], source.audio[:, 1])
            for label, source in zip(MUSDB18.LABELS, self.sources)
        }


class MUSDB18(object):

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
        batch_size = self.param.batch_size
        sample_length = self.param.sample_length

        tracks = self.tracks if not is_valid else self.valids
        for _ in range(num_batches):
            mixture = np.zeros([2 * batch_size, sample_length], dtype=np.float32)
            sources = np.zeros([2 * batch_size, 4, sample_length], dtype=np.float32)

            for idx in range(batch_size):
                idx_l, idx_r = 2 * idx, 2 * idx + 1

                track = Track(random.choice(tracks))
                start = random.randint(0, track.length - sample_length)
                end = start + sample_length
                mixture[idx_l] = track.mixture[0][start:end]
                mixture[idx_r] = track.mixture[1][start:end]

                for c, stem in enumerate(MUSDB18.LABELS):
                    sources[idx_l][c] = track.sources[stem][0][start:end]
                    sources[idx_r][c] = track.sources[stem][1][start:end]

    def make_dataset(self) -> Tuple[tf.data.Dataset, Union[tf.data.Dataset, None]]:
        param = self.param

        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape([2 * param.batch_size, param.sample_length]),
            tf.TensorShape([2 * param.batch_size, 4, param.sample_length]),
        )

        if self.num_valids != 0:
            valid_batches = int(param.validation_split * param.num_batches)
            valid_dataset = tf.data.Dataset.from_generator(
                lambda: self.generate(valid_batches, is_valid=True),
                output_types=output_types,
                output_shapes=output_shapes,
            )
        else:
            valid_batches = 0
            valid_dataset = None

        track_batches = param.num_batches - valid_batches
        track_dataset = tf.data.Dataset.from_generator(
            lambda: self.generate(track_batches, is_valid=False),
            output_types=output_types,
            output_shapes=output_shapes,
        )

        return track_dataset, valid_dataset


@beartype
def make_dataset(
    param: MUSDB18Param, mode: Literal["train", "test"] = "train"
) -> Tuple[tf.data.Dataset, Union[tf.data.Dataset, None]]:
    return MUSDB18(param, mode).make_dataset()
