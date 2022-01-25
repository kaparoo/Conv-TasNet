# -*- coding: utf-8 -*-

from beartype import beartype
from dataclasses import dataclass
from dataclasses_io import dataclass_io

from os import PathLike
from pathlib import Path
from typing import Union


__all__ = ["load_param", "make_param", "MUSDB18Param"]


@dataclass_io
@dataclass
class MUSDB18Param:
    dataset_root: str
    num_batches: int = 200
    batch_size: int = 400
    sample_length: int = 2048
    validation_split: float = 0.0


@beartype
def load_param(path: Union[str, PathLike, Path]) -> MUSDB18Param:
    return MUSDB18Param.load(path)


@beartype
def make_param(
    dataset_root: str,
    num_batches: int = 200,
    batch_size: int = 400,
    sample_length: int = 2048,
    validation_split: float = 0.0,
) -> MUSDB18Param:
    return MUSDB18Param(
        dataset_root, num_batches, batch_size, sample_length, validation_split
    )
