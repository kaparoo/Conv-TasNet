# -*- coding: utf-8 -*-

from beartype import beartype
from dataclasses import dataclass
from dataclasses_io import dataclass_io

from os import PathLike
from pathlib import Path
from typing import Union


__all__ = ["ConvTasNetParam", "load_param", "make_param"]


@dataclass_io
@dataclass
class ConvTasNetParam:
    """Container for Conv-TasNet hyperparameters."""

    T: int = 2048
    C: int = 4
    N: int = 512
    L: int = 16
    B: int = 128
    S: int = 128
    H: int = 512
    P: int = 3
    X: int = 8
    R: int = 3
    causal: bool = False


@beartype
def load_param(path: Union[str, PathLike, Path]) -> ConvTasNetParam:
    return ConvTasNetParam.load(path)


@beartype
def make_param(
    T: int = 2048,
    C: int = 4,
    N: int = 512,
    L: int = 16,
    B: int = 128,
    S: int = 128,
    H: int = 512,
    P: int = 3,
    X: int = 8,
    R: int = 3,
    causal: bool = False,
) -> ConvTasNetParam:
    return ConvTasNetParam(T, C, N, L, B, S, H, P, X, R, causal)
