# -*- coding: utf-8 -*-

from dataclasses import dataclass
from dataclasses_io import dataclass_io


__all__ = ["ConvTasNetParam"]


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
