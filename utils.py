# -*- coding: utf-8 -*-

from absl import flags
from beartype import beartype

import conv_tasnet
import musdb18

from pathlib import Path

import sys

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks

from typing import List, Literal, Optional, Tuple, Union

if sys.version_info >= (3, 9):
    List = list
    Tuple = tuple


__all__ = [
    "define_flags",
    "load_model",
    "prepare_callbacks",
    "prepare_dataset",
    "prepare_model",
]

Model = conv_tasnet
ModelParam = conv_tasnet.ConvTasNetParam
ModelParamKWArgs = lambda: {
    "T": _FLAGS.input_length,
    "C": _FLAGS.num_sources,
    "N": _FLAGS.num_filters,
    "L": _FLAGS.filter_length,
    "B": _FLAGS.residual_channels,
    "S": _FLAGS.skipconn_channels,
    "H": _FLAGS.convblock_channels,
    "P": _FLAGS.dconv_kernel,
    "X": _FLAGS.num_blocks,
    "R": _FLAGS.num_repeats,
    "causal": _FLAGS.causal,
}

Dataset = musdb18
DatasetParam = musdb18.MUSDB18Param
DatasetParamKWArgs = lambda: {
    "dataset_root": _FLAGS.dataset_root,
    "batch_size": _FLAGS.batch_size,
    "sample_length": _FLAGS.input_length,
    "validation_split": _FLAGS.validation_split,
}

_FLAGS = flags.FLAGS
_CONFIG_FILENAME = "config.json"
_CSVLOG_FILENAME = "log.csv"


@beartype
def define_flags(mode: Literal["train", "test"] = "train"):
    flags.DEFINE_string("checkpoint_dir", None, "Directory of results", required=True)
    if mode == "train":
        flags.DEFINE_integer(
            "num_epochs", 3, "Number of epochs", lower_bound=0, upper_bound=99999
        )
    else:  # mode == "test"
        flags.DEFINE_integer(
            "checkpoint_idx",
            0,
            "Index of checkpoint for evaluation",
            lower_bound=0,
        )
    Model.define_flags()
    Dataset.define_flags()


def _initialize_model_param(checkpoint_dir: Path) -> ModelParam:
    model_param = Model.make_param(**ModelParamKWArgs())
    model_param.save(checkpoint_dir / _CONFIG_FILENAME, overwrite=False)
    return model_param


@beartype
def prepare_model(checkpoint_dir: Path) -> Tuple[keras.Model, int]:
    model, initial_epoch = None, 0

    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_param = _initialize_model_param(checkpoint_dir)
    elif not checkpoint_dir.is_dir():
        raise FileExistsError("not a directory: %s" % checkpoint_dir)
    else:
        try:
            model_param = Model.load_param(checkpoint_dir / _CONFIG_FILENAME)
        except (FileNotFoundError, KeyError):
            model_param = _initialize_model_param(checkpoint_dir)
        except Exception as e:
            raise e
        finally:
            checkpoints = sorted(checkpoint_dir.glob("*.ckpt*"), reverse=True)
            for checkpoint in checkpoints:
                if checkpoint.is_file():
                    latest_checkpoint = checkpoint.with_suffix("")
                    model = Model.make_model(model_param)
                    model.load_weights(latest_checkpoint)
                    initial_epoch = int(latest_checkpoint.stem)
                    break

    if not model:
        model = Model.make_model(model_param)

    return model, initial_epoch


@beartype
def load_model(
    checkpoint_dir: Path, checkpoint_idx: Optional[int] = None
) -> keras.Model:
    if not checkpoint_dir.exists():
        raise FileNotFoundError("no such directory: %s" % checkpoint_dir)
    elif not checkpoint_dir.is_dir():
        raise FileExistsError("not a directory: %s" % checkpoint_dir)
    else:
        model_param = Model.load_param(checkpoint_dir / _CONFIG_FILENAME)
        if checkpoint_idx is not None and checkpoint_idx > 0:
            checkpoint_path = checkpoint_dir / f"{checkpoint_idx:05d}.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError("no such file: %s" % checkpoint_path)
        else:
            checkpoints = sorted(checkpoint_dir.glob("*.ckpt*"), reverse=True)
            for checkpoint in checkpoints:
                if checkpoint.is_file():
                    latest_checkpoint = checkpoint.with_suffix("")
                    model = Model.make_model(model_param)
                    model.load_weights(latest_checkpoint)
                    return model
            else:
                raise FileNotFoundError("no *.ckpt file: %s" % checkpoint_dir)


def _initialize_dataset_param(checkpoint_dir: Path) -> DatasetParam:
    dataset_param = Dataset.make_param(**DatasetParamKWArgs())
    dataset_param.save(checkpoint_dir / _CONFIG_FILENAME, overwrite=False)
    return dataset_param


@beartype
def prepare_dataset(
    checkpoint_dir: Path, mode: Literal["train", "test"] = "train"
) -> Tuple[tf.data.Dataset, Union[tf.data.Dataset, None]]:
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dataset_param = _initialize_dataset_param(checkpoint_dir)
    elif not checkpoint_dir.is_dir():
        raise FileExistsError("not a directory: %s" % checkpoint_dir)
    else:
        try:
            dataset_param = Dataset.load_param(checkpoint_dir / _CONFIG_FILENAME)
        except (FileNotFoundError, KeyError):
            dataset_param = _initialize_dataset_param(checkpoint_dir)
        except Exception as e:
            raise e

    dataset_root = Path(dataset_param.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError("no such directory: %s" % dataset_root)
    else:
        return Dataset.make_dataset(dataset_param, mode)


@beartype
def prepare_callbacks(
    path: Path,
    callback_list: List[callbacks.Callback] = [],
    mode: Literal["train", "test"] = "train",
) -> List[callbacks.Callback]:
    callback_list.append(callbacks.TerminateOnNaN())
    if mode == "train":
        callback_list.extend(
            [
                callbacks.CSVLogger(path / _CSVLOG_FILENAME, append=True),
                keras.callbacks.ModelCheckpoint(
                    filepath=path / "{epoch:05d}.ckpt",
                    verbose=1,
                    save_weigthts_only=True,
                ),
            ]
        )
    return callback_list
