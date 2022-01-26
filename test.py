# -*- coding: utf-8 -*-

from absl import app
from absl import flags
import pathlib
import utils


FLAGS = flags.FLAGS


def train(_):
    checkpoint_dir = pathlib.Path(FLAGS.checkpoint_dir)
    model = utils.load_model(checkpoint_dir, FLAGS.checkpoint_idx)
    dataset, _ = utils.prepare_dataset(checkpoint_dir, mode="test")
    model.evaluate(dataset)


if __name__ == "__main__":
    utils.define_flags(mode="test")
    app.run(train)
