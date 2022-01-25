# -*- coding: utf-8 -*-

from absl import app
from absl import flags
import pathlib
import utils


FLAGS = flags.FLAGS


def train(_):
    try:
        checkpoint_dir = pathlib.Path(FLAGS.checkpoint_dir)
        model, initial_epoch = utils.prepare_model(checkpoint_dir)
        train_dataset, valid_dataset = utils.prepare_dataset(checkpoint_dir)
        callback_list = utils.prepare_callbacks(checkpoint_dir)
        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            initial_epoch=initial_epoch,
            epochs=FLAGS.num_epochs + initial_epoch,
            callbacks=callback_list,
        )
    except KeyboardInterrupt:
        model.save(checkpoint_dir / "model")


if __name__ == "__main__":
    utils.define_flags()
    app.run(train)
