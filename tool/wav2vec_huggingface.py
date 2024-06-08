# Standard imports
import glob
import os

# Third-party imports
from torch.utils import data
import tensorflow as tf
import torch

# Custom imports
from data.format import speech_data_helper
from typing import Any, Dict, List, Optional, Union

#db_top_dir="/home/chanwcom/databases/"
db_top_dir = "/home/chanwcom/speech_database"
train_top_dir = os.path.join(db_top_dir, "stop/music_train_tfrecord")
test_top_dir = os.path.join(db_top_dir,
                            "stop/test_0_music_random_300_tfrecord")

# yapf: disable
op = speech_data_helper.SpeechDataToWave()
train_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(train_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.map(op.process)
# yapf: enable

# yapf: disable
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.map(op.process)
# yapf: enable

def to_torch(inputs: dict):
    for key in inputs.keys():
        inputs[key] = torch.tensor(inputs[key].numpy())

    return inputs


class IterDataset(data.IterableDataset):
    def __init__(self, tf_dataset):
        self._dataset = tf_dataset
        op = speech_data_helper.SpeechDataToWave()

        # The following line is neede, otherwise ..dataset.map will not work

        # Parses the serialized data.
        #self._dataset = self._dataset.map(op.process)

    def __iter__(self):
        for data in self._dataset:
            #torch.as_tensor(val.numpy()).to(device)
            output = {}
            output["input_values"] = tf.squeeze(data[0]["SEQ_DATA"],
                                                axis=2).numpy()
            output["input_length"] = tf.squeeze(data[0]["SEQ_LEN"]).numpy()
            output["labels"] = data[1]["SEQ_DATA"]

            yield (output)


iter_dataset = IterDataset(test_dataset)

import pdb

pdb.set_trace()

for elem in iter_dataset:
    print(elem)

count = 0
for elem in iter_dataset:
    print(count)
    print(elem)
    count += 1

    import pdb
    pdb.set_trace()

    if count > 5:
        break
