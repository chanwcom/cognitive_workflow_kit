# Standard imports
import glob
import os

# Third-party imports
from torch.utils import data
import tensorflow as tf

# Custom imports
from data.format import speech_data_helper


#top_dir = "/home/chanwcom/speech_database/stop/music_train_tfrecord"
train_top_dir = "/home/chanwcom/databases/stop/music_train_tfrecord"
test_top_dir =  "/home/chanwcom/databases/stop/test_0_music_random_300_tfrecord"

train_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(train_top_dir, "*tfrecord-*"))
    , compression_type="GZIP")
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "*tfrecord-*"))
    , compression_type="GZIP")


class IterDataset(data.IterableDataset):
    def __init__(self, tfrecord):
        self._dataset = tfrecord
        op = speech_data_helper.SpeechDataToWave()

        # Parses the serialized data.
        self._dataset = self._dataset.map(op.process)

    def __iter__(self):
        for data in self._dataset:
            #torch.as_tensor(val.numpy()).to(device)
            yield (data)

iter_dataset = IterDataset(dataset)
#
count = 0
for elem in iter_dataset:
    print (elem)
    count += 1

    if count > 5:
        break

