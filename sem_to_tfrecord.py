# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Script to download the Imagenet dataset and upload to gcs.

To run the script setup a virtualenv with the following libraries installed.
- `gcloud`: Follow the instructions on
  [cloud SDK docs](https://cloud.google.com/sdk/downloads) followed by
  installing the python api using `pip install gcloud`.
- `google-cloud-storage`: Install with `pip install google-cloud-storage`
- `tensorflow`: Install with `pip install tensorflow`

Once you have all the above libraries setup, you should register on the
[Imagenet website](http://image-net.org/download-images) to get your
username and access_key.

Make sure you have around 300GB of disc space available on the machine where
you're running this script. You can run the script using the following command.
```
python imagenet_to_gcs.py \
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/IMAGENET_DIR" \
  --local_scratch_dir="./imagenet" \
  --imagenet_username=FILL_ME_IN \
  --imagenet_access_key=FILL_ME_IN \
```

Optionally if the raw data has already been downloaded you can provide a direct
`raw_data_directory` path. If raw data directory is provided it should be in
the format:
- Training images: train/n03062245/n03062245_4620.JPEG
- Validation Images: validation/ILSVRC2012_val_00000001.JPEG
- Validation Labels: synset_labels.txt
"""

import math
import os
import random
import tarfile
import urllib

from PIL import Image

from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string(
    'raw_data_dir', None, 'Directory path for raw Imagenet dataset. '
    'Should have train and validation subdirectories inside it.')

FLAGS = flags.FLAGS

TRAINING_SHARDS = 1024
VALIDATION_SHARDS = 128

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'test'


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, height, width):
    """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    assert synset == None

    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/colorspace': _bytes_feature(colorspace),
                'image/channels': _int64_feature(channels),
                'image/class/label': _int64_feature(label),
                'image/format': _bytes_feature(image_format),
                'image/filename': _bytes_feature(str.encode(os.path.basename(filename) + ".JPEG")),
                'image/encoded': _bytes_feature(image_buffer)
            }))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data)

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        image = image[:, :, 0:3]
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
    # Read the image file.
    print(filename + ".JPEG")
    with tf.gfile.FastGFile(filename + ".JPEG", 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
    """Processes and saves list of images as TFRecords.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    output_file: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
  """
    assert synsets == None
    assert labels == None

    writer = tf.python_io.TFRecordWriter(output_file)

    for filename in filenames:
        image_buffer, height, width = _process_image(filename, coder)
        label = int(os.path.basename(
            os.path.dirname(filename)))  # use the parent dir name as label
        example = _convert_to_example(filename, image_buffer, label, None,
                                      height, width)
        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(filenames, synsets, labels, output_directory, prefix,
                     num_shards):
    """Processes and saves list of images as TFRecords.

  Args:
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: map of string to integer; id for all synset labels
    output_directory: path where output files should be created
    prefix: string; prefix for each file
    num_shards: number of chucks to split the filenames into

  Returns:
    files: list of tf-record filepaths created from processing the dataset.
  """
    assert synsets == None
    assert labels == None

    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize:(shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files, None,
                                   labels)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


def convert_to_tf_records(raw_data_dir):
    """Convert the Imagenet dataset into TF-Record dumps."""

    # Shuffle training records to ensure we are distributing classes
    # across the batches.
    random.seed(0)

    def make_shuffle_idx(n):
        order = list(range(n))
        random.shuffle(order)
        return order

    # Glob all the training files
    training_files = tf.gfile.Glob(
        os.path.join(raw_data_dir, TRAINING_DIRECTORY, '*', '*.tiff'))

    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]

    # Glob all the validation files
    validation_files = sorted(
        tf.gfile.Glob(
            os.path.join(raw_data_dir, VALIDATION_DIRECTORY, '*', '*.tiff')))

    print(training_files)

    # convert TIFF to JPEG using PIL
    for tiff_image in training_files + validation_files:
        im = Image.open(tiff_image)
        print("Generating jpeg for %s" % tiff_image)
        out = im.convert("RGB")
        out.save(tiff_image + ".JPEG", "JPEG", quality=100)

    # Create training data
    tf.logging.info('Processing the training data.')
    training_records = _process_dataset(
        training_files, None, None,
        os.path.join(FLAGS.raw_data_dir, TRAINING_DIRECTORY),
        TRAINING_DIRECTORY, TRAINING_SHARDS)

    # Create validation data
    tf.logging.info('Processing the validation data.')
    validation_records = _process_dataset(
        validation_files, None, None,
        os.path.join(FLAGS.raw_data_dir, VALIDATION_DIRECTORY),
        VALIDATION_DIRECTORY, VALIDATION_SHARDS)

    return training_records, validation_records


def main(argv):  # pylint: disable=unused-argument
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.raw_data_dir is None:
        raise ValueError('SEM data directory path must be provided.')

    # Convert the raw data into tf-records
    training_records, validation_records = convert_to_tf_records(
        FLAGS.raw_data_dir)


if __name__ == '__main__':
    app.run(main)
