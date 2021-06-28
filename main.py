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
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import resnet_model
import vgg_preprocessing

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument(
    '--data_dir',
    type=str,
    default='/tmp/cifar10_data',
    help='The path to the CIFAR-10 data directory.')

parser.add_argument(
    '--model_dir',
    type=str,
    default='/tmp/cifar10_model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size',
    type=int,
    default=50,
    choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_epochs',
    type=int,
    default=100,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval',
    type=int,
    default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size',
    type=int,
    default=32,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--COT',
    '-c',
    action='store_true',
    help='Using Complement Objective Training (COT)')

parser.add_argument(
    '--data_format',
    type=str,
    default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 255

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500

sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True))


def complement_entropy_func(logits, onehot_labels):
    batch_size = FLAGS.batch_size
    labels = tf.argmax(onehot_labels, axis=1)
    softmax_logits = tf.nn.softmax(logits, axis=1)
    Yg = tf.math.reduce_sum(softmax_logits * onehot_labels, axis=1)
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = softmax_logits / tf.expand_dims(Yg_, axis=1)
    Px_log = tf.math.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = tf.ones_like(onehot_labels) - onehot_labels
    output = Px * Px_log * y_zerohot
    loss = tf.math.reduce_sum(output)
    loss /= tf.cast(batch_size, tf.float32)
    loss /= tf.cast(_NUM_CLASSES, tf.float32)
    return loss


def filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return [
            os.path.join(data_dir, 'train', 'train-%05d-of-01024' % i)
            for i in range(1024)
        ]
    else:
        return [
            os.path.join(data_dir, 'test', 'test-%05d-of-00128' % i)
            for i in range(128)
        ]


def record_parser(value, is_training):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
        tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
        tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
        tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
        tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
        tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
        tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
        tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = vgg_preprocessing.preprocess_image(
        image=image,
        output_height=_DEFAULT_IMAGE_SIZE,
        output_width=_DEFAULT_IMAGE_SIZE,
        is_training=is_training)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

    return image, tf.one_hot(label, _NUM_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input function which provides batches for train or eval."""
    dataset = tf.data.Dataset.from_tensor_slices(
        filenames(is_training, data_dir))

    if is_training:
        dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(
        lambda value: record_parser(value, is_training), num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


def resnet_model_fn(features, labels, mode, params):
    """Our model_fn for ResNet to be used with our Estimator."""
    tf.summary.image('images', features, max_outputs=6)

    network = resnet_model.imagenet_resnet_v2(
        params['resnet_size'], _NUM_CLASSES, params['data_format'])
    logits = network(
        inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    complement_entropy = complement_entropy_func(
        logits=logits, onehot_labels=labels)

    # Create a tensor named complement_entropy for logging purposes.
    tf.identity(complement_entropy, name='complement_entropy')
    tf.summary.scalar('complement_entropy', complement_entropy)

    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n([
        tf.nn.l2_loss(v) for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

    cot_loss = complement_entropy
    if params['COT']:
        loss += cot_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.

        # TODO: need warm-up trainig for large minibatch
        #if epoch <= 9 and lr > 0.1:
        #    lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.

        initial_learning_rate = 0.1 * params['batch_size'] / 128
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        global_step = tf.train.get_or_create_global_step()

        global_step_scheduling = tf.cast(global_step, tf.int32)
        if params['COT']:
            global_step_scheduling = tf.cast(global_step / 2, tf.int32)

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [
            int(batches_per_epoch * epoch) for epoch in [100, 150, 200]
        ]
        values = [
            initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]
        ]
        learning_rate = tf.train.piecewise_constant(global_step_scheduling,
                                                    boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=_MOMENTUM)

        # Scale the learning rate linearly with the batch size. When the batch size
        # is 128, the learning rate should be 0.1.

        #TODO: need warm-up trainig for large minibatch
        #if epoch <= 9 and lr > 0.1:
        #    lr = 0.1 + (complement_learning_rate - 0.1) * epoch / 10.

        cot_initial_learning_rate = 0.1 * params['batch_size'] / 128

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        cot_boundaries = [
            int(batches_per_epoch * epoch) for epoch in [100, 150, 200]
        ]
        cot_values = [
            initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]
        ]
        cot_learning_rate = tf.train.piecewise_constant(
            global_step_scheduling, cot_boundaries, cot_values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(cot_learning_rate, name='cot_learning_rate')
        tf.summary.scalar('cot_learning_rate', cot_learning_rate)

        cot_optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            orig_train_op = optimizer.minimize(loss, global_step)
            train_op = tf.group(orig_train_op)
            if params['COT']:
                cot_train_op = optimizer.minimize(cot_loss, global_step)
                train_op = tf.group(orig_train_op, cot_train_op)

    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


#feature_spec = {'x': tf.FixedLenFeature([4],tf.float32)}
#
#def serving_input_receiver_fn():
#    serialized_tf_example = tf.placeholder(dtype=tf.string,
#                                         shape=[None],
#                                         name='input_tensors')
#    receiver_tensors = {'inputs': serialized_tf_example}
#
#    features = tf.parse_example(serialized_tf_example, feature_spec)
#    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
        save_checkpoints_secs=1e9, session_config=sess_config)
    resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params={
            'resnet_size': FLAGS.resnet_size,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'COT': FLAGS.COT
        })

    #exporter = tf.estimator.BestExporter(
    #    name="best_exporter",
    #    serving_input_receiver_fn=serving_input_receiver_fn,
    #    exports_to_keep=5)

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'complement_entropy': 'complement_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        print('Starting a training cycle.')
        resnet_classifier.train(
            input_fn=
            lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
            hooks=[logging_hook])

        # Evaluate the model and print results
        print('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
        print(eval_results)

        #export_model_path = os.path.join(FLAGS.model_dir, '../export_model')
        #exporter.export(estimator=cifar_classifier,
        #        export_path=export_model_path,
        #        checkpoint_path=export_model_path,
        #        eval_result=eval_results,
        #        is_the_final_export=False)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
