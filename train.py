"""Train fingerprint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import smile as sm

from tensorflow.models.rnn.translate import data_utils

from unsupervised import seq2seq_model

# TODO: in the future we need to implement the build model option with data script.
with sm.app.flags.Subcommand("build", dest="action"):
    sm.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
    sm.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
    sm.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
    sm.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                              "Learning rate decays by this much.")
    sm.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                              "Clip gradients to this norm.")

with sm.app.flags.Subcommand("train", dest="action"):
    sm.app.flags.DEFINE_string("model_dir", "", "model path of the seq2seq fingerprint.",
                               required=True)
    sm.app.flags.DEFINE_string("train_data", "", "train_data for seq2seq fp train.",
                               required=True)
    sm.app.flags.DEFINE_string("test_data", "", "test data path of the seq2seq fp eval.",
                               required=True)
    sm.app.flags.DEFINE_integer("batch_size", 128,
                                "Batch size to use during training.")
    sm.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                                "How many training steps to do per checkpoint.")



FLAGS = sm.app.flags.FLAGS

def read_data(source_path, buckets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in source.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            source = source_file.readline()
    return data_set

def train(train_data, test_data): # pylint: disable=too-many-locals
    """Train script."""
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size

    with tf.Session() as sess:
        # Create model.
        model = seq2seq_model.Seq2SeqModel.load_model_from_dir(
            model_dir, False, sess=sess)
        buckets = model.buckets
        model.batch_size = batch_size

        # Read data into buckets and compute their sizes.
        print("Reading train data from %s..." % train_data)
        train_set = read_data(train_data, buckets)
        print("Reading test data from %s..." % test_data)
        test_set = read_data(test_data, buckets)

        # Subject to remove in the future.
        # if FLAGS.train_with_dev:
        #     print("Training with development (testing) set for pretrain use...")
        #     train_set = [a + b for a, b in zip(train_set, test_set)]
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print ("global step %d learning rate %.4f step-time %.6f perplexity "
                       "%.6f" % (model.global_step.eval(), model.learning_rate_op.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                model.save_model_to_dir(model_dir, sess=sess)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("  eval: bucket %d perplexity %.6f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def main(_):
    """Entry function for the script."""
    if FLAGS.action == "build":
        raise NotImplementedError("Model build action not implemented.")
    elif FLAGS.action == "train":
        train(FLAGS.train_data, FLAGS.test_data)
    else:
        print("Unsupported action: %s" % FLAGS.action)

if __name__ == "__main__":
    sm.app.run()