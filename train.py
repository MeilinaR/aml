# Copyright 2016 Google Inc.
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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sys import argv
from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
from aml_fixed_meta import AMLFixedOptimizer
from aml_learned_meta import AMLLearnedOptimizer
import util
from aml_utils import set_lambda, get_lambda

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 100, "Log period.")
flags.DEFINE_integer("evaluation_period", 1000, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")

flags.DEFINE_string("which_aml", None, "Custom optimizer to use, 'fixed' or 'learned'.")
flags.DEFINE_float("fixed_lambda", 0.01, "Value for the 'fixed' lambda optimizer.")

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  set_lambda(FLAGS.fixed_lambda)
  if FLAGS.which_aml == 'fixed':
    print(f"LAMBDA set to {get_lambda()}")

  if FLAGS.save_path is not None:
    if os.path.exists(FLAGS.save_path):
      print("Folder {} already exists".format(FLAGS.save_path))
    else:
      os.mkdir(FLAGS.save_path)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)

  # Optimizer setup.
  optimizer = {
    None: meta.MetaOptimizer(**net_config),
    'fixed': AMLFixedOptimizer(**net_config),
    'learned': AMLLearnedOptimizer(**net_config),
  }[FLAGS.which_aml]

  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)
  step, update, reset, cost_op, _ = minimize

  with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls)
      total_time += time
      total_cost += cost

      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                      num_unrolls)
          eval_time += time
          eval_cost += cost

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)

        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          infix = FLAGS.which_aml
          if infix is None:
            infix = ""
          suffix = ""
          if FLAGS.which_aml == 'fixed':
            suffix = str(get_lambda())
          savefile_name = f"cw{infix}{suffix}.l2l"
          for f in os.listdir(FLAGS.save_path):
            if f == savefile_name:
              os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
          optimizer.save(sess, FLAGS.save_path)
          best_evaluation = eval_cost


if __name__ == "__main__":
  tf.app.run()
