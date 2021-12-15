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
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

from aml_utils import set_lambda
import problems


def run_epoch(sess, cost_op, ops, reset, num_unrolls, cost_history_path=None):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  cost_histories = []

  for _ in xrange(num_unrolls):
    cost = sess.run([cost_op] + ops)[0]
    if cost_history_path is not None:
      cost_histories.append(cost)

  if cost_history_path is not None:
    with open(cost_history_path, 'a') as f:
      f.write(','.join(str(cost) for cost in cost_histories))
      f.write('\n')
  return timer() - start, cost


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_net_path(name, path, file_suffix=None):
  if path is None:
    return None

  if file_suffix is None:
    file_suffix = ""
  elif file_suffix == 'fixed':
    possible_nets = os.listdir(os.path.join(path))
    if len(possible_nets) == 1:
      net_path = possible_nets[0]
    else:
      (print(f"{i}\t{fn}") for i, fn in enumerate(possible_nets))
      n = input("Enter the number of the file you want to use: ")
      net_path = possible_nets[n]

    LAMBDA = net_path.split("fixed")[-1].split('.l2l')[0]
    print(f"Extracted LAMBDA value {LAMBDA}")
    set_lambda(LAMBDA)
    return os.path.join(path, net_path)
  elif file_suffix == 'learned':
    net_path = os.listdir(os.path.join(path))[0]
    print(f"Using the configuration stored in {net_path}")
    # TODO: read correct parameter settings once implemented
    return os.path.join(path, net_path)

  else: # Original L2L optimizer
    return os.path.join(path, name + ".l2l")


def get_default_net_config(name, path, file_suffix=None):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": get_net_path(name, path, file_suffix=file_suffix)
  }


def get_config(problem_name, path=None, file_suffix=None):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": get_net_path("cw", path, file_suffix=file_suffix)
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": get_net_path("cw", path, file_suffix=file_suffix)
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.1}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=10)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path, file_suffix=file_suffix)
    }}
    net_assignments = None
  elif problem_name == "mnist":
    mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, file_suffix=file_suffix)}
    net_assignments = None
  elif problem_name == "cifar":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, file_suffix=file_suffix)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
        "conv": get_default_net_config("conv", path, file_suffix=file_suffix),
        "fc": get_default_net_config("fc", path, file_suffix=file_suffix)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  return problem, net_config, net_assignments
