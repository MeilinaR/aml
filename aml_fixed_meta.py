from meta import MetaOptimizer
import tensorflow as tf
from aml_utils import get_lambda
import os
import networks

class AMLFixedOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super().__init__(save_file_suffix="fixed", **kwargs)

  def update_parameters(self, x_next, deltas, x, j, idx):
    """Function that returns the update for x_next[j]."""
    LAMBDA = get_lambda()
    return x_next[j] + deltas[idx] + tf.math.multiply(x[j], LAMBDA)

  def save(self, sess, path=None):
    """Save meta-optimizer."""
    result = {}
    for k, net in self._nets.items():
      if path is None:
        filename = None
        key = k
      else:
        filename = os.path.join(path, f"{k}{self.save_file_suffix}{get_lambda()}.l2l")
        key = filename
      net_vars = networks.save(net, sess, filename=filename)
      result[key] = net_vars
    return result