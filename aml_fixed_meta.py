from meta import MetaOptimizer
from aml_utils import get_lambda

class AMLFixedOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super().__init__(save_file_suffix="fixed", **kwargs)

  def update_parameters(self, x_next, deltas, x, j, idx):
    """Function that returns the update for x_next[j]."""
    LAMBDA = 0.1 #get_lambda()
    return x_next[j] + deltas[idx] + LAMBDA * x[j]