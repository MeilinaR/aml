from .meta import MetaOptimizer

class AMLOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super.__init__(self, **kwargs)

  # TODO: override the functions to match our proposed update rule