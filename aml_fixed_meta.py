from meta import MetaOptimizer

class AMLFixedOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super().__init__(save_file_suffix="fixed", **kwargs)

  # TODO: override the functions to match our proposed update rule