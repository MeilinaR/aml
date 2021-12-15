from meta import MetaOptimizer

class AMLLearnedOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super().__init__(save_file_suffix='learned', **kwargs)

  # TODO: override the functions to match our proposed update rule