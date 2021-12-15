from meta import MetaOptimizer

class AMLLearnedOptimizer(MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  # TODO: override the functions to match our proposed update rule