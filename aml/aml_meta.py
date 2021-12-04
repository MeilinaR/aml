# See https://stackoverflow.com/questions/35166821/valueerror-attempted-relative-import-beyond-top-level-package
import os
import sys
sys.path.append(os.path.realpath('.'))

from .. import meta

class AMLOptimizer(meta.MetaOptimizer):
  """Modified Learning to learn (meta) optimizer."""
  
  def __init__(self, **kwargs):
    super.__init__(self, **kwargs)

  # TODO: override the functions to match our proposed update rule