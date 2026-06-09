"""AdaCoNet: Adaptive Compositional Network Inference.

A microbial co-occurrence network inference algorithm combining:
1. Dirichlet-Multinomial posterior correlation
2. Spearman correlation on Bayesian CLR (principled zero-handling)
3. Proportionality for composition-aware association
4. Adaptive ensemble with StARS stability selection
"""

__version__ = "0.1.0"

from .pipeline import AdaCoNetPipeline

__all__ = ["AdaCoNetPipeline"]
