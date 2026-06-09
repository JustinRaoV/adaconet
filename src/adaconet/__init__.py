"""AdaCoNet: Adaptive Compositional Network Inference.

A microbial co-occurrence network inference algorithm combining:
1. Dirichlet-Multinomial posterior correlation
2. Spearman correlation on Bayesian CLR (principled zero-handling)
3. VLR Proportionality (composition-aware association)
4. Gaussian Copula correlation (latent normal-space estimation)

Model-based adaptive ensemble: |alpha|/p selects appropriate layers.
"""

__version__ = "0.1.0"

from .pipeline import AdaCoNetPipeline

__all__ = ["AdaCoNetPipeline"]
