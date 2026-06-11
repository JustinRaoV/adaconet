"""AdaCoNet: Adaptive Compositional Network Inference.

A microbial co-occurrence network inference framework based on the
Compositional Copula Model (CCM).  Integrates four complementary layers:

1. Dirichlet-Multinomial posterior correlation
2. Spearman correlation on Bayesian CLR (when CLR variance inflation is low)
3. VLR Proportionality (composition-aware association)
4. Compositional Copula Model — EM-based latent correlation estimation

Theory-driven adaptive ensemble: per-layer weights derived from the
CLR Variance Inflation Theorem (1 + p/α₀), with a Phase Transition
at α₀/p ≈ c* separating reliable from unreliable Spearman regimes.
"""

__version__ = "0.1.0"

from .pipeline import AdaCoNetPipeline

__all__ = ["AdaCoNetPipeline"]
