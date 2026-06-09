## Literature Review: Microbial Co-occurrence Network Inference

### 1. Existing Methods and Their Limitations

#### 1.1 Marginal Correlation Methods

**SparCC** (Friedman & Alm, 2012, PLoS Comput Biol) exploits the log-ratio variance identity t_ij = omega_i + omega_j - 2*Sigma_ij to estimate basis correlations from compositional data. Under a sparsity assumption, it iteratively solves a p(p-1)/2 x p linear system while excluding strongly correlated pairs. Key weaknesses: (1) no guarantee that estimated correlations fall in [-1, 1]; (2) performance degrades sharply when >30% of pairs are truly correlated; (3) sensitive to zero handling via pseudocounts; (4) only estimates marginal (pairwise) correlations, not conditional dependencies; (5) high computational cost from iterative exclusion and permutation p-values. Time complexity: O(p^2 * I).

**CCLasso** (Fang et al., 2015, Bioinformatics) formulates the same problem as a penalized least-squares optimization: min sum(t_ij - omega_i - omega_j + 2*Sigma_ij)^2 + lambda*||Sigma||_1. This guarantees a positive semi-definite output but costs O(p^3) per iteration due to eigenvalue decomposition. Cross-validation for lambda adds significant overhead. The L1 penalty over-shrinks large correlations.

**REBACCA** (Ban et al., 2015, Bioinformatics) constructs a similar linear system but reformulates relative to a reference component, then applies lasso-type penalization. The system has deficient rank (rank p-1), requiring regularization. Performance is comparable to CCLasso; both generally outperform SparCC.

**propr** (Quinn et al., 2017, GigaScience) abandons correlation entirely, using proportionality: rho_p = 1 - VLR/(Var(log x_i) + Var(log x_j)). While inherently valid for compositional data, proportionality captures a different relationship than correlation — features can be correlated but not proportional.

**FastSpar** (Watts et al., 2019, Bioinformatics) is a C++ reimplementation of SparCC achieving 1000x+ speedup via Eigen library and OpenMP parallelism, but inherits all mathematical limitations.

**SparXCC** (Jensen et al., 2024, PLOS ONE) extends SparCC for cross-correlations between two compositional datasets, but accuracy depends on within-dataset SparCC quality (error propagation).

#### 1.2 Conditional Dependency Methods

**SPIEC-EASI** (Kurtz et al., 2015, PLoS Comput Biol) applies CLR transformation then estimates the precision matrix via graphical lasso or Meinshausen-Buhlmann neighborhood selection, with StARS for sparsity selection. It infers conditional dependencies (direct interactions) rather than marginal correlations. Weaknesses: (1) CLR introduces a singularity (rank-deficient covariance); (2) Gaussian assumption on CLR data is approximate; (3) difficulty recovering scale-free topologies; (4) StARS is O(100 * p^3); (5) sensitive to pseudocounts.

**gCoda** (Fang et al., 2017, J Comput Biol) uses a logistic normal generative model with a majorization-minimization algorithm wrapping iterative graphical lasso problems. More faithful to compositional structure than SPIEC-EASI but O(K * p^3) and non-convex.

**FlashWeave** (Tackmann et al., 2019, Cell Systems) uses local-to-global learning with conditional independence testing and a feed-forward heuristic. Scales to >500,000 samples but does not rigorously handle compositionality — relies on normalization and rank statistics.

#### 1.3 Recent Hybrid/Ensemble Methods

**OneNet** (2024, PLoS Comput Biol) combines 7 GGM-based methods via stability selection. **CMiNet** (2024) integrates 10 methods with threshold-based consensus. **HARMONIES** (2020) combines ZINB modeling with Dirichlet process priors and GGMs via Bayesian hierarchical inference.

#### 1.4 Deep Learning Approaches

**SIMBA-GNN** (2025) uses a heterogeneous graph transformer with simulation-augmented training for cross-feeding inference. **MNLVAE** uses longitudinal VAEs with GP priors. These are primarily designed for specific tasks (cross-feeding, temporal dynamics) rather than general co-occurrence network inference.

### 2. Systematic Weakness Analysis

| Weakness | SparCC | CCLasso | SPIEC-EASI | gCoda | FlashWeave | propr |
|---|---|---|---|---|---|---|
| Compositional data | Approximate | Good | Approximate | Good | Poor | Excellent |
| Zero inflation | Poor | Poor | Poor | Poor | Fair | Fair |
| Nonlinear relations | No | No | No | No | Partial | No |
| Conditional deps | No | No | Yes | Yes | Yes | No |
| Scalability (p>2000) | Moderate | Poor | Poor | Poor | Excellent | Good |
| Network topology fidelity | Low | Low | Medium | Medium | Medium | Low |
| GPU acceleration | No | No | No | No | No | No |

### 3. Opportunities for Innovation

Based on this analysis, a superior algorithm should:

1. **Proper generative model**: Use Dirichlet-Multinomial (not log-ratio approximations) to handle both compositionality and zero inflation in a unified Bayesian framework.

2. **Nonlinear capture**: Introduce a Variational Autoencoder with neural network decoder whose Jacobian reveals nonlinear interaction structure in latent space.

3. **Information-theoretic complement**: Estimate mutual information via Kraskov-Stoegbauer-Grassberger (KSG) on Bayesian-smoothed posteriors, capturing arbitrary dependency forms.

4. **Adaptive ensemble**: Learn optimal weights for combining heterogeneous signals, with automatic per-edge confidence calibration.

5. **Computational efficiency**: O(p^2 * n) core complexity with GPU-accelerated VAE training and randomized SVD for dimensionality reduction.

### 4. Key References

- Friedman & Alm (2012). PLoS Comput Biol. DOI: 10.1371/journal.pcbi.1002687
- Kurtz et al. (2015). PLoS Comput Biol. DOI: 10.1371/journal.pcbi.1004226
- Fang et al. (2015). Bioinformatics. DOI: 10.1093/bioinformatics/btv349
- Quinn et al. (2017). GigaScience. DOI: 10.1093/gigascience/gix054
- Watts et al. (2019). Bioinformatics. DOI: 10.1093/bioinformatics/bty751
- Tackmann et al. (2019). Cell Systems. DOI: 10.1016/j.cels.2019.03.004
- Jensen et al. (2024). PLOS ONE. DOI: 10.1371/journal.pone.0305032
- OneNet (2024). PLoS Comput Biol. DOI: 10.1371/journal.pcbi.1012627
