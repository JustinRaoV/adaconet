# AdaCoNet

**Ada**ptive **Co**mpositional **Net**work inference — a multi-signal ensemble framework for microbial co-occurrence network inference from compositional sequencing data.

## Overview

AdaCoNet integrates four complementary statistical layers with a model-based adaptive weighting scheme to robustly infer microbial co-occurrence networks across diverse data-generating mechanisms:

| Layer | Method | Captures |
|-------|--------|----------|
| 1 | **DM Posterior Correlation** | Count-level Bayesian covariance (handles zeros naturally) |
| 2 | **Spearman on CLR** | Rank correlation in log-ratio space (compositional correction) |
| 3 | **VLR Proportionality** | Constant-ratio preservation between taxa pairs |
| 4 | **Gaussian Copula** | Latent Gaussian correlations (robust to zero-inflation) |

The **model-based adaptive ensemble** uses the Dirichlet-Multinomial sufficient statistic |α|/p to automatically determine which layers are appropriate for the given data:
- |α|/p ≥ 0.05 (multinomial regime) → all 4 layers, equal weights
- |α|/p < 0.05 (copula regime) → Spearman excluded, remaining 3 layers equally weighted

This eliminates empirical tuning while adapting to the data's generative properties.

## Key Results

Benchmarked against 8 competing methods (SparCC, REBACCA, CCLasso, FastSpar, Spearman CLR, Proportionality, Graphical Lasso, SPIEC-EASI):

| Simulator | AdaCoNet AUROC | Best Competitor | Speedup vs SparCC |
|-----------|:-:|:-:|:-:|
| v4 direct-covariance (N=200, P=50) | **0.863** | SparCC 0.840 | 7× |
| v4 direct-covariance (N=500, P=500) | **0.799** | SparCC 0.792 | 570× |
| SparseDOSSA2 Gaussian copula (N=500) | **0.865** | Proportionality 0.899 | 280× |

## Installation

```bash
# Clone the repository
git clone https://github.com/JustinRaoV/adaconet.git
cd adaconet

# Create virtual environment (requires Python ≥ 3.9)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```python
from adaconet import AdaCoNetPipeline
import numpy as np

# Load your count matrix (samples × taxa)
counts = np.loadtxt("your_data.csv", delimiter=",", skiprows=1)

# Run AdaCoNet
pipe = AdaCoNetPipeline(verbose=True)
pipe.fit(counts)

# Get results
results = pipe.get_intermediate_results()
W = results["W"]              # Ensemble score matrix (p × p)
adjacency = results["adjacency"]  # Binary network (p × p)
tau = results["tau"]          # Selected StARS threshold
alpha_p = results["alpha_sum"] / counts.shape[1]  # DM model fit diagnostic
```

## Running Benchmarks

### Simulated data (v4 direct-covariance)

```bash
python run_benchmarks.py --n-repeats 3 --output-dir results/
```

### SparseDOSSA2 validation

```bash
# Step 1: Generate data with R
Rscript run_sparsedossa2_benchmark.R

# Step 2: Run benchmark
python run_sparsedossa2_benchmark.py
```

## Project Structure

```
adaconet/
├── src/adaconet/           # Core algorithm
│   ├── __init__.py         # Package exports
│   ├── pipeline.py         # Main AdaCoNetPipeline (4-layer + adaptive ensemble)
│   ├── dm_foundation.py    # Dirichlet-Multinomial model fitting
│   ├── ensemble.py         # Ensemble combination + StARS
│   └── utils.py            # CLR transforms, normalization, filtering
├── benchmarks/             # Benchmark framework
│   ├── simulator.py        # MicrobialNetworkSimulator
│   ├── metrics.py          # NetworkMetrics (AUROC, AUPRC, F1, topology)
│   └── runner.py           # BenchmarkRunner
├── run_benchmarks.py       # 9-method benchmark entry point
├── run_sparsedossa2_benchmark.R  # SparseDOSSA2 data generation
├── run_sparsedossa2_benchmark.py # SparseDOSSA2 evaluation
├── docs/
│   ├── paper_short.tex     # Short paper (OUP template, 4 pages)
│   ├── paper_short.pdf     # Compiled paper
│   └── figures/            # Publication figures
└── pyproject.toml          # Project configuration
```

## Dependencies

- numpy ≥ 1.24
- scipy ≥ 1.10
- scikit-learn ≥ 1.3
- pandas ≥ 2.0
- networkx ≥ 3.0
- matplotlib ≥ 3.7
- seaborn ≥ 0.12

## Citation

If you use AdaCoNet in your research, please cite:

> Rao, J. & Liang, X. (2026). AdaCoNet: Model-Adaptive Ensemble Inference for Microbial Co-occurrence Networks. *Briefings in Bioinformatics*, XX(x).

## License

MIT
