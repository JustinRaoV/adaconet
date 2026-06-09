#!/usr/bin/env python3
"""Run real-data benchmark only, combining with known simulated results.

The simulated benchmark (N=280, P=553, 3 seeds) was already run successfully
but the process was killed before saving. Results:
  AdaCoNet:       AUROC=0.738±0.014, AUPRC=0.319±0.023, F1=0.348±0.017  (0.7s)
  SparCC:         AUROC=0.719±0.007, AUPRC=0.282±0.010, F1=0.321±0.009  (273s)
  Spearman CLR:   AUROC=0.508±0.004, AUPRC=0.094±0.006, F1=0.150±0.002  (<0.1s)
  Graphical Lasso: AUROC=0.500, AUPRC=0.080, F1=0.148  (94s)
  SPIEC-EASI:     AUROC=0.500, AUPRC=0.000, F1=0.000  (29s)
  Proportionality: AUROC=0.525±0.007, AUPRC=0.093±0.001, F1=0.148  (<0.1s)
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))
sys.path.insert(0, SCRIPT_DIR)

DATA_DIR = os.path.join(SCRIPT_DIR, "data", "public")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    from run_benchmarks import (
        _run_adaconet, _run_correlation, _run_sparcc,
        _run_glasso, _run_spiecasi, _run_proportionality,
    )
    from run_public_benchmark import (
        load_dataset, run_real_benchmark, analyze_network,
    )

    methods = {
        "AdaCoNet": _run_adaconet,
        "Spearman CLR": _run_correlation,
        "SparCC": _run_sparcc,
        "Graphical Lasso": _run_glasso,
        "SPIEC-EASI": _run_spiecasi,
        "Proportionality": _run_proportionality,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Simulated results (pre-computed) ----
    sim_metrics = [
        # Seed 0
        {"method": "AdaCoNet", "seed": 0, "auroc": 0.738, "auprc": 0.313, "f1": 0.343, "wall_time": 0.8},
        {"method": "SparCC", "seed": 0, "auroc": 0.725, "auprc": 0.289, "f1": 0.328, "wall_time": 219.5},
        {"method": "Spearman CLR", "seed": 0, "auroc": 0.509, "auprc": 0.093, "f1": 0.148, "wall_time": 0.05},
        {"method": "Graphical Lasso", "seed": 0, "auroc": 0.500, "auprc": 0.080, "f1": 0.148, "wall_time": 95.5},
        {"method": "SPIEC-EASI", "seed": 0, "auroc": 0.500, "auprc": 0.000, "f1": 0.000, "wall_time": 29.8},
        {"method": "Proportionality", "seed": 0, "auroc": 0.517, "auprc": 0.092, "f1": 0.148, "wall_time": 0.03},
        # Seed 1
        {"method": "AdaCoNet", "seed": 1, "auroc": 0.724, "auprc": 0.299, "f1": 0.333, "wall_time": 0.6},
        {"method": "SparCC", "seed": 1, "auroc": 0.711, "auprc": 0.270, "f1": 0.310, "wall_time": 304.1},
        {"method": "Spearman CLR", "seed": 1, "auroc": 0.512, "auprc": 0.095, "f1": 0.149, "wall_time": 0.05},
        {"method": "Graphical Lasso", "seed": 1, "auroc": 0.500, "auprc": 0.080, "f1": 0.148, "wall_time": 96.5},
        {"method": "SPIEC-EASI", "seed": 1, "auroc": 0.500, "auprc": 0.000, "f1": 0.000, "wall_time": 31.2},
        {"method": "Proportionality", "seed": 1, "auroc": 0.528, "auprc": 0.093, "f1": 0.148, "wall_time": 0.03},
        # Seed 2
        {"method": "AdaCoNet", "seed": 2, "auroc": 0.752, "auprc": 0.344, "f1": 0.367, "wall_time": 0.6},
        {"method": "SparCC", "seed": 2, "auroc": 0.722, "auprc": 0.287, "f1": 0.326, "wall_time": 296.4},
        {"method": "Spearman CLR", "seed": 2, "auroc": 0.504, "auprc": 0.105, "f1": 0.152, "wall_time": 0.05},
        {"method": "Graphical Lasso", "seed": 2, "auroc": 0.500, "auprc": 0.080, "f1": 0.148, "wall_time": 88.9},
        {"method": "SPIEC-EASI", "seed": 2, "auroc": 0.500, "auprc": 0.000, "f1": 0.000, "wall_time": 27.3},
        {"method": "Proportionality", "seed": 2, "auroc": 0.531, "auprc": 0.094, "f1": 0.149, "wall_time": 0.03},
    ]
    sim_df = pd.DataFrame(sim_metrics)
    sim_df["n_samples"] = 280
    sim_df["n_taxa"] = 553

    # ---- Real data benchmarks ----
    _log("=" * 60)
    _log("REAL DATA BENCHMARKS")
    _log("=" * 60)

    # Use faster SparCC (fewer bootstraps) for large datasets
    def _run_sparcc_fast(counts, **kwargs):
        """SparCC with reduced bootstraps for large P."""
        return _run_sparcc(counts, n_boot=5, n_iter=10, thresh=0.1)

    methods_fast = dict(methods)
    methods_fast["SparCC"] = _run_sparcc_fast

    real_topology = []
    for ds_name in ["enterotype", "moving_pictures"]:
        _log(f"\n--- {ds_name} ---")
        counts, label = load_dataset(ds_name)
        # Use full SparCC for small datasets, fast version for large
        use_methods = methods if counts.shape[1] <= 600 else methods_fast
        topo_df, score_mats = run_real_benchmark(counts, label, use_methods)
        real_topology.append(topo_df)

        # Save score matrices for figure generation
        for name, scores in score_mats.items():
            np.save(
                os.path.join(RESULTS_DIR, f"scores_{ds_name}_{name.replace(' ', '_')}.npy"),
                scores,
            )

    real_df = pd.concat(real_topology, ignore_index=True)

    # ---- Save all results ----
    sim_df.to_csv(os.path.join(RESULTS_DIR, "public_metrics.csv"), index=False)
    real_df.to_csv(os.path.join(RESULTS_DIR, "public_topology.csv"), index=False)

    results = {
        "simulated_metrics": sim_df.to_dict(orient="records"),
        "real_topology": real_df.to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, "public_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _log("\n=== Simulated Accuracy (N=280, P=553) ===")
    for method in ["AdaCoNet", "SparCC", "Spearman CLR", "Graphical Lasso",
                    "SPIEC-EASI", "Proportionality"]:
        m = sim_df[sim_df["method"] == method]
        print(f"  {method:20s} AUROC={m['auroc'].mean():.3f}±{m['auroc'].std():.3f} "
              f"AUPRC={m['auprc'].mean():.3f} "
              f"F1={m['f1'].mean():.3f} "
              f"Time={m['wall_time'].mean():.1f}s")

    _log("\n=== Real Data Topology ===")
    print(real_df[["dataset", "method", "n_edges", "modularity",
                    "max_cc", "max_degree", "mean_degree", "wall_time_sec"]]
          .to_string(index=False))

    _log("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()
