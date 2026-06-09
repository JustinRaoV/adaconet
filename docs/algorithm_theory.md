## AdaCoNet: Adaptive Compositional Network Inference

### Mathematical Framework

AdaCoNet is a four-layer hybrid algorithm that combines a Dirichlet-Multinomial generative model, information-theoretic association scoring, variational autoencoder-based latent space analysis, and an adaptive ensemble mechanism. The algorithm name stands for **Ada**ptive **Co**mpositional **Net**work inference.

---

### Layer 1: Dirichlet-Multinomial Foundation

**Problem statement.** Let X in N^{n x p} be the observed count matrix (n samples, p taxa). Each row x_i follows a multinomial sampling model with sample-specific total N_i = sum_j x_{ij} and unknown relative abundances pi_i:

    x_i | pi_i ~ Multinomial(N_i, pi_i)

The relative abundances pi_i lie on the (p-1)-simplex and vary across samples due to both biological variation and sampling noise. We model this variation with a Dirichlet prior:

    pi_i ~ Dirichlet(alpha_1, ..., alpha_p)

This yields the marginal Dirichlet-Multinomial (DM) distribution:

    P(x_i | alpha) = Gamma(N_i + 1) / prod_j Gamma(x_{ij} + 1) * Gamma(|alpha|) / Gamma(N_i + |alpha|) * prod_j Gamma(x_{ij} + alpha_j) / Gamma(alpha_j)

where |alpha| = sum_j alpha_j is the concentration parameter.

**Parameter estimation.** We use the method-of-moments estimator (Manna & Subedi, 2021) for computational efficiency. Let m_j = mean(x_{ij}/N_i) and s_j^2 = var(x_{ij}/N_i). Define the moment ratio:

    r = (sum_j s_j^2 - sum_j m_j(1-m_j)/N_bar) / (sum_j m_j^2 - sum_j m_j^2/N_bar)

Then |alpha| = (1 - r) / r and alpha_j = m_j * |alpha|. For improved accuracy, we apply one step of Newton-Raphson refinement on the marginal log-likelihood.

**Posterior correlation.** Given the Dirichlet posterior pi_i | x_i ~ Dirichlet(x_{i1} + alpha_1, ..., x_{ip} + alpha_p), the posterior mean for sample i, taxon j is:

    E[pi_{ij} | x_i] = (x_{ij} + alpha_j) / (N_i + |alpha|)

Let theta_{ij} = x_{ij} + alpha_j and Theta_i = N_i + |alpha|. The posterior variance and covariance are:

    Var[pi_{ij} | x_i] = theta_{ij}(Theta_i - theta_{ij}) / (Theta_i^2 (Theta_i + 1))
    Cov[pi_{ij}, pi_{ik} | x_i] = -theta_{ij} * theta_{ik} / (Theta_i^2 (Theta_i + 1))

The DM correlation matrix is computed from the posterior second moments across all samples:

    mu_j = (1/n) sum_i E[pi_{ij} | x_i]
    S_{jk} = (1/n) sum_i (E[pi_{ij}|x_i] - mu_j)(E[pi_{ik}|x_i] - mu_k)
    R^{DM}_{jk} = S_{jk} / sqrt(S_{jj} * S_{kk})

**Zero handling.** We adopt a structural/sampling zero model. For taxon j, if the prevalence (fraction of non-zero samples) is below threshold tau_zero (default 0.05), zeros are treated as structural (taxon truly absent). Otherwise, zeros are treated as sampling zeros and handled naturally by the Bayesian smoothing (alpha_j > 0 provides shrinkage toward the prior mean). No pseudocounts are needed.

**Computational cost:** O(n * p) for parameter estimation, O(n * p^2) for posterior correlation matrix. Memory: O(p^2).

---

### Layer 2: Information-Theoretic Association Scoring

**Motivation.** Pearson-type correlations (including DM posterior correlations) capture only linear associations. Mutual information (MI) captures arbitrary dependency forms, including nonlinear relationships common in microbial ecology (e.g., threshold effects, saturation).

**Bayesian-smoothed CLR transform.** We define a robust CLR-like transform using the DM posterior means:

    z_{ij} = log(E[pi_{ij} | x_i]) - (1/p) sum_k log(E[pi_{ik} | x_i])

This avoids pseudocounts entirely — the Dirichlet prior provides natural smoothing. The posterior mean E[pi_{ij} | x_i] = (x_{ij} + alpha_j) / (N_i + |alpha|) is strictly positive when alpha_j > 0.

**MI estimation via KSG.** For each pair (j, k), we estimate MI using the Kraskov-Stoegbauer-Grassberger (KSG) k-nearest-neighbor estimator:

    I(z_j; z_k) = psi(n) - <psi(n_j + 1) + psi(n_k + 1)> + psi(k)

where psi is the digamma function, k is the number of nearest neighbors (default k=6), n_j and n_k are the number of points within the Chebyshev distance to the k-th neighbor in the marginal spaces, and <> denotes the sample average.

**Significance testing.** MI significance is assessed via permutation testing with efficient early stopping. For B permutations (default B=100), we shuffle sample labels and recompute MI. The p-value is (1 + #{permuted MI >= observed MI}) / (B + 1). We apply BH-FDR correction across all p(p-1)/2 pairs.

**MI-based association score:**

    A^{MI}_{jk} = I(z_j; z_k) * I(p_{jk} < alpha_FDR)

where I(.) is the indicator function and alpha_FDR is the FDR threshold (default 0.05).

**Computational cost:** O(n * p^2 * log(n)) for all-pairs KSG estimation (using KD-tree for neighbor search). Memory: O(p^2).

---

### Layer 3: Variational Autoencoder Latent Space Analysis

**Motivation.** Microbial communities are shaped by latent environmental and host factors not captured in the observed taxa. A VAE discovers these latent factors, and the decoder's Jacobian reveals how taxa co-respond to latent changes — capturing nonlinear interaction structure.

**Architecture.**

Encoder: q(z|x) = N(mu_phi(x), diag(sigma_phi(x)^2))

    h = ReLU(W_1 * z_clr(x) + b_1)       # h in R^h
    mu = W_mu * h + b_mu                    # mu in R^d
    log_sigma^2 = W_sigma * h + b_sigma     # log_sigma^2 in R^d

where z_clr(x) is the Bayesian-smoothed CLR vector (Layer 2), h is the hidden dimension (default h = min(p/2, 256)), and d is the latent dimension (default d = min(p/4, 64)).

Decoder: p(x|z) parameterized by a neural network:

    h' = ReLU(V_1 * z + c_1)
    logits = V_2 * h' + c_2                 # logits in R^p
    pi_hat = softmax(logits / tau)           # temperature-scaled softmax

The temperature tau (default 0.5) controls output sharpness.

**Loss function.**

    L = L_recon + beta * L_KL + gamma * L_jacobi

Reconstruction loss (negative log-likelihood under DM):

    L_recon = -(1/n) sum_i sum_j [x_{ij} * log(pi_hat_{ij}) + alpha_j * log(pi_hat_{ij})]

KL divergence (standard VAE):

    L_KL = -(1/2) sum_l (1 + log(sigma_l^2) - mu_l^2 - sigma_l^2)

Jacobian regularizer (encourages structured latent-taxa mapping):

    L_jacobi = ||J^T J - I_d||_F^2

where J = d(pi_hat)/dz in R^{p x d} is the decoder Jacobian. This encourages the latent dimensions to have approximately orthogonal effects on the taxa composition, improving interpretability.

**Jacobian-based association score.** After training, we compute the decoder Jacobian J_i = d(pi_hat)/dz at each sample's latent encoding. The Jacobian similarity between taxa j and k is:

    S^{VAE}_{jk} = (1/n) sum_i corr(J_i[j,:], J_i[k,:])

where J_i[j,:] is the j-th row of the Jacobian at sample i (how taxon j responds to all latent dimensions). High S^{VAE}_{jk} means taxa j and k co-respond similarly to latent factors — evidence of ecological association.

**Computational cost:** Training: O(E * n * h * d) per epoch, E epochs (default E=100). Jacobian computation: O(n * p * d) via automatic differentiation. Total training is typically O(n * p * d * E) which for p=3000, n=5000, d=64 is very fast on GPU. Memory: O(n * d + p * d) for the model.

---

### Layer 4: Adaptive Ensemble Integration

**Score normalization.** Each score matrix is normalized to [0, 1]:

    R*^{DM}_{jk} = (R^{DM}_{jk} + 1) / 2                    # correlation to [0,1]
    A*^{MI}_{jk} = A^{MI}_{jk} / max(A^{MI})                 # MI to [0,1]
    S*^{VAE}_{jk} = (S^{VAE}_{jk} + 1) / 2                  # Jacobian similarity to [0,1]

Additionally, we include proportionality as a fourth signal:

    rho_p(j,k) = 1 - Var(log(pi_j/pi_k)) / (Var(log(pi_j)) + Var(log(pi_k)))
    R*^{prop}_{jk} = (rho_p(j,k) + 1) / 2

**Adaptive weighting.** The final association score is a weighted geometric mean:

    W_{jk} = [R*^{DM}_{jk}]^{w_1} * [A*^{MI}_{jk}]^{w_2} * [S*^{VAE}_{jk}]^{w_3} * [R*^{prop}_{jk}]^{w_4}

The weights w = (w_1, w_2, w_3, w_4) are learned by maximizing agreement with a held-out validation set. Specifically:

1. Split data into K=5 folds.
2. For each fold, compute individual score matrices on the training portion.
3. Optimize w to maximize the average AUPRC across folds, using the held-out fold's DM correlation as a proxy target (since true ground truth is unavailable).
4. Final weights are averaged across folds.

Alternatively, if a small labeled set (from literature-curated interactions) is available, the weights are optimized directly against known interactions.

**Threshold determination.** The final network is obtained by thresholding:

    G_{jk} = I(W_{jk} > tau)

The threshold tau is selected via:
- **StARS-like stability selection** (default): subsample at multiple rates (80% samples), compute networks at a grid of tau values, select tau that minimizes instability.
- **FDR-based**: select tau to achieve a target FDR (using permutation null distribution of W_{jk}).

**Edge direction (optional).** For edges in the final network, we estimate directionality using:
1. **Partial correlation asymmetry**: |rho(j,k | rest) - rho(k,j | rest)| via residual analysis.
2. **Temporal precedence** (if longitudinal data available).
3. **Abundance-based causality**: higher-abundance taxa more likely to be "drivers."

---

### Complexity Analysis

| Layer | Time | Space | GPU-friendly |
|---|---|---|---|
| L1: DM Foundation | O(n * p^2) | O(p^2) | Yes (batched matrix ops) |
| L2: MI Scoring | O(p^2 * n * log(n)) | O(p^2) | Partial (neighbor search) |
| L3: VAE Training | O(E * n * p * d) | O(p * d + n * d) | Yes (full GPU) |
| L3: Jacobian Score | O(n * p * d) | O(p^2) | Yes |
| L4: Ensemble | O(p^2 * K) | O(p^2) | No (trivial) |
| **Total** | **O(n*p^2 + p^2*n*log(n) + E*n*p*d)** | **O(p^2)** | **—** |

For the benchmark sizes:
- N=500, P=500: ~seconds on GPU, ~minutes on CPU
- N=1000, P=1000: ~minutes on GPU
- N=5000, P=3000: ~10-20 minutes on GPU, ~1-2 hours on CPU

Compare to SparCC (with permutations): O(p^2 * I * B) = O(p^2 * 10 * 100) = O(1000 * p^2).
Compare to SPIEC-EASI (with StARS): O(100 * p^3).
AdaCoNet is faster for p > 1000 due to avoiding p^3 operations.

---

### Pseudocode

```
function AdaCoNet(X, options):
    Input: X in N^{n x p} — count matrix
           options: {alpha_fdr, k_mi, d_latent, E_epochs, tau_zero, n_folds, method_threshold}
    
    // === Layer 1: Dirichlet-Multinomial Foundation ===
    alpha = estimate_dm_params(X)                        // O(n*p)
    posterior_means = compute_posterior_means(X, alpha)  // O(n*p)
    R_DM = compute_posterior_correlation(posterior_means) // O(n*p^2)
    
    // === Layer 2: Information-Theoretic Scoring ===
    Z_CLR = bayesian_clr(X, alpha)                       // O(n*p)
    A_MI = zeros(p, p)
    for j = 1 to p:
        for k = j+1 to p:
            mi = ksg_mi(Z_CLR[:,j], Z_CLR[:,k], k=options.k_mi)  // O(n*log(n))
            pval = permutation_test_mi(Z_CLR[:,j], Z_CLR[:,k], B=100)
            A_MI[j,k] = mi * (pval < alpha_fdr)
    A_MI = A_MI + A_MI^T
    
    // === Layer 3: VAE Latent Space ===
    vae = train_vae(Z_CLR, d=options.d_latent, E=options.E_epochs)  // GPU-accelerated
    J_all = compute_jacobians(vae, Z_CLR)               // O(n*p*d), GPU
    S_VAE = zeros(p, p)
    for j = 1 to p:
        for k = j+1 to p:
            S_VAE[j,k] = mean([corr(J_i[j,:], J_i[k,:]) for J_i in J_all])
    S_VAE = S_VAE + S_VAE^T
    
    // === Layer 3b: Proportionality ===
    R_prop = compute_proportionality(Z_CLR)              // O(n*p^2)
    
    // === Layer 4: Adaptive Ensemble ===
    // Normalize all scores to [0, 1]
    R_DM_norm = (R_DM + 1) / 2
    A_MI_norm = A_MI / max(A_MI)
    S_VAE_norm = (S_VAE + 1) / 2
    R_prop_norm = (R_prop + 1) / 2
    
    // Learn optimal weights via cross-validation
    w = learn_ensemble_weights(R_DM_norm, A_MI_norm, S_VAE_norm, R_prop_norm,
                                n_folds=options.n_folds)
    
    // Compute final score
    W = R_DM_norm^w[0] * A_MI_norm^w[1] * S_VAE_norm^w[2] * R_prop_norm^w[3]
    
    // Determine threshold via StARS-like stability selection
    tau = select_threshold(W, X, method='stars')
    
    // Construct final network
    G = (W > tau)
    
    // Edge weights (signed)
    W_signed = G .* R_DM  // use DM correlation for sign
    
    return G, W_signed, W, {
        'R_DM': R_DM,
        'A_MI': A_MI,
        'S_VAE': S_VAE,
        'R_prop': R_prop,
        'weights': w,
        'threshold': tau,
        'alpha_dm': alpha
    }
```

---

### Evaluation Metrics

For benchmarking against ground truth networks:

**Edge-level metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * Precision * Recall / (Precision + Recall)
- AUPRC: area under the precision-recall curve (primary metric for sparse networks)
- AUROC: area under the ROC curve

**Network topology metrics:**
- Degree distribution similarity: KS-test statistic between inferred and true degree distributions
- Clustering coefficient correlation: Pearson correlation of node clustering coefficients
- Modularity (Q): Newman-Girvan modularity of the inferred network
- Community detection accuracy: NMI (normalized mutual information) between true and inferred modules
- Hub recovery rate: fraction of true hubs (degree > 90th percentile) correctly identified

**Biological validation (real data):**
- Functional coherence: enrichment of KEGG/COG pathways within inferred modules
- Taxonomic coherence: enrichment of known symbiotic/competitive relationships
- Cross-study reproducibility: Jaccard similarity of edges across independent studies of the same habitat
