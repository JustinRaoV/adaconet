# Peer Review Report -- Methodology

## Reviewer Information
- **Role**: Peer Reviewer 1 (Methodology)
- **Identity**: Prof. Hongzhe Li, University of Pennsylvania
- **Expertise**: High-dimensional statistics, compositional data analysis, Dirichlet-Multinomial models, sparse graphical models, multiple testing
- **Focus**: Statistical validity of the three theorems, rigor of proof sketches, experimental design, reproducibility

---

## Overall Assessment

### Recommendation: Major Revision
### Confidence: 4/5

### Summary

AdaCoNet addresses a genuine and important problem: the lack of principled guidance for choosing among competing microbial co-occurrence estimators. The Compositional Copula Model (CCM) provides a useful conceptual unification, framing existing methods as different estimators of a single latent correlation matrix. The benchmarking effort is substantial, spanning two simulators, two real datasets, eight competitors, and a ten-variant ablation study.

However, the manuscript's theoretical claims substantially exceed what is actually established. Theorem 1 (CLR Variance Inflation) is a routine Delta-method calculation whose core result has been known in the compositional data literature for over a decade. Theorem 2 (Phase Transition) provides only an upper bound with unspecified constants -- not a sharp transition in any rigorous sense. Theorem 3 (Optimal Ensemble Weights) introduces a heuristic rational function but labels it "optimal" and cites Chen & Li (2009) for a parameter value (c_ref = 0.05) that paper never discusses in the context of ensemble weights or phase transitions. The proof sketch references supplementary material that was not provided. Additionally, the StARS edge selection procedure degenerates in practice, and the v4 simulator creates a self-favoring bias for the DM layer. These issues collectively undermine the claim of a "rigorous theoretical foundation."

The work has genuine potential. With honest reframing of the theoretical contributions, proper proof or removal of the theorems, correction of the c_ref misattribution, and resolution of the StARS degeneracy, this could make a solid contribution.

---

## Strengths

**S1. Conceptually valuable unification.** The CCM framework, which positions the DM posterior, Spearman-on-CLR, proportionality, and Gaussian copula as four estimators of a single latent correlation matrix Sigma, is a genuinely useful organizing principle. Even if the individual components are well known, framing them as competing estimators under one generative model clarifies *why* different methods excel under different conditions. This conceptual contribution stands independently of the theorems and is the paper's strongest element.

**S2. Thorough and well-structured benchmarking.** The evaluation spans two simulators with fundamentally different data-generating mechanisms (direct-covariance multinomial vs. SparseDOSSA2 Gaussian copula), two real microbiome datasets, eight competing methods, and multiple metrics (AUROC, AUPRC, top-k precision, P@FPR). The inclusion of 10 random seeds per configuration with standard deviations (Table 1) is good practice. The cross-simulator inversion in ablation patterns (Table 3) -- Spearman dominant on v4, copula/proportionality dominant on SD2 -- is a compelling piece of evidence.

**S3. Honest acknowledgment of limitations.** The authors acknowledge the v4 self-favoring bias (DM is the Bayes estimator under the true generative model), the StARS degeneracy (tau = 1.0 yielding at most 1 edge), and the MovingPictures modularity shortfall. This level of transparency is unusual and commendable.

**S4. The zero-fraction guard addresses a real and subtle problem.** The observation that prevalence filtering can inflate alpha_0/p by removing low-concentration taxa, thereby masking CLR unreliability, is insightful. The dual diagnostic (alpha_0/p and f_0) is a practical contribution, even if its current form is ad hoc.

**S5. Computational efficiency is competitive.** AdaCoNet runs in under 6 seconds at P = 1000, compared to 88-112 s for SparCC, while achieving comparable or better accuracy. This is practically important for microbiome studies with hundreds of taxa.

---

## Weaknesses

### W1. Theorem 1 (CLR Variance Inflation) presents a standard result as novel

**Problem.** The CLR variance inflation factor (1 + p/|alpha|) is derived via a first-order Delta-method Taylor expansion, as the proof sketch itself shows. This calculation is standard in the compositional data literature. The variance of log-ratio transformed multinomial data, including the contribution of the geometric mean denominator, has been discussed in various forms by Aitchison (1986), Egozcue et al. (2003), and in the context of CLR-based network inference by Friedman & Alm (2012) and Fang et al. (2015, CCLasso). The specific factor (1 + p/|alpha|) under uniform pi is a textbook-level consequence of the multinomial covariance structure.

**Why it matters.** Labeling this as "Theorem 1" implies a novel theoretical contribution. Readers and reviewers will evaluate the paper's theoretical depth based on this result, and finding it standard undermines confidence in the subsequent theorems.

**Suggestion.** Reframe Theorem 1 as a "Proposition" or "Lemma" that formalizes a known result for the specific CCM setting, with an explicit acknowledgment that the Delta-method derivation is standard. Cite the prior work where this or closely related variance expressions appear. The value is in making the inflation factor explicit within the CCM framework, not in the derivation itself.

**Severity: Medium.** This is a framing issue rather than a technical error, but it sets a problematic precedent for the novelty claims of Theorems 2 and 3.

---

### W2. Theorem 2 (Phase Transition) provides an upper bound, not a sharp transition

**Problem.** The theorem states an MSE upper bound:

    E[(rho_hat_S - rho)^2] <= (C_1/n)(1 + p/|alpha|)^2 + C_2 * g(alpha_0, p)

with unspecified constants C_1, C_2 and an unspecified function g(alpha_0, p) that is only characterized by its limit g -> 0 as alpha_0/p -> infinity. The critical threshold c* is asserted to exist but never defined, computed, or bounded. There is no matching lower bound.

This is not a phase transition in any rigorous statistical sense. A genuine phase transition -- such as the BBP transition in random matrix theory or the computational-statistical gap in sparse PCA -- requires either (a) a matching lower bound showing that the MSE is bounded *below* by a large quantity on one side of the threshold, or (b) a demonstrable discontinuity or non-analyticity in an order parameter as a function of the control parameter. The current result shows only that MSE is *at most* some quantity that transitions smoothly from large to small. This is an MSE *rate bound*, not a sharp transition.

Furthermore, the decomposition into "multinomial regime" and "copula regime" is presented as a consequence of the theorem, but it is actually an *interpretation* layered on top of a continuous bound. The bound itself does not exhibit any discontinuity.

**Why it matters.** The phrase "Phase Transition Theorem" is the paper's central theoretical claim and appears in the title. If the result is merely an upper bound with unspecified constants, the title and all downstream claims (including the ensemble weight derivation) are overstated.

**Suggestion.** (a) Either prove a matching lower bound that establishes a genuine sharp transition (with an explicit c* value), or (b) rename the result to "MSE Bound for Spearman-on-CLR" and present it as a rate characterization rather than a phase transition. In case (b), the ensemble weight derivation in Theorem 3 should be reframed as a heuristic motivated by this bound, not as a theorem. The constants C_1 and C_2 should be made explicit, or their omission should be justified.

**Severity: High.** This is the paper's flagship result. Its overstatement cascades through the entire narrative.

---

### W3. Theorem 3 (Optimal Ensemble Weights) conflates heuristic design with optimal theory, and misattributes c_ref

**Problem.** Three distinct issues converge here:

*(a) The weight formula is not derived from the MSE-minimization principle.* The theorem states that MSE-minimizing weights satisfy w_k proportional to 1/MSE_k (which is indeed a standard result for combining unbiased estimators with uncorrelated errors). However, the specific formula w_S = alpha_0 / (alpha_0 + c_ref) is not derived from this principle. It is a rational function *proposed* as an approximation, and the connection to 1/MSE_S is asserted but never demonstrated. For this to follow from the inverse-MSE principle, one would need to show that MSE_S is proportional to (alpha_0 + c_ref)/alpha_0, which requires a specific model for how MSE_S depends on alpha_0 -- a model that is never stated or derived.

*(b) c_ref = 0.05 is misattributed.* The manuscript cites Chen & Li (2009) for c_ref = 0.05. I am intimately familiar with this paper, as I am a co-author. Chen & Li (2009) addresses variable selection for the Dirichlet-Multinomial distribution using a Dirichlet-multinomial regression framework. It does not discuss ensemble weights, phase transition thresholds, or any quantity that could be interpreted as c_ref in the present context. The value 0.05 appears in Chen & Li (2009) only in the standard context of significance levels for hypothesis testing, not as a concentration threshold. This citation does not support the claimed justification.

*(c) The "optimality" label is not warranted.* The weight formula w_S = alpha_0/(alpha_0 + c_ref) is a one-parameter sigmoid-like shrinkage function. Calling it "optimal" requires a precise statement of the loss function, the class of admissible weights, and a proof that this formula achieves the minimum within that class. None of these are provided.

**Why it matters.** This is the paper's operational result -- the formula that practitioners actually use. If it is a heuristic (which it appears to be), it should be presented as such, with empirical validation serving as the primary justification rather than an unproven optimality claim.

**Suggestion.** (a) Explicitly derive w_S = alpha_0/(alpha_0 + c_ref) from the inverse-MSE principle by stating and proving a model for MSE_S(alpha_0). If this derivation is not possible, present the formula as a principled heuristic motivated by the MSE bound in Theorem 2. (b) Remove the Chen & Li (2009) citation for c_ref and instead justify the value empirically (e.g., via the sensitivity analysis mentioned in the Discussion) or through a theoretical derivation of c*. (c) Rename "Optimal Ensemble Weights" to "Theory-Guided Ensemble Weights" or "Adaptive Ensemble Weights."

**Severity: High.** The misattribution of c_ref is a factual error that will be immediately apparent to any reader familiar with the cited work. The "optimal" label overstates the theoretical guarantee.

---

### W4. StARS edge selection degenerates, undermining the end-to-end pipeline

**Problem.** The manuscript acknowledges that StARS consistently selects tau = 1.0, yielding at most 1 edge, across all subsample counts from 5 to 50. This means the StARS-based edge selection -- described in the Methods section as a key component of the pipeline --fails entirely on the ensemble score matrix. The authors then report AUROC and AUPRC based on raw score rankings rather than StARS-selected edges, which means the benchmarked pipeline is not the same as the described pipeline.

The stated cause -- min-max normalized scores with a skewed near-zero distribution causing instability minimization to collapse -- is plausible but indicates a fundamental mismatch between StARS's assumptions and the ensemble score distribution. StARS was designed for sparse precision matrix estimation where the regularization path smoothly controls sparsity. Applying it to a fixed ensemble score matrix with a different thresholding mechanism is not a straightforward adaptation.

**Why it matters.** The Methods section presents StARS as the edge selection mechanism, but the Results section evaluates a different pipeline (raw score ranking). This discrepancy means the end-to-end method as described does not work as advertised. For real-data applications (Table 4), edges are selected by fixed-density thresholding, which has no theoretical justification from the CCM framework.

**Suggestion.** (a) Either fix the StARS degeneracy (e.g., by applying StARS to individual layers rather than the ensemble, or by using a modified stability criterion appropriate for ensemble scores), or (b) remove StARS from the Methods description and explicitly adopt fixed-density or FDR-based thresholding as the edge selection mechanism, with a clear theoretical or empirical justification. The current situation -- describing StARS but evaluating without it -- is not acceptable for a methods paper.

**Severity: High.** This is a reproducibility and validity issue. A reader implementing the described pipeline would encounter the same degeneracy.

---

### W5. Proof sketches reference supplementary material that is not provided

**Problem.** The proof sketch for Theorem 2 states "Full proofs are in Supplementary Material." No supplementary material accompanies this manuscript. The proof sketch itself is incomplete: it establishes the variance inflation factor under uniform pi but does not derive the MSE bound, does not prove the existence of c*, and does not address the non-uniform pi case. Theorem 3 has no proof at all.

**Why it matters.** Without full proofs, the theorems are unsubstantiated claims. The proof sketch for Theorem 1 is adequate for the uniform case but the general case is not addressed. For Theorem 2, the sketch does not bridge the gap between the variance inflation factor and the MSE bound. For Theorem 3, no derivation connects the inverse-MSE principle to the specific rational function.

**Suggestion.** Either (a) provide the supplementary material with complete proofs, or (b) include proof sketches sufficient to verify each theorem's key steps within the main text. If full proofs are not yet available, the results should be labeled as "Conjectures" or "Proposed Bounds" rather than "Theorems."

**Severity: High.** Unverifiable theoretical claims cannot be accepted in a methods paper.

---

## Detailed Comments

### Methods Section (Section 2)

**The CCM generative model (Section 2.5).** The model eta_i ~ N(0, Sigma_tilde), pi_i = softmax(eta_i), x_i | pi_i ~ Mult(N_i, pi_i) is a reasonable generative model for compositional data. However, the constraint Sigma_tilde * 1 = 0 (required for the CLR parameterization) means that Sigma_tilde is rank-deficient (rank p-1 at most). The implications of this rank deficiency for the downstream correlation estimation are not discussed. Additionally, the softmax mapping from R^p to the simplex does not preserve the correlation structure of eta in a simple way: Corr(pi_j, pi_k) is not a simple function of Sigma_tilde_{jk}. The claim that all four layers are "different estimators of the same latent correlation matrix Sigma" is therefore an approximation whose accuracy depends on the concentration regime, and this approximation quality should be quantified.

**DM layer (Section 2.1).** The description is clear and standard. The Newton-Raphson refinement on |alpha| is well-established. However, the method-of-moments initialization can produce unreliable estimates when p is large relative to N, which is precisely the regime where the DM layer is most needed (e.g., N=200, P=50). The sensitivity of the DM correlation to estimation error in alpha is not discussed.

**Spearman layer (Section 2.2).** The N/P-dependent strategy selection (Bayesian CLR vs. raw CLR with Ledoit-Wolf shrinkage) is pragmatic but introduces a discontinuity at N/P = 2 that could affect the smoothness of the ensemble weight as a function of the data. The pseudocount of 0.5 is arbitrary; the sensitivity to this choice is not explored.

**Ensemble weight structure.** The weight assignment -- DM and proportionality receive baseline weight 1.0, Spearman receives w_S, copula receives (1 - w_S) -- is asymmetric. The fixed weight of 1.0 for DM and proportionality implies these layers are "always moderately informative," but the ablation study (Table 3) shows that DM-only achieves AUROC 0.607-0.659 on v4 data and proportionality-only achieves 0.522-0.560. These are not consistently "moderately informative" -- proportionality is near-random on v4 data. The justification for fixing their weights at 1.0 while adapting Spearman and copula is unclear.

**Min-max normalization.** Score matrices are min-max normalized to [0,1] before weighted combination. This normalization is sensitive to outliers: a single extreme score can compress the entire distribution into a narrow range, reducing discriminability. The impact of this normalization on the final edge ranking is not evaluated.

### Results Section (Section 3)

**v4 benchmark (Table 1).** The results are clearly presented with standard deviations across 10 seeds. The AUROC advantage of AdaCoNet over the second-best method is modest in several configurations: at N=500, P=200, AdaCoNet (0.896 +/- 0.048) vs. FastSpar (0.866 +/- 0.028), the difference is 0.030 with a pooled standard error of approximately 0.020 -- a difference of only 1.5 standard errors. At N=1000, P=500, AdaCoNet (0.811 +/- 0.114) vs. SparCC (0.808 +/- 0.098) is a difference of 0.003, which is well within noise. The paper would benefit from formal paired statistical tests (e.g., paired t-tests or Wilcoxon signed-rank tests across seeds) to distinguish genuine improvements from sampling variability.

**Self-favoring bias of v4.** The authors acknowledge this in the Discussion but do not attempt to quantify or mitigate it. A simple ablation -- removing the DM layer from the ensemble and re-evaluating on v4 -- would show how much of the advantage comes from the Bayes estimator matching the true model. Table 3 shows that removing DM from the adaptive ensemble reduces AUROC from 0.919 to 0.906 at N=200, P=50 (a drop of only 0.013), suggesting the DM layer's contribution is modest. However, the DM layer may still influence the ensemble through its interaction with the weight normalization. A more informative comparison would be to evaluate AdaCoNet on a simulator where *none* of the four layers matches the true generative model.

**SparseDOSSA2 results (Table 2).** AdaCoNet achieves AUROC 0.714 at N=500, second to proportionality alone at 0.899. This is a substantial gap (0.185 AUROC). The fact that a single layer (proportionality) dramatically outperforms the ensemble on this dataset is concerning. It suggests that the ensemble mechanism, even with theory-driven weights, cannot fully suppress the influence of poorly-performing layers. The copula layer achieves 0.721 alone on SD2, which is also better than AdaCoNet's 0.714. This raises the question: does the ensemble ever outperform its *best* individual layer, or does it only outperform its *average* layer?

**Ablation study (Table 3).** Several observations warrant comment:

- On v4 N=200, P=50: copula-only AUROC is 0.554, yet removing copula from the full ensemble *improves* AUROC from 0.919 to 0.923. This directly contradicts the claim that all four layers are needed and suggests the theory-driven weight for copula (1 - w_S) is still too large.
- Spearman-only (0.952) exceeds the full adaptive ensemble (0.919) on v4 N=200, P=50. If the best single layer outperforms the ensemble, the ensemble is actively harmful in this regime. The paper does not address this.
- No standard deviations are reported for the ablation study, making it impossible to assess whether the observed differences (e.g., 0.919 vs. 0.923 for with/without copula) are statistically meaningful.

**Real data (Table 4).** The real-data evaluation is limited to two datasets and relies on network topology metrics (modularity, component structure) rather than ground-truth edge validation. While ground truth is unavailable for real microbiome data, the paper could strengthen this section by evaluating overlap with known ecological interactions (e.g., from the literature or curated databases) or by assessing reproducibility across related datasets.

The MovingPictures result (Q = 0.163, far below proportionality's 0.438) is a significant practical failure that deserves more prominent discussion. The post-hoc explanation (temporal dynamics favor proportionality) is plausible but unfalsifiable without additional experiments.

### Discussion Section (Section 4)

**Limitation acknowledgment.** As noted in the Strengths, the transparency is commendable. The discussion of the v4 self-favoring bias, the sublinear growth of |alpha| with p, and the MovingPictures modularity failure are all valuable.

**Sensitivity analysis claim.** The Discussion states that results are robust to c_ref in [0.01, 0.12] on v4 data and that the zero-fraction guard makes the SD2 result insensitive to c_ref entirely. This is a valuable claim, but it appears only in the Discussion with a reference to "Supplementary Fig. S1" which, like the supplementary proofs, is not provided. This analysis should be in the main Results section with the actual figure or table included.

**Missing comparison with cross-validation.** The Discussion mentions "cross-validated weight learning" as future work. However, a simple cross-validated baseline -- e.g., choosing weights by leave-one-out AUROC maximization on the v4 training data -- would provide a meaningful comparison point. Does the theory-driven weight outperform a purely data-driven weight? Without this comparison, the value of the theoretical framework is unclear.

---

## Questions for Authors

**Q1.** Theorem 2 claims the existence of a critical threshold c* such that a "sharp transition" occurs. Can you provide (a) an explicit formula or numerical value for c*, (b) a matching lower bound on the MSE that demonstrates the transition is indeed sharp rather than smooth, and (c) a precise definition of what "sharp" means in this context? Without these, the result appears to be a standard MSE rate bound rather than a phase transition.

**Q2.** The value c_ref = 0.05 is attributed to Chen & Li (2009). As a co-author of that paper, I am not aware of any result in it that establishes 0.05 as a concentration threshold for ensemble weighting or phase transitions. Can you specify exactly which result in Chen & Li (2009) you are citing, and explain how a variable selection threshold translates to an ensemble weight parameter? If this is a misattribution, what is the correct justification for c_ref = 0.05?

**Q3.** The ablation study (Table 3) shows that on v4 N=200, P=50, Spearman-only (AUROC 0.952) outperforms the full adaptive ensemble (0.919), and removing copula improves performance (0.923 vs. 0.919). Under what conditions does the ensemble *outperform its best individual layer*? If the ensemble only outperforms the average layer, the practical recommendation should be to use the best single layer (selected by the phase transition diagnostic) rather than the ensemble.

**Q4.** The StARS degeneracy (tau = 1.0, at most 1 edge) means the edge selection mechanism described in Methods does not function. For reproducibility: what edge selection mechanism should a practitioner actually use? If the answer is fixed-density thresholding, this should replace StARS in the Methods description, and the sensitivity to the density parameter should be evaluated.

---

## Minor Issues

**M1.** Equation (1): The notation Sigma_tilde * 1 = 0 uses bold 1 for the all-ones vector. This should be defined explicitly for clarity.

**M2.** Section 2.2: "Bayesian CLR uses posterior means" -- the posterior is with respect to the DM model from Section 2.1. This connection should be made explicit, as it creates a dependency between layers that violates the independence assumption implicit in the inverse-MSE weighting.

**M3.** Table 1: Several cells show results to three decimal places (e.g., 0.919 +/- 0.012), but with only 10 seeds, the standard error of the standard deviation is itself approximately 30% of its value. Consider reporting to two decimal places.

**M4.** The abstract states AdaCoNet achieves AUROC "up to 0.94" but Table 1 shows the maximum is 0.919 +/- 0.012. The value 0.94 does not appear in the results. If this is from a different configuration or metric, it should be clarified; otherwise, this appears to be an error.

**M5.** Section 3.2: "SparCC, FastSpar, and CCLasso were excluded at P=1000 due to computational cost." The exact computational budgets and hardware specifications should be stated for reproducibility.

**M6.** The zero-fraction guard uses a hard threshold f_0 > 0.5 to apply a "soft penalty." The functional form of this soft penalty is not specified in the Methods section. Is it a multiplicative factor on w_S? A continuous function of f_0? This should be stated explicitly.

**M7.** The reference to "Supplementary Table S2" and "Supplementary Fig. S1" in the Discussion implies supplementary material exists, but none was provided with the manuscript. All supplementary references should either be included or removed.

**M8.** On SparseDOSSA2, the ground truth is defined as "the empirical Spearman correlation of log(a_spiked + 1)." This choice of ground truth favors Spearman-based methods by construction, yet Spearman-on-CLR achieves only 0.443-0.447 AUROC on this data. A brief explanation of why the CLR transformation degrades Spearman correlation even when the ground truth is Spearman-based would strengthen the paper.

---

## Dimension Scores

| Dimension | Weight | Score (0-100) | Weighted Score | Justification |
|---|---|---|---|---|
| **Originality** | 20% | 68 | 13.6 | CCM unification is conceptually valuable; "phase transition" framing is overclaimed; Theorem 1 is standard; ensemble idea is not new but the theoretical motivation is |
| **Methodological Rigor** | 25% | 45 | 11.3 | Missing proofs; c_ref misattribution; StARS degeneracy unresolved; simulator self-favoring bias; no formal statistical tests; ensemble outperformed by best single layer in key regimes |
| **Evidence Sufficiency** | 25% | 62 | 15.5 | Two simulators and two real datasets with 10 seeds is adequate but not strong; v4 bias unmitigated; no significance tests; ablation lacks standard deviations; sensitivity analysis not shown |
| **Argument Coherence** | 15% | 65 | 9.8 | Good narrative from CCM to ensemble, but logical gaps in c_ref justification, "optimality" claim, and StARS workaround weaken the chain of reasoning |
| **Writing Quality** | 15% | 75 | 11.3 | Well-structured with clear notation and good figures; some overclaiming in terminology; abstract discrepancy (0.94 vs 0.919) |
| **Overall** | 100% | | **61.4** | |

**Decision Mapping**: 61.4 falls in the **Major Revision** range (50-64).

---

## Summary of Required Revisions

The following issues must be addressed before the manuscript can be considered for acceptance:

1. **Rename or prove the theorems.** Either provide rigorous proofs (with supplementary material) that establish genuine sharp transitions and optimal weights, or relabel as "Propositions" / "Heuristic Guidelines" with honest characterization of what is established vs. conjectured.

2. **Correct the c_ref attribution.** Remove the Chen & Li (2009) citation for c_ref = 0.05 and provide a valid justification, either theoretical (deriving c* from the MSE bound) or empirical (sensitivity analysis in the main text, not supplementary).

3. **Resolve the StARS degeneracy.** Either fix StARS for ensemble scores or replace it with a working edge selection mechanism and update the Methods section accordingly.

4. **Address the ensemble-vs-best-layer issue.** Demonstrate regimes where the ensemble outperforms its best individual layer, or revise the practical recommendation to use the phase transition diagnostic for layer selection rather than ensemble combination.

5. **Quantify the v4 self-favoring bias.** At minimum, evaluate a simulator where no layer matches the true generative model, or report the DM layer's individual contribution more prominently.

6. **Add statistical testing.** Report paired significance tests for key comparisons (AdaCoNet vs. second-best method) across seeds.

7. **Provide supplementary material** or remove all references to it.
