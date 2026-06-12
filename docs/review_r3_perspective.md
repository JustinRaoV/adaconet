# Peer Review Report — Perspective

## Reviewer Information
- **Role:** Peer Reviewer 3 (Cross-disciplinary Perspective)
- **Identity:** Prof. Larry Wasserman, Carnegie Mellon University
- **Expertise:** Statistical learning theory, high-dimensional inference, ensemble methods, stability-based model selection
- **Focus:** Theoretical novelty of the phase transition claim, validity of the CCM unification, comparison with principled ensemble methodology, and fidelity of the StARS adaptation

---

## Overall Assessment

### Recommendation: Major Revision
### Confidence: 5 (Very High — co-author of StARS; deep familiarity with the theoretical and methodological questions at issue)

### Summary

AdaCoNet addresses a genuine and important problem: the absence of a principled framework for selecting among competing microbial co-occurrence estimators. The idea of placing existing methods under a single generative model (the CCM) and using that model to derive data-adaptive ensemble weights is intellectually appealing, and the empirical results are encouraging — particularly the cross-simulator ablation study, which provides the strongest evidence in the paper. However, the theoretical contributions are significantly overstated. The "Phase Transition Theorem" (Theorem 2) is, in its current form, an upper bound on MSE that varies continuously with alpha_0; there is no matching lower bound, no proof of discontinuity, and no precise characterization of the critical threshold c*. The result is closer to a reliability threshold analysis — a smoothed bias-variance tradeoff — than a phase transition in any sense recognizable from statistical physics or random matrix theory. The CCM, while a useful organizing device, is introduced post-hoc and constrains the correlation structure (Sigma times 1 equals 0) in ways that are not fully discussed. The ensemble weight derivation resembles a simple shrinkage estimator, and no comparison is made with stacking, Bayesian model averaging, or cross-validated weight selection — all of which have extensive theoretical and empirical support. Finally, as a co-author of StARS, I must note that the StARS adaptation is problematic: applying stability selection to min-max normalized continuous scores is a non-standard use case, and the acknowledged degeneracy (tau = 1.0) is a symptom of a deeper mismatch between the method's assumptions and this data regime.

---

## Strengths

1. **Well-motivated problem with clear practical value.** The observation that no single co-occurrence estimator dominates across all data-generating mechanisms is well-supported by the literature and by the authors' own cross-simulator results. The desire for a theory-guided, adaptive ensemble is entirely reasonable, and the paper articulates this motivation clearly.

2. **Thorough and well-structured empirical evaluation.** The benchmark design is strong: two fundamentally different simulators (direct-covariance MVN and SparseDOSSA2 Gaussian copula), multiple sample sizes and dimensionalities, 10 seeds per configuration, ablation studies, top-k precision analysis, and real-data network topology. The cross-simulator inversion in ablation patterns (Spearman dominant on v4, copula/proportionality dominant on SD2) is the most convincing result in the paper and provides genuine evidence that different regimes require different estimator combinations.

3. **Useful organizing framework in the CCM.** Even if the CCM is not a fully novel generative model (see Weakness 2), the idea of treating all four layers as estimators of the same latent correlation matrix Sigma is a productive unification. It provides a coherent vocabulary for discussing when each method should be preferred and could serve as a useful pedagogical tool for the field.

4. **Transparent reporting of limitations.** The authors acknowledge the self-favouring bias of the v4 simulator (whose generative process matches the DM layer), the StARS degeneracy, and the poor performance at P=1000. This level of honesty is commendable and unusual in methods papers.

5. **Computational efficiency.** Achieving competitive or superior accuracy at 5-85x speedup over SparCC is a genuine practical contribution, particularly for large-scale microbiome studies where computational cost is a binding constraint.

---

## Weaknesses

### W1: The "Phase Transition" Is Not a Phase Transition
**Problem:** The paper uses the term "Phase Transition Theorem" and invokes the language of sharp, discontinuous transitions ("sharp reliability transition at a critical Dirichlet-Multinomial concentration threshold"). However, Theorem 2 provides only an MSE *upper bound* that is a continuous function of alpha_0/p. There is no matching lower bound, no proof that the MSE itself (rather than the bound) exhibits discontinuous behavior, and no precise characterization of c* beyond asserting its existence.

**Why this matters:** In statistical physics and random matrix theory, a phase transition refers to a genuine discontinuity in some order parameter (e.g., the BBP transition in spiked covariance models, where the top eigenvalue separates from the bulk at a precise threshold). The MSE bound here, equation (4), is of the form C_1/n times (1 + p/|alpha|)^2 plus a residual term g(alpha_0, p) that vanishes continuously. This is a standard bias-variance decomposition showing that the variance inflation factor (1 + p/|alpha|) degrades estimator quality as concentration decreases — a well-known consequence of the delta method applied to log-ratio transforms. Calling this a "phase transition" overstates the theoretical contribution and may mislead readers into expecting a sharp qualitative change in behavior.

**Suggestion:** Either (a) prove a matching lower bound showing that the MSE itself exhibits a discontinuity or non-analytic behavior at some critical c*, which would constitute a genuine phase transition, or (b) reframe the result as a "reliability threshold" or "efficiency boundary" analysis. The latter is still a valuable contribution — providing a quantitative criterion for when CLR-based methods degrade — but it should not be marketed as a phase transition. The proof sketch in the paper relies on standard delta-method variance calculations and asymptotic Spearman efficiency results; this is solid statistical analysis, but it does not establish the kind of sharp threshold that the "phase transition" label implies.

**Severity:** High. This is the central theoretical claim of the paper and the basis for the ensemble weight derivation. Overstating it undermines the paper's credibility with theoretically sophisticated readers.

---

### W2: The CCM Is a Post-Hoc Rationalization with a Strong Structural Constraint
**Problem:** The CCM (equation 1) is introduced as a "unified generative model," but it is in fact constructed to rationalize the four existing estimators, not derived from first principles or validated as a generative model for real microbiome data. More critically, the constraint that Sigma-tilde times the all-ones vector equals zero is a very strong structural assumption: it forces the latent correlation structure to be compatible with the CLR transform, which is itself an artifact of compositional measurement. This constraint is not a property of the true ecological correlation structure but a property of the measurement process.

**Why this matters:** A truly unifying generative model should generate data that looks like real microbiome data and should be testable as such. The CCM assumes (i) a latent Gaussian structure, (ii) a softmax link to the simplex, and (iii) multinomial sampling. This is a reasonable approximation in some regimes but a poor fit for zero-inflated, overdispersed, or multimodal communities. The fact that the v4 simulator (which closely matches the DM layer's assumptions) favors AdaCoNet, while SparseDOSSA2 (which does not) produces more mixed results, is itself evidence that the CCM's generative assumptions are not universally valid. The authors acknowledge this partially in the limitations but do not provide any goodness-of-fit test for the CCM on real data.

**Suggestion:** (a) Provide a goodness-of-fit assessment: simulate data from the fitted CCM and compare summary statistics (zero fraction, diversity indices, correlation structure) to real data. (b) Discuss the Sigma-tilde times 1 equals 0 constraint explicitly: what ecological correlation structures are incompatible with this constraint? (c) Acknowledge that the CCM is an organizing framework for understanding estimator behavior under a specific generative assumption, not a universally valid generative model for microbiome data.

**Severity:** Medium-High. The CCM is the conceptual backbone of the paper. Its limitations should be discussed with the same rigor as its strengths.

---

### W3: The Ensemble Weights Lack Comparison with Principled Alternatives
**Problem:** The weight formula w_S = alpha_0 / (alpha_0 + c_ref) is presented as "provably near-optimal without empirical tuning" (Theorem 3), but this claim is problematic on several grounds. First, c_ref = 0.05 is itself an empirical choice, drawn from Chen and Li (2009) for a different purpose (variable selection under DM). The claim of being "without empirical tuning" is therefore contradictory. Second, the formula is a simple shrinkage estimator — it has the form of a James-Stein-type shrinkage toward zero for the Spearman weight — and there is a vast literature on principled ensemble combination that is entirely ignored.

**Why this matters:** The ensemble methods literature offers several well-studied alternatives: (a) Stacking (Wolpert 1992, Breiman 1996), which uses cross-validated predictions to learn optimal combination weights; (b) Bayesian model averaging (Hoeting et al. 1999), which weights models by their marginal likelihood; (c) Cross-validated weight selection, which directly optimizes a held-out performance metric; (d) EXPOL and exponential weighting schemes (Catoni 1997, Juditsky and Nemirovski 2000). Each of these comes with theoretical guarantees (e.g., oracle inequalities for stacking, consistency for BMA under correct specification). The paper provides no comparison with any of these alternatives, either theoretically or empirically. The ablation study compares against equal-weight ensembles and leave-one-out variants, but not against data-driven weight learning.

**Suggestion:** (a) Add a stacking baseline: use K-fold cross-validation to learn optimal weights for the four layers and compare with the theory-driven weights. (b) Add a simple convex-combination optimization: minimize held-out MSE over the weight simplex. (c) Discuss theoretically how w_S = alpha_0/(alpha_0 + c_ref) relates to shrinkage estimators and under what conditions it would outperform or underperform stacking. (d) Remove or qualify the claim "provably near-optimal without empirical tuning."

**Severity:** High. This is the core methodological contribution, and the absence of comparison with well-established alternatives is a significant gap.

---

### W4: The StARS Adaptation Is Methodologically Unsound
**Problem:** As a co-author of the original StARS paper (Liu, Roeder, and Wasserman, NeurIPS 2010), I feel obligated to scrutinize the adaptation of StARS to this setting carefully. StARS was designed for sparse graph selection via a regularization path: one varies a regularization parameter lambda, computes edge presence/absence across subsamples, and selects the lambda that minimizes total instability while maintaining sparsity. The key insight of StARS is that the instability curve as a function of lambda has a characteristic U-shape for well-specified problems, and the optimal lambda lies near the minimum of this curve.

AdaCoNet applies StARS to min-max normalized continuous ensemble scores rather than to a regularization path. This is a fundamentally different use case. The normalized scores are bounded in [0,1] and follow a highly skewed distribution with a long tail of near-zero values (as the authors acknowledge). The threshold tau sweeps over score values, not regularization parameters, and the resulting instability does not exhibit the U-shape that StARS relies on. The consequence is the acknowledged degeneracy: StARS selects tau = 1.0, yielding at most 1 edge, across all subsample counts from 5 to 50.

**Why this matters:** The StARS degeneracy is not merely an inconvenient detail — it means that the paper's primary edge selection mechanism fails. The authors handle this by falling back to raw score ranking (AUROC/AUPRC) for evaluation and fixed-density thresholds for real data, which is a reasonable pragmatic workaround but means that StARS plays no functional role in the method. Given that StARS is listed as a key component of the architecture (Figure 1), this is misleading. The real-data results in Table 4 all use a fixed 5% density threshold, not StARS-selected edges, so the contribution of StARS to the reported results is nil.

**Suggestion:** (a) Replace StARS with an edge selection method appropriate for continuous score matrices, such as FDR control via the Benjamini-Hochberg procedure applied to a suitable null distribution, or a permutation-based threshold. (b) If StARS is retained, reformulate its application: rather than sweeping a threshold over normalized scores, define a regularization path (e.g., by varying the minimum ensemble weight or the number of included layers) and apply StARS to this path. (c) At minimum, be transparent in the main text (not just the Discussion) that StARS fails on this data type and that all reported results use alternative edge selection.

**Severity:** High. StARS is presented as a core architectural component but contributes nothing to the actual results. This misrepresents the method.

---

### W5: Finite-Sample Behavior of the Weight Formula Is Under-Explored
**Problem:** The weight formula w_S = alpha_0 / (alpha_0 + c_ref) with c_ref = 0.05 produces weights that are sensitive to the estimation of alpha_0, particularly in the regime where alpha_0 is near c_ref. The paper reports that alpha_0/p ranges from 0.03 to 0.86 across v4 configurations, meaning alpha_0 itself is often in the range [0.03, 0.12] — close to c_ref = 0.05. In this regime, small perturbations in the DM fit (which is itself estimated from data via method-of-moments plus Newton-Raphson) can produce large changes in w_S.

**Why this matters:** The sensitivity analysis in the Discussion varies c_ref but does not vary the estimated alpha_0. In practice, alpha_0 is estimated with error, and this estimation error propagates into the weights. The paper provides no confidence intervals for alpha_0, no analysis of the variance of the DM estimator of alpha_0, and no discussion of how estimation error in alpha_0 affects downstream network inference. At P=1000, where alpha_0/p drops to approximately 0.03-0.04, the method is in the low-weight regime where w_S is small and sensitive to perturbations.

**Suggestion:** (a) Report the standard error or bootstrap confidence interval of the estimated alpha_0 across seeds. (b) Conduct a sensitivity analysis: perturb alpha_0 by plus/minus one standard error and report the resulting change in AUROC. (c) Consider a Bayesian treatment that places a prior on alpha_0 and integrates over posterior uncertainty rather than plugging in a point estimate.

**Severity:** Medium. The empirical results suggest the method is reasonably robust (the sensitivity analysis on c_ref is partially reassuring), but the analysis is incomplete.

---

## Detailed Comments

### On the theoretical results

**Theorem 1 (CLR Variance Inflation).** The variance inflation factor (1 + p/|alpha|) is a known consequence of the delta method applied to log-ratio transforms of multinomial or Dirichlet-multinomial data. See, for example, Aitchison (1986, "The Statistical Analysis of Compositional Data") and subsequent work on the variance structure of log-ratio coordinates. The derivation in the proof sketch is standard. This is a useful result to state explicitly, but it should be attributed as a known consequence rather than presented as a novel theorem.

**Theorem 2 (Phase Transition).** As discussed in W1, the MSE bound is a continuous function of alpha_0. The statement "there exists a critical threshold c*" is not proven — it is asserted. The proof sketch does not derive c* or show that the MSE exhibits qualitatively different behavior above and below c*. What is shown is that the MSE bound decreases as alpha_0 increases, which is expected. The binary classification into "multinomial regime" and "copula regime" is a useful heuristic but lacks a formal decision-theoretic justification (e.g., minimax optimality of one estimator class over another in different parameter regions).

**Theorem 3 (Optimal Ensemble Weights).** The claim that MSE-minimizing weights satisfy w_k proportional to 1/MSE_k is correct under the assumption of uncorrelated estimator errors — a standard result in ensemble theory (see, e.g., Bates and Granger 1969 for the forecasting literature). The specialization to w_S = alpha_0/(alpha_0 + c_ref) follows from substituting the MSE bound from Theorem 2, but the derivation is not fully shown and relies on the unspecified function g(alpha_0, p). The zero-fraction guard f_0 > 0.5 is a heuristic overlay, not derived from the theory.

### On the empirical results

**Self-favouring bias of v4.** The authors acknowledge this but I want to emphasize its severity. The v4 simulator generates data via MVN sampling, exponentiation, normalization, and multinomial draws — precisely the generative process that the DM layer is designed to capture. AdaCoNet's advantage on v4 is therefore partly an artifact of the DM layer being the Bayes-optimal estimator under the true generative model. A more convincing evaluation would use a simulator that does not match any single layer's assumptions. The SparseDOSSA2 results partially address this, but here AdaCoNet is not the top performer (proportionality alone achieves 0.899 vs AdaCoNet's 0.714).

**The P=1000 regime.** At P=1000, REBACCA slightly outperforms AdaCoNet (0.739 vs 0.729). Given that REBACCA runs in under 0.01 seconds while AdaCoNet takes several seconds, this is an important practical result that deserves more discussion. What does the theory predict in this regime, and why does the 3-layer ensemble (DM + Prop + Copula) underperform a simple closed-form estimator?

**Real data interpretation.** The real-data results are difficult to interpret without ground truth. Modularity Q is a proxy for network quality but a high modularity score does not imply that the edges are biologically meaningful. The MovingPictures result, where AdaCoNet achieves Q=0.163 while proportionality achieves Q=0.438, is concerning and suggests that the ensemble may not adapt well to temporal or longitudinal structure.

### On the StARS degeneracy

I want to elaborate on why StARS fails in this setting, drawing on the theory developed in our original paper and subsequent work (Liu et al. 2012, "High-dimensional semiparametric Gaussian copula graphical models"). StARS relies on the following property: as the regularization parameter lambda decreases, the graph becomes denser, and the instability (measured as the average disagreement across subsamples) first decreases (as true edges stabilize) and then increases (as spurious edges enter). This U-shape is essential for the method to work.

When applied to min-max normalized continuous scores, the threshold tau plays the role of lambda, but the relationship is inverted and the score distribution matters critically. With a highly skewed score distribution (most edges have near-zero scores, a few have high scores), the instability at any threshold tau < 1 is dominated by the instability of the many near-zero edges, which are randomly included or excluded across subsamples. This drives the total instability up for any tau < 1, and the minimum instability is achieved at tau = 1 (no edges). The U-shape degenerates into a monotone function.

This is not a bug that can be fixed by tuning the number of subsamples or the subsample size — it is a fundamental mismatch between the score distribution and StARS's assumptions. The correct approach would be to either transform the scores to produce a more uniform distribution before applying StARS, or to use a different edge selection framework entirely.

---

## Questions for Authors

1. **On the phase transition claim:** Can you provide a precise definition of c* and a proof of its existence? Specifically, can you show that the MSE of Spearman-on-CLR exhibits a non-analytic behavior (discontinuity in some derivative) at c*, or is the "transition" simply a continuous degradation that you discretize for heuristic purposes? If the latter, would "efficiency boundary" be a more accurate term?

2. **On the ensemble weights:** How does the theory-driven weight w_S = alpha_0/(alpha_0 + c_ref) compare empirically to stacking weights learned via cross-validation? In particular, on the SparseDOSSA2 data where the CCM's generative assumptions are violated, would data-driven weights outperform the theory-driven ones? Have you tested this?

3. **On the CCM constraint:** The constraint that Sigma-tilde times 1 equals zero means the latent Gaussian variables sum to zero, which is the CLR parameterization. What ecological or biological correlation structures are incompatible with this constraint? For example, if a subset of taxa are truly independent of all others (a block-diagonal Sigma with a zero-correlation block), can the CCM represent this?

4. **On the role of StARS:** Given that StARS degenerates to tau = 1.0 and contributes no edges to any reported result, would it be more accurate to describe AdaCoNet as a score-combination method with fixed-density edge selection, rather than as a method that uses StARS for graph selection? What is the intended role of StARS in the architecture going forward?

---

## Dimension Scores

| Dimension                     | Score (1-5) | Notes                                                                                       |
|-------------------------------|:-----------:|---------------------------------------------------------------------------------------------|
| Novelty / Originality         | 3           | CCM is a useful organizing device; "phase transition" is overstated                         |
| Theoretical Rigor             | 2           | Upper bounds only; no matching lower bounds; c* not precisely characterized                  |
| Technical Soundness           | 3           | Delta-method derivations are correct; ensemble weight claims exceed what is proven           |
| Empirical Evaluation          | 4           | Thorough benchmark design with two simulators, ablations, and real data; self-favouring bias |
| Comparison with Prior Work    | 2           | No comparison with stacking, BMA, or cross-validated ensembles; limited baseline methods     |
| StARS Implementation          | 1           | Fundamentally misapplied; degeneracy acknowledged but not resolved; no functional role        |
| Writing and Presentation      | 4           | Clear, well-organized; overclaiming in theoretical section                                  |
| Reproducibility               | 4           | Code and data available; simulation parameters well-documented                               |
| Practical Utility             | 4           | Fast, competitive accuracy; useful for large-scale microbiome studies                        |
| Significance to the Field     | 3           | Addresses a real need but theoretical overclaiming limits impact                             |

---

## References Cited in This Review

- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.
- Bates, J.M. and Granger, C.W.J. (1969). The combination of forecasts. *Journal of the Operational Research Society*, 20(4), 459-468.
- Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49-64.
- Catoni, O. (1997). A mixture approach to adaptive model selection. *Annals of Statistics*.
- Hoeting, J.A. et al. (1999). Bayesian model averaging: a tutorial. *Statistical Science*, 14(4), 382-417.
- Juditsky, A. and Nemirovski, A. (2000). Reliable accuracy estimates from k-fold cross validation. *Mathematics of Operations Research*.
- Liu, H., Roeder, K., and Wasserman, L. (2010). Stability approach to regularization selection (StARS) for high dimensional graphical models. *NeurIPS*, 23, 1432-1440.
- Liu, H., Han, F., Yuan, M., Lafferty, J., and Wasserman, L. (2012). High-dimensional semiparametric Gaussian copula graphical models. *Annals of Statistics*, 40(4), 2293-2326.
- Wolpert, D.H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259.

---

*Review completed: June 2026*
