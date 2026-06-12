# Editorial Decision

## Manuscript Information
- **Title**: AdaCoNet: A Phase-Transition Theory for Adaptive Compositional Network Inference
- **Journal**: Briefings in Bioinformatics
- **Decision Date**: 2026-06-12
- **Review Round**: 1

---

## Decision

### Major Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC (Prof. Janet Kelso) | Editor-in-Chief | Minor Revision | 4/5 |
| R1 (Prof. Hongzhe Li) | Methodology | Major Revision | 4/5 |
| R2 (Prof. Rob Knight) | Domain Expert | Major Revision | 4/5 |
| R3 (Prof. Larry Wasserman) | Cross-disciplinary | Major Revision | 5/5 |
| DA (Dr. "Skeptic") | Devil's Advocate | CRITICAL issues found | — |

**Weighted scores**: EIC 73.7, R1 61.4, R2 68.0, R3 ~60 (estimated from 1-5 scale). Panel average: **~65**, at the boundary between Minor and Major Revision. However, 3 CRITICAL issues identified by the Devil's Advocate preclude Acceptance (Iron Rule #4), and 3 of 4 peer reviewers recommend Major Revision.

---

## Consensus Analysis

### Points of Agreement (Consensus)

**[CONSENSUS-5]** (All reviewers agree):

1. **The CCM unification is conceptually valuable.** All reviewers acknowledge that framing four existing estimators (DM posterior, Spearman-CLR, proportionality, Gaussian copula) as different estimators of a single latent correlation matrix is a productive organizing principle. EIC calls it "a genuine conceptual advance" (S1); R1 says it is "the paper's strongest element" (S1); R2 calls it "a genuine conceptual contribution" (S1); R3 says it "provides a coherent vocabulary" (S3); DA acknowledges "the CCM as a unifying vocabulary has conceptual value" (Observations #6).

2. **The "Phase Transition Theorem" is overstated.** Theorem 2 provides only an MSE upper bound with unspecified constants C_1, C_2 and unspecified function g. No matching lower bound, no proof of discontinuity, no precise characterization of c*. R1 (co-author of Chen & Li 2009): "this is an MSE rate bound, not a sharp transition in any rigorous sense" (W2). R3 (co-author of StARS): "this is a standard delta-method variance inflation analysis" (W1). DA: "an upper bound with loose asymptotic characterisation dressed in the language of statistical physics" (C-D2). EIC notes the title may "overstate what is essentially a threshold result" (Detailed Comments).

3. **The v4 simulator creates a self-favouring bias.** The generative mechanism (MVN → exp → normalize → multinomial) matches the DM layer's prior family, making the DM posterior the Bayes estimator under the true model. All reviewers flag this as a credibility issue for the primary benchmark.

4. **StARS is non-functional and should be removed or fixed.** StARS degenerates to τ=1.0 (≤1 edge) across all subsample counts. R3 provides a detailed technical explanation of why min-max normalized scores with skewed distributions are fundamentally incompatible with StARS's U-shaped instability requirement (W4). All reviewers agree the architecture diagram should not depict StARS as a core component when it contributes nothing to reported results.

5. **Real-data evaluation is insufficient.** Only 2 datasets (both human gut, both >10 years old), only topological metrics (modularity, MaxCC), no biological validation (keystone species, functional enrichment, known interactions), and AdaCoNet underperforms proportionality dramatically on MovingPictures (Q=0.163 vs 0.438).

### Points of Disagreement

**Disagreement 1: Severity of theoretical overclaiming**
- **EIC view**: Minor Revision — the theoretical contribution is genuine, just needs reframing and supplementary material.
- **R1/R3 view**: Major Revision — the "theorems" are either standard results (Theorem 1), unproven bounds (Theorem 2), or heuristic formulas mislabeled as "optimal" (Theorem 3). The c_ref = 0.05 attribution to Chen & Li (2009) is a factual error (R1 is a co-author of that paper and confirms this).
- **DA view**: CRITICAL — the theoretical apparatus is "a post-hoc rationalisation of a heuristic design, not a genuine predictive theory" (C-D2).
- **Editor's Resolution**: The panel majority (R1, R3, DA) correctly identifies that the theoretical claims exceed what is established. The c_ref misattribution is a verifiable factual error. **Major Revision required for theoretical reframing.**

**Disagreement 2: Whether the ensemble adds value over its best single layer**
- **R1/DA view**: CRITICAL — Spearman-only (0.952) beats the full ensemble (0.919) on v4 N200,P50. The theory assigns w_S ∈ [0.16, 0.32] when the optimal weight would be ~1.0. This directly contradicts "provably near-optimal."
- **EIC/R2 view**: The ensemble's value is cross-simulator robustness, not single-regime optimality. AdaCoNet's advantage is that it performs well across both simulators without requiring prior knowledge of the generative regime.
- **Editor's Resolution**: Both perspectives have merit. The cross-regime robustness argument is valid, but the paper must address the ablation paradox explicitly: under what conditions does the ensemble outperform its best individual layer? If the answer is "never on a single simulator, only across simulators," this should be stated clearly.

---

## Decision Rationale

AdaCoNet addresses a genuine and important problem — the absence of principled guidance for choosing among competing microbial co-occurrence estimators. The CCM framework is a valuable conceptual contribution that unifies existing methods under a single generative model. The empirical evaluation is substantial: two distinct simulation frameworks, 10 seeds per configuration, 9 competing methods, and a thorough ablation study.

However, the manuscript has three categories of issues that require major revision. First, the theoretical claims significantly exceed what is established: Theorem 1 is a standard result, Theorem 2 provides only an upper bound (not a phase transition), Theorem 3 misattributes c_ref = 0.05 to a paper that does not discuss this quantity, and full proofs are not provided. Second, the pipeline has a non-functional component (StARS) depicted as integral to the architecture. Third, the empirical evaluation has significant gaps: a self-favouring primary simulator, limited real-data analysis with no biological validation, and an unresolved performance failure on longitudinal data.

The decision is Major Revision rather than Minor because: (1) the theoretical reframing requires substantive rewriting of the Methods section and potentially relabeling the theorems; (2) the StARS removal/replacement affects both the Methods and architecture diagram; (3) at least one additional simulator or DM-excluded analysis is needed to address the self-favouring bias; and (4) the real-data section requires biological validation beyond topology metrics.

---

## Required Revisions (Must Fix)

| # | Revision Item | Source | Severity | Section | Effort |
|---|--------------|--------|----------|---------|--------|
| R1 | Reframe theorems: rename "Phase Transition" to "MSE Reliability Bound"; relabel Theorems as Propositions or provide full proofs with matching lower bounds | R1, R3, DA | Critical | Methods (§2.6) | 2 weeks |
| R2 | Correct c_ref attribution: remove Chen & Li (2009) citation; justify c_ref empirically or derive c* from the MSE bound; rename "Optimal Weights" to "Theory-Guided Weights" | R1 (co-author of cited paper) | Critical | Methods (§2.6), Results | 1 week |
| R3 | Remove or fix StARS: either develop a working edge selection for ensemble scores, or remove StARS from Methods, architecture diagram, and re-describe the pipeline as score combination + fixed-density/FDR thresholding | R1, R3, DA, EIC | Critical | Methods, Fig. 1 | 1 week |
| R4 | Address the ablation paradox: explicitly analyze when the ensemble outperforms its best single layer; if never on a single simulator, reframe the value proposition as cross-regime robustness | R1, DA | Major | Results (§3.3), Discussion | 1 week |
| R5 | Add paired statistical tests: report Wilcoxon signed-rank or paired t-tests across seeds for key comparisons (AdaCoNet vs. second-best) | R1 | Major | Results (§3.2–3.4) | 3 days |
| R6 | Expand literature review: add CoNet (Faust et al. 2012), MENAP (Deng et al. 2012), NetCoMi (Peschel et al. 2021); explicitly position AdaCoNet relative to CoNet as a prior ensemble | R2 | Major | Introduction | 3 days |
| R7 | Add biological validation for real data: hub taxa analysis, module enrichment, recovery of known pairwise associations, edge overlap between methods | R2, EIC | Major | Results (§3.5) | 2 weeks |
| R8 | Report CCM diagnostics (α₀/p, f₀, layer weights) for Enterotype and MovingPictures datasets | R2, DA | Major | Results (§3.5) | 1 day |

## Suggested Revisions (Should Fix)

| # | Revision Item | Source | Priority | Section |
|---|--------------|--------|----------|---------|
| S1 | Add cross-validated weight baseline (stacking or K-fold CV) to compare with theory-driven weights | R3, DA | P2 | Results |
| S2 | Quantify v4 self-favouring bias: evaluate a DM-excluded 3-layer ensemble with recalibrated weights, or add a third simulator neutral to all layers | EIC, DA | P2 | Results |
| S3 | Add at least one non-gut real dataset (e.g., HMP oral/skin, EMP soil) | R2 | P2 | Results (§3.5) |
| S4 | Add practical guide/decision flowchart for practitioners translating theory into recommendations | R2, EIC | P2 | Discussion |
| S5 | Report standard errors/bootstrap CIs for estimated α₀ across seeds; conduct sensitivity analysis on α₀ perturbation | R3 | P2 | Results |
| S6 | Fix abstract: AUROC "up to 0.94" → correct to 0.919 (Table 1 maximum); resolve all manuscript placeholders ([Lab name], grant number) | R1, EIC | P3 | Abstract, Acknowledgments |
| S7 | Provide supplementary material (proofs, sensitivity Fig. S1, StARS Table S2) or remove all references to it | EIC, R1 | P3 | Supplementary |
| S8 | Address zero-fraction guard functional form: explicitly state the penalty formula for f₀ > 0.5 | R1 | P3 | Methods |

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Estimated: 3–4 weeks)
- [ ] R1: Reframe theorems — rename, relabel, and either provide full proofs or downgrade to Propositions with honest characterization
- [ ] R2: Correct c_ref attribution — remove misattributed citation, provide valid justification
- [ ] R3: Remove/fix StARS — update Methods, architecture diagram (Fig. 1), and real-data pipeline description
- [ ] R4: Address ablation paradox — add explicit analysis of ensemble-vs-best-layer performance

### Priority 2 — Content Supplementation (Estimated: 2–3 weeks)
- [ ] R5: Add paired statistical tests across seeds
- [ ] R6: Expand literature review (CoNet, MENAP, NetCoMi)
- [ ] R7: Add biological validation for real-data networks
- [ ] R8: Report CCM diagnostics for real datasets
- [ ] S1: Add cross-validated weight baseline
- [ ] S2: Quantify or mitigate v4 self-favouring bias

### Priority 3 — Text and Formatting (Estimated: 3–5 days)
- [ ] S4: Add practical decision guide for practitioners
- [ ] S6: Fix abstract numbers and manuscript placeholders
- [ ] S7: Provide or remove supplementary material references
- [ ] S8: Specify zero-fraction guard penalty formula
- [ ] Minor language/formatting corrections from all reviewers

### Total Estimated Effort
- **Major Revision: 6–8 weeks**

---

## Revision Deadline

- **Recommended deadline**: 2026-08-07 (8 weeks)
- **Extension policy**: If extension is needed, notify 1 week before the deadline

---

## Response Letter Instructions

Please respond to every reviewer comment item by item using the R→A→C format:
- **R (Reviewer comment)**: Quote the reviewer's concern
- **A (Author response)**: Your response and explanation
- **C (Change made)**: What was changed in the revised manuscript, with page/line numbers

Must include:
1. Response for each Required Revision (R1–R8)
2. Response for each Suggested Revision (S1–S8, adopted or reason for not adopting)
3. Change markup in the revised manuscript (highlighted or tracked)

---

## Closing

We encourage you to carefully consider the reviewers' comments and submit a substantially revised manuscript. The reviewers unanimously recognize the conceptual value of the CCM framework and the thoroughness of the empirical evaluation. The required revisions, while significant, do not require re-engineering the method itself — they primarily involve honest reframing of theoretical claims, removal of a non-functional component (StARS), and strengthening the empirical evaluation. Please note that the revised manuscript will undergo another round of review.

We look forward to receiving your revision.

---

## Appendix: Full Reviewer Reports

The 5 independent reviewer reports are available as separate files:
- `docs/review_eic.md` — EIC (Prof. Janet Kelso)
- `docs/review_r1_methodology.md` — Methodology (Prof. Hongzhe Li)
- `docs/review_r2_domain.md` — Domain (Prof. Rob Knight)
- `docs/review_r3_perspective.md` — Perspective (Prof. Larry Wasserman)
- `docs/review_devils_advocate.md` — Devil's Advocate (Dr. "Skeptic")
