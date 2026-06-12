# Peer Review Report — EIC

## Reviewer Information
- **Role:** Editor-in-Chief
- **Identity:** Prof. Janet Kelso, EIC of *Briefings in Bioinformatics*
- **Expertise:** Computational genomics and metagenomics
- **Focus:** Journal fit, originality, significance, readership relevance

---

## Overall Assessment

### Recommendation: Minor Revision

### Confidence: 4 (High — familiar with the field and its methodological landscape)

### Summary

Rao and Liang present AdaCoNet, a four-layer ensemble framework for microbial co-occurrence network inference, underpinned by a novel theoretical construct — the Compositional Copula Model (CCM) and an associated Phase Transition Theorem. The central idea is elegant: rather than treating the choice among compositional correlation estimators (SparCC, proportionality, Spearman-on-CLR, copula) as an empirical hyperparameter, the CCM unifies them as distinct estimators of a single latent correlation matrix and identifies the Dirichlet–Multinomial concentration ratio as the key quantity governing estimator reliability. Theory-driven ensemble weights then follow without empirical tuning.

The paper is well-motivated and addresses a genuine problem — no single co-occurrence method consistently dominates across data-generating regimes, and existing ensemble approaches lack principled justification. The benchmarking is reasonably comprehensive (two simulation frameworks, two real datasets, eight competing methods), and the ablation study directly validates the theoretical predictions. The manuscript is clearly written and well-organized, with effective use of figures and tables.

However, several issues temper enthusiasm. The v4 simulator's generative mechanism (multinomial draws from a Dirichlet prior) is identical to the distributional assumption of the DM layer, creating a self-favouring bias that the authors acknowledge but do not fully counterbalance. Key methods (gCoda, FlashWeave, CoNI) are excluded from benchmarking for logistical reasons, weakening the completeness claim. The real-data evaluation is limited, and the StARS edge-selection failure on ensemble scores is methodologically troubling. These issues are addressable in revision.

---

## Strengths

### S1: Novel and well-motivated theoretical framework
The CCM and Phase Transition Theorem represent a genuine conceptual advance. Unifying disparate co-occurrence estimators under a single generative model and deriving adaptive weights from first principles is both elegant and practically useful. To my knowledge, no prior work has provided this level of theoretical justification for ensemble composition in compositional network inference. The identification of the DM concentration ratio α₀/p as the regime-governing quantity (Theorem 2, lines 123–133) is a specific, testable, and insightful claim.

### S2: Rigorous ablation study validating theoretical predictions
The ablation study (Table 3, lines 254–281) is the strongest empirical element. The cross-simulator inversion — Spearman dominant on v4, copula/proportionality dominant on SparseDOSSA2 — directly validates the Phase Transition Theorem's predictions. The quantified cost of ignoring the theorem (0.034 AUROC loss on SD2, line 279) provides concrete evidence that theory-driven weighting matters.

### S3: Comprehensive simulation benchmarking across distinct data-generating mechanisms
Using two fundamentally different simulation frameworks (direct-covariance embedding and SparseDOSSA2 Gaussian copula) is a methodological strength. The cross-simulator comparison (Fig. 3, lines 213–217) demonstrates that AdaCoNet's adaptive mechanism responds appropriately to different generative regimes, which is the central claim of the paper.

### S4: Strong practical performance with competitive computational efficiency
AdaCoNet's speed–accuracy profile (Fig. 4, lines 294–298) is compelling: 5–85× faster than SparCC while achieving equal or better AUROC. For the *Briefings in Bioinformatics* readership — which includes practitioners deploying these tools on real microbiome datasets — this practical accessibility matters.

### S5: Clear, well-organized manuscript with effective visual communication
The four figures are well-designed and informative. Fig. 1 (architecture overview) effectively communicates the framework, and Fig. 3 (cross-simulator comparison) is particularly effective at conveying the core message. The manuscript structure follows a logical progression from motivation to theory to empirical validation.

---

## Weaknesses

### W1: Self-favouring bias in the primary simulation benchmark
**Problem:** The v4 simulator generates data through "MVN sampling, exponentiation, normalization, and multinomial draws" (line 205) — effectively multinomial draws from a Dirichlet prior, which is the exact generative model assumed by AdaCoNet's DM layer. This creates a circular evaluation advantage.
**Why it matters:** The DM posterior correlation is the Bayes estimator under the true generative model in v4, so AdaCoNet's advantage may partly reflect model-matching rather than genuine inferential superiority. The authors acknowledge this (limitation i, lines 400–401) and note SparseDOSSA2 as a counterpoint, but the primary benchmark's credibility is diminished.
**Suggestion:** Include at least one additional simulation framework with a generative mechanism that does not favour any single layer — e.g., a Dirichlet-tree multinomial, a logistic-normal with non-Dirichlet covariance structure, or a zero-inflated negative binomial model. Alternatively, present a version of AdaCoNet without the DM layer to isolate the DM contribution from the ensemble benefit.
**Severity:** Moderate — the SparseDOSSA2 results partially mitigate this, but the primary claims rest heavily on v4.

### W2: Incomplete benchmarking — key methods excluded
**Problem:** gCoda (Fang et al. 2017), FlashWeave (Tackmann et al. 2019), CoNI (Klaus et al. 2022), and SparXCC (Jensen et al. 2024) are all mentioned in the Introduction (lines 67–68) and Discussion (line 396) but excluded from benchmarking. The justification — "primarily available as R packages, while our benchmark infrastructure is implemented in Python" (line 396) — is a logistical limitation, not a scientific one.
**Why it matters:** For a methods paper claiming to advance the state of the art in compositional network inference, excluding multiple prominent competitors weakens the completeness of the evaluation. The *Briefings in Bioinformatics* readership will expect comprehensive method comparison.
**Suggestion:** At minimum, include gCoda (a direct extension of SparCC with compositionally robust estimation) in the benchmark, as it is the most directly comparable method. For others, a brief qualitative comparison under the CCM framework would partially address this gap.
**Severity:** Moderate — affects the paper's claim to comprehensiveness but does not invalidate the core contribution.

### W3: Limited real-data evaluation and weak MovingPictures performance
**Problem:** Only two real datasets are tested (Enterotype: N=280, P=550; MovingPictures: N=1967, P=926). On MovingPictures, AdaCoNet achieves modularity Q=0.163 versus proportionality's Q=0.438 (Table 4, lines 322–326) — a substantial gap. The authors provide a plausible explanation (lines 398–399) but this represents a notable practical failure on a widely used benchmark dataset.
**Why it matters:** Real-data performance is where practitioners form their tool-selection decisions. The MovingPictures result raises questions about AdaCoNet's practical utility for temporal/longitudinal microbiome data — a common use case for the *Briefings in Bioinformatics* readership.
**Suggestion:** Include at least one additional real dataset with known biological ground truth (e.g., a mock community or a dataset with validated interactions) to enable quantitative rather than solely topological evaluation. The current evaluation relies entirely on network topology (modularity, connectivity), which is an indirect proxy for correctness.
**Severity:** Moderate–High — affects practical recommendations.

### W4: StARS edge-selection failure undermines practical deployment
**Problem:** The authors report that StARS "consistently selected τ = 1.0 (yielding ≤1 edge) across all subsample counts" (lines 405–406) on ensemble scores, recommending fixed-density thresholds instead. This is a significant practical limitation for a method that includes StARS as part of its pipeline (Fig. 1, line 158).
**Why it matters:** Edge selection is a critical step in network inference. If the included edge-selection mechanism produces degenerate results, the framework is incomplete for end-to-end deployment. The Discussion acknowledges this (lines 404–407) but the fix (fixed-density thresholds) is ad hoc and removes the data-adaptive advantage that StARS was meant to provide.
**Suggestion:** Either (a) develop or adopt an edge-selection method that works reliably with ensemble scores (e.g., FDR-based thresholding, permutation-based cutoffs), or (b) remove StARS from the pipeline description and recommend external edge selection from the outset. Presenting a broken component as part of the architecture is misleading.
**Severity:** Moderate — does not affect the simulation evaluation (which uses AUROC on raw scores) but affects real-data utility.

### W5: Incomplete manuscript elements and presentation issues
**Problem:** Several manuscript elements contain unresolved placeholders: "[Lab name]" (line 423), "[institution, to be confirmed]" (line 423), "grant number to be confirmed" (line 414). The reference to "Supplementary Fig. S1" (line 402) and "Supplementary Table S2" (line 406) implies supplementary material that is not provided with this submission. The constant c_ref = 0.05 (line 144) is attributed to Chen and Li (2009) but that paper concerns variable selection for DM distributions — the connection to this specific threshold value is not established in the main text.
**Why it matters:** Incomplete placeholders signal a pre-submission draft and reduce confidence in the manuscript's readiness. Missing supplementary material prevents verification of key claims (sensitivity analysis, StARS details).
**Suggestion:** Resolve all placeholders before resubmission. Include supplementary material or move critical results (sensitivity analysis of c_ref, StARS analysis) into the main text or an appendix. Clarify the derivation of c_ref = 0.05 from Chen and Li (2009).
**Severity:** Minor–Moderate — presentation issues that are easily addressed but currently detract from professionalism.

---

## Detailed Comments

### Title and Abstract
- The title "AdaCoNet: A Phase-Transition Theory for Adaptive Compositional Network Inference" is clear and descriptive, though the claim of a "Phase-Transition Theory" may overstate what is essentially a threshold result on the CLR variance inflation factor. The term "phase transition" in statistical physics implies a thermodynamic limit and symmetry-breaking; here it refers to a regime change in estimator MSE. Consider softening to "A Phase-Transition Framework" or "A Regime-Adaptive Theory" to avoid over-claiming.
- The abstract (lines 53–56) is well-structured (Motivation/Results/Availability) and informative. It effectively summarizes the key contributions. However, the claim of AUROC "up to 0.94" is somewhat selective — the abstract should note that this peak occurs at the most favorable configuration (N=200, P=50) and that performance degrades at higher dimensions.
- The abstract's statement that AdaCoNet is "5–85× faster than SparCC" (line 54) is accurate but potentially misleading since REBACCA is even faster (<0.01s) and achieves competitive AUROC. The speed comparison should include REBACCA as a reference point.

### Introduction
- The Introduction (lines 63–71) provides a concise and accurate survey of the methodological landscape. The categorization of methods into distinct statistical frameworks (log-ratio, penalized, information-theoretic, proportionality) is helpful.
- The motivation paragraph (lines 69–70) — "no single method demonstrates consistent superiority" — is well-supported by the subsequent results and is the paper's strongest motivational argument.
- However, the Introduction does not discuss the existing ensemble approaches in the field (e.g., CoNI's ML-based integration), which would help position AdaCoNet's theoretical contribution relative to prior ensemble attempts.

### Results
- Table 1 (lines 172–203) is well-constructed with appropriate reporting of mean ± std across 10 seeds. The exclusion of SparCC/FastSpar/CCLasso at P=1000 is justified by computational cost but should be explicitly noted in the table caption rather than only in the table notes.
- The high-confidence edge precision analysis (Table 5, lines 346–374) is a valuable addition that goes beyond standard AUROC/AUPRC reporting. The P@FPR metrics are particularly informative for practitioners.
- The claim that the zero-fraction guard "correctly downweight[s] Spearman on seeds with anomalous zero inflation" (line 207) is well-illustrated by the specific example (seed 1 at N=500, P=500: f₀ = 0.58, w_S = 0.17) but would benefit from systematic reporting of how often this guard activates across all seeds and configurations.

### Discussion
- The Discussion (lines 382–407) is substantive and addresses most key issues. The honest treatment of the v4 self-favouring bias (limitation i, lines 400–401) and the StARS failure (lines 404–407) is commendable.
- The sensitivity analysis of c_ref (lines 402–403) is important and should be elevated — currently it is buried in the Discussion rather than presented alongside the main results. A reader encountering Theorem 3 (line 139) would naturally wonder about robustness to this constant.
- The paragraph on MovingPictures (lines 398–399) provides a reasonable hypothesis for the low modularity but stops short of proposing a concrete solution. The suggestion that "a modified weight scheme that upweights proportionality" could help is speculative without supporting analysis.
- Limitation (v) (lines 400–401) — that α₀/p drops below c_ref at P=1000 and the ensemble underperforms REBACCA — is a significant caveat for high-dimensional applications that deserves more prominent placement (perhaps in the Results section alongside Table 1).

---

## Questions for Authors

**Q1.** The v4 simulator generates data from a Dirichlet–Multinomial distribution, which is the same model assumed by AdaCoNet's DM layer. Can the authors provide results from an ablation where the DM layer is excluded from the ensemble (not just "w/o DM" in Table 3, but a fully re-weighted 3-layer ensemble with recalibrated theory weights), to isolate the ensemble benefit from the model-matching benefit? Alternatively, can the authors benchmark on a simulation framework with a non-Dirichlet generative mechanism (e.g., logistic-normal with arbitrary covariance)?

**Q2.** The constant c_ref = 0.05 is attributed to Chen and Li (2009), but that paper addresses variable selection for DM distributions, not CLR variance inflation thresholds. Can the authors clarify the precise connection? Is c_ref derived from their work, or is it chosen based on the authors' own analysis? If the latter, a more thorough justification — beyond the sensitivity analysis in the Discussion — is warranted in the Methods section.

**Q3.** The StARS edge-selection component fails on ensemble scores, and the authors recommend fixed-density thresholds instead. Given that StARS is depicted as part of the AdaCoNet architecture (Fig. 1), do the authors plan to develop a robust edge-selection method for ensemble scores, or should StARS be considered an optional external component rather than an integral part of the framework?

**Q4.** How does AdaCoNet perform when the number of taxa P substantially exceeds the number of samples N (the P >> N regime common in metagenomic studies)? The current benchmarks include N=200, P=50 and N=500, P=500, but the P >> N case (e.g., N=50, P=500) is not tested. This regime is where compositional effects are most severe and where the theoretical predictions of the Phase Transition Theorem would be most informative.

---

## Dimension Scores

| Dimension | Score | Weight | Weighted | Descriptor |
|-----------|-------|--------|----------|------------|
| Originality | 82 | 20% | 16.4 | **Strong** — The CCM and Phase Transition Theorem represent a genuine conceptual contribution that unifies existing methods under a single generative framework. The theory-driven weighting approach is novel in this domain. |
| Methodological Rigor | 70 | 25% | 17.5 | **Adequate** — The theoretical results are interesting but proofs are deferred to supplementary material (not provided). The v4 self-favouring bias, exclusion of key competitor methods, and StARS failure reduce methodological completeness. |
| Evidence Sufficiency | 68 | 25% | 17.0 | **Adequate** — Two simulation frameworks and two real datasets provide reasonable coverage, but the primary benchmark's circularity, limited real-data evaluation, absence of biological ground truth, and missing high-dimensional regime testing leave gaps. |
| Argument Coherence | 78 | 15% | 11.7 | **Strong** — The paper builds a coherent narrative from motivation through theory to validation. The ablation study's cross-simulator inversion elegantly supports the central claim. Minor issues with buried results and unclarified constants. |
| Writing Quality | 74 | 15% | 11.1 | **Adequate** — Generally clear and well-organized with effective figures and tables. Unresolved placeholders, missing supplementary material, and the "phase transition" terminology over-claim detract from polish. |
| **Weighted Average** | **73.7** | **100%** | **73.7** | **Minor Revision** |

---

## Editorial Note

The manuscript presents a genuinely interesting theoretical contribution that has the potential to influence how practitioners think about compositional network inference. The CCM framework is elegant, and the Phase Transition Theorem provides actionable guidance for ensemble design. However, the empirical evaluation has notable gaps — particularly the self-favouring v4 benchmark, the exclusion of several relevant competitor methods, and the limited real-data analysis — that must be addressed before the paper can be considered for publication in *Briefings in Bioinformatics*.

I recommend **Minor Revision** with the expectation that the authors will: (1) include at least one non-Dirichlet simulation framework or provide a DM-excluded ablation analysis; (2) incorporate gCoda or another directly comparable competitor into the benchmark; (3) resolve all manuscript placeholders and include supplementary material; and (4) clarify the derivation of c_ref and the status of StARS within the framework.

The theoretical contribution alone is of interest to the *Briefings in Bioinformatics* readership, and with a strengthened empirical evaluation, this manuscript could make a valuable addition to the journal.
