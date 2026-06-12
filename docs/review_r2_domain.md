# Peer Review Report -- Domain Expert

## Reviewer Information
- **Role**: Peer Reviewer 2 (Domain Expert)
- **Identity**: Prof. Rob Knight, UCSD
- **Expertise**: Microbial ecology, diversity metrics, compositional data analysis, co-occurrence networks, QIIME/phyloseq ecosystem
- **Focus Areas**: Literature coverage, biological relevance of real-data results, practical utility for microbiome researchers, ecological validity of the framework

---

## Overall Assessment

### Recommendation: Major Revision
### Confidence: 4/5
### Summary

AdaCoNet presents a theoretically motivated ensemble framework for microbial co-occurrence network inference, grounded in a Compositional Copula Model (CCM) and a Phase Transition Theorem that predicts when rank-based versus copula-based estimators should be preferred. The theoretical ambition is commendable, and the simulation benchmarks are thorough, covering two distinct data-generating mechanisms with multiple seeds and configurations. The ablation study is particularly well-designed and provides genuine insight into layer contributions.

However, from the perspective of a microbial ecologist who would use or recommend such tools, the paper has significant gaps. The literature review omits several foundational and widely used microbial network methods (CoNet, MENAP, LSA), undermining the claim of comprehensive coverage. The real-data evaluation reports only network topology metrics (modularity, connected components) with no biological validation whatsoever -- no assessment of whether inferred modules correspond to known functional guilds, whether hub taxa are established keystone species, or whether edges overlap with known metabolic interactions. The two real datasets are both human gut microbiome studies, one dating to 2011, and neither represents the environmental, marine, or soil microbiomes where network inference is most actively applied. The MovingPictures result (Q=0.163 vs. proportionality at Q=0.438) raises concerns about the framework's suitability for longitudinal data, which the discussion acknowledges but does not adequately resolve. For *Briefings in Bioinformatics*, the theoretical machinery requires more accessible exposition for the journal's mixed computational/biological readership.

---

## Strengths

**S1. Principled theoretical unification.** The CCM framework, which recasts existing estimators (DM posterior, Spearman-CLR, proportionality, Gaussian copula) as different estimators of a single latent correlation matrix, is a genuine conceptual contribution. This unification has practical value: it gives researchers a principled basis for understanding why method A works on dataset X but fails on dataset Y, rather than treating method selection as an empirical black box.

**S2. Rigorous cross-simulator benchmarking.** The use of two fundamentally different simulation frameworks -- a direct-covariance Dirichlet-Multinomial simulator and SparseDOSSA2's zero-inflated truncated log-normal copula -- is excellent practice. The inversion of method rankings across simulators (AdaCoNet dominates on v4, proportionality dominates on SD2) is the most compelling result in the paper and directly validates the phase transition intuition. Ten seeds per configuration is adequate for variance estimation.

**S3. Informative ablation study.** The 10-variant ablation (Table 4) provides genuine mechanistic insight. The observation that removing Spearman causes the largest drop on v4 but has negligible effect on SD2, while removing copula hurts SD2 but not v4, is a clean demonstration that the ensemble is adapting its composition rather than simply averaging noise. This table alone provides more insight than many ensemble methods papers.

**S4. Practical computational efficiency.** Achieving competitive or superior accuracy in under 6 seconds at P=1000, while being 5-85x faster than SparCC, addresses a real bottleneck in microbial network analysis. The Pareto front analysis (Fig. 4) is a useful way to present the speed-accuracy trade-off.

**S5. High-confidence edge precision analysis.** Table 5 (P@k and P@FPR) is a valuable addition that goes beyond AUROC. For practitioners who will use only the top-ranked edges for experimental validation, this is arguably the most relevant metric, and its inclusion reflects awareness of real-world usage patterns.

---

## Weaknesses

### W1. Incomplete literature coverage of microbial network methods

**Problem**: The Introduction (Section 1) surveys SparCC, SPIEC-EASI, proportionality, CCLasso, REBACCA, gCoda, SparXCC, CoNI, and FlashWeave, but omits several foundational and widely cited methods in microbial co-occurrence network inference.

**Why this matters**: CoNet (Faust et al., *Nature Methods* 2012) is one of the most cited microbial network tools and was specifically designed as an ensemble that integrates multiple similarity measures (Pearson, Spearman, mutual information, Bray-Curtis, Kullback-Leibler) with a bootstrap-based significance framework. Its conceptual relationship to AdaCoNet -- both are ensemble approaches to microbial network inference -- demands direct discussion. MENAP (Deng et al., *ISME J* 2012) introduced random matrix theory-based network construction for microbial ecology and has been applied in hundreds of studies, particularly in environmental microbiomes. LSA (Biddle et al., *BMC Bioinformatics* 2008; Ruan et al., *Bioinformatics* 2006) pioneered local similarity analysis for time-series microbial data and is directly relevant to the MovingPictures application. More recent tools such as NetCoMi (Peschel et al., *Briefings in Bioinformatics* 2021) for network construction and comparison, and HiCorNet (2022) for higher-order correlation networks, are also absent.

**Suggestion**: Add a paragraph to the Introduction surveying these methods and position AdaCoNet relative to them. In particular, CoNet's ensemble philosophy should be directly compared to AdaCoNet's: CoNet uses bootstrap resampling to assess edge significance across multiple measures, while AdaCoNet uses theory-driven weights. The distinction is important for readers familiar with CoNet.

**Severity**: Major. A methods paper in *Briefings in Bioinformatics* that surveys the landscape incompletely risks misleading readers about what already exists and what gap the new method fills.

---

### W2. No biological validation of real-data network results

**Problem**: Section 3.5 reports only topological metrics for the real-data networks: edge count, density, MaxCC, number of components, and modularity Q. There is no assessment of biological meaning.

**Why this matters**: Modularity is a graph-theoretic property, not a biological one. A high modularity score does not necessarily indicate biologically meaningful community structure -- it could reflect technical artefacts, phylogenetic clustering, or batch effects. For microbiome researchers to trust AdaCoNet's output, the paper should demonstrate that:
- **Known keystone taxa occupy hub positions** in the inferred network (e.g., *Bacteroides*, *Faecalibacterium prausnitzii*, methanogens in the gut).
- **Modules are enriched for phylogenetically or functionally related taxa** (e.g., do module members share metabolic pathways or occupy the same ecological niche?).
- **Known positive associations are recovered** (e.g., cross-feeding pairs such as *Bifidobacterium* and butyrate producers) and **known negative associations** (e.g., competition between *Bacteroides* and *Prevotella* enterotypes).
- **Edge overlap between methods** is quantified (e.g., Jaccard index of top-k edges between AdaCoNet and proportionality).

The field has moved well beyond reporting modularity as the sole quality metric. Recent methods papers (e.g., FlashWeave, CoNI, NetCoMi) include at least some form of biological ground-truthing.

**Suggestion**: For the Enterotype dataset, analyze the top-50 edges from each method for: (i) hub taxa identity and their known ecological roles; (ii) module composition at the phylum/family level with enrichment tests; (iii) recovery of well-documented pairwise associations. For MovingPictures, assess whether temporal modules correspond to taxa with known co-fluctuation patterns.

**Severity**: Major. Without biological validation, the real-data section cannot convince a microbial ecologist that AdaCoNet produces more useful networks -- only that it produces more modular ones.

---

### W3. MovingPictures modularity gap raises unresolved concerns

**Problem**: On the MovingPictures dataset, AdaCoNet achieves Q=0.163, substantially lower than proportionality (Q=0.438), SparCC (Q=0.382), and SPIEC-EASI (Q=0.415). The discussion attributes this to temporal dynamics favoring ratio-preservation over rank correlation, but this explanation is incomplete.

**Why this matters**: MovingPictures is a longitudinal dataset with repeated sampling, and many microbial network applications involve temporal or spatially structured data. If AdaCoNet systematically underperforms on longitudinal data -- the very setting where ecological interactions are most directly observable -- this is a significant practical limitation. The discussion notes that "a modified weight scheme that upweights proportionality could improve modularity," but does not implement or test this. Moreover, modularity alone is insufficient: the AdaCoNet network has 482 components (the most fragmented among non-degenerate methods), suggesting that the ensemble produces a diffuse network with many isolated small components rather than coherent ecological modules.

The paper should address: Is the low modularity a consequence of the theory weights assigning too much weight to Spearman on temporal data? Could the zero-fraction guard or the CCM diagnostics be extended to detect longitudinal structure and adjust weights accordingly?

**Suggestion**: (i) Report the actual weights assigned to each layer on MovingPictures. (ii) Test a simple modification: re-run AdaCoNet with manually increased proportionality weight on MovingPictures and report whether modularity improves. (iii) Discuss whether the CCM framework could incorporate a temporal correlation structure (e.g., AR(1) process) to handle longitudinal data natively.

**Severity**: Major. A method that underperforms on longitudinal microbiome data -- one of the most common and informative study designs -- will have limited adoption.

---

### W4. Only two real datasets, both human gut, one outdated

**Problem**: The real-data evaluation uses only the Enterotype dataset (Arumugam et al., 2011; N=280, P=550) and MovingPictures (Caporaso et al., 2011; N=1967, P=926). Both are human gut microbiome studies, and both are over a decade old.

**Why this matters**: The field standard for methods papers in this area is typically 3+ real datasets representing diverse body sites or environments. The Enterotype dataset, while historically important, has well-known limitations: it was assembled from multiple studies with different 16S primer sets and sequencing platforms, introducing batch effects that confound network inference. Neither dataset represents soil, marine, oral, skin, or built-environment microbiomes, where network inference is increasingly applied and where compositional properties (sparsity, diversity, sequencing depth) differ substantially from stool.

The CCM's phase transition depends on alpha_0/p and zero fraction f_0, both of which vary dramatically across environments. Soil microbiomes, for instance, are far more diverse (P >> 1000) and sparser than gut microbiomes. Marine microbiomes have different depth distributions. Without demonstrating that the theory-driven weights adapt correctly across these regimes, the generality claim remains unvalidated on real data.

**Suggestion**: Add at least one non-gut dataset. Good candidates include: (i) the Earth Microbiome Project (a natural fit given its scale and diversity), (ii) the HMP oral or skin sites (readily available, same 16S protocol), or (iii) any soil or marine dataset with N > 100 and P > 200. Report the CCM diagnostics (alpha_0/p, f_0) and predicted regime for each, and verify that the theory-driven weights produce sensible networks.

**Severity**: Major. Two gut-only datasets are insufficient to validate a framework that claims generality across "diverse data-generating mechanisms."

---

### W5. Accessibility of theoretical content for the target journal

**Problem**: The paper contains formal theorems (Theorems 1-3), proof sketches, and notation-heavy mathematical exposition (Dirichlet-Multinomial concentration, copula theory, variance inflation factors, phase transition thresholds). While mathematically sound, this level of formalism may be inaccessible to a substantial fraction of *Briefings in Bioinformatics* readers, many of whom are applied bioinformaticians or biologists.

**Why this matters**: *Briefings in Bioinformatics* publishes reviews and methods papers aimed at a broad audience. The "Briefings" in the title implies pedagogical value. A reader who encounters Theorem 2 (Phase Transition for Spearman-on-CLR) with its bound involving C_1, C_2, g(alpha_0, p), and the critical threshold c* needs substantial mathematical background to understand when and why the result applies. The Discussion section does a reasonable job of translating theory into intuition, but the gap between Section 2 (Methods) and Section 4 (Discussion) is large.

**Suggestion**: Add a "Practical guide" subsection or box (perhaps 1 paragraph in the Discussion) that translates the theory into actionable recommendations: "If your dataset has zero fraction < 0.5 and > 200 samples, AdaCoNet will likely produce the most informative network. If your zero fraction exceeds 0.5, consider using proportionality alone or AdaCoNet with Spearman downweighted." Include a decision flowchart (alpha_0/p vs f_0) that practitioners can follow without reading the theorems. This would substantially increase the paper's impact and accessibility.

**Severity**: Minor-to-Moderate. The theory is a strength, but its presentation should be calibrated to the journal's audience.

---

## Detailed Comments

### Introduction: Literature Coverage and Motivation

The motivation is well-articulated: the observation that no single method dominates across all data-generating regimes is correct and well-supported by the simulation results. The enumeration of methods in paragraph 2 of the Introduction is the most comprehensive I have seen in a microbial network methods paper, covering parametric (SparCC, CCLasso, REBACCA, gCoda), non-parametric (CoNI, FlashWeave), inverse covariance (SPIEC-EASI), and proportionality approaches.

However, the omissions are notable. Beyond the methods listed in W1, the paper does not discuss:
- **Correlation threshold selection strategies** beyond StARS: how do practitioners currently choose thresholds, and why is this problem hard for ensemble scores? The StARS degeneracy (discussed in Section 4, paragraph 8) is important but comes too late -- it should be foreshadowed in the Introduction.
- **Network comparison and differential network methods**: NetCoMi, DINGO, and related tools that compare networks across conditions are increasingly relevant. AdaCoNet produces a single network; how would one compare AdaCoNet-inferred networks between case and control groups?
- **The relationship to differential abundance analysis**: methods like ANCOM, ALDEx2, and DESeq2 address a related but distinct question. The relationship between co-occurrence network inference and differential abundance is underexplored in the literature and could be briefly acknowledged.

The framing of "ensemble approaches that combine multiple estimators" should explicitly acknowledge CoNet (Faust et al. 2012), which is the most widely used ensemble in microbial ecology and predates AdaCoNet by over a decade. Without this citation, readers may incorrectly conclude that AdaCoNet is the first ensemble approach for microbial networks.

### Results: Real-Data Interpretation

**Enterotype (Table 3, top panel)**: AdaCoNet's modularity of Q=0.510 is the highest among all methods, which is a strong result. However, I have concerns about interpretation:

1. The SparCC network has 12,797 edges at 8.5% density, compared to AdaCoNet's 5,917 edges at 5%. The difference in edge count makes modularity comparison less informative -- sparser networks tend to have higher modularity simply because inter-module edges are more easily pruned. The paper should report modularity at matched edge densities (which it partially does for MovingPictures, where all methods converge to 21,413 edges).

2. SPIEC-EASI and Graphical Lasso both returned degenerate complete graphs (150,975 edges, density 1.0). This is a known failure mode of regularization selection when the data violate model assumptions, but it is reported without analysis. What does this failure tell us about the Enterotype data's structure? Is this a warning about applying these methods naively?

3. The Enterotype dataset has known batch effects from combining multiple studies. Were these corrected before network inference? If not, the inferred network may capture study-level artefacts rather than biological associations.

**MovingPictures (Table 3, bottom panel)**: The convergence of all non-degenerate methods to exactly 21,413 edges is interesting and suggests that StARS is selecting the same density across methods. However, the key result -- AdaCoNet's low modularity of 0.163 -- is the paper's most problematic real-data finding. The explanation in the Discussion (temporal dynamics favor proportionality) is plausible but raises a deeper question: if the CCM's theory-driven weights do not account for temporal autocorrelation, then the framework may be systematically mis-specified for longitudinal data. This deserves more than a paragraph in the Discussion; it should be addressed empirically (see W3 suggestions).

The paper does not report the layer weights actually assigned to MovingPictures. This is a critical omission: the reader cannot evaluate whether the theory correctly diagnosed the data regime without seeing alpha_0/p, f_0, and the resulting w_S for this dataset.

### Discussion: Biological Implications

The Discussion is the strongest section of the paper. The interpretation of the cross-simulator inversion, the StARS degeneracy analysis, and the sensitivity analysis of c_ref are all thoughtful and honest. The acknowledgment of the v4 simulator's self-favouring bias (paragraph 6) is unusually candid for a methods paper.

However, the Discussion lacks any discussion of the **biological implications** of the results:

1. **What do the AdaCoNet networks tell us about microbial ecology?** The Enterotype network (Q=0.510) has 179 components. What do these components represent ecologically? Are they phylogenetically coherent? Do they correspond to known enterotype-associated guilds?

2. **How should a practitioner interpret edge signs?** The paper focuses on edge presence/absence, but microbial ecologists care about positive (mutualistic, cross-feeding) vs. negative (competitive, antagonistic) interactions. Does AdaCoNet preserve edge sign information? Are positive and negative edges evaluated separately?

3. **What is the relationship between statistical association and ecological interaction?** This is a foundational question in microbial network inference. Co-occurrence networks capture statistical patterns that may arise from shared environment, trophic interactions, or phylogenetic relatedness. The paper should briefly acknowledge this distinction and discuss whether the CCM framework has any implications for disentangling direct from indirect associations.

4. **The StARS degeneracy on ensemble scores** (discussed in the final substantive paragraph) is an important practical finding that deserves more prominence. If StARS -- the most widely used edge selection method in the field -- fails on AdaCoNet's output, this significantly limits the method's out-of-the-box usability. The recommendation to use fixed-density thresholds is pragmatic but ad hoc; the paper should discuss whether a modified stability selection procedure could be developed for ensemble scores.

---

## Questions for Authors

**Q1.** Can you report the actual CCM diagnostics (alpha_0/p, f_0, and resulting layer weights w_k) for both the Enterotype and MovingPictures datasets? Without these, the reader cannot assess whether the theory correctly diagnosed the data regime in the real-data setting. The MovingPictures case is particularly important: if the theory assigned high weight to Spearman (w_S ~ 0.3) but proportionality achieved much higher modularity, this suggests the CCM diagnostics may not capture all relevant aspects of the data-generating process.

**Q2.** How does AdaCoNet's performance change when the input includes phylogenetically structured noise? In real microbiome data, phylogenetically related taxa share similar abundances due to shared biology, not necessarily ecological interaction. Have you considered evaluating whether AdaCoNet edges are enriched for phylogenetically distant pairs (suggesting genuine ecological interaction) versus close pairs (suggesting phylogenetic signal)?

**Q3.** The paper treats network inference as a single-network problem, but many microbiome studies compare networks across conditions (e.g., healthy vs. disease). How would one use AdaCoNet for differential network analysis? Does the CCM framework extend naturally to comparing two or more networks, or would one need to run AdaCoNet independently on each condition and compare post-hoc?

**Q4.** The v4 simulator embeds direct covariance in the Dirichlet-Multinomial, which favors the DM layer (the Bayes estimator under the true model). You acknowledge this self-favouring bias in the Discussion. Have you considered a third simulation framework that is neutral to all four layers -- for instance, a neutral agent-based model or a simulation based on generalized Lotka-Volterra dynamics (e.g., the community simulator from Venturelli et al., *Molecular Systems Biology* 2018)? This would strengthen the claim of generality.

---

## Dimension Scores

| Dimension | Score (0-100) | Comments |
|---|---|---|
| **Novelty / Originality** | 78 | The CCM unification and Phase Transition Theorem are genuinely novel contributions. The ensemble idea itself is not new (CoNet, 2012), but the theoretical grounding is. |
| **Technical Soundness** | 82 | Theorems appear correct; proof sketches are reasonable; simulation design is rigorous with adequate seeds and configurations. The StARS degeneracy is honestly reported. Minor concern: finite-sample behavior of c* is acknowledged but not resolved. |
| **Literature Coverage** | 62 | Good coverage of recent parametric and non-parametric methods, but misses foundational ecological network tools (CoNet, MENAP, LSA). This is a significant gap for a methods paper targeting a bioinformatics audience. |
| **Experimental Design (Simulations)** | 85 | Two distinct simulators, 10 seeds, multiple N/P configurations, ablation study, top-k precision analysis. This is strong. The only gap is the absence of a third, ecologically motivated simulator. |
| **Real-Data Evaluation** | 48 | Only two datasets (both gut, both old), topology-only metrics, no biological validation, unresolved MovingPictures modularity gap. This is the weakest aspect of the paper and falls below field standards. |
| **Biological Relevance** | 45 | No assessment of keystone species, functional modules, known interactions, or ecological interpretability. The paper speaks to computational scientists but not to microbial ecologists who would use the tool. |
| **Practical Utility** | 70 | Fast runtime, Python implementation, and GitHub availability are positives. The StARS degeneracy and lack of practical guidance (decision rules, parameter recommendations) reduce out-of-the-box usability. |
| **Writing Quality** | 75 | Clear and well-structured overall. The theory section is dense but rigorous. The Discussion is strong. Accessibility for the target journal's broad readership could be improved with a practical guide or decision flowchart. |
| **Reproducibility** | 80 | Code and data availability stated; simulation parameters well-documented. Missing: actual layer weights for real-data experiments, exact preprocessing steps (filtering thresholds, batch correction). |
| **Significance for the Field** | 72 | The theoretical framework has potential to shift how researchers think about method selection for compositional networks. However, the gap between theory and practice (no biological validation, limited real-data demonstration) limits immediate impact. |

**Overall Score: 68/100**

---

## Summary of Required Revisions

To warrant acceptance in *Briefings in Bioinformatics*, the following revisions are essential:

1. **Expand the literature review** to include CoNet (Faust et al. 2012), MENAP (Deng et al. 2012), LSA (Biddle et al. 2008), and NetCoMi (Peschel et al. 2021), with explicit positioning of AdaCoNet relative to these tools.

2. **Add biological validation** to the real-data analysis: hub taxa identification, module enrichment, recovery of known pairwise associations, and edge overlap between methods.

3. **Add at least one non-gut real dataset** (e.g., HMP oral/skin, Earth Microbiome Project soil) to demonstrate generality across environments.

4. **Report CCM diagnostics and layer weights** for all real datasets, and address the MovingPictures modularity gap empirically (e.g., test modified weight schemes).

5. **Add a practical guide or decision flowchart** for practitioners that translates the theoretical results into actionable recommendations without requiring readers to parse the theorems.

These revisions are achievable without re-engineering the method itself and would substantially strengthen the paper's contribution to both the computational and ecological communities.
