# References Audit — FYP_v12_fable/references.bib

**Date:** 2026-06-10
**Method:** Every entry checked against a primary source (arXiv abstract page, IEEE/ACM/PMLR/AAAI/NeurIPS/JMLR/Annual Reviews/Springer listing, dblp record, or the ISO catalogue). No metadata was retained on trust.
**Output:** `references_verified.bib` (drop-in replacement; all BibTeX keys unchanged).

## Verdict summary

| Verdict | Count |
|---|---|
| verified-correct | 29 |
| corrected | 11 |
| replaced (wrong authors/title/venue) | 6 |
| UNVERIFIABLE | 0 |
| NEW entries added | 6 |
| **Total entries in new bib** | **52** |

The six *replaced* entries had fabricated or placeholder metadata; in every case the real paper was found via its arXiv ID, so **no entry needs to be dropped** (including `cbfrl2024`, whose old author list was invented but which maps to a real 2025 paper).

## Per-entry audit

| Key | Verdict | What changed | Source URL used |
|---|---|---|---|
| `iso15066` | verified-correct | Nothing. ISO/TS 15066:2016 "Robots and robotic devices — Collaborative robots", standard 62996. | https://www.iso.org/standard/62996.html |
| `iso10218` | verified-correct | Added catalogue URL (standard 73933). Title, year 2025, and "integrates TS 15066" note all confirmed (3rd edition incorporates collaborative-operation requirements). | https://www.iso.org/standard/73933.html ; https://www.automate.org/robotics/blogs/updated-iso-10218-faq |
| `iso25785` | corrected | Designation updated WD→CD (stage 30.20 effective 2026-05-12); full official title restored incl. "(legged, wheeled, or other forms of locomotion)"; year→2026; catalogue URL added. | https://www.iso.org/standard/91469.html ; https://iss.rs/en/project/show/iso:proj:91469 |
| `svarny2019unified` | corrected | Was cited as arXiv preprint; actually published at IROS 2019, pp. 7574–7581. Converted to @inproceedings, arXiv ID kept in note. Authors/title confirmed. | https://arxiv.org/abs/1908.03046 |
| `svarny2020collision` | corrected | Was @article with `journal=ICRA`; fixed to @inproceedings, ICRA 2021, pp. 3829–3835 added. Authors/title confirmed (arXiv:2009.01036). | https://arxiv.org/abs/2009.01036 |
| `marvel2017implementing` | **NEW** (verified) | Added: Marvel & Norcross, RCIM vol. 44, pp. 144–155, 2017. DOI 10.1016/j.rcim.2016.08.001. Canonical ISO-15066 SSM implementation reference. | https://www.sciencedirect.com/science/article/abs/pii/S0736584516302617 |
| `altman1999constrained` | verified-correct | Nothing. Chapman & Hall/CRC, 1999. | https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780849303821 |
| `achiam2017cpo` | verified-correct | Nothing. ICML 2017, pp. 22–31 (PMLR 70). | https://dblp.org (publ API: "Constrained Policy Optimization") |
| `ray2019safetygym` | verified-correct | Cosmetic: `publisher`→`howpublished`; URL pointed at the canonical report PDF instead of the GitHub repo. Authors/title/year confirmed. | https://openai.com/index/benchmarking-safe-exploration-in-deep-reinforcement-learning/ ; https://cdn.openai.com/safexp-short.pdf |
| `stooke2020pid` | verified-correct | Nothing. ICML 2020, pp. 9133–9143. | https://dblp.org (publ API) |
| `yang2021wcsac` | verified-correct | Nothing. AAAI 2021, 35(12):10639–10646 — exact match. | https://ojs.aaai.org/index.php/AAAI/article/view/17272 |
| `kumar2020cql` | verified-correct | Nothing. NeurIPS 2020, vol. 33, pp. 1179–1191. | https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html |
| `bellemare2017c51` | verified-correct | Nothing. ICML 2017, pp. 449–458. | https://dblp.org (publ API) |
| `dabney2018qrdqn` | verified-correct | Nothing. AAAI 2018, 32(1). DOI 10.1609/aaai.v32i1.11791. | https://ojs.aaai.org/index.php/AAAI/article/view/11791 |
| `thananjeyan2021recovery` | verified-correct | Nothing. RA-L 6(3):4915–4922, 2021. | https://dblp.org (publ API: "Recovery RL") |
| `srinivasan2020learning` | verified-correct | Nothing. arXiv:2010.14603; never published at a venue ("in submission" on arXiv), so arXiv-preprint form is right. | https://arxiv.org/abs/2010.14603 |
| `brunke2022safelearning` | verified-correct | Nothing. Annu. Rev. Control Robot. Auton. Syst. 5:411–444, 2022. | https://www.annualreviews.org/content/journals/10.1146/annurev-control-042920-020211 |
| `wachi2024constraint` | **NEW** (verified) | Added: Wachi, Shen, Sui, "A Survey of Constraint Formulations in Safe Reinforcement Learning", IJCAI 2024 Survey Track, pp. 8262–8271, DOI 10.24963/ijcai.2024/913. Used for current-state safe-RL constraint-formulation claims. | https://www.ijcai.org/proceedings/2024/913 |
| `garcia2015comprehensive` | corrected | Dropped spurious `number={1}` (dblp/JMLR cite as 16:1437–1480). Everything else confirmed. | https://www.semanticscholar.org/paper/c0f2c4104ef6e36bb67022001179887e6600d24d |
| `ames2019cbf` | verified-correct | Nothing. ECC 2019, pp. 3420–3431. | https://researchr.org/publication/AmesCENST19 |
| `bansal2017hamilton` | **NEW** (verified) | Added: Bansal, Chen, Herbert, Tomlin, CDC 2017, pp. 2242–2253. HJ-reachability positioning reference. | https://dl.acm.org/doi/10.1109/CDC.2017.8263977 |
| `alshiekh2018shielding` | **NEW** (verified) | Added: Alshiekh, Bloem, Ehlers, Könighofer, Niekum, Topcu, AAAI 2018, 32(1):2669–2678. DOI 10.1609/aaai.v32i1.11797. Canonical shielding reference. | https://ojs.aaai.org/index.php/AAAI/article/view/11797 |
| `shield2025` | **replaced** | Author list was fabricated ("Choi, Castillo, Sreenath, Ames"). Real authors: **Yang, Werner, Cosner, Fridovich-Keil, Culbertson, Ames** (Caltech/UT Austin/Cornell — matches the report's description; Unitree G1 demo). Title casing confirmed ("SHIELD: Safety on Humanoids via CBFs In Expectation on Learned Dynamics"); venue upgraded from arXiv preprint to **IROS 2025** (accepted, conference held Oct 2025). | https://arxiv.org/abs/2505.11494 |
| `cbfrl2024` | **replaced** | Old entry was fabricated ("Anand, Mayank and Seetharaman, Karthik and Hardik", RSS 2025). Real paper found: **"CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions", Lizhi Yang, Blake Werner, Massimiliano de Sa, Aaron D. Ames, arXiv:2510.14959 (2025), preprint** (no verified venue). Key kept for the .tex. Do **not** drop — the real paper matches the report's usage (CBF filtering during RL training). | https://arxiv.org/abs/2510.14959 |
| `chernyadev2024bigym` | verified-correct | Pages 4201–4217 added (CoRL 2024 = PMLR v270); booktitle expanded to "Proceedings of the 8th Conference on Robot Learning (CoRL)". Authors/title/year confirmed. | https://proceedings.mlr.press/v270/chernyadev25a.html |
| `sferrazza2024humanoidbench` | verified-correct | Nothing. RSS 2024 confirmed (arXiv:2403.10506; Semantic Scholar venue record). | https://arxiv.org/abs/2403.10506 ; https://www.semanticscholar.org/paper/75850000ac9e056ce604c1bcd7631a3ae73d0458 |
| `james2020rlbench` | corrected | Was @inproceedings with `booktitle=IEEE Robotics and Automation Letters`; fixed to **@article**, journal RA-L 5(2):3019–3026, 2020. Official title uses "&": "The Robot Learning Benchmark \& Learning Environment". | https://ui.adsabs.harvard.edu/abs/2020IRAL....5.3019J/abstract (DOI 10.1109/LRA.2020.2974707) |
| `yuan2022safecontrolgym` | corrected | Title was truncated — official title ends "…and Reinforcement Learning **in Robotics**". Vol/issue/pages confirmed (7(4):11142–11149). | https://arxiv.org/abs/2109.06325 ; DOI 10.1109/LRA.2022.3196132 |
| `ji2023safetygymnasium` | verified-correct | Volume 36 + "Datasets and Benchmarks Track" added. Authors (incl. Juntao Dai) confirmed. Title rendered "Safety Gymnasium" per NeurIPS camera-ready (arXiv hyphenates it). | https://openreview.net/forum?id=WZmlxIuIGR ; https://arxiv.org/abs/2310.12567 |
| `seo2024cqnas` | corrected | Author spelling fixed ("Uruc"→"Uruç"); pages 2866–2894 added (PMLR v270); booktitle expanded. This is the **CQN** paper (the CQN-AS paper is the new `seo2025actionsequence`). | https://proceedings.mlr.press/v270/seo25a.html ; https://arxiv.org/abs/2407.07787 |
| `seo2025actionsequence` | **NEW** (verified) | Added: **CQN-AS** — Seo & Abbeel, "Coarse-to-fine Q-Network with Action Sequence for Data-Efficient Reinforcement Learning", **NeurIPS 2025** (poster), arXiv:2411.12155. (Early arXiv versions used the title "…for Data-Efficient Robot Learning".) | https://arxiv.org/abs/2411.12155 ; https://openreview.net/forum?id=VoFXUNc9Zh |
| `zhao2023act` | verified-correct | Nothing. RSS 2023 (Daegu), ALOHA/ACT paper. | https://www.roboticsproceedings.org/rss19/p016.pdf |
| `chi2023diffusion` | verified-correct | Nothing. RSS 2023; the 7-author list matches the conference version (Tedrake was only added on the IJRR 2024/25 journal version, which reorders authors). | https://arxiv.org/abs/2303.04137 ; https://github.com/real-stanford/diffusion_policy |
| `yarats2021drqv2` | corrected | Was arXiv preprint; published at **ICLR 2022** (poster). Converted to @inproceedings, year→2022, arXiv ID kept in note. | https://iclr.cc/virtual/2022/poster/6275 ; https://dblp.org/rec/conf/iclr/YaratsFLP22.html |
| `narvekar2020curriculum` | verified-correct | Nothing. JMLR 21(181):1–50, 2020. | https://jmlr.org/papers/v21/20-212.html |
| `openai2019solving` | verified-correct | Nothing. arXiv:1910.07113; first nine named authors match the arXiv listing, "and others" covers the rest. | https://arxiv.org/abs/1910.07113 |
| `henning2022bodyslam` | verified-correct | Nothing. ECCV 2022, pp. 656–673 (DOI 10.1007/978-3-031-19842-7_38). | https://link.springer.com/chapter/10.1007/978-3-031-19842-7_38 |
| `henning2023bodyslampp` | verified-correct | Nothing. IROS 2023, pp. 3781–3788 (DOI 10.1109/IROS55552.2023.10342291). | https://arxiv.org/abs/2309.01236 ; TUM FIS record |
| `romero2017smplh` | corrected | Was @misc with journal info stuffed into `publisher`; fixed to **@article**: ACM TOG (Proc. SIGGRAPH Asia) 36(6), Article 245, pp. 245:1–245:17, Nov 2017. | https://arxiv.org/abs/2201.02610 ; ACM TOG 36(6) Article 245 record |
| `mahmood2019amass` | verified-correct | Nothing. ICCV 2019, pp. 5442–5451. | https://dblp.org/rec/conf/iccv/MahmoodGTPB19.html ; https://arxiv.org/abs/1904.03278 |
| `unitree_g1` | verified-correct | Note refined to match the live spec page: ~35 kg with battery, height 1320 mm, **23–43 joint motors depending on configuration** (was "up to 43-DOF" — same claim, now phrased per the page). URL live. | https://www.unitree.com/g1 |
| `unitree_h1` | verified-correct | Nothing. Product page live; H1 launched 2023. | https://www.unitree.com/h1 |
| `auditing2021` | **replaced** | Author list was wrong ("Chinchali, Krishnan, Pavone, Katti"). Real author: **Homanga Bharadhwaj** (sole author). Title/year confirmed; note added: Blue Sky paper at CoRL 2021. | https://arxiv.org/abs/2110.05702 |
| `tobin2017domainrand` | verified-correct | Nothing. IROS 2017, pp. 23–30. | https://dl.acm.org/doi/10.1109/IROS.2017.8202133 |
| `zhao2020sim2real` | corrected | Was @article with `journal=SSCI`; fixed to **@inproceedings** (IEEE SSCI 2020, pp. 737–744). | https://www.semanticscholar.org/paper/5a1b92aa50797a7c1e99b8840ff01aad66038596 |
| `todorov2012mujoco` | verified-correct | Nothing. IROS 2012, pp. 5026–5033. | https://www.semanticscholar.org/paper/b354ee518bfc1ac0d8ac447eece9edb69e92eae1 |
| `trautman2010freezing` | **NEW** (verified) | Added: Trautman & Krause, IROS 2010, pp. 797–803 — canonical freezing-robot-problem citation. | https://www.semanticscholar.org/paper/0ae705501edfa94f610f10424faa7d4e7615a1f8 ; https://las.inf.ethz.ch/files/trautman10unfreezing.pdf |
| `tian2022confidence` | verified-correct | Nothing. Title exact; authors Tian, Sun, Bajcsy, Tomizuka, Dragan; published ICRA 2022 (DOI 10.1109/ICRA46639.2022.9812048). | https://arxiv.org/abs/2109.14700 ; https://dl.acm.org/doi/10.1109/ICRA46639.2022.9812048 |
| `permissivefilter2025` | **replaced** | Placeholder authors and paraphrased title replaced. Real metadata: **"Provably Optimal Reinforcement Learning under Safety Filtering", Donggeon David Oh, Duy P. Nguyen, Haimin Hu, Jaime F. Fisac**, arXiv:2510.18082 (cs.LG); accepted at IASEAI 2026. | https://arxiv.org/abs/2510.18082 |
| `nakamura2025latent` | corrected | Title hyphenation fixed to arXiv form: "…Beyond **Collision-Avoidance** via Latent-Space Reachability Analysis". Authors and RSS 2025 venue confirmed. | https://arxiv.org/abs/2502.00935 |
| `latentcbf2025` | **replaced** | Placeholder authors and paraphrased title replaced. Real metadata: **"How to Train Your Latent Control Barrier Function: Smooth Safety Filtering Under Hard-to-Model Constraints", Kensuke Nakamura, Arun L. Bishop, Steven Man, Aaron M. Johnson, Zachary Manchester, Andrea Bajcsy**, arXiv:2511.18606 (cs.RO), preprint. | https://arxiv.org/abs/2511.18606 |
| `dontfreeze2026` | **replaced** | Placeholder authors and paraphrased title replaced. Real metadata: **"Don't Freeze, Don't Crash: Extending the Safe Operating Range of Neural Navigation in Dense Crowds", Jiefu Zhang, Yang Xu, Vaneet Aggarwal**, arXiv:2603.06729 (cs.LG), preprint. | https://arxiv.org/abs/2603.06729 |

## Notes for the .tex

1. **No keys changed**, so no `\cite{}` edits are needed. To adopt: `cp references_verified.bib references.bib` (or point the `\bibliography{}` at the new file).
2. If the report text describes `shield2025` or `cbfrl2024` by their (previously fabricated) author names anywhere in prose, those sentences need updating — both are now Ames-lab papers (Lizhi Yang first author on both).
3. `auditing2021` is single-authored (Bharadhwaj); any prose saying "Chinchali et al." must change.
4. `iso25785` is now a **committee draft** (stage 30.20, May 2026), no longer a working draft — if the prose says "working draft", consider "under development (committee draft as of May 2026)".
5. `seo2024cqnas` = CQN (CoRL 2024); `seo2025actionsequence` = CQN-AS (NeurIPS 2025). Cite the latter for the action-sequence backbone.
6. Three entries remain arXiv-preprint-only by design (`srinivasan2020learning`, `latentcbf2025`, `dontfreeze2026`, plus `cbfrl2024`): no published venue exists for them as of 2026-06-10.

---

# Addendum: v16 audit (2026-06-12)

**Scope:** `FYP_v16_fable/references.bib` (was 87 entries; now 62 = exactly the keys cited in `main.tex`).
**Method:** the 2026-06-10 v12 audit covered 50 of the 87; the remaining 37 were either verified now against primary sources (Crossref/doi.org, IEEE Xplore, dblp, official proceedings/OpenReview, arXiv abs, publisher pages) or removed as uncited. DOI fields back-filled across the whole file.

## Removed (25 entries, uncited in v16 main.tex)
achiam2017cpo, auditing2021, bena2025poisson, dontfreeze2026, ganai2024hjsurvey,
garcia2015comprehensive, he2023autocost, iso25785, joseph2026mujoco, kim2022offtrc,
kim2023sdac, konighofer2025shields, latentcbf2025, marvel2017implementing,
morton2025oscbf, muratore2022randomized, nakamura2025latent, prudencio2024offlinesurvey,
rudin2022walkminutes, shin2024wham, sun2025spark, wabersich2023datadriven, xu2022cpq,
yang2023wcsacjournal, zhuang2024parkour
(All still recoverable from git history / `references_verified.bib` if prose re-cites them.)

## Key corrections found in this pass
- `svarny2019unified` — **pages were wrong** (7574–7581 → **7580–7587**; IEEE/Crossref + dblp agree for DOI 10.1109/IROS40897.2019.8968463).
- `shield2025` — **now published**: IROS 2025 proceedings on Xplore, pp. 203–210, DOI 10.1109/IROS60139.2025.11247065 (was "accepted").
- `tian2022confidence` — ICRA 2022 pages 11229–11235 + DOI added.
- `cbfrl2024` — **accepted at IEEE ICRA 2026** (held June 2026); no Xplore record yet → kept as arXiv with 10.48550 DOI + acceptance note. Re-check ~July/Aug 2026 for the proceedings DOI.
- `permissivefilter2025` — IASEAI'26 held Feb 2026; still no archival DOI → arXiv form kept, note updated.
- `puig2024habitat3` — expanded "and others" to the full 23-author ICLR 2024 list; OpenReview URL added.
- `zhang2025safevla` — NeurIPS 2025 (Spotlight) confirmed, vol. 38; OpenReview URL.
- `tobin2017domainrand`, `unitree_g1` — drift vs the v12 audit reconciled (DOI added; access date added).
- All 10 previously-unaudited 2025/2026 entries (hartmann2026iso, koczi2025humanoidsafety, ieee2025pathway, hundt2025llmrobots, cai2026humanoidcbf, …) verified to exist; none fabricated.

## DOI / URL coverage policy (now complete: 62/62)
- 44 entries carry publisher DOIs (incl. RSS 10.15607, AAAI 10.1609, arXiv-only works as 10.48550/arXiv.\<id\>).
- 18 carry official URLs where the venue issues no DOI: PMLR/CoRL, NeurIPS, ICLR, JMLR, TMLR (OpenReview/proceedings pages), ISO catalogue pages, Unitree product pages, OpenAI reports.

## Style
`\bibliographystyle{vancouver}` (already set in main.tex) + `vancouver.bst` (v1.0, Folkert van der Beek, NLM/ICMJE implementation) now vendored in this directory so the build does not depend on the TeX installation; it formats doi/url/eprint fields. natbib `[numbers,sort&compress,square]` is compatible.
