# Static audit of the historical experimental repository

Date of the source audit: 2026-10-13. Repository: [Ensayos-Energy-Graph](https://github.com/cero1979/Ensayos-Energy-Graph). Inspected commit: [`186e27400f94d551432759fa4830e0c114a2aa69`](https://github.com/cero1979/Ensayos-Energy-Graph/tree/186e27400f94d551432759fa4830e0c114a2aa69).

This is the complete English translation of the historical static code audit, with portable provenance. Filenames such as `EXPARTFINAL.ipynb`, `README.md`, `requirements.txt`, and `table*.csv` below refer to that pinned historical repository. No historical notebook is bundled here; consult the pinned online source when a finding requires inspection of its original code. References to `manuscrito.tex` concern the original manuscript, not the revised English manuscript in this bundle. The audit can be used by any researcher or AI assistant and does not require access to the original author's machine or software session.

During the source audit, a snapshot was downloaded by a read-only clone into the logical historical location `revision_RAM/evidencia/Ensayos-Energy-Graph_snapshot`. This is a provenance location, not a required local directory in this handoff. No cell was executed, no training module was imported, no dataset was opened, and no historical notebook, CSV, figure, or checkpoint was modified. Cell JSON was analyzed as text using the standard library, and sources/CSV files were read. The snapshot's `git status --porcelain` was empty after inspection. A defect in a code path does not establish that the path was activated in every historical run.

SHA-256 values of consulted files:

| Historical file | SHA-256 |
|---|---|
| EXPARTFINAL.ipynb | `2e072d70a135cf372d2bc49d146aac5605907b0bcc770d084d9daa0eed1e9b20` |
| README.md | `a5c946a601bbe920a9540d501ffacc7ac3706d678bc4221dc6f3d6f4e6b4d49a` |
| requirements.txt | `58a795426d337fa3fe1d3f9a49f4a08660e6a7d33791c8af98a59cd04a73ae40` |

Location convention: **cell C = the index in the `cells` array plus one, counting both Markdown and code cells**. Line L = index plus one within `''.join(cell['source']).splitlines()`. It is neither the number in a “Cell ...” heading nor `execution_count`. There are 35 cells. The notebook contains some old outputs, but many decisive cells have no outputs, including cells 29 and 31. It cannot, by itself, reconstruct the provenance of all CSV files.

The original LaTeX file consulted was logically `historical-manuscript/PaperEnergyElsevier/manuscrito.tex`. Its line references identify the preserved original, not the corrected manuscript. The mathematical audit's M coordinates use this same original. These historical names preserve evidence coordinates; they are not missing local handoff documents. The revised English manuscript and proofs are available under `manuscript/manuscrito_RAM.pdf`, `manuscript/manuscrito_RAM.tex`, and `manuscript/sections/`.

## Verified findings

“Verified” in this section means observed by static inspection of the cited code or files. It does not mean that a defect was observed executing in a historical training run, that every historical result was reproduced, or that the proposed new implementation exists.

| ID | Location | Observed fact and diagnosis | Consequence / required action |
|---|---|---|---|
| C01 | C3:L17 | `warnings.filterwarnings('ignore')` globally suppresses warnings. | Remove it in new code; record and classify warnings, without asserting that historical warnings can be recovered. |
| C02 | C5:L8–20 | `_safe_svdvals` adds random noise after SVD errors and ultimately returns a zero vector. | If activated, this can report false energy/certification. Record the original error and each attempt; never substitute zeros. Historical activation is unknown. |
| C03 | C5:L2–6 | `_safe_svd` retries the SVD of W plus noise of scale 1e-6. | That SVD no longer corresponds exactly to W; the computed projection is not the exact projection of the original input. Measure and record the error, or fail. |
| C04 | C11:L47–55 | The differentiable SVD for NUC also adds noise and returns zeros after exhausting retries. | This can disable the penalty/gradient without a clear signal. E0 must inject the failure and require a traceable error. |
| C05 | C5:L26–29,46–57; C11:L75–92 | Energy and spectra iterate over `named_parameters`; SN is registered using `torch.nn.utils.parametrizations.spectral_norm`. | With SN, these tensors are the original parameters, not the effective weights used in the forward pass. Historical SN energy and sigma_max do not measure the weights whose accuracy is reported; recompute from an effective-weight checkpoint if one exists. |
| C06 | C31:L53–64 | To normalize SN effective rank, the code looks for parametrized names in a new MLP without SN; if no match is found, it does not initialize/recompute `mr`. | It can reuse `mr` from a previous layer/method. The usual flow retains 10 from the last NUC layer. Do not treat SN values in table7 as a verified derivation from the code; require dimensions obtained directly from the module. |
| C07 | C9:L5–21 | The object called `val_loader` uses `train=False`, meaning the official test split. | Validation and test are not separated. The split is repeatedly used for evaluation and narrative selection; do not present this accuracy as a held-out test result. |
| C08 | C9:L17–18,30–33; C17:L7–8; C29:L13–14 | Subsets are sampled with the global RNG for each seed and regenerated, without index files. | The seed mixes initialization variability with subset variability. Freeze indices/hashes and use separate generators. |
| C09 | C11:L169–175 | rho changes with clipping to [.1,10], but U is always divided/multiplied by 2. | This breaks y=rho*u when clipping makes the actual change differ from a factor of 2. Examples: 8→10 and .1→.1. Use the actual rho_old/rho_new ratio. |
| C10 | C11:L101–114 | Z is created at initialization and is not refreshed after warmup; the transition tau is recalculated from current energy. | The transition is not interpolation between two frozen endpoints. It is a different variant from the usual description. The new protocol fixes Z=W3, u=0, and tau0 once at the warmup transition. |
| C11 | C11:L116–160 | The W update uses one epoch of SGD with momentum; Z is projected at the end of the epoch. | This is not an exact ADMM argmin. Do not directly apply exact ADMM/GD theorems; declare the frequency and variant. |
| C12 | C5:L94–100; C11:L147–152 | Bisection returns the midpoint of the interval, not its feasible endpoint; its stopping rule uses width 1e-5. | It can leave a small excess in the singular-value sum. Exact feasibility does not follow from this stopping rule. Measure the output and use a scaled tolerance. |
| C13 | C19:L36,43; C29:L43,52,61 | `satisfies = energy <= tau*1.01`. | The historical label allows 1% excess. It is neither exact Omega<=tau nor a tolerance explicitly stated in the article; preserve raw violation. |
| C14 | C19:L58–61; C31:L18–21 | The feasibility table omits STD and WD. | This biases the comparative interpretation. Aggregated WD values suggest it might satisfy .8·STD, but without raw pairing, do not certify counts; report all methods. |
| C15 | C11:L177–180 | The history contains W energy and current violation, but not Omega(Z), current and final tau, or separate W/Z/P(W) evaluations. | CSV files alone cannot audit the final budget during transition or the behavior of Z. Save every field in the new suite. |
| C16 | C13:L48–50; C15:L31–35 | The smallest positive Hessian eigenvalue is taken as mu/SC, ignoring zero and negative eigenvalues; Synthetic uses fallback mu=1e-6. | This proves neither SC nor a QG region. The interpretation that MLP hypotheses were validated is invalidated; replace these trials with problems having an analytic optimum/constant. |
| C17 | C13:L64; C15:L49 | `delta_L=max(L_k-loss_star,1e-12)`. | This hides negative gaps relative to the L-BFGS candidate. Save the original gap and investigate the reference; do not clip. |
| C18 | C13:L66–68; C15:L51–53 | R is computed by counting singular values >1e-6. | Numerical rank can underestimate the theorem's constant. Use analytic rank or conservative D. |
| C19 | C15:L125–126 | The Synthetic table writes `SC met=50/50` and `All<1=True` without a specific trial-by-trial check of those properties. | These counts do not certify the hypotheses; do not reuse them as proof of a theorem. |
| C20 | C17:L28–33 | The code fits the log of positive violations using a compressed index, and prints “converged” if there are fewer than 3. | Removing gaps/zeros changes the time axis; this is not a valid diagnostic of asymptotic convergence. |
| C21 | C17:L40; C15:L105 | Zeros are changed only for log plotting; the MNIST ratio figure imposes ylim 0–1.5 although the CSV reports 4001.0547. | Source data must preserve the values; the figure can hide the magnitude of the violation. Do not call a ratio >1 a “vacuous bound.” |
| C22 | C17:L6–14; C19:L9–39; C23:L7–18; C25:L7–14; C29:L12–57 | STD/ADMM/WD are trained again for configurations and seeds already used. | These repeat configurations rather than providing additional independent evidence. “20 independent runs” is unsupported: the 5 configurations in historical E2 are included among the 15 in historical E6. |
| C23 | C19:L27–32; C29:L34–39 | Every NUC candidate is trained, then the winner is trained again; there is no configuration cache. | This creates hidden cost and avoidable duplication. Save/reuse candidate checkpoints and count calibration cost. |
| C24 | C19:L9; C25:L7; C29:L12 | Historical Experiment 3 uses three seeds; spectra in historical E5 use one; the historical E6 extension uses five. | Do not attribute five seeds to every table/figure without mapping its origin. |
| C25 | C19:L63–65; C31:L25–27 | `np.std` uses default ddof=0, and results are rounded before being written to CSV. | This is not sample SD or a confidence interval. Original statistics cannot be recovered from rounded aggregates alone. |
| C26 | C9:L2–6 | Fashion-MNIST would use MNIST normalization (.1307,.3081); the main execution loads MNIST. | Loader support is not evidence that a Fashion-MNIST experiment was run. New normalization is computed only on the training split of each dataset. |
| C27 | C3:L23; requirements.txt | CPU is forced; dependencies have only minimum-version bounds, without a lockfile. | This does not establish Linux/CUDA reproducibility. Record and pin the new environment after compatibility checks. |

The source audit consulted the official [PyTorch 2.8 SN documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.parametrizations.spectral_norm.html), which confirms implementation by parametrization and estimation of sigma by power iteration, rather than exact clipping of every singular value. Its example even shows an effective norm slightly greater than 1. C05 rests on the code's measurement path and those semantics; it is not an inference from poor accuracy.

Historical experiment numbers in code-provenance statements (especially C22 and C24) refer to the old notebook's experiments. Action items mentioning the new E0–E6 refer to the redesigned suite in `EXPERIMENTAL_PROTOCOL.md`; the two numbering systems must not be conflated.

## Unconfirmed suspicions and correct aspects

- **Unconfirmed: moving NUC to CPU breaks the gradient.** The current historical penalty at C11:L66–69 uses `_safe_svdvals_grad(p...)` and `.backward()` on tensors, without `.cpu()`, `detach`, NumPy, or a float conversion in the nominal path. Do not attribute a gradient failure to that nonexistent transfer. The C04 fallbacks do require correction. E0 will check autograd in the new implementation.
- **The main scaling is correct in the nominal path:** C11:L134 uses `half_tau=cur_tau/2`; the individual allocation at C11:L159 also uses tau/2. Do not claim that this particular factor is wrong in every path; the new suite tests it to prevent regressions.
- **The nominal joint projection is conceptually correct:** It implicitly concatenates spectra and uses one threshold. The objection concerns tolerance, fallback behavior, and logging, not the imposition of independent layer radii in the `joint=True` path.
- **Whether SVD failure actually occurred:** No record was found quantifying how often zero/noise fallbacks were activated; do not claim that every historical low-energy result arose from this defect.
- **Historical checkpoints:** The inspected versioned tree contains no model checkpoints, ADMM state files, or per-run manifests. Files may exist outside the repository; request them from the collaborator if historical numbers are to be recovered. Their existence is not presumed.
- **Effective GPU determinism, OOM behavior, and autograd on the collaborator's machine:** Pending; the code was not run on that machine, and its hardware is unknown.
- **Exact zeroing under individual allocation:** Not demonstrated by the code/table. With a nonzero input and positive radius, an exact individual projection does not produce zero; low accuracy does not identify the causal mechanism.

## Architectures and historical results: traceability

The definitions at C7:L1–40 match original manuscript L876–878: 3L=784–256–128–10; 6L=784–256–128–64–64–32–10; 9L=784–256–128–128–64–64–32–32–16–10. Purely arithmetic weight-plus-bias counts confirm 235146, 248682, and 266618, respectively; no models were instantiated or trained to check them.

| Historical source | Recorded content | Reuse classification |
|---|---|---|
| table1_holder_mc.csv / LaTeX L947–969 | 50 Synthetic trials, 10 binary trials; maximum binary ratio 4001.0547. | Historical data preserved; the interpretation “SC validated/vacuous bound” is invalidated by C16–19. |
| table2_admm.csv / LaTeX L980–994 | Five rows; rounded tau, energy, residuals, and accuracy. | Historical observation pending weights/unrounded configuration. It proves neither convergence nor mathematically exact feasibility. |
| table3_comparison.csv | Three seeds; ADMM 9L 86.88±5.74. | A different table from the five-seed extension; do not merge it with table6. |
| table6_sn_comparison.csv / LaTeX L1021–1041 | Five seeds; ADMM 9L 88.21±5.43. | Historical accuracies pending reproducibility; SN metrics do not correspond to effective weights because of C05. |
| table7_sn_spectral.csv | Mean SN sigma values 1.59, 1.92, 2.15; ratios .921, .839, .811. | The interpretation of effective SN control is invalidated. The current historical code also has C06; exact provenance remains pending. |
| table4_projection.csv / LaTeX L1094–1096 | One-seed ablation; 3L split 69.05, joint 95.30, WD 96.20. | A historical observation, not a law of depth or proof of zeroing. |
| README, Experiment 4 | Reports 3L split 94.10, joint 95.90, WD 96.00. | Verified discrepancy against CSV/text; do not select whichever number is more favorable. |
| README, Experiment 1 | Describes 200 trials. | The code would run 50+10; the CSV matches 50+10. Correct the documentation before reproducing. |
| README, figure structure | Lists fig8–fig11; cell C33 writes them. | The versioned tree does not contain those figures. Do not claim the historical package supplies everything advertised. |
| LaTeX L1110 | ADMM 9L 86.9±5.7 within the projection discussion. | A figure from the three-seed table3, although the central table uses five-seed table6; identify the denominator/origin. |
| LaTeX L1148,1202 | SN “redistributes/flattens” proportions through rescaling. | A mathematically false interpretation for positive uniform rescaling; remove it regardless of retraining. |
| LaTeX L1177,1223 | “20 independent training runs.” | The 3L configurations repeat; do not use them as 20 independent replicates. |

Historical execution was not verified, and no new accuracies were calculated. A number preserved in a CSV is insufficient to reconstruct its configuration, effective weights, per-seed budget, and failures. New modules must generate unrounded files with complete provenance. The README asserts an approximate duration, but **it was not measured in this audit** and cannot be used to budget CUDA runtime.

## Record of read-only actions

- Remote `git ls-remote` access and clone of the cited commit: **Existing and verified** in the source audit; only a snapshot was downloaded.
- Inspection of JSON/CSV/source files and SHA-256 hashing with the Python standard library: **Existing and verified**; no code contained in the notebook was executed.
- Dimension/parameter counts from arithmetic on width lists: **Existing and verified**; the values are reported above.
- `Run All Cells`, training, import of experimental functions, and dependency installation: **Not performed**.
- All commands for the new experimental workflow: **Interface pending implementation by the collaborator**. Their description in the protocol is not evidence that a runnable suite or fresh experiment exists.

An additional read-only provenance check during preparation of this English handoff fetched the three pinned files from GitHub as data, recomputed their SHA-256 values, and confirmed all three values above and the 35-cell count. The cited SVD fallback, rho update, and stale-`mr` source paths were inspected as text, and the official PyTorch 2.8 documentation was rechecked. This check neither executed the notebook nor reproduced historical training; it does not upgrade pending evidence to verified execution.

Implementation priority is E0 (projections, NUC gradient, effective SN weights, dual updates, and failure behavior), E1 with analytic hypotheses, a cost pilot, a frozen manifest, and then E2–E6. Preserve these findings even if new results favor ADMM. Use the local `MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md` for mathematical contracts, `EXPERIMENTAL_PROTOCOL.md` for the redesigned suite, `CLAIMS_AND_EVIDENCE_MATRIX.md` for evidence status, and `AI_ASSISTANT_PROMPT.md` for generic implementation instructions. Mathematical proofs and counterexamples are in `MATHEMATICAL_AUDIT.md` and the bundled revised manuscript; no separate historical proof-note file is required.
