# Experimental protocol for the RAM revision

Protocol RAM-2026-10-13-v1. This document assigns future work to the colleague who will implement and run the experiments on Linux/CUDA. The current revision comprised a static inspection of the repository and small mathematical checks; it did not run the experimental suite or correct the historical notebook. The old results are historical evidence whose provenance remains to be established, not results of the new protocol. The historical repository is available online; its location and audited commit are documented in `HISTORICAL_CODE_AUDIT.md`. Its notebook is not a bundled implementation of this protocol. The supplied small checks are in `mathematical_checks/verify_mathematics.py`, with `recorded_results.json` and `recorded_execution.log` in the same directory; they are not a new training package.

The scientific objective is to evaluate the tradeoff among loss, accuracy, nuclear-norm budget, and cost, and to check feasibility diagnostics. Validity depends on following the protocol, not on ADMM outperforming other methods. Carlos retains responsibility for theory and writing; the colleague must flag discrepancies and must not silently change formulas to accommodate results.

The documents governing implementation are `MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md`, `MATHEMATICAL_AUDIT.md`, `CLAIMS_AND_EVIDENCE_MATRIX.md`, and `HISTORICAL_CODE_AUDIT.md`. Experimental identifiers E0–E6 remain stable even if the article's numbering changes.

## Sequence, gates, and decisions before test-set access

1. Freeze the shared code, environment, and data; preserve the historical `EXPARTFINAL.ipynb`, results, figures, and CSV files in the historical repository. Implement new modules on a shared task branch, with CPU/MPS/CUDA profiles; Docker is optional. Do not maintain different algorithms on different machines.
2. Pass E0 and E1 on CPU in float64 before any training in the experimental suite. Correct the identified failures in new code and document the differences, without rewriting historical files.
3. Run a bounded pilot without consulting the test set: one seed, 42; both datasets and all three architectures; five epochs for each method: STD, WD with lambda=1e-3, NUC with lambda=1e-3, SN, ADMM with alpha=.8, and PSGD-M with alpha=.8: 36 short runs. The pilot uses 10,000 training and 2,000 validation examples, taken as stratified prefixes of the main partitions; these are not main-study replicates. Its tau is calculated using the corresponding five-epoch pilot STD and serves only the pilot; it does not replace the main STD reference. Include SVDs of the matrices in all layers and timed evaluations. If additional technical tests are needed, record them as such.
4. Deliver a measured budget showing hours per method/architecture/dataset, peak memory, and the costs of evaluation, calibration, checkpoints, and E4. Confirm the compute budget with the authors before launching hundreds of runs. Do not invent a GPU, VRAM capacity, driver, CUDA version, or duration.
5. Freeze the manifest, main-study size, hyperparameters, tolerances, failure policy, and model-selection policy before opening test results. Run deduplicated reference and calibration runs; validate completeness; run E2–E4 and derive E5–E6.
6. Only after validation-based selection is closed, evaluate the test set once for each preselected checkpoint. If a technical defect requires a code correction, create a new version of the protocol/manifest, disclose prior access to the test set, and do not present the result as a first blind evaluation.

**Pilot timing limitation:** with the unchanged three-epoch warmup and five-epoch transition, a five-epoch pilot reaches only warmup and part of the transition. It does not measure the fixed-budget phase beginning at epoch 9. Label extrapolated fixed-phase costs as estimates. If additional technical timing measurements at the target budget are needed, declare and log their scope and cost separately; do not silently compress the schedule or count those measurements as main benchmark replicates. The pilot count of 36 five-epoch runs remains unchanged.

## Fixed common configuration

| Aspect | Operational decision |
|---|---|
| Datasets | MNIST and Fashion-MNIST, official downloads through torchvision; save file SHA-256 hashes and version |
| Main partitions | From the official training set: 50,000 train and 10,000 validation; reserve the complete official test set of 10,000 |
| Reproducible split | For each dataset and class: permute with NumPy `Generator(PCG64(20261013))`; reserve the appropriate number per class to obtain 10,000 validation examples (1,000/class for Fashion; for MNIST, allocate proportionally with the remainder assigned in class order); use the rest for train. Publish indices and hashes; all seeds and methods share indices |
| Pilot | Stratified prefixes of 10,000 train/2,000 validation drawn from those partitions; no test access |
| Normalization | ToTensor in [0,1]; pixel mean/standard deviation calculated only on the main train partition of each dataset, frozen and saved; no augmentation |
| Training seeds | **42, 123, 456, 789, 1024**, without replacing a seed that produces poor accuracy |
| Initialization | Initial `nn.Linear` tensors saved per architecture/seed before applying SN parametrization; the same initial checkpoint for every method and alpha |
| Minibatch order | Train permutations precomputed per seed/epoch with a separate generator; evaluation does not consume the training RNG; `drop_last=false` |
| Architectures | 3L: 784–256–128–10; 6L: 784–256–128–64–64–32–10; 9L: 784–256–128–128–64–64–32–32–16–10 |
| Parameters | Including biases: 235,146, 248,682, 266,618; weights: 234,752, 248,128, 265,888. Hidden ReLU, linear output; no dropout, BatchNorm, or skip connections |
| Horizon | 25 epochs; batch 64; constant LR .01; momentum .9, no Nesterov, except in controlled E1 experiments |
| Precision | Train in float32, without AMP/TF32. Projection and spectral auditing in float64; record the profile/device and copying back. Choose compatible versions after checking installation; pin them in a lockfile |
| Reference | One final STD per dataset/architecture/seed; tau(alpha,s)=alpha·Omega(W_STD_ref,s), alpha=.6,.8,1.0; save the unrounded value and checkpoint hash |
| ADMM | One SGD-M sweep per W step; Z/dual once per epoch; 3 warmup, 5 transition, and 17 fixed-budget epochs; initial rho 1, limits [.1,10], balance factor 10, and scaling factor 2; rescale u by rho_old/rho_new |
| Projected method | **PSGD-M** with momentum .9; global projection at the start of each active epoch and after every minibatch; the same schedule. Use this name in tables, not GD |
| Primary selection | Last scheduled epoch, with weights W and biases from the same state; no early stopping or switching to Z for a better result |
| ADMM secondary evaluations | Z with the same biases; projection of W onto final tau with the same biases; retain all W metrics |

The proposed main evaluation uses 50,000 examples per dataset to avoid conclusions that depend on the historical subset. The pilot must determine whether it fits the budget. If it does not, a reduced main phase of 10,000/2,000 may be agreed before test access while retaining the reserved test set; it must be explicitly labeled exploratory and must not be mixed with the full main study. Expansion is the predefined move from that reduced phase to 50,000/10,000 if the measured cost fits the agreed budget; it is not decided by test accuracy. An expansion has new data hashes and new STD references, so it cannot reuse tau or count as the same runs. Do not selectively reduce seeds or only unfavorable configurations.

### Methods and energy calibration

STD: cross-entropy without regularization. WD: the same optimizer and a (lambda/2)||theta||² penalty on weights **and biases**. NUC: add lambda·sum_l||W_l||_* to every minibatch, retaining autograd and without regularizing biases. SN: PyTorch parametrization of linear weights, one power iteration per access during training; measure only effective weights during evaluation. ADMM and PSGD-M use exactly the global set defined in the specification.

For WD and NUC, use the predeclared grid of five coefficients `{1e-5,1e-4,1e-3,1e-2,1e-1}` for each method. Train each candidate only once per dataset/architecture/seed, independently of alpha. Record time, losses, energy, and failures for every candidate. Select one coefficient per dataset/architecture/alpha, shared by all five seeds, by minimizing the mean over the five seeds of `abs(Omega/tau-1)`; ties within 1e-12 are resolved by lower mean validation loss, then by the smaller coefficient. Energy is obtained from weights, without test access. A candidate is eligible if it completes all five seeds with finite metrics; ineligible candidates retain their failures in the report.

The main table compares the selections at each alpha; all five candidates remain in the complete accuracy–Omega and loss–Omega curves. A pair is declared to have similar energy only if its difference is at most 5% relative to tau. If no candidate reaches that neighborhood, state “no energy match”; do not describe it as an equal-budget comparison or expand the grid after seeing test results. Do not select ADMM hyperparameters based on beating baselines. A new search requires an amendment before test access, an additional inventory, and reporting of all attempts.

## Work inventory and cost

| Block | Calculation | Unique runs |
|---|---|---:|
| E2 reference STD | 2 datasets × 3 architectures × 5 seeds | 30 |
| E2 joint ADMM | 2 × 3 × 3 alpha × 5 | 90 |
| E2 PSGD-M | 2 × 3 × 3 alpha × 5 | 90 |
| WD, all candidates | 2 × 3 × 5 seeds × 5 coefficients | 150 |
| NUC, all candidates | 2 × 3 × 5 seeds × 5 coefficients | 150 |
| SN | 2 × 3 × 5 seeds; reuse against all three alpha values | 30 |
| **E2 core** | 180 constrained + 30 references + 300 calibration + 30 SN | **540** |
| E3 | Diagnostics from the same runs, without repeating training | 0 |
| Mandatory E4, two allocations | 2 datasets × 3 architectures × 5 seeds × 2 allocations, alpha=.8 | 60 |
| Mandatory E4 longer horizon | Joint ADMM 9L, both datasets, five seeds, alpha=.8: continue from epoch 25 to 100 | 10 continuations of 75 epochs |
| E5/E6 | Checkpoint analysis, small tests, and measurements; no new replicates | 0 training runs |
| Preliminary pilot | 2 × 3 × 6 methods, seed 42, five epochs | 36 short runs |
| Optional E4 uniform allocation | 2 × 3 × 5, alpha=.8 | +30 |
| Optional E4 fixed width | 2 datasets × depths 3/6/9 × 5 seeds × {STD,ADMM,PSGD-M}, alpha=.8 | +90 |
| Optional E3 fixed rho | 2 × 3 × 5, ADMM rho=1, alpha=.8 | +30 |

Mandatory total: **600 runs of 25 epochs**, **10 continuations of 75 epochs**, **36 pilots of five epochs**, plus E0/E1/analysis and evaluation. The 300 calibration runs include the selected candidates: **do not add another 180 WD/NUC runs** or retrain the winner. SN is not tripled by budget. E3 reuses E2. Present the core cost with and without calibration; the cost of a continuation counts even if it shares the parent run_id. If SN is removed from the comparison and the manuscript's conclusions by explicit prior agreement, subtract its 30 runs and six pilots and document the change.

The pilot estimates times separately for each dataset/architecture/method cell. Budget = sum of remaining epochs/runs × measured time for a comparable phase + evaluation/checkpoint costs + E0/E1/E5/E6 + an operational margin of 20%. Present a range based on the measured minimum/maximum, hours already spent, and hours remaining; do not confuse an estimate with a guaranteed duration. Reserve 10% of the measured budget for infrastructure retries, without creating new seeds. Once measured, the authors confirm that it fits before the experimental suite is launched.

## E0 — Audit and tests before training

**Priority:** blocking. **Question:** do the tensors actually optimized, projected, and measured correspond to the formulas? **Claims:** joint projection, spectral projection, weight/dual accounting, and absence of false measurements. **Data:** small synthetic float64 matrices, without datasets or training. **Seeds:** the five fixed seeds; add exact deterministic cases.

Mandatory tests:

- Joint projection for shapes (1,1), (2,3), (3,2), (4,4), product spaces with one and four layers, zero/nonzero matrices, and mixed dimensions. Radii 0, .1·Omega(Y), .8·Omega(Y), Omega(Y), 1.2·Omega(Y). Verify finiteness, feasibility, identity in the interior, idempotence, empirical continuity when crossing the boundary, and a unique output under rotations of repeated singular subspaces.
- Repeated spectra: diag(3,3,1,0), orthogonally rotated matrices, and two identical layers. Check the same common threshold and reconstruction; test a layer zeroed by global projection and an individual ball with a positive radius that does not zero a nonzero input.
- Independent comparison: matrices no larger than 4×4; solve the matrix projection using a convex nuclear-norm program distinct from the code's SVD/thresholding algorithm. Pending interface: CVXPY with the CLARABEL solver, absolute, relative, and feasibility tolerances of 1e-10, maximum 200 iterations; verify support/compatibility during installation and do not assume it is already available. Require optimal status and documented residuals; compare solutions with tolerance 1e-7·max(1,||Y||_F), specific to the external oracle, and compare objective and feasibility. Retain the stricter internal E0 tolerances from the specification. If the solver does not certify the precision, mark the test inconclusive and resolve the configuration before the experimental suite. An additional scalar oracle that exhaustively searches simplex faces checks the singular-value subproblem, but does not replace the matrix test.
- Spectral projection of diag(2,1) onto radius 1: output diag(1,1), squared distance 1; rescaling gives diag(1,.5), squared distance 1.25. Zero case without NaN. Positive rescaling preserves the effective rank of a nonzero matrix.
- Counterexamples: a chain of scalar weights 1,1 gives E(A)=2sqrt(2), Omega=4; dual increase in the Lagrangian =rho||r||²; penalty example F_lambda(w)=f(w)+lambda·max(Omega(w)-tau,0), with f(w)=(w-2)²/2, Omega=2|w|, tau=2, lambda=1: penalized minimum w=1, F*=1/2; w=1.5 gives violation 1 and gap 5/8, refuting the old universal bound.
- Inject SVD failure, NaN, and Inf: these must produce a traceable error state, never zeros, silent random noise, or `filterwarnings('ignore')`. Check that an authorized numerical fallback preserves the input without perturbing it and is logged.
- NUC gradient: use diagonal matrices with distinct positive singular values and central perturbations; compare the autograd gradient with the analytical derivative/finite difference; check for a nonzero gradient in the actual weight. Test the CPU/CUDA path if used; a `detach`/NumPy/float path cannot participate in the NUC loss.
- SN: record original and effective weights in separate columns; verify predictions using the effective copy; use module dimensions for relative rank, without resolving dimensions through parametrized names. A power-method estimate may yield a norm exceeding 1: measure it, do not hide it.
- Test the factor of 2: tau=2 implies nuclear-norm sum<=1. Test rho at its limits; for example, rho 8→10 requires u×.8, and rho .1→.1 requires unchanged u; equality y_before=y_after within tolerance.
- Resume test: a two-step toy training run with synthetic data, continuous versus saved/resumed; equality of configuration/state/tensors within the declared tolerance. This small test is not the training suite.
- Reconcile notebook/CSV/text by hash and origin. Detect a repeated configuration in the manifest and reject a new identical run; separate `attempt_id` from scientific identity.

**Outputs:** `E0/test_report.json`, `E0/projection_oracle.csv`, `E0/counterexamples.csv`, `E0/gradient_check.csv`, a complete log of injected failures, and the environment. **Validity:** specification tolerances, an oracle with explicit status, and all critical cases passed; merely finishing the test without an exception is insufficient. **Favorable interpretation:** the implementation agrees. **Neutral:** a documented accurate fallback with higher cost. **Unfavorable:** a failure blocks the experimental suite; correct it and repeat E0, without discarding the case.

## E1 — Verification under analytical assumptions

**Priority:** mandatory. **Question:** are nuclear-norm variation and the QG/SC bound calculated correctly relative to the relevant optimum? **Claim:** local stability of Omega under quadratic growth (QG), with strong convexity (SC) as a special case; not validation of SC for an MLP.

**SC case:** two matrices of shapes (3,2),(2,3), biases of sizes 3,2; known theta_*, zero by default. Second fixed rational case: W1_*=[[1,0],[0,2],[1,-1]], W2_*=[[1,0,-1],[0,1,1]], b1_*=[.5,0,-.5], b2_*=[.25,-.25]. f(theta)=(gamma/2)||theta-theta_*||², gamma=2, f*=0. L=gamma. The strong-convexity constant is exact, and all parameters are regularized. For each seed, generate 100 normalized Gaussian directions and radii 10^j, j=-8,-6,-4,-2,0. Use D=4 in the bound. Add the sharpness case of one matrix W=tI_2, theta_*=0, without biases, which gives ratio 1 using R=D=2.

**QG case without uniqueness:** W of size 2×2, P=diag(1,0), W0=[[1,2],[0,0]], f(W)=(3/2)||P(W-W0)||². S={W:PW=PW0}; f*=0; projection W^+=W-P(W-W0). The Hessian is PSD, with eigenvalues 3 in the first row and 0 in the second. Global QG with gamma=3 relative to S; not SC over the whole space. Here rank(W-W^+)<=1 is an analytical bound usable without thresholding. Use five seeds, 100 Gaussian matrices, and the same five radii in normal and tangential components. Compare energies with the projection corresponding to each W, not with an arbitrary minimum-energy reference.

**Controlled ADMM diagnostic:** the same SC quadratic in weights and biases, using the rational theta_* above, positive tau fixed at .8·Omega(W_*), and fixed rho=1; initialize W=Z=0, u=0, b=0. Solve the W step by the formula (gamma·W_*+rho(Z-u))/(gamma+rho), b=b_*. Exact joint Z update. Run 100 small iterations and record r,s,e and the stationarity bound using L=gamma. The constrained weight solution is P_C(W_*); this provides an oracle. This example does not transfer a guarantee to ReLU.

**Metrics/outputs:** raw gap, distance to S, gamma/L/D/analytical rank, Omega(W), Omega(W^+), left-hand side, bound, ratio with special status for 0/0, error relative to the oracle; `E1/sc_samples.csv`, `E1/qg_samples.csv`, `E1/controlled_admm.csv`, plots of ratio versus radius and the geometry of S. **Validity:** nonnegative gap within explicitly stated rounding error; no value is replaced by zero in storage; ratios<=1+tolerance. If ratio>1, review the calculation/assumptions/reference; do not call it a vacuous bound. **Favorable:** formulas and equality cases are reproduced. **Neutral:** a loose but correct bound. **Unfavorable:** a reproducible discrepancy requires stopping use of that measurement; it does not by itself refute a proof without auditing the arithmetic. Simulations do not replace proofs.

## E2 — Main comparison under a global budget

**Priority:** mandatory. **Question:** what predictive/spectral/cost tradeoff does ADMM offer relative to STD, WD, NUC, SN, and PSGD-M, and how does it vary with alpha and architecture? **Claim:** empirical evaluation of the method, not universal superiority or an asymptotic guarantee. **Data/configuration/seeds:** common table, inventory of 540, fixed calibration, paired initializations and references.

**Metrics:** data cross-entropy on train/validation/test, accuracy, generalization gap defined as train minus test (and train minus validation separately), Omega, raw/relative violation and feasibility within tolerance for **all** methods, training/evaluation/calibration time, memory, spectra, and projection count. For SN use an effective-weight snapshot. For ADMM the primary comparison uses W and publishes Z/P(W) in secondary panels. The test set does not determine hyperparameters, horizon, variants, or selected figures.

**Outputs:** `E2/runs_manifest.jsonl`, `E2/raw_metrics.parquet` or full-precision CSV, `E2/calibration.csv`, `E2/paired_differences.csv`, `E2/failures.csv`; figures `accuracy_omega`, `loss_omega`, `feasibility_all_methods`, `time_accuracy`, separated by dataset/architecture, showing points for each seed and curves by alpha. Tables with mean±sample standard deviation, five observations, and paired differences, without treating epochs as replicates.

**Validity:** data separation, five planned seeds, frozen configuration, no duplicate identity, energy/accuracy from the same checkpoint, uniform tolerance, visible failures. A run that converges to low accuracy remains valid. **Favorable:** ADMM achieves a better tradeoff than some comparators at similar energy, with uncertainty and cost specified. **Neutral:** small or uncertain differences; report that no advantage is identified. **Unfavorable:** worse predictive performance, high cost, or failures by architecture are published as limitations. PSGD-M may match or outperform ADMM while satisfying the budget; do not reserve the word feasibility for ADMM.

## E3 — Feasibility and dynamics

**Priority:** mandatory; reuses E2. **Question:** which weights satisfy the budget, and which diagnostics explain the difference among W/Z/final projection? **Claims:** residual feasibility certificate; stationarity only where the smoothness assumptions are justified.

**Record for each active epoch:** current/final tau, Omega(W), Omega(Z), violations of W and Z relative to both tau values, r=W-Z, Frobenius norm and nuclear-norm sum of the differences, s=rho_used(Z-Z_old), rho_used/rho_next, dual scaling factor, phase, threshold, actual projection frequency, and gradients for each layer and biases. Evaluate W,Z,P_final(W) on train and validation every epoch; test only on selected final states. Save spectra at epochs 0,3,8,15,25. After every PSGD-M projection, save the violation and the maximum per epoch; at least at the beginning/end of each phase, perform a complete float64 measurement and check that intermediate logging does not alter the state.

Certificate: compare the direct W violation with `violation(Z)+2 sum nuclear(W-Z)` and with `violation(Z)+2 sqrt(D)||r||`. Preserve the raw difference bound minus violation. Stationarity: check the formula in E1 with the full gradient, biases, and known L. For MLPs, calculate the full gradient at checkpoints 3,8,15,25 and the epoch's e if Z_old/u_old/rho_used were retained, but do not call ||e||+||s|| a certificate without a valid L, or call a minibatch gradient a certificate.

**Outputs:** `E3/dynamics.csv`, `E3/feasibility_certificates.csv`, `E3/stationarity_applicability.csv`; one figure per seed showing shaded phases, tau, W/Z, residuals, and rho; a final table by variant. Do not hide zeros/negative values through replacements for log plots: retain the source values and use a symlog scale or mark zeros.

**Validity:** formula/bound evaluated with the same tensors/tau; finite values; save failures and distinguish the primary network from the auxiliary network. **Favorable:** W is feasible within tolerance and residuals are small. **Neutral:** Z is feasible by construction, W is slightly infeasible, and the certificate is conservative. **Unfavorable:** only Z satisfies the budget or a persistent W-Z gap exists; state this explicitly, without reporting W accuracy as the feasible accuracy of Z. Small or decreasing residuals over 25 epochs do not prove asymptotic convergence or universal monotonicity.

## E4 — Allocation across layers, depth, and horizon

**Priority:** mandatory. **Question:** how do spectral allocation and learning change when the feasible set changes, and how much is explained by an insufficient horizon? **Data:** both datasets, the three original architectures, alpha .8, five seeds. **Method:** joint ADMM reused from E2 versus two ADMM variants identical except for Z projection, with radii allocated by number of weights or by the energies of the paired STD. Exact formulas are in the specification; preserve total tau, schedule, and rho. Adds 60 runs, without repeating the reference or joint variant.

**Metrics:** spectrum per layer, nuclear-norm sum, sigma_max, effective rank, relative distance from initialization, norms of data gradients and augmented-term gradients separately, bias norm, zero ReLU activations, and percentage of positive activations. “Nearly zeroed” is predefined as ||W_l||_F/max(||W_l_initial||_F,1e-12)<=1e-3; also record the absolute scale and whether W or Z is exactly zero. Inspect before/after projection. A positive individual radius does not justify exact zeroing; distinguish this from a layer eliminated by the global threshold, SVD failure, ReLU saturation, and proximal thresholding.

**Mandatory diagnostic:** continue all global ADMM 9L alpha .8 runs for both datasets and all five seeds from epoch 25 to 100, retaining the original final tau, rho/dual/momentum, and LR=.01. Do not start only poor-performing seeds. Report this as a longitudinal extension of the same seed, not ten new replicates. Evaluate whether validation loss/accuracy improves; the primary E2 comparison remains at epoch 25. If the longer horizon does not help, this does not prove a pure depth effect.

**Optional controlled width:** MLP 784–[128 repeated m-1 times]–10 for m=3,6,9; the same protocol and five seeds, STD/ADMM/PSGD-M alpha .8, 90 runs. Keeping width fixed changes the parameter count: report it. The original architectures change both width and depth, so no comparison is titled “isolated effect of depth.”

**Outputs:** `E4/layer_metrics.csv`, `E4/allocations.json`, `E4/horizon_continuations.csv`; figures of radii/energies/gradients by layer and the 25→100 trajectory, paired tables. **Validity:** radii sum to tau/2, sets are identified as distinct, five seeds, and learning problems are not filtered out. **Favorable:** the joint set better preserves accuracy for certain configurations. **Neutral:** similar allocations or an advantage that disappears with a longer horizon. **Unfavorable:** per-layer allocation improves results, the global set eliminates layers, or the advantage is unstable. Do not conclude that per-layer alternatives are mathematically invalid or always worse.

## E5 — Graph, surrogate, and parametrization

**Priority:** mandatory while the two-sided relationship and the ReLU example are retained. **Question:** what distinguishes adjacency energy, the surrogate, and the represented function? **Data:** chains of small random matrices, five seeds, consecutive dimensions 3–4–2–3; zero cases; E2 checkpoints for seed 42, both datasets, three architectures, methods STD/ADMM/PSGD-M/SN at alpha .8, epochs 0 and 25. Reuse STD/SN; for ADMM measure W/Z.

Construct signed symmetric adjacency without biases; compute E(A) using float64 eigvalsh or the nuclear norm of A, without diagonalizing in minibatches. Record Omega=0 as a separate case, with expected E(A)=0 and ratio `null`; otherwise measure E(A)/Omega and compare against [.5,1]. If the cost of checkpoint adjacency matrices is excessive, run the mandatory small cases and record which checkpoints were not calculated; do not substitute Omega for E(A) under the same name.

Parametrization example: a ReLU network 3–4–2 without biases, small fixed/Gaussian weights for five seeds, 100 Gaussian inputs, c={.25,.5,1,2,4}; compare (cW1,c^-1 W2). Maximum relative/absolute prediction error <=1e-10 in float64, unless the scale is explicitly reviewed; save Omega and graph energy. Data loss is the same if predictions coincide; WD may change. Do not claim that Omega changes for absolutely every c (coincidences are possible); demonstrate that it is not invariant in general.

**Outputs:** `E5/graph_ratios.csv`, `E5/rescaling.csv`, a plot of E(A) versus Omega, and an Omega-versus-c curve. **Validity:** symmetric, dimensionally correct adjacency, separate zero case, predictions from the same model/input. **Favorable:** identities and bounds are reproduced. **Neutral:** ratios far from the extremes; no performance implication. **Unfavorable:** a ratio outside the tolerated interval or a changed function requires reviewing the implementation/scope, not reinterpreting the bound.

## E6 — Cost and compression

**Priority:** costs mandatory; compression conditional on retaining those claims. **Question:** how much does each method cost, and does a lower-rank spectrum translate into a useful compressed artifact? **Data/seeds:** the same E2–E4 runs and final checkpoints for five seeds; the same Linux/CUDA machine and profile. Do not compare historical Mac times with CUDA times as a method effect.

Record actual hardware (GPU model/VRAM, CPU, RAM), driver, CUDA runtime and toolkit if present, OS, versions, BLAS, threads, clocks/power caps if fixed, precision, and determinism. Time after a warmup of 20 synthetic steps with equivalent shapes; synchronize CUDA at the start and end of the timed interval. Separate forward/backward/SGD, NUC gradient, SVD, thresholding/reconstruction, CPU/GPU transfers, evaluation, and serialization; count SVDs per matrix and projection calls. Total user time includes initialization and warmup; report time per stage separately. Measure peak CUDA allocated/reserved memory and CPU RSS. Do not measure each operation with extra synchronization during the primary run without quantifying instrumentation cost; use stage intervals or an identical profiled pilot.

If compression is retained: for each layer and checkpoint, use SVD truncation with retained Frobenius-energy fractions { .90,.95,.99 } selected by validation; retain biases and factorize W≈A B with r(m+n) parameters versus mn. Do not factorize a layer if r(m+n)>=mn; record the decision according to the policy, not the test set. Evaluate without retraining and save a real artifact in a fixed format/dtype, with a comparable dense file. Report the count of stored numeric values, disk bytes, median latency and percentiles over 100 passes after 20 warmups, loss/accuracy, and prediction error. Use effective-weight snapshots for SN. If fine-tuning is added, it is a new protocol and cost, not a free improvement.

**Outputs:** `E6/costs.csv`, `E6/hardware.json`, `E6/compression.csv`, dense/factorized files and hashes; cost/accuracy/bytes figures. **Validity:** same machine, comparable phases, synchronization, accounting for calibration and conversion, and compression realized in an artifact. **Favorable:** actual savings with a predictive loss accepted before test access (operational threshold .5 percentage points on validation). **Neutral:** lower effective rank without fewer parameters/bytes. **Unfavorable:** higher cost, more storage, or degradation; withdraw compression promises and restrict the text to spectral properties if compression is not measured.

## Identity, storage, deduplication, and resumption

Create `config_canonical.json` (UTF-8, sorted keys, unrounded numbers, no volatile fields). `config_hash=SHA256(bytes)`; use a descriptive `run_id` plus the first 16 hex characters of the hash. Include in the hash: commit, diff hash if the code is dirty, protocol version, architecture, data/indices/normalization, seed/initialization, optimizer, objective, horizon, schedule, alpha/tau and STD hash, numerical profile, and model selection. Save the dependency manifest and hardware; relevant environment differences create a new profile/hash. Deduplication must recognize that WD/NUC/SN/STD do not depend on alpha: tables reference the same run_id, not configuration clones with an irrelevant alpha.

Each run directory delivers: configuration; commit and `git status`/diff; data and index hashes; initialization/STD hashes; lockfile, `environment.json`; structured epoch/minibatch/projection log; full-precision metrics; spectra; a checkpoint containing W,b,Z,u,rho, momentum, scheduler if any, and all RNGs; final state and checkpoints 0,3,8,15,25; references to E4 if continued. For SN, include originals, parametrization/power-iteration state, and an effective copy. Temporary checkpoints are written atomically and verified by hash.

A configuration in the run matrix is the unit of scientific identity; `attempt_id=run_id/attempt-N` identifies attempts with the same configuration. Before launching, consult the registry: if a successful run with valid artifacts/hashes exists, reuse it; if a valid incomplete checkpoint exists, resume; if it is running, do not duplicate it. A continuation from 25→100 is a derived configuration referencing the parent checkpoint; its epoch-25 data are reused, not measured again as an independent sample.

## Failures and analysis rules

| Status | Predefined action |
|---|---|
| `numerical_failure` (NaN/Inf/nonconvergent SVD) | Stop; save the last state and minimal failing tensor/configuration; no zeros or noise. One diagnostic retry with the same input on CPU in float64, without changing data, is recorded separately. If the algorithm/profile changes, create a new hash; it does not silently replace the primary result |
| `infrastructure_failure` (power outage/preemption/I/O) | Up to two resumptions from the last valid checkpoint, with the same run_id and seed; preserve logs; then mark incomplete |
| `resource_failure` (OOM) | Record memory and stop; do not reduce the batch size only for that method. Adjust a global policy in a new version before test access and budget for comparable repetitions |
| `implementation_failure` | Block the affected block, correct code, repeat E0 and affected runs under a new commit/hash; preserve the previous ones |
| `completed_low_accuracy` | A scientifically valid result; do not repeat it to seek a better seed |
| `completed_infeasible` | A valid result if the implementation passed E0; report violation, variant, and failure of the practical objective |
| `duplicate_configuration` | Reuse the existing artifact or resolve the hash conflict; do not count it as a replicate |

Do not exclude a failure on accuracy grounds. Report attempted/completed/failed counts per method. Means over successes are labeled conditional on completion; do not impute the mean for failures or conclude superiority from survivors. Publish paired comparisons only for available pairs, together with the denominator; the primary table indicates missing data. Retain every relevant warning and classify it; do not suppress warnings globally.

Report sample SD (`ddof=1`); never call it a confidence interval. For five complete paired differences, the descriptive 95% t CI is mean±2.776·SD/sqrt(5), with its assumptions and substantial uncertainty; if pairs are missing, report n and recalculate the quantile, or do not produce a CI. Do not treat images/epochs as independent replicates or present “not significant” as evidence of equivalence. For practical interpretation, predefine a band of ±.5 accuracy percentage points: a small difference compatible with that band is descriptive; speak of formal equivalence only if an analysis designed for that purpose supports it. Avoid declaring a global winner from a single alpha or depth.

Permitted wording: if ADMM wins, “in these configurations and at this cost”; if it ties, “no advantage is identified with five seeds”; if it loses, “the baseline achieves the better observed tradeoff”; if it fails at depth, “the configuration exhibits failures and sensitivity that limit its scope”; if only Z satisfies the budget, “the auxiliary Z is feasible; W does not satisfy the budget at the fixed tolerance.” No numerical output authorizes a conclusion of general asymptotic convergence.

## Proposed commands and deliverables

**No CLI for this protocol exists in the audited repository.** All commands below are a **pending interface to be implemented by the colleague**; they have not been executed or tested. The `ram_experiments` package, `configs/ram_v1.json`, and `frozen_selection.json` below are proposed interface/configuration placeholders, not supplied executable artifacts:

```bash
python -m ram_experiments.audit --manifest configs/ram_v1.json
python -m ram_experiments.verify --suite E0,E1 --device cpu --dtype float64
python -m ram_experiments.pilot --manifest configs/ram_v1.json --no-test
python -m ram_experiments.inventory --manifest configs/ram_v1.json --deduplicate
python -m ram_experiments.run --manifest configs/ram_v1.json --stage references,calibration,E2,E4 --resume
python -m ram_experiments.analyze --manifest configs/ram_v1.json --stage E3,E5,E6 --no-test
python -m ram_experiments.evaluate-test --selection frozen_selection.json
python -m ram_experiments.export --manifest configs/ram_v1.json --include-failures
```

The historical README proposes running the entire notebook: that command was not verified by execution and is outside this assignment. The colleague will first deliver new reviewable code, the E0/E1 report, and the pilot budget; then the frozen manifest and complete results, including failures. Final delivery: tables without premature rounding, figures with source data, models/states for resumption, environment, hashes, logs, instructions actually verified, and an explicit comparison with the claims permitted by the evidence matrix. The authors will decide scientific/editorial changes based on the complete results.
