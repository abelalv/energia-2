# Claims and evidence matrix

Version RAM-2026-10-13-v1. The main manuscript is `manuscript/manuscrito_RAM.tex`; its LaTeX labels are stable. E0–E6 are future protocols, except for the small checks expressly identified as already run in `mathematical_checks/`. A mathematical proof does not require favorable training results; an empirical claim does require the planned runs.

## Retained claims and required evidence

| ID | Claim and mathematical result | Assumptions and limits | Experiment | Expected evidence file | Admissible wording |
|---|---|---|---|---|---|
| AF01 | Block identity and `Omega/2 <= E(A) <= Omega`, Prop.1 `prop:graph` | Real signed matrices, consecutive layers only, biases excluded from the graph. Optimality of the lower constant is not proved | E0/E5 | `E0/counterexamples.csv`, `E5/graph_ratios.csv`; proof in `manuscript/sections/01_foundations.tex` | “For consecutive-layer signed graphs, the additive surrogate bounds adjacency energy within the stated factors.” No universal equality or electricity consumption claim |
| AF02 | Nuclear perturbation, Lemma2 `lem:perturbation` | Algebraic rank of the pair or bound D; product norm; biases included in theta | E0/E1 | `E1/sc_samples.csv`, `E1/qg_samples.csv` | “The surrogate is Lipschitz in parameter space, with a dimension-uniform bound and a pair-dependent rank refinement.” |
| AF03 | Square-root loss-gap stability, Theorem4 `thm:qg` and Cor.5 `cor:sc` | Nonempty/closed S, fixed f, explicit QG on a region, reference given by a projection onto S. SC requires the full assumptions; it is not verified for MLPs | E1 | The same files, with gamma/L/S/reference saved | “Under the stated local QG assumption, the spectral variation is bounded by the square root of the loss gap.” Do not infer quality from low energy |
| AF04 | Existence and projection, Lemma3 `lem:existence` | Continuous and bounded-below f_data; regularize all parameters for coercivity. Projection onto a nonempty closed set exists; it is not necessarily unique | E1, illustrative | Proof in the source; `E1/qg_samples.csv` | Existence under these assumptions; do not call a multivalued projection canonical or a set with unconstrained biases compact |
| AF05 | Joint projection, Prop.6 `prop:projection` | SVD with every singular value, product Frobenius norm, tau>=0; threshold unique only in the positive active case | E0 | `E0/projection_oracle.csv`, `E0/test_report.json` | “The exact Euclidean projection uses one threshold across all layers.” Bisection is an alternative; no novelty claim for thresholding |
| AF06 | Spectral projection and effective rank, Prop.7 `prop:spectral` | Frobenius norm and spectral ball; positive scaling of a nonzero matrix; convention r_eff(0)=0 | E0/E6 | `E0/counterexamples.csv`, `E4/layer_metrics.csv` | Singular-value clipping is distinct from rescaling/SN. Do not attribute spectral flattening to uniform scaling |
| AF07 | Necessary KKT conditions, Prop.8 `prop:kkt` | Locally differentiable f; tau>0 for the scalar formula through Slater's condition; direct cone for tau=0; bias gradients | E1; conditional diagnostic in E3 | `E1/controlled_admm.csv`, `E3/stationarity_applicability.csv` | Necessary conditions; no global optimality claim for a nonconvex network |
| AF08 | Finite-iteration feasibility, Prop.9 `prop:feasibility` | Any W,Z; retain Z violation and factor 2; known R or D | E3 | `E3/feasibility_certificates.csv`, `E3/dynamics.csv` | “The measured residual supplies a numerical upper bound on W's violation, within stated tolerances.” Feasibility of Z alone is insufficient |
| AF09 | A posteriori stationarity, Prop.10 `prop:stationarity` | Full gradient, L justified between x,z, fixed rho within the step, exact projection; add normal-cone/gradient defects if approximate | E1; E3 only if applicable | `E1/controlled_admm.csv`, `E3/stationarity_applicability.csv` | Certificate at z=(Z,b). For an MLP without a proof of L: “diagnostics”, not a smooth certificate or convergence |
| AF10 | Valid penalty, Prop.11 `prop:penalty` | An actual lower bound a, or a global saddle point and lambda>nu*. Do not confuse with NUC or ADMM | E0 | `E0/counterexamples.csv`; proof in `manuscript/sections/02_optimization.tex` | Use `(F-a)/lambda` or `(F-f*)/(lambda-nu*)`, according to the assumptions. No universal loss-gap/lambda bound |
| AF11 | Dependence on ReLU parametrization | Two layers without biases, c>0; the same function but Omega generally changes; WD is not invariant | E5 | `E5/rescaling.csv` | “The budget depends on the parameterization even when the represented function is unchanged.” |
| AF12 | ADMM predictive trade-off | Frozen configuration; separate splits; paired seeds; energy and accuracy from the same W; calibration costs included | E2 | `E2/raw_metrics.parquet` or CSV, `E2/calibration.csv`, `E2/paired_differences.csv`, `E2/failures.csv` | Conclusions limited to the configurations/datasets, reporting uncertainty and unfavorable results |
| AF13 | W/Z/P(W) differences and dynamics | Models labeled before testing, same biases, current/final tau, rho_used, no hidden final projection | E3 | `E3/dynamics.csv`, table by variant and hashes | Identify exactly which model is feasible; do not mix W accuracy with Z energy |
| AF14 | Allocation across layers and architecture | Radii sum to tau/2; different sets; multiple seeds; different widths; horizon diagnostic | E4 | `E4/allocations.json`, `E4/layer_metrics.csv`, `E4/horizon_continuations.csv` | Observed configuration effect; no universal causal claim about depth or claim that layerwise allocation is invalid |
| AF15 | Cost and compression | Same hardware/profile; synchronization; separate costs; actual factorization if compression is claimed | E6 | `E6/costs.csv`, `E6/hardware.json`, `E6/compression.csv`, models with hashes | Lower effective rank = a spectral property; savings only if parameters/bytes/latency/prediction are measured |

AF06 also uses the layer spectra already recorded in `E4/layer_metrics.csv` as supporting evidence. This cross-reference reuses the prescribed E4 measurements; it does not add a training run or change the E0/E6 validation tasks.

## Checks already run

`mathematical_checks/verify_mathematics.py`, `mathematical_checks/recorded_results.json`, and `mathematical_checks/recorded_execution.log` record 227 CPU/float64 checks, seed 20261013. They include AF01–AF03, AF05–AF06, AF08–AF11, and the stationarity/dual identities. They are not equivalent to E0 for a future implementation: they do not test the new training code, its resumption, SN/autograd, or GPU execution. The existing independent comparison solves the convex singular-value subproblem using SLSQP; the colleague's E0 additionally requires an independent matrix-level comparison. E2–E6 have not been run on new networks.

## Assumptions versus implementation

| Condition | Checks/quadratic control | Planned primary training variant | Consequence |
|---|---|---|---|
| Closed proper convex f | E1 quadratic: yes | ReLU MLP: not established | Do not import convex ADMM convergence |
| L-Lipschitz gradient on the relevant region | Analytic L in E1 | Not automatically justified | Prop.10 certified only in the control; the MLP records a diagnostic |
| First block solved exactly | Closed-form formula in E1 | One SGD-M epoch | Compute e; do not assume e=0 |
| SVD with every singular value and exact projection in real arithmetic | Proven formula; approximate float64 computation | SVD with every singular value in float64, float32 return | Distinguish the exact formula from evaluation with tolerances |
| Fixed rho/tau | Yes in E1 | Adaptive rho and transitioning tau | Do not claim a fixed Lyapunov function or an asymptotic rate |
| Full gradient | Available in E1 | Only at predefined checkpoints | Do not call a minibatch gradient the full residual |
| Global saddle point/convexity for the penalty | Verifiable in a controlled example | Not proved | Do not transfer the saddle-point version of Prop.11 to training |
| Known loss QG and optimum | E1: yes | MLP: no | MLP figures do not validate SC/QG through a partially positive Hessian |

## Interpretation rules, fixed before the results

| Possible result | Permitted wording in the article | Condition for use |
|---|---|---|
| ADMM improves the trade-off | “At comparable measured nuclear budgets, ADMM achieved [observed difference] in these settings, with [observed cost and uncertainty].” | Fill placeholders exclusively from data; define comparability according to the protocol; show comparators and failures |
| Small/uncertain differences | “The five-seed comparison did not identify a clear predictive advantage for ADMM.” | Do not turn nonsignificance into formal equivalence |
| ADMM loses | “The projected-gradient or calibrated penalty baseline achieved a better observed trade-off under this protocol.” | Identify the method/configuration and costs; retain results for all alpha values |
| Failure in 9L or another configuration | “Failures and sensitivity in the specified architecture limit the scope of the method.” | Do not exclude failed seeds; publish the denominator and categories |
| Only Z is feasible | “The auxiliary Z model satisfies the numerical budget, while the primary W model remains infeasible at the declared tolerance.” | Publish accuracy for both; do not assign Z's guarantee to W |
| Small residuals | “Finite-run residuals are small in the reported configurations.” | Never replace this with “ADMM converges” |
| A longer horizon removes the difference | “The observed difference is sensitive to the optimization horizon.” | Continuations do not count as additional seeds |
| Lower effective rank without savings | “The method changes the singular-value distribution; deployment compression has not been established.” | Withdraw promises of storage/electricity savings |
| Theoretical ratio >1 | “This numerical check failed and requires an audit of the assumptions, reference optimum and computation.” | Do not call the bound vacuous or clip a negative gap |

## Withdrawn claims and conditions for reconsideration

- Global identity `E(A)=Omega`: refuted; it cannot be restored without an additional proved architecture/case.
- General nonconvex convergence in the former result 9: incorrect proof and insufficient conditions. A future claim requires a theorem applicable to the specific algorithm, its assumptions checked term by term, and evidence that they hold; finite-run residuals are no substitute.
- Linear rate in the former Theorem 10: mixes a nonsmooth objective/algorithms and uses insufficient summability; it is not retained.
- Corollaries 12–13 with a universal nonnegative sign and global domain: incorporated as a local absolute estimate in Theorem 4; activity can establish a sign in specific cases.
- Universal Corollary 14: refuted and replaced by Prop.11 with explicit assumptions.
- Exclusive ADMM advantage for feasibility: withdrawn; PSGD-M projects onto the same set.
- SN without nuclear control and flattening through simple normalization: mathematically withdrawn; moreover, the historical metrics do not adequately measure effective weights.
- Exact annihilation by a small positive individual nuclear radius: refuted. Investigate alternative mechanisms using spectra/gradients.
- Twenty independent historical runs and exactly zero violation: unsupported by historical traceability or tolerance. Do not reuse without independent reconstruction.

The audit and code documents specify the locations and evidence for each withdrawal. New evidence may change the applied conclusion, but must never silently alter the definitions, tolerances, or assumptions to favor a method.
