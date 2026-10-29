# Mathematical audit and traceability — RAM revision

Date of the source audit: 2026-10-13. The audit record was created before the manuscript body was changed and completed during the revision. The decisions below document the mathematical replacements. The final tables identify revised results by stable labels, independently of their editorial numbering.

This is a complete English translation of the mathematical audit, with portable references for this handoff. It preserves the original findings, counterexamples, hypotheses, decisions, source coordinates, and distinctions between demonstrated results and pending experiments. The Spanish filenames in the provenance below name historical sources; they are not instructions to find missing documents inside this bundle. Any AI assistant or researcher can use the handoff without access to the original author's machine or software session.

## Provenance and preservation

- The read-only original manuscript collection is referred to logically as `historical-manuscript/PaperEnergyElsevier/`. This is a provenance name for the original collection, not a required local bundle directory.
- The collection contained two standalone entry points: `main_energia_elsevier.tex` (external bibliography) and `manuscrito.tex` (embedded bibliography, a local PDF, and a compilation log). They were compared by content, not by filename.
- At the time of the source audit, the original directory was not a Git repository. The revision workspace was a repository without commits, on branch `master`, with no changes at the start. No reset, push, or publication was performed. These statements describe that historical audit environment, not the recipient's environment.
- Revised sources and deliverables were originally saved under the logical revision root `revision_RAM/`; original figures, data, and sources were preserved unchanged. The English revised manuscript is bundled here as `manuscript/manuscrito_RAM.pdf`, with `manuscript/manuscrito_RAM.tex` and `manuscript/sections/` as its sources.
- The historical environment had `pdflatex`, `latexmk`, and `bibtex`; the final compilation was verified in that revision. The source audit cited a separate historical final-verification record, `evidencia/VERIFICACION_FINAL.md`. That record is provenance, not a required file in this handoff, and this statement does not assert that those tools are installed on the recipient's machine.
- The experimental suite was not run, and its notebook was not imported.
- For implementation and new experiments, use the local `MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md`, `EXPERIMENTAL_PROTOCOL.md`, `CLAIMS_AND_EVIDENCE_MATRIX.md`, and `AI_ASSISTANT_PROMPT.md`. The static historical code audit is `HISTORICAL_CODE_AUDIT.md`; its repository provenance is pinned to commit `186e27400f94d551432759fa4830e0c114a2aa69`.

## Findings register

The original `manuscrito.tex` was read in full (lines 1–1589), including its abstract, interpretations, tables, captions, conclusions, and bibliography. A full diff against `main_energia_elsevier.tex` confirmed the same scientific body: from the title through the end of the body, a line in `main_energia_elsevier.tex` is the corresponding line in `manuscrito.tex` minus 4. The differences concern the preamble, figure paths, and embedded versus external bibliography. Locations below use **M** = the original `manuscrito.tex` and **F** = the original `main_energia_elsevier.tex`, both in the logical historical manuscript collection defined above. They do not denote lines in the revised manuscript.

The original category codes are retained for traceability: **ED**, demonstrated error; **HA**, missing hypothesis; **DI**, incomplete proof; **IE**, overinterpretation; **MI**, discrepancy between mathematics and implementation/documentation; **CC**, correct but known. A result can require more than one category. The source audit referred to a separate historical independent-proof record. That record is not needed here: the corresponding complete revised proofs are bundled in `manuscript/sections/01_foundations.tex` and `manuscript/sections/02_optimization.tex`, and rendered in `manuscript/manuscrito_RAM.pdf`. The stable labels below locate those proofs.

### M01 — Dimensions, indices, signs, and biases

- **Location and original claim:** M156–179/F152–175 uses `m` matrices between `m+1` layers, but the adjacency ends at `W^(N)` and the sum ends at `N−1`. M427–452/F423–448 identifies bias-free networks with networks whose biases do not contribute to the energy; M493/F489 does define the parameters as `(W,b)`.
- **Diagnosis:** ED for dimensional consistency and IE for equivalence. Omitting biases from a functional is not equivalent to removing them from the network.
- **Argument:** An affine layer has `W_l ∈ R^(n_l×n_(l−1))` and `b_l ∈ R^n_l`; the compatible adjacency has `A_(l,l−1)=W_l`, `A_(l−1,l)=W_l^T`, with layers `0,…,m`. Negative weights are signed edges; replacing them with their absolute values defines another matrix and another problem.
- **Correction:** Fix this convention, retain biases in the full parameter space, and explicitly exclude them from Ω. With consecutive-layer connections, the full graph is bipartite by layer parity.
- **Dependencies:** Definitions preceding all results.
- **Experimental consequence:** Measure effective weights and preserve exactly the same biases when evaluating W, Z, and a final projection; check dimensions and parameter counts.

### M02 — Full graph energy and the additive surrogate

- **Location and original claim:** M98/F94 and M122/F118 assert equality between network energy and twice the sum of nuclear norms. M196–219/F192–215 correctly distinguishes the block identity from the global bound; M354/F350 again calls the nuclear-norm sum `E(W)`.
- **Diagnosis:** ED for the global equality; CC for the block identity and upper bound. Criticizing the body for never distinguishing the two quantities would be incorrect: it does distinguish them at M207–219.
- **Counterexample:** `W_1=W_2=[1]` gives the spectrum of A as `{√2,0,−√2}`; `E(A)=2√2`, whereas `Ω=4`.
- **Correction:** Reserve `E(A)=||A||_*` and `Ω(W)=2Σ||W_l||_*`. Retain `E(B_l)=2||W_l||_*` and prove `Ω/2≤E(A)≤Ω` using the two alternating edge families. The revised result is `prop:graph`; optimality of the lower constant is not claimed.
- **Dependencies:** M01; orthogonal invariance and the nuclear-norm triangle inequality.
- **Experimental consequence:** E5 compares E/Ω only when Ω>0, treats zero separately, and does not interpret Ω as electrical consumption.

### M03 — Parameter space and distinct losses

- **Location and original claim:** M229–246/F225–242, M319–329/F315–325, and M387–390/F383–386 use `θ=vec(W)`; M493/F489 adds biases. M457/F453 introduces a regularized loss; M519/F515 minimizes data loss, and M567–578/F563–574 adds a one-sided penalty.
- **Diagnosis:** HA/MI. The Frobenius–vectorization equality is CC for weights alone; it would be ED if biases were included while their norm was omitted.
- **Argument:** For `θ=(W,b)`, `||Δθ||²=Σ||ΔW_l||_F²+Σ||Δb_l||²`, hence `||ΔW||_F≤||Δθ||`. The minimizers of `f_data`, `f_data+λ||θ||²/2`, and `F_λ` can differ.
- **Correction:** Use the full product norm and name the objective f in each theorem; do not interchange minima or gaps. Include the correct inequalities in `lem:perturbation` and `thm:qg`.
- **Dependencies:** M01.
- **Experimental consequence:** Every gap requires a compatible objective, reference point, and optimum value; no silent change of regularization is allowed.

### M04 — Unifying SC and QG, ranks, and domain

- **Location and original claim:** Lemmas 1–3, M236–255/F232–251; Theorem 4, M257–305/F253–301; hypotheses and Theorem 5, M331–410/F327–406. R is introduced within the proof, and Ω/E are used interchangeably; the assumption written at M227 is quadratic growth at a point, not the full definition of strong convexity.
- **Diagnosis:** CC for nuclear-norm inequalities and Cauchy–Schwarz; DI/HA for presentation and scope. The central estimate is valid once its hypotheses are specified.
- **Argument:** `|Ω(W)−Ω(V)|≤2Σ||W_l−V_l||_*≤2√R(W,V)||W−V||_F≤2√D||W−V||_F`, with `D=Σmin(n_l,n_(l−1))`. For a nonempty closed set S of minimizers of the same f and local QG `f−f_*≥γ dist(θ,S)²/2`, any nearest-point projection θ+ gives `|Ω(W)−Ω(W+)|≤2√(2R/γ)√(f−f_*)`. R depends on the pair; D is uniform.
- **Correction:** Use `lem:perturbation`, `thm:qg`, and `cor:sc`. Define the domain U where QG holds and, for constrained minima, use `U∩(C_τ×R^B)`. SC on a suitable convex set, with a minimizer and a first-order condition, is a sufficient special case.
- **Dependencies:** M03, M06. No general equivalence among QG, PL, KL, and SC is assumed.
- **Experimental consequence:** E1 uses quadratics with proved constants; D or conservative rank bounds replace a thresholded numerical rank that underestimates R.

### M05 — Equality, sharpness, and energy on the optimum set

- **Location and original claim:** M413–425/F409–421 acknowledges that `E(W+)` can differ from `E_min`, but lists equality conditions without the reverse triangle inequality and requires QG equality along the whole segment.
- **Diagnosis:** CC for the warning about `E_min`; DI/IE for equality. The revision agenda must not attribute an automatic substitution of `E_min` to the original: the correct residual term already appears there.
- **Argument:** In the nontrivial equality case with R>0, equality is also needed in `|Σ(norm W−norm W+)|≤Σ||W−W+||_*`, along with aligned signs across layers and, when biases are present, equality in `||ΔW||≤||Δθ||`. QG equality at the evaluated point is sufficient; equality along the entire segment is unnecessary. If R=0, the bound is trivially an equality even when biases change, and those nondegenerate conditions are not required. Sharp example: `f=γ||θ||²/2`, θ*=0, zero biases, and all positive singular values of every W equal to c; then `Ω=2Rc`, the gap is `γRc²/2`, and the bound is attained.
- **Correction:** Retain only the fully calculated sharp example and explain that it is an elementary consequence of known inequalities. Do not replace Ω(W+) with the minimum of Ω over S.
- **Dependencies:** M04.
- **Experimental consequence:** Do not claim novelty for the exponent 1/2 or compare against an unidentified energy minimizer.

### M06 — Existence, coercivity, and projections

- **Location and original claim:** Lemma 6, M456–470/F452–466: continuity of `f_data` and λ>0 supposedly imply coercivity of `f_data+λ||θ||²/2`; the proof introduces the unstated condition `f_data≥0` and calls θ+ canonical.
- **Diagnosis:** ED/HA; the interpretation confuses existence and uniqueness of a projection.
- **Counterexamples:** `f_data(θ)=−λ||θ||²` is continuous; the regularized objective is `−λ||θ||²/2`, which is not coercive and has no minimum. For `S={−1,1}`, θ=0 has two projections. A budget on W leaves free biases unbounded.
- **Correction:** `lem:existence` adds a finite lower bound for f_data and regularization of all relevant parameters. Continuity and coercivity give a compact minimizer set. In finite dimensions, a nonempty closed S already guarantees existence of a projection, without compactness; convexity of S suffices for uniqueness. Specify an arbitrary selection explicitly, without calling it canonical.
- **Dependencies:** M03 and the finite-dimensional setting.
- **Experimental consequence:** E1 QG problems with nonunique minimizers use an explicit projection; do not infer existence of cross-entropy minima without regularization.

### M07 — Direction of stability and the meaning of complexity

- **Location and original claim:** M130/F126 asserts proximity to the “optimal value” of energy; M308/F304 and M470–472/F466–468 interpret the theorem as stability/certification; M118–149/F114–145, M480/F476, and M1214–1238/F1210–1234 connect energy, capacity, and compression.
- **Diagnosis:** IE. The bound concerns a parameter descriptor under local hypotheses, not generalization or physical resource limits.
- **Argument:** For `f(w)=(w−1)²`, Ω=2|w|, w=−1 and w*=1 have the same Ω but loss gaps 4 and 0. Equal energy does not imply close losses. M24 also proves dependence on parametrization.
- **Correction:** Describe the implication small loss gap ⇒ bounded spectral variation relative to the nearest optimum. Avoid the reverse implication, a unique optimum energy, or claims about unmeasured FLOPs, memory, electricity, or actual compression.
- **Dependencies:** M04, M24, M27.
- **Experimental consequence:** E1 checks the correct direction; E6 must measure actual compression if that promise is to be retained.

### M08 — Joint projection and boundary cases

- **Location and original claim:** M608–626/F604–622, M637/F633, and M907–912/F903–908 assert a common threshold and a unique root of a strictly decreasing function, always found by bisection.
- **Diagnosis:** CC for the mechanism; HA/DI for derivation and boundary cases; IE if bisection is presented as the only option.
- **Argument:** The von Neumann trace inequality reduces the projection to `min_(z≥0,Σz≤τ/2) ||z−σ||²/2`. KKT gives `z=(σ−t)+`. For feasible input, t=0; for τ>0 and exterior input, there is a unique root `h(t)=Σ(σ−t)+=τ/2` in `(0,maxσ)`. For maxσ>0, h is strictly decreasing on `[0,maxσ]` and constantly zero afterward. If τ=0, the projection is zero and every `t≥maxσ` represents it. If all matrices are zero, they are feasible for τ≥0.
- **Correction:** `prop:projection` includes zero matrices and repeated spectra; the projected matrix is unique even when the SVD is not. With multiplier η for Ω≤τ and objective `ρ||Z−Y||²/2`, `t=2η/ρ`. Reserve Q,V for the SVD and u for the dual variable. Sorting can also compute the threshold.
- **Dependencies:** M01; M21 for approximate SVDs.
- **Experimental consequence:** E0 covers interior and exterior inputs, zero radius, repeated singular values, mixed dimensions, feasibility, idempotence, and an independent comparison.

### M09 — Incorrect spectral projection

- **Location and original claim:** Lemma 8, M627–644/F623–640, especially M638/F634; algorithm M657–658/F653–654 and cost M859/F855. `Π(Y)=min(1,a/σ_max(Y))Y` is asserted to be the Frobenius projection onto a spectral-norm ball.
- **Diagnosis:** ED.
- **Counterexample:** Y=diag(2,1), a=1. Clipping to diag(1,1) gives squared distance 1; rescaling to diag(1,1/2) gives 5/4.
- **Correction:** `prop:spectral`: `Π(Y)=Q diag(min(σ_i,a))V^T`, a≥0, with a proof using the trace inequality and separable minimization. Distinguish projection, rescaling, and SN parametrization.
- **Dependencies:** The spectral reduction in M08.
- **Experimental consequence:** E0 explicitly compares both outputs. Computing only σ_max does not generally produce the exact Frobenius projection.

### M10 — Nuclear-norm control induced by SN/Frobenius bounds

- **Location and original claim:** M149/F145 says WD cannot certify any nuclear-norm bound; M1046–1049/F1042–1045, M1191/F1187, and M1232/F1228 attribute a complete lack of global control to spectral bounds.
- **Diagnosis:** ED/IE. A penalty does not guarantee every prescribed τ, but spectral or Frobenius bounds do control Ω through dimension-dependent bounds.
- **Argument:** `||W_l||_*≤d_l||W_l||_2` and `||W_l||_*≤√d_l||W_l||_F`; hence `Ω≤2Σd_l a_l` if `||W_l||_2≤a_l`, and `Ω≤2√D||W||_F`. If data loss ≥a and the WD objective ≤c, then `||θ||²≤2(c−a)/λ` when λ regularizes all of θ.
- **Correction:** State these bounds, distinguish compliance with the chosen τ from the existence of any control, and include projected gradient as a feasible method.
- **Dependencies:** M03, M04.
- **Experimental consequence:** Check WD and SN feasibility against every τ; do not omit it because the method uses a penalty.

### M11 — Layerwise projection, different sets, and zeroing

- **Location and original claim:** M94/F90, M136/F132, M908/F904, M1108/F1104, and M1229/F1225 say proportional allocation “provably starves” and eliminates singular values of a nonzero layer.
- **Diagnosis:** ED for exact elimination of an entire layer at positive radius; IE for universal inferiority.
- **Argument:** If Y≠0 and a>0, some εY with `0<ε<min(1,a/||Y||_*)` is feasible and closer to Y than zero, so Π_a(Y)≠0. By contrast, joint projection of scalar values 3 and 1 with total nuclear radius 1 has t=2 and output `(1,0)`: an entire layer can vanish. `Π_product` projects onto `{||W_l||_*≤a_l}`, a subset of C_τ if Σa_l=τ/2; it is generally not projection onto C_τ.
- **Correction:** Distinguish strong shrinkage, exact zeroing under joint projection, fixed-penalty proximal operators, and numerical failures. The common-threshold projection is exact for the chosen set; layerwise alternatives are different, valid problems.
- **Dependencies:** M08.
- **Experimental consequence:** E4 compares sets and several allocations, spectra, and gradients using multiple seeds; it does not presuppose the cause of failure.

### M12 — KKT, the normal cone, biases, and the factor of two

- **Location and original claim:** M543–559/F539–555 gives KKT conditions and an abbreviated nuclear-norm subgradient.
- **Diagnosis:** CC with HA concerning regularity, parameter space, and details; IE if interpreted as global nonconvex optimality.
- **Argument:** On D_τ=C_τ×R^B, a local minimum of differentiable f satisfies `0∈∇f(θ)+N_Dτ(θ)` by the directional condition on a convex set. For τ>0, Slater's condition for the single convex constraint gives a multiplier ν≥0, complementarity, `∇_b f=0`, and `0∈∇_(W_l)f+2ν∂||W_l||_*`. For a compact SVD QΣV^T, the subdifferential is `{QV^T+H: Q^TH=0, HV=0, ||H||_2≤1}`; at W=0 it is the spectral unit ball.
- **Correction:** `prop:kkt`; the normal cone is defined at a feasible point; τ=0 is handled directly by the normal to `{0}×R^B`. KKT is necessary and globally sufficient only under convexity of the objective.
- **Dependencies:** M01, M03, M08, M20.
- **Experimental consequence:** Evaluate stationarity in the full parameter space, including biases; do not use only a weight-gradient norm or a minibatch gradient as a complete certificate.

### M13 — Augmented Lagrangian for an inequality

- **Location and original claim:** M582–591/F578–587 declares `f+νg+ρg²/2` to be an augmented Lagrangian for g≤0.
- **Diagnosis:** ED/HA. That is the equality formula; without eliminating a slack variable, it also penalizes negative constraint slack.
- **Argument:** With g+s=0, s≥0, minimizing over s gives `f+([(ν+ρg)+]²−ν²)/(2ρ)`. For g<−ν/ρ, it is flat with respect to g, unlike the original quadratic.
- **Correction:** Remove this detour and retain only the split formulation `W=Z`, `Z∈C_τ`, whose augmented Lagrangian is well defined.
- **Dependencies:** M12.
- **Experimental consequence:** Do not interchange one-sided-penalty training, a scalar-multiplier method, and split ADMM.

### M14 — Pseudocode and adaptive parameters

- **Location and original claim:** The algorithm at M646–664/F642–660, titled global, projects using radii τ_l and permits a spectral alternative; M661/F657 combines loss evaluated at θ with energy of Z. M900–902/F896–898 adds SGD, adaptive rho, warmup, and varying τ.
- **Diagnosis:** MI/ED in the method definition. Solving an argmin exactly and taking SGD steps with momentum are different algorithms.
- **Argument:** Fixed layer radii do not generally represent C_τ (M11); `f(W,b), Ω(Z)` does not describe the same model unless W=Z. Changing rho while holding u fixed changes the unscaled dual variable y=rho u.
- **Correction:** Separate exact pseudocode from the implementation: θ-step, common projection onto C_τ, then u update. If rho changes, use `u←(rho_old/rho_new)u` to preserve y. Record τ_k, phases, frequency, subproblem error, and the model evaluated.
- **Dependencies:** M08, M11, M22–M23.
- **Experimental consequence:** E2–E3 evaluate W, Z, and the final projection with identical biases; predefine the primary model; measure loss and energy on the same weights.

### M15 — Proposition 9: sign, factor, and convergence

- **Location and original claim:** M670–708/F666–704. M693–704/F689–700 calculates the dual increase as `ρ||r||²/2` and turns it into a telescoping decrease of the same magnitude.
- **Diagnosis:** Algebraic ED and DI in the theorem. The stated smoothness, exact projection, and compactness assumptions do not justify the proof; they also do not control the SGD approximation.
- **Argument:** `L=f+I_C+ρ||r||²/2+ρ<u,r>` with W,Z,rho fixed. Thus `L(u+r)−L(u)=ρ||r||²`, which is positive. Primal descent ≤0 plus dual increase ≥0 does not imply net descent or summability. Moreover, Z is feasible immediately after projection, not merely asymptotically, and M671 mixes Ω with a sum of radii without the factor 2.
- **Correction:** Remove the general asymptotic guarantee and unverified KL/PL arguments. Replace them with `prop:feasibility` and `prop:stationarity`, which concern finite iterations and have precise scope. Do not retain an external theorem without checking all its hypotheses and the exact method.
- **Dependencies:** M08, M14, M20, M22–M23.
- **Experimental consequence:** Finite residuals and feasibility of Z do not prove convergence of W; monotonicity is not assumed. The historical primal-residual figure is interpreted as a diagnostic, not a theorem.

### M16 — Theorem 10: PL, a nonsmooth function, and errors

- **Location and original claim:** M712–743/F708–739 assumes F_λ is L-smooth, mixes a GD step on F_λ with errors in Z, and asserts `O((1−αμ)^k)` for α∈(0,2/L) from summability of errors.
- **Diagnosis:** HA/ED/MI. At M710/F706 the text already says this is not a direct ADMM theorem, so claiming that this caveat is entirely absent would be inaccurate. Nevertheless, the formulation mixes algorithms and does not prove the rate.
- **Counterargument:** F_λ is generally nonsmooth because of the nuclear norm and positive part. For exact GD on an L-smooth PL objective, the calculation gives `q=1−2αμ(1−αL/2)`, not generally `1−αμ` when α>1/L. With `Δ_(k+1)≤qΔ_k+ε_k`, the solution is `q^kΔ_0+Σq^(k−1−j)ε_j`. For equality and `ε_j=(j+1)^−2`, Δ_(k+1)≥ε_k, so there is no geometric rate despite Σε<∞. Even requiring summable error norms is insufficient: take `ε_j=(j+1)^−4`.
- **Correction:** Remove Theorem 10, without inventing proximal-PL theory to preserve numbering. An alternative result would have to specify the invariant region, function, and method.
- **Dependencies:** M14–M15, M20.
- **Experimental consequence:** Do not fit a geometric rate as validation of a nonexistent theorem, or call F_λ a Lyapunov function for the implemented ADMM.

### M17 — Corollary 11: convexity and a positive multiplier

- **Location and original claim:** M749–777/F745–773 allows ν*≥0 and divides by it; the proof invokes convexity of f_data without assuming it.
- **Diagnosis:** HA/ED.
- **Counterexamples:** An active constraint does not imply ν*>0: `f(w)=(w−1)²/2`, Ω=2|w|, τ=2, w*=1, ν*=0. Even with ν*>0, KKT is insufficient without convexity/globality: `f(w)=1−w³`, C=[−1,1], w*=1, ν*=3/2; at w=0, the Ω deficit is 2 but gap/ν is 2/3.
- **Correction:** Remove it as a nonconvex corollary. In the discussion of `prop:penalty`, retain only the version under a global saddle condition and ν*>0: for feasible θ, `ν*(τ−Ω(θ))≤f(θ)−f_*` when the optimum is active. Convexity of f plus KKT is sufficient for that saddle condition.
- **Dependencies:** M12, M19.
- **Experimental consequence:** Do not estimate this ratio in an MLP using a zero ν or an unjustified presumed saddle point.

### M18 — Corollaries 12–13: sign and neighborhood

- **Location and original claim:** M779–836/F775–832 asserts `0≤Ω(θ*)−Ω(θ)` for all feasible θ despite local SC/QG assumptions.
- **Diagnosis:** ED/HA.
- **Counterexample:** `f(w)=w²/2`, Ω=2|w|, τ=2, w*=0. For feasible w=1, Ω(w*)−Ω(w)=−2. QG and SC are global here, so the sign is the actual problem. A local hypothesis also does not permit a conclusion outside its domain.
- **Correction:** Absorb both into `thm:qg` and `cor:sc`, with an absolute value, a well-defined projection, and θ inside U (and feasible for constrained minima). Nonnegativity requires, for example, an active optimum and feasible θ.
- **Dependencies:** M03–M06.
- **Experimental consequence:** Record the distance/reference and domain; do not treat a budget as automatic justification of the sign.

### M19 — Corollary 14: penalty and violation

- **Location and original claim:** M838–856/F834–852: for every λ>0, violation ≤(F_λ−F_λ*)/λ. The proof presupposes a feasible minimizer and `f(θ)−f(θ*)≥0` even at infeasible points.
- **Diagnosis:** ED, including for convex problems and a penalty large enough to be exact.
- **Requested counterexample, verified:** `f(w)=(w−2)²/2`, Ω=2|w|, τ=2, λ=1. The penalized minimizer is w*=1, F*=1/2. At w=3/2, F=9/8, the gap is 5/8, and violation is 1. Thus `1≤5/8` is false. Furthermore, with λ=1/10, the minimizer w*=9/5 is infeasible and its violation 8/5 cannot be bounded by a zero gap.
- **Correction:** `prop:penalty`: (a) if f≥a, `λ(Ω−τ)+≤F_λ−a`; (b) if `(θ*,ν*)` is a GLOBAL saddle point, λ>ν* gives `(λ−ν*)(Ω−τ)+≤F_λ−f_*`, with F_λ*=f_* and feasible penalized minimizers. Prove both; include ν*=0. λ=ν* is insufficient to conclude that every penalized minimizer is feasible.
- **Dependencies:** M12; there is no automatic promise for a nonconvex MLP.
- **Experimental consequence:** Do not infer ADMM feasibility from a one-sided function that ADMM does not minimize.

### M20 — Smoothness in ReLU networks

- **Location and original claim:** M671/F667 and M713/F709 use L-smoothness; M871/F867 uses ReLU; M931/F927 uses a Hessian for local hypotheses.
- **Diagnosis:** HA/IE.
- **Argument:** For datum x=1 and objective `f(w)=(ReLU(w)−1)²`, the left derivative at zero is 0 and the right derivative is −2. Thus neither differentiability nor a globally Lipschitz gradient follows from using ReLU. Almost-everywhere differentiability does not repair this jump.
- **Correction:** Smooth stationarity applies only when ∇f exists and is L-Lipschitz on the relevant region, or along the segment between compared parameters. A stable activation pattern and bounded derivatives on a compact region may justify a local constant; it is not assumed automatically.
- **Dependencies:** M12, M23.
- **Experimental consequence:** A computable residual is not a numerical stationarity bound unless L and smoothness are certified; an arbitrary autodiff subgradient does not substitute for the smooth theorem.

### M21 — Truncated SVD and inexact projection

- **Location and original claim:** M537/F533 and M858–860/F854–856 assert the use of truncated SVD/heuristics alongside exact projection in the convergence claim.
- **Diagnosis:** HA/MI. A truncated SVD can be exact for the projection only if every omitted singular value is certified to lie below the correct threshold and the threshold is determined without omitting active mass.
- **Argument:** Truncating Y to Yhat and projecting Yhat is generally not equivalent to projecting Y. If `||Y−Yhat||_F≤δ`, nonexpansiveness guarantees `||Π_C(Y)−Π_C(Yhat)||_F≤δ`; this is an approximation error, not equality. The output Π_C(Yhat) remains feasible if projected exactly, but it is not optimal for Y.
- **Correction:** Use full SVDs in the primary specification; approximate variants record the tail/error and check feasibility/optimality. If `||Ztilde−Π_C(Y)||≤ε`, violation ≤2√D ε; position error alone does not certify a normal-cone residual.
- **Dependencies:** M08, M15, M22–M23.
- **Experimental consequence:** E0 rejects hidden SVD failures; approximations are identified ablations and are not called “exact” on the basis of an informal tolerance.

### M22 — Candidate B: a finite-iteration feasibility certificate

- **Location and original claim:** A necessary replacement for Proposition 9, M670–708/F666–704, and the algorithm diagnostic at M661/F657.
- **Diagnosis:** An additional proved result, an elementary consequence of M04 and the 1-Lipschitz property of the positive part.
- **Proof:** `vτ(W)≤vτ(Z)+|Ω(W)−Ω(Z)|≤vτ(Z)+2Σ||W_l−Z_l||_*≤vτ(Z)+2√R(W,Z)||W−Z||_F≤vτ(Z)+2√D||W−Z||_F`. With feasible Z, vτ(Z) vanishes; convergence is unnecessary.
- **Correction:** `prop:feasibility` is valid at every iteration and for each evaluated τ. The nuclear-residual version is usually tighter; R requires the true rank or a conservative upper bound; D depends only on the architecture.
- **Dependencies:** M04; smoothness and convexity of the loss are unnecessary.
- **Experimental consequence:** E3 compares raw violation and the three bounds for the current and final τ; call a floating-point evaluation a “numerical certificate with tolerance,” not a computer-assisted proof with enclosed rounding error.

### M23 — Candidate C: a posteriori stationarity including biases

- **Location and original claim:** M536/F532 promises a stationarity residual without developing it; this replaces part of Proposition 9.
- **Diagnosis:** An additional proved result, conditional on smooth f, rho fixed within the iteration, and exact projection.
- **Proof:** At θ+=(W+,b+), zbar=(Z+,b+), let `e=(∇W f(θ+)+ρ(W+−Zold+uold), ∇b f(θ+))`, `s=(ρ(Z+−Zold),0)`, and `r=W+−Z+`. Exact projection gives `n=(ρunew,0)∈N_(Cτ×R^B)(zbar)`. Then `∇f(θ+)+n=e−s`; Lipschitz continuity gives `dist(0,∇f(zbar)+N(zbar))≤||e||+||s||+L||r||`.
- **Correction:** `prop:stationarity`; evaluate the normal cone at Z+, never at possibly infeasible W+. For feasible Ztilde with normal-cone defect η, add η; for an approximate gradient with error δ, add δ. If Ztilde is infeasible, its normal cone is empty and a different feasible point is required.
- **Dependencies:** M08, M12, M14, M20–M21.
- **Experimental consequence:** Calculate e using the full gradient of the relevant objective; a minibatch gradient is only an estimator unless an explicit error bound is provided. The bound implies neither asymptotic convergence nor global optimality.

### M24 — Candidate D: dependence on parametrization

- **Location and original claim:** M118/F114, M127/F123, M480/F476, and the conclusions interpret the descriptor as network/function complexity without discussing ReLU symmetry.
- **Diagnosis:** An additional proved result and an interpretive limitation.
- **Proof:** For h(x)=W_2 ReLU(W_1x), c>0, `c^−1W_2 ReLU(cW_1x)=h(x)`, but `Ω(c)=2(c||W_1||_*+c^−1||W_2||_*)` generally changes. Scalar example W1=W2=1: Ω(1)=4 and Ω(2)=5, with identical predictions. For a,b>0, the minimum over the orbit is `4√(ab)` at `c=√(b/a)`.
- **Correction:** Explain that Ω measures a parametrization and is not a function invariant. WD and other regularizers can break equality of objectives even when f_data is unchanged.
- **Dependencies:** M01–M02.
- **Experimental consequence:** E5 verifies predictions and Ω and separately reports changes in the regularized objective.

### M25 — Historical SC/QG verification and a ratio greater than one

- **Location and original claim:** M925/F921 defines μ by the smallest positive Hessian eigenvalue; M931–957/F927–953 declares SC valid in a trial and calls a ratio of approximately 4001 “vacuous”; M1180/F1176, M1199/F1195, and M1220/F1216 present this as validation.
- **Diagnosis:** ED/IE in the interpretation; historical evidence remains pending an audit of the calculation/reference.
- **Counterargument:** H=diag(1,0) has smallest positive eigenvalue 1 and no positive full-space SC constant. H=diag(1,−1) does not even describe a minimum. A Hessian at one point does not certify the region traversed by SGD. If numerator/bound>1, the computed inequality fails; an excessively large, vacuous bound would give a small ratio, not 4001.
- **Correction:** Invalidate that theoretical validation and retain its data only as historical observations pending identification of the error. Use E1 with known H≻0 and singular H⪰0, explicit minima, and explicit projections. For a PSD quadratic, λ_min^+(H) does certify QG relative to the null directions ker(H), not SC.
- **Dependencies:** M04, M06, M20.
- **Experimental consequence:** Do not clip negative gaps, do not estimate MLP constants by discarding zero/negative eigenvalues, and investigate every ratio outside tolerance.

### M26 — Inferences and consistency of historical results

- **Location and original claim:** M94/F90, M991–998/F987–994, and M1177/F1173 assert convergence/monotonicity/exact zero; M1021–1038/F1017–1034 omits WD feasibility; M1110/F1106 cites 86.9±5.7 versus 88.21±5.43 in the table; M1223/F1219 calls 20 runs independent; M1271/F1267 retains an old description of six experiments and 3 seeds.
- **Diagnosis:** IE/MI; textual discrepancies are verified, while run independence and metrics require comparison against files/code.
- **Argument:** Twenty-five epochs and small residuals do not prove a limit; a rounded 0.0000 does not prove exact zero. WD means are below the displayed mean budgets, so omitting its feasibility prevents a claim that ADMM uniquely achieves feasibility. Exact per-seed counts cannot be inferred from means. Depth also changes widths and does not permit attributing causality solely to depth.
- **Correction:** Classify historical material as reusable only after audit, pending, or supporting invalidated interpretations. Remove asymptotic promises, literal exactness, and untested causality. Refer code/CSV traceability to the separate experimental evidence audit, translated here as `HISTORICAL_CODE_AUDIT.md`.
- **Dependencies:** M14–M15, M22, M25.
- **Experimental consequence:** E2–E4 include feasibility of all methods, a unique seed/run identity, paired references, tolerances, and multiple seeds for ablations; do not count duplicates as replicates.

### M27 — Effective rank, SN, and compression

- **Location and original claim:** M1116–1120/F1112–1116 defines entropy effective rank without the zero case; M1148/F1144, M1157/F1153, and M1202/F1198 attribute spectral “flattening” to division by σmax. The table at M1139–1141/F1135–1137 shows mean SN σmax of 1.59–2.15, incompatible with the declared exact normalization if the same effective weights were measured.
- **Diagnosis:** ED for the instantaneous effect of rescaling; HA for zero; MI for the SN metric; IE for compression.
- **Proof:** If c>0 and Y≠0, `cσ_i/Σcσ_j=σ_i/Σσ_j`; all proportions and entropy effective rank remain identical. For Y≠0, `1≤r_eff≤rank(Y)≤d`; the normalized ratio lies in `[1/d,1]`, and zero requires an additional convention. Changes during training can arise from the dynamics, not from the simple act of scaling.
- **Correction:** Define r_eff(0)=0 as an explicit convention; remove the false explanation and suspend SN conclusions until effective weights are measured. A lower effective rank does not prove savings in storage, FLOPs, or memory.
- **Dependencies:** M09–M10, M24.
- **Experimental consequence:** E0 inspects SN; E4 inspects spectra; E6 promises compression only if it performs factorization/truncation and measures representation cost and predictive loss.

### M28 — References, originality, and contribution scope

- **Location and original claim:** M118–143/F114–139 announces an unexplored area and ADMM guarantees; M535/F531 generically cites KL/PL for ADMM; M1414–1417 and M1488–1495 duplicate *Graph Energy* with different authors; M1444–1450 uses a Reddi key for a Huang–Chen–Lu entry.
- **Diagnosis:** Bibliographic IE/MI. Matching labels do not verify authorship or applicability; an internal historical citation key alone does not establish that a reference is false.
- **Argument:** The bipartite identity, nuclear norm, spectral/nuclear projection, KKT, and ADMM are known tools. The new results in this draft are proofs of consequences and their organization for this budget; the audit does not establish bibliographic priority. Prior work by Alvarez–Salzmann and Zimmer–Spiegel–Pokutta prevents presenting nuclear-norm training/compression as entirely new.
- **Correction:** Limit the contribution to the quantitative graph–surrogate relationship, a consistent formulation, local bounds with hypotheses, finite-iteration certificates, and a reproducible protocol. Primary-source verification and the numbering of invoked theorems must be recorded in the project's bibliography register; do not transfer a cited nonconvex theorem to the code without verified hypotheses.
- **Dependencies:** All retained results, independent bibliographic review, and the official RAM guidance.
- **Experimental consequence:** Do not design the evaluation to confirm superiority; technical validity is independent of which method wins.

## Complete mapping of original results 1–14

The numbering is the **article's internal numbering**, shared by `lemma`, `theorem`, `proposition`, and `corollary`; these are not fourteen lemmas. `Assumption` and `Remark` have separate counters. Physical pages in a package are not this numbering. Labels are used so traceability survives final layout.

| Original number and type | Original content and label | Location M / F | Decision / revised destination | IDs |
|---|---|---|---|---|
| 1, Lemma | Nuclear–Frobenius, `lem:nuclear-fro` | 236–241 / 232–237 | Absorbed, with proof, into `lem:perturbation`; a known inequality | M04 |
| 2, Lemma | Layerwise Frobenius vectorization, `lem:fro-stack` | 243–248 / 239–244 | Product-norm definition and inequality including biases; used in `lem:perturbation` | M01, M03–M04 |
| 3, Lemma | Cauchy–Schwarz across layers, `lem:cs-layers` | 250–255 / 246–251 | Absorbed into the proof of `lem:perturbation`; not a new independent result | M04 |
| 4, Theorem | Surrogate–loss bound under SC, `thm:main` | 257–305 / 253–301 | Unified in `thm:qg` and its special case `cor:sc` | M03–M05 |
| 5, Theorem | Bound relative to the optimum set under QG, `thm:qg` | 356–410 / 352–406 | Retained and corrected in `thm:qg`, including Ω, biases, domain, R, and D | M03–M06 |
| 6, Lemma | Existence and projection, `lem:proj-exists` | 456–470 / 452–466 | Replaced by `lem:existence`, with a lower bound and a selection that need not be unique | M06 |
| 7, Lemma | Bipartite energy = twice the nuclear norm, `lem:bipartita-nuclear` | 500–514 / 496–510 | Integrated into `prop:graph`, with the full two-sided bound | M01–M02 |
| 8, Lemma | Nuclear/spectral projection, `lem:proj-opt` | 627–644 / 623–640 | Split into `prop:projection` and `prop:spectral`; clipping corrected | M08–M09, M21 |
| 9, Proposition | Primal residual and ADMM convergence, `prop:residuo` | 670–708 / 666–704 | Removed; replaced with results of different scope: `prop:feasibility` and `prop:stationarity` | M14–M15, M20–M23 |
| 10, Theorem | Local linear PL convergence, `thm:pl` | 712–743 / 708–739 | Removed; neither the rate nor an application to ADMM is retained | M16 |
| 11, Corollary | Energy gap / multiplier, `cor:energy_loss_linear` | 749–777 / 745–773 | Removed in its stated generality; version with a global saddle condition and ν*>0 in the discussion of `prop:penalty` | M17, M19 |
| 12, Corollary | Rank-based gap under SC, `cor:energy_loss_rank_SC` | 779–807 / 775–803 | Absorbed into `cor:sc`, with absolute value and correct domain | M18 |
| 13, Corollary | Gap under constrained QG, `cor:energy_loss_QG` | 809–836 / 805–832 | Absorbed as an explicit constrained case of `thm:qg` | M18 |
| 14, Corollary | Violation / penalized gap, `cor:violation_penalty` | 838–856 / 834–852 | Removed and replaced by `prop:penalty`, with correct hypotheses and denominators | M19 |

The justified new developments are the two-sided comparison (`prop:graph`, candidate A), finite-iteration feasibility (`prop:feasibility`, B), a posteriori stationarity (`prop:stationarity`, C), and ReLU symmetry/parametrization discussion (D). `prop:kkt` is added to fully express conditions that previously appeared only as formulas without a numbered result. No claim of bibliographic novelty is made for these consequences.

The revised numbering was checked against `manuscrito_RAM.aux` generated on 2026-10-13. The pages below are internal pages of that revised article; they are historical layout coordinates, while the labels remain the authoritative lookup in the bundled sources.

| Revised result | Label | Internal page |
|---|---|---:|
| Proposition 1, block identity and two-sided comparison | `prop:graph` | 2 |
| Lemma 2, perturbation estimate | `lem:perturbation` | 3 |
| Lemma 3, existence and projection | `lem:existence` | 3 |
| Theorem 4, QG stability | `thm:qg` | 3 |
| Corollary 5, SC case | `cor:sc` | 4 |
| Proposition 6, joint projection | `prop:projection` | 5 |
| Proposition 7, spectral projection and scaling | `prop:spectral` | 6 |
| Proposition 8, necessary KKT conditions | `prop:kkt` | 6 |
| Proposition 9, finite-iteration feasibility | `prop:feasibility` | 7 |
| Proposition 10, a posteriori stationarity | `prop:stationarity` | 7 |
| Proposition 11, valid penalty bound | `prop:penalty` | 8 |

Sharing a number with an original result does not mean its content is retained: the new Proposition 9 does not assert convergence, and the new Proposition 10 is not the removed PL theorem.

## Independent review after integration

The revised draft's `sections/01_foundations.tex`, `sections/02_optimization.tex`, and `sections/03_evidence.tex` were read in full. Their copies in this handoff are under `manuscript/sections/`. Particular checks covered the least-squares projection `θ+=θ−B†(Bθ−d)` (also for an inconsistent system), the approximate-projection objective-excess bound `dε+ε²/2`, the normal-cone conditions, biases in the stationarity bound, and the global penalty inequality. The formulas and proofs are valid under their stated hypotheses. LaTeX presentation details, terminology for the objective f, and the R=0 exception to the description of nontrivial equality were communicated and corrected.

The mathematical-check script and its execution record were inspected in full. In the original revision these were named `comprobaciones/verificar_matematica.py` and `comprobaciones/ejecucion.log`; these are historical provenance names. This handoff provides the script as `mathematical_checks/verify_mathematics.py` and the recorded numerical results as `mathematical_checks/recorded_results.json`. The historical record reports 227 passed checks, seed 20261013, NumPy 1.26.4, Python 3.11.8, float64, and zero training runs. The script does not import the notebook; the suite includes projections, graph/surrogate comparisons, QG, residuals, penalties, ReLU rescaling, and counterexamples. Checking code and recorded outputs does not turn randomized checks into proofs; the proofs are in the bundled revised manuscript. This paragraph reports the recorded run, not a new execution on the recipient's hardware.

## Audit conclusions

The revision retains the norm inequalities, block identity, correctly localized QG/SC stability, existence under justified coercivity, joint nuclear-norm projection, and necessary KKT conditions. It corrects spectral projection, interpretations of SN/WD, biases, ranks, boundary cases, and penalties. It removes unproved nonconvex ADMM convergence, a PL rate based only on summable errors, corollaries with incorrect sign/scope, and historical SC validation based on a positive eigenvalue.

The user's review agenda was largely correct, but this audit acknowledges what the original already did correctly: the block/full-graph distinction appears in its Section 2; the warning about varying Ω on the optimum set is present; the sentence before Theorem 10 clarifies that it is not directly ADMM; and Lemmas 1–3 and the QG estimate are essentially valid once the hypotheses are fixed. It also clarifies that compactness of S is sufficient, but not necessary, for existence of a projection in finite dimensions.

This mathematical audit does not declare the historical experimental suite verified or the article ready for submission. The implementation, CSV files, primary bibliography, and final compilation have separate records; no pending result becomes executed evidence here. Use `HISTORICAL_CODE_AUDIT.md` for the complete translated static code findings and `CLAIMS_AND_EVIDENCE_MATRIX.md` to track what new evidence is still needed.
