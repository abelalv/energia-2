# Mathematical implementation specification

Protocol version: RAM-2026-10-13-v1. Intended reader: the colleague responsible for implementing and running the Linux/CUDA experiments. This specification defines pending work; no training has been run and the historical notebook has not been modified in this revision. The audited code reference is `cero1979/Ensayos-Energy-Graph`, commit `186e27400f94d551432759fa4830e0c114a2aa69`. Also read `EXPERIMENTAL_PROTOCOL.md` and `HISTORICAL_CODE_AUDIT.md`.

## 1. Objects the code must represent

The network has $m$ matrices $W_\ell\in\mathbb R^{n_\ell\times n_{\ell-1}}$, biases $b_\ell\in\mathbb R^{n_\ell}$, ReLU hidden activations, and a linear output. In PyTorch, `Linear.weight` already has this orientation. The norm of the full space is

$$
\|\theta\|^2=\sum_\ell\|W_\ell\|_F^2+\sum_\ell\|b_\ell\|_2^2,\qquad\theta=(W,b).
$$

Weights may be signed. The adjacency matrix $A(W)$ is symmetric, with consecutive blocks $A_{\ell,\ell-1}=W_\ell$, $A_{\ell-1,\ell}=W_\ell^T$, zero diagonal blocks and zero other blocks; it neither incorporates biases nor takes entrywise absolute values. Distinguish:

$$
E(A)=\|A\|_*,\quad E(B_\ell)=2\|W_\ell\|_*,\quad
\Omega(W)=2\sum_\ell\|W_\ell\|_*.
$$

For this consecutive-layer architecture, the two-sided relationship retained in the revision is $\Omega/2\le E(A)\le\Omega$. This is neither an identity nor a measure of electricity consumption, FLOPs, memory, or the intrinsic complexity of the function. Define $D=\sum_\ell d_\ell$, $d_\ell=\min(n_\ell,n_{\ell-1})$. For the original 3L, 6L, and 9L architectures, $D=394,554,730$.

The primary data objective is the mean training cross-entropy over all examples, $f_{\rm data}(W,b)$. Do not confuse it with $f_{\rm WD}=f_{\rm data}+(\lambda_{\rm WD}/2)\|\theta\|^2$, with $f_{\rm NUC}=f_{\rm data}+\lambda_{\rm NUC}\sum_\ell\|W_\ell\|_*=f_{\rm data}+(\lambda_{\rm NUC}/2)\Omega$, or with a violation penalty. Every loss record must carry an `objective_id` and separate components. The optimizer does not add WD by default to STD, NUC, SN, ADMM, or PSGD-M.

## 2. Joint nuclear projection, exact in real arithmetic

The correct set is $C_\tau=\{W:2\sum_\ell\|W_\ell\|_*\le\tau\}$, with $\tau\ge0$. Solve

$$
\min_{Z\in C_\tau}\frac12\sum_\ell\|Z_\ell-Y_\ell\|_F^2.
$$

Orthogonal invariance and von Neumann's trace inequality allow the singular vectors to be retained. Let the SVDs include every singular value (economy SVD without truncating the tail): $Y_\ell=Q_\ell\operatorname{diag}(\sigma_\ell)V_\ell^T$, $\sigma_{\ell i}\ge0$, and let $a$ be their concatenation. It suffices to project $a$ onto $\{z\ge0:\sum_i z_i\le\tau/2\}$. The matrix solution is unique by strict convexity of the distance objective, even when the SVD and a multiplier are not unique.

1. If $\sum_i a_i\le\tau/2$, return **the input**, with threshold $t=0$; do not reconstruct unnecessarily.
2. If $\tau=0$, return all-zero matrices. Any $t\ge\max_i a_i$ reproduces this output; by convention, set $t=\max_i a_i$, or $0$ if all values are zero. The threshold is not unique in this case.
3. If $0<\tau/2<\sum_i a_i$, there is a unique $t\in(0,\max_i a_i)$ such that $h(t)=\sum_i(a_i-t)_+=\tau/2$. $h$ is continuous, piecewise linear, and strictly decreasing on $[0,\max a_i]$ when $\max a_i>0$; it is identically zero to the right of that interval. Return $Z_\ell=Q_\ell\operatorname{diag}((\sigma_\ell-t)_+)V_\ell^T$.

Zero matrices contribute zero singular values. Multiplicities do not invalidate the uniqueness of $Z$: apply the same scalar function to the entire repeated singular subspace. A layer becomes zero under the joint projection if $\sigma_{\max}(Y_\ell)\le t$. In contrast, the projection of a nonzero matrix onto its own nuclear-norm ball with a strictly positive radius is nonzero.

Reference threshold procedure: sort $a_{(1)}\ge\cdots\ge a_{(p)}$, set $c_j=\sum_{i\le j}a_{(i)}$, $j_*=\max\{j:a_{(j)}>(c_j-\tau/2)/j\}$, and $t=(c_{j_*}-\tau/2)/j_*$. Handle the interior and zero-radius cases first. Bisection is an alternative: maintain $h(t_{\rm lo})>\tau/2$, $h(t_{\rm hi})\le\tau/2$, and return the feasible endpoint `hi`, documenting its tolerance. An absolute threshold tolerance alone is insufficient to establish energy accuracy when there are many singular values.

If the projection objective is $(\rho/2)\|Z-Y\|^2$ and the multiplier term $\nu(2\sum\|Z_\ell\|_*-\tau)$ is added, the scalar KKT condition gives **$t=2\nu/\rho$**. For the objective with coefficient $1/2$, $t=2\nu$. If the constraint is written without the factor 2 and with radius $\tau/2$, its multiplier is $\rho t$. Save `threshold_t` and the convention; never interchange these three quantities.

Do not use $U$ for both singular vectors and the dual variable: reserve $Q,V$ for the SVD and $u$ for the scaled dual variable.

### Numerical implementation

Primary training uses float32 without AMP, with TF32 disabled. Projection: copy $Y$ to float64 on the declared device, compute the SVD with every singular value, compute the threshold with float64 accumulation, reconstruct, and convert to the training dtype. Remeasure the matrix actually returned in float64. This implementation is a floating-point approximation to the exact projection, not a proof using rigorous rounding enclosures. If the float32 copy exceeds the feasibility tolerance, repeat the float64 computation once without noise, using a more conservative feasible endpoint; record the correction and the distance from the first output. If the violation persists, mark a failure and do not claim exact projection.

Primary feasibility tolerance, fixed before viewing results:

$$
v_{\rm raw}=\max(\Omega(X)-\tau,0),\quad
\varepsilon_{\rm feas}(\tau)=10^{-6}+10^{-5}\max(1,|\tau|),\quad
\texttt{feasible}=v_{\rm raw}\le\varepsilon_{\rm feas}(\tau).
$$

Also save `signed_slack=Omega-tau`, $v_{\rm raw}/\max(1,|\tau|)$, and the dtype. In float64 E0, use stricter tolerances: feasibility $10^{-10}+10^{-10}\max(1,\tau)$, idempotence and solution comparisons $10^{-9}\max(1,\|Y\|_F)$. A zero radius must return exact zeros. These are numerical tolerances; never write “exactly zero violation” on the basis of a rounded CSV.

Truncated SVD is not permitted in the primary experiment suite. In a future ablation, report $\delta=\|\widetilde P(Y)-P(Y)\|_F$, the feasibility of $\widetilde P(Y)$, the omitted tail, and the projection objective gap. From $\delta$, one obtains $(\Omega(\widetilde P)-\tau)_+\le2\sqrt D\,\delta$, but an unknown tail does not provide $\delta$. Verify optimality through a gap/distance check, not only feasibility; many feasible points are not the projection.

## 3. Distinct spectral operations

The Frobenius projection onto $\|Z\|_2\le a$, $a\ge0$, is $Q\operatorname{diag}(\min(\sigma_i,a))V^T$. It is not radial rescaling $\min(1,a/\sigma_{\max})Y$. For $Y=\operatorname{diag}(2,1)$, $a=1$, the squared distances are $1$ and $5/4$, respectively. For $Y=0$, return zero without dividing.

SN parametrizes an original weight $T$ through an effective weight $W_{\rm eff}=T/\widehat\sigma(T)$, usually using power iteration for the estimate. SN with power iteration does not enforce an exact spectral ball. For the baseline, fix `n_power_iterations=1`, `eps=1e-12`; save both, the power-iteration vectors, and the version. For evaluation: set the network to `eval`, capture the **effective `module.weight`** for each linear module in an immutable copy, copy the biases, evaluate predictions using that same copy, and measure its spectrum. Do not use `named_parameters()` to measure SN: it exposes the original parameters. Measure the actual excess of $\|W_{\rm eff}\|_2$ over 1. The official [PyTorch 2.8 documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.parametrizations.spectral_norm.html) confirms the parametrization and power-iteration estimate; the new version must be checked when creating the environment.

For a nonzero matrix, $p_i=\sigma_i/\sum_j\sigma_j$, $r_{\rm eff}=\exp(-\sum_{p_i>0}p_i\log p_i)$. Declared convention for an exactly zero matrix: $r_{\rm eff}=0$. Save the spectrum without thresholding; numerical rank is a different metric with an explicit tolerance. Uniform multiplication by a positive scalar preserves $p_i$ and $r_{\rm eff}$. Do not attribute spectral “flattening” to rescaling alone.

The bounds $\|W_\ell\|_*\le d_\ell\|W_\ell\|_2$ and $\|W_\ell\|_*\le\sqrt{d_\ell}\|W_\ell\|_F$ do provide dimension-dependent control of $\Omega$. They do not, by themselves, guarantee the particular $\tau$ used in the comparison. WD is a penalty, not automatic enforcement of a predetermined Frobenius bound.

## 4. Mathematical ADMM and the experimental variant

Split problem: minimize $f(W,b)+I_{C_\tau}(Z)$, subject to $W=Z$, with unconstrained biases. The scaled Lagrangian is

$$
\mathcal L_\rho=f(W,b)+I_C(Z)+\frac\rho2\|W-Z+u\|_F^2-\frac\rho2\|u\|_F^2.
$$

Exact ADMM, with fixed $\rho,\tau$:

```text
(W_new,b_new) = argmin f(W,b) + (rho/2)||W-Z_old+u_old||_F^2
Z_new = P_Ctau(W_new+u_old)
r = W_new-Z_new
s = rho*(Z_new-Z_old)
u_new = u_old+r
```

Consensus is not imposed on biases; they are optimized in the $W,b$ step. The $W,b$ problem may be nonconvex, and the `argmin` notation requires a solution; it does not automatically represent an SGD pass. When only $u$ is updated, with the other blocks fixed, $\mathcal L_\rho(u+r)-\mathcal L_\rho(u)=\rho\|r\|_F^2$; therefore, do not infer telescoping descent from blockwise descent alone.

The experiment suite uses **inexact stochastic ADMM with momentum**, whose $W,b$ step is one epoch of minibatch SGD on the preceding objective, with $Z,u,\rho$ frozen throughout the epoch. The revision contains no general convergence guarantee for this ReLU variant with momentum, adaptive $\rho$, and a changing budget.

### Unambiguous implementation order and schedule

Epoch indices $e=1,\ldots,25$. Constant LR 0.01; batch size 64; momentum 0.9; dampening 0; Nesterov false; initial biases and matrices paired with STD. Initialization uses the standard `nn.Linear.reset_parameters` behavior of the pinned version and saves the initial tensors. Momentum is preserved between epochs.

- Epochs 1–3: data training without an ADMM term or projection; `phase=warmup`, current budget `null`, but record violation relative to the final $\tau$.
- At the end of epoch 3: save $\tau_0=\max(\Omega(W^3),\tau_{\rm final})$, set $Z=W^3$, $u=0$, $\rho=1$. Freeze $\tau_0$; do not recompute it in each transition epoch.
- Epochs 4–8: $\tau_e=(1-j/5)\tau_0+(j/5)\tau_{\rm final}$, $j=e-3$. `phase=transition`. The target is reached exactly in epoch 8.
- Epochs 9–25: $\tau_e=\tau_{\rm final}$, `phase=fixed`.
- In each active epoch: save `rho_used`, `u_old`, `Z_old`; perform an SGD pass on the data objective plus the augmented term; project $W+u_{old}$ with that epoch's budget; compute $r,s$ using `rho_used`; update $u\leftarrow u_{old}+r$. Do not overwrite $W\leftarrow Z$.
- After the preceding steps, adapt $\rho$: if $\|r\|>10\|s\|$, the candidate is $2\rho$; if $\|s\|>10\|r\|$, the candidate is $\rho/2$; otherwise, retain it. Clip the candidate to $[0.1,10]$. **Always apply $u\leftarrow(\rho_{old}/\rho_{new})u$** using the actual values after clipping, preserving $y=\rho u$. If both residuals are zero, change nothing. Save `rho_next` and `dual_rescale_factor`. Do not blindly multiply by 2 when a limit is reached.
- Measure the three network variants and save a checkpoint. A checkpoint for resumption contains the state before/after adaptation, all RNG states, sampler state, and momentum. An end-of-epoch checkpoint is sufficient to resume from that boundary.

Changing $\tau$ changes the problem; separate the phases in figures. Adapting $\rho$ preserves $y$, not a proven Lyapunov function. The procedure does not use the incorrect expression $f+\nu g+(\rho/2)g^2$ as the augmented Lagrangian of an inequality.

## 5. Projected baseline and layerwise allocation

Primary baseline **PSGD-M**: SGD with momentum 0.9, the same LR, batch size, and budget schedule; after **every minibatch** in active epochs, replace $W\leftarrow P_{C_{\tau_e}}(W)$; preserve biases and the momentum buffer. Do not project the buffer or silently clear it. There is no projection during warmup. Apply an initial projection at the beginning of every active epoch so that the point preceding the first gradient belongs to the new set, and count it. Thus, this is projected SGD with momentum, not deterministic GD or SGD without momentum. Label `projection_frequency=minibatch_plus_epoch_entry`; feasibility between steps means after each projection, not while computing a candidate.

Distinct optional variants: PSGD uses momentum 0; PGD uses the full gradient with a declared step size. Neither inherits the other's guarantees. Projecting only once per epoch is another variant and does not replace the primary baseline.

E4 changes only the ADMM $Z$ step: $C_{\boldsymbol r}=\prod_\ell\{Z_\ell:\|Z_\ell\|_*\le r_\ell\}$, where $\sum r_\ell=\tau_e/2$. Allocation by parameter count: $q_\ell=n_\ell n_{\ell-1}/\sum_j n_jn_{j-1}$; allocation by reference: $q_\ell=\|W^{STD,ref}_\ell\|_*/\sum_j\|W^{STD,ref}_j\|_*$. Set $r_\ell=q_\ell\tau_e/2$. Biases do not enter the count. Use weights from the same paired final STD model and freeze $q$ before training the ablation. If the reference is entirely zero, use $q_\ell=1/m$ and record this case. Layerwise radii may be zero. $C_{\boldsymbol r}\subseteq C_\tau$, generally strictly; this is a different, valid problem, not the same global projection. The optional uniform allocation is $q_\ell=1/m$.

## 6. Models and certificates at each checkpoint

**Prespecified primary model: $(W,b)$ from the last complete epoch**, without replacing it based on its result. Secondary models: $(Z,b)$ and $(P_{C_{\tau_{final}}}(W),b)$. During warmup, mark $Z$ as undefined until it is initialized; do not invent zero diagnostics. Save the cost and distance of the final projection. Freeze the copy: accuracy, loss, energy, biases, and spectra belong to the same `model_variant` and `weights_sha256`. Do not retrospectively choose W, Z, or P$W$ according to which favors ADMM. All methods, including WD and STD, are evaluated against the three final budgets; non-ADMM methods have no fictitious Z.

### Feasibility

For any $W,Z$, with $R_{WZ}=\sum_\ell\operatorname{rank}(W_\ell-Z_\ell)$,

$$
(\Omega(W)-\tau)_+\le(\Omega(Z)-\tau)_++2\sum_\ell\|W_\ell-Z_\ell\|_*
\le(\Omega(Z)-\tau)_++2\sqrt{R_{WZ}}\|W-Z\|_F
\le(\Omega(Z)-\tau)_++2\sqrt D\|W-Z\|_F.
$$

Save the nuclear and dimension-based bounds and their difference from the direct violation, all with the same current or final $\tau$, clearly labeled. Thresholded numerical rank is not a safe substitute for $R_{WZ}$. The bound using $D$ is conservative and operational. If Z satisfies the budget, the bound reduces to $2\sqrt D\|r\|$. Do not confuse feasibility of Z with feasibility of W.

### Smooth stationarity, conditional scope

Define $\mathcal C=C_\tau\times\mathbb R^{\dim b}$. Within the iteration, with fixed $\rho$, exact projection, and $f$ having an $L$-Lipschitz gradient on a region containing the points used, take $x=(W^{k+1},b^{k+1})$, $z=(Z^{k+1},b^{k+1})$,

$$
e=\big(\nabla_W f(x)+\rho(W^{k+1}-Z^k+u^k),\ \nabla_b f(x)\big),\quad
s=\rho(Z^{k+1}-Z^k),\quad r=W^{k+1}-Z^{k+1}.
$$

By the projection, $y^{k+1}=\rho(u^k+r)\in N_{C_\tau}(Z^{k+1})$; the cone is evaluated **at Z**, not W. Moreover, $e_W=\nabla_W f(x)+y^{k+1}+s$. Thus

$$
\operatorname{dist}(0,\nabla f(z)+N_{\mathcal C}(z))
\le\|e\|+\|s\|+L\|r\|.
$$

Do not omit $\nabla_b f$; do not replace the full training gradient with a minibatch gradient. Summing minibatch gradients weighted by their fraction of the total number of examples reproduces the full gradient; compute its norm afterward, rather than summing the norms. In E1, $L$ is known analytically. For ReLU MLPs, there is no global $L$ or automatic smoothness; save the full gradient and residuals as **diagnostics** at the selected checkpoints, but set `stationarity_certificate_applicable=false` and `bound=null`, unless a smooth region and constant are additionally proved. Differentiability almost everywhere is insufficient.

If the projection is inexact and Z is feasible, a defect $\eta=\|y-n_Z\|$ must be added for some actually certified $n_Z\in N_C(Z)$: the bound becomes $\|e\|+\|s\|+L\|r\|+\eta$. A projection-distance error alone does not justify this normal-cone term. If Z is infeasible, the normal cone of this set at Z is empty and this version does not apply. Distinguish an exact certificate, numerical evaluation with tolerance, and an uncertified diagnostic.

## 7. KKT and stability that can be checked

For differentiable $f$, a necessary condition for a feasible local minimum is $0\in\nabla_W f(W,b)+N_{C_\tau}(W)$, $\nabla_b f(W,b)=0$. For $\tau>0$, Slater's condition for the nuclear set allows the formulation $\nu\ge0$, $\nu(\Omega(W)-\tau)=0$, $\nabla_Wf+2\nu(G_\ell)_\ell=0$, where, for a compact SVD $W_\ell=Q_r\Sigma_rV_r^T$,

$$
G_\ell=Q_rV_r^T+H_\ell,\quad Q_r^TH_\ell=0,\ H_\ell V_r=0,\ \|H_\ell\|_2\le1.
$$

At rank zero, this becomes the entire spectral ball $\|G_\ell\|_2\le1$. For $\tau=0$, use the normal cone directly; do not invoke Slater's condition. Activity does not imply $\nu>0$, and KKT does not provide a global guarantee for the nonconvex MLP.

The perturbation bound is

$$
|\Omega(W)-\Omega(V)|\le2\sum\|W_\ell-V_\ell\|_*\le2\sqrt{R(W,V)}\|W-V\|_F\le2\sqrt D\|\theta-\vartheta\|.
$$

If a nonempty closed optimal set $S$ and a specified region satisfy $f(\theta)-f_*\ge(\gamma/2)\operatorname{dist}(\theta,S)^2$, then, for any projection $\theta^+\in P_S(\theta)$ and $\theta$ within that region,

$$
|\Omega(W)-\Omega(W^+)|\le2\sqrt{2R(W,W^+)/\gamma}\sqrt{f(\theta)-f_*}.
$$

Use $D$ if the exact rank is unknown. $\Omega(W^+)$ is not necessarily the lowest energy on S. SC is the special case with a unique minimizer and $\gamma=\mu$, when its assumptions are justified; an isolated positive eigenvalue of an MLP Hessian does not establish SC or local QG. A negative gap requires investigation of the reference/computation; never replace it with zero to hide it. An observed-value/upper-bound ratio above one fails the check; it does not mean that the “bound is vacuous.”

The existence of an optimum for $f_{data}+(\lambda/2)\|\theta\|^2$, $\lambda>0$, is justified if $f_{data}$ is continuous and bounded below and all parameters are regularized. A budget on W with unconstrained biases does not make the full $\theta$ space compact. Regularized, penalized, and data losses have different optima.

## 8. Pending interface contract

The following names are **interfaces pending implementation by the colleague**, not verified repository functions:

```text
project_joint(weights, tau, numerical_policy) -> weights_projected, projection_record
effective_weights(model, model_variant) -> immutable_weight_snapshot
run_one(config, resume=None) -> run_directory
evaluate_checkpoint(checkpoint, variants, split) -> metrics_records
validate_manifest(manifest) -> duplicate/conflict_report
```

`project_joint` must be pure with respect to its input and must not treat biases as budgeted matrices. The ADMM adapter performs assignments under `no_grad`; NUC preserves autograd along the training path. Moving a tensor to the CPU using `.to('cpu')` may preserve autograd; `detach`, NumPy, or conversion to float in the penalty breaks it. E0 must check the actual gradient, not infer it from a function's name. Do not execute the notebook on import: separate definitions from the execution entry point. CPU/MPS/CUDA profiles share the algorithm, configuration, and results format; a device limitation is recorded as a distinct numerical profile, never as a silent mathematical correction.
