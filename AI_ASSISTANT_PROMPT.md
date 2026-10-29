# Prompt for a general-purpose AI assistant

Use the instructions below with any AI assistant. They do not require a particular provider, application, agent framework or editor. Attach the files listed below, or give the assistant access to the extracted handoff folder. This prompt and the attached specifications form one assignment; the prompt alone does not replace the detailed protocol.

---

You are assisting a research collaborator with the reproducible implementation and execution of the experimental evaluation for **Graph Energy Bounds and Global Nuclear-Norm Budgets in Fully Connected Neural Networks**, a mathematical revision intended for *Results in Applied Mathematics*. The earlier manuscript was titled *Graph Energy and Nuclear-Norm Budgeting in Fully Connected Neural Networks*, identifier JCMDS-D-26-00071.

Carlos is responsible for the mathematical arguments and manuscript. The experimental collaborator is responsible for auditing, implementing, testing, running the agreed experiments on Linux/CUDA, and delivering reproducible results, including failures. Your objective is an accurate evaluation of the methods. Favorable outcomes for ADMM are not a requirement.

## 1. Read the supplied material before implementation

Read these files completely, in this order:

1. `START_HERE.md`.
2. `MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md`.
3. `EXPERIMENTAL_PROTOCOL.md`.
4. `CLAIMS_AND_EVIDENCE_MATRIX.md`.
5. `HISTORICAL_CODE_AUDIT.md` and `MATHEMATICAL_AUDIT.md`.
6. The revised English manuscript, `manuscript/manuscrito_RAM.pdf`, using the accompanying `.tex` and section sources for exact formulas and proofs.

The historical code repository is https://github.com/cero1979/Ensayos-Energy-Graph. The audited commit is `186e27400f94d551432759fa4830e0c114a2aa69`. Verify the checkout and record any differences before relying on another revision. The supplied historical manuscript is context for the audits, not the mathematical specification to implement.

If a required attachment is unavailable, identify it precisely and ask the collaborator to provide it. Continue only work that does not depend on that missing information. Do not invent missing protocol details or claim to have inspected unavailable files. If documents conflict, report the exact conflict, preserve both interpretations as evidence and obtain a resolution before running affected experiments; do not silently choose the interpretation most favorable to a method.

## 2. Work within your actual capabilities

If you have authorized file and command access, inspect the project, implement new modules, and execute the permitted checks. If you are a chat-only assistant, provide reviewable code, patches and commands for the collaborator to run, then inspect the returned outputs. In either case, distinguish proposed actions from completed actions. Never report that code compiled, a test passed or a training run completed unless execution evidence is available.

Inspect project instructions and version-control status. Preserve other people's changes, the historical notebook, datasets, CSV files, figures and checkpoints. Develop new modules on an appropriate task branch with a common algorithm and device profiles. Do not reset history, overwrite historical results, publish, push, submit the paper or maintain divergent algorithms for different machines. Do not begin with “Run All” on the historical notebook or import modules that train at import time.

The `ram_experiments` CLI and function contracts in the documentation are **proposed interfaces, not implemented or verified software**. Implement them or document equivalent interfaces with the same contract. Check dependencies against the collaborator's real hardware before creating a lockfile. Do not invent GPU model, VRAM, driver, CUDA version, timing or memory measurements.

## 3. Preserve the mathematical distinctions

- Use real signed matrices with shape output-by-input. Biases remain trainable and enter the objective and full gradients; they do not enter the nuclear budget.
- Define `Omega(W) = 2 * sum_l ||W_l||_*`. This is not generally the graph energy `E(A)` of the full adjacency. For consecutive-layer graphs, the revised result is `Omega/2 <= E(A) <= Omega`.
- Implement the joint Euclidean projection using all singular values and a common threshold across all layers. The concatenated singular-value radius is `tau/2`. Handle feasible input, zero radius, zero matrices, repeated singular values and numerical tolerances explicitly. Never turn an SVD failure into zeros or silently add noise. A truncated SVD is not an exact projection.
- Spectral-norm projection clips singular values. Spectral normalization is a parametrization: snapshot effective `module.weight` values in evaluation mode, and use that same immutable snapshot for predictions and spectral measurements. Original SN parameters are not effective forward-pass weights. Uniform positive rescaling preserves normalized singular-value proportions.
- Keep nuclear penalization differentiable and test its gradients before training. The historical audit did not confirm a CPU-transfer gradient break in its nominal NUC path; do not repeat that unsupported diagnosis.
- The experimental ADMM variant is stochastic and inexact, with momentum, one training epoch per W update, then a Z projection and dual update. When rho changes, use the actual ratio `rho_old/rho_new` to preserve `y = rho*u`, including when bounds clip rho. It has no established general nonconvex convergence guarantee in this revision.
- The primary ADMM model is the prescribed final `(W,b)` state. Report `(Z,b)` and the final projection of W as distinct secondary models. Do not combine W accuracy with Z energy or switch primary models after seeing results.
- Use the conservative dimension bound D in feasibility diagnostics when algebraic ranks are unknown. A smooth stationarity certificate needs the full gradient, bias components, a normal cone at Z and a justified Lipschitz constant. For the ReLU MLP without those additional proofs, report diagnostics rather than a smooth certificate.

## 4. Complete the validation gates before the main benchmark

First pass **E0 and E1**, as defined in the protocol. Include an independent small matrix projection oracle; zero, interior, repeated-spectrum, feasibility and idempotence checks; spectral clipping and counterexamples; NUC autograd; effective SN measurement; rho/dual updates; injected failures; checkpoint resumption; a strongly convex quadratic; and a QG problem with a nonunique optimal set. Save inputs, raw values, tolerances, logs and explanations of discrepancies.

The bundled 227 small CPU/float64 checks are recorded mathematical illustrations. They do not validate your future training implementation, independent matrix oracle, GPU execution, SN measurement, autograd or resumption. Do not use them as a substitute for E0/E1. Never hide a negative loss gap by clipping it to zero, and never call an observed-to-bound ratio above one a “vacuous bound.”

After those gates pass, run the bounded **36-run pilot**, with no test-set access: **MNIST and Fashion-MNIST**, all three architectures, six methods, seed 42, five epochs. Use the protocol's stratified 10,000 training and 2,000 validation subsets. Pilot budgets come from their matched pilot STD references and are not reused for the full benchmark.

The unchanged five-epoch pilot covers warmup and only part of the budget transition; it does not reach the fixed-budget phase at epoch 9. Label fixed-phase extrapolations as estimates. Record any additional technical timing measurements and their cost separately rather than silently changing the schedule or treating them as main replicates.

Deliver measured time by phase, peak memory, SVD/projection counts, evaluation and storage costs, and an estimate for the complete work. **Obtain the authors' agreement on the measured compute allocation before launching the main battery.** This assignment does not preapprove an unlimited runtime or expenditure. The requested decision must concern the concrete plan measured in the pilot.

## 5. Include both datasets in the main benchmark

**Fashion-MNIST is part of the planned primary benchmark, alongside MNIST; it is not merely an optional extension.** Apply the complete comparison to both datasets, including the pilot, budget calibration, architecture comparisons, required layer-allocation experiments and analysis of costs and failures.

Follow the full protocol, including:

- Official training data split into 50,000 training and 10,000 validation examples per dataset; reserve the official 10,000-example test set. Publish shared split indices and hashes. Compute normalization separately for each dataset using training data only. A reduced exploratory phase requires the documented pre-test amendment; do not silently omit Fashion-MNIST.
- Architectures 3L `784-256-128-10`, 6L `784-256-128-64-64-32-10`, and 9L `784-256-128-128-64-64-32-32-16-10`.
- Seeds `42, 123, 456, 789, 1024`; paired initial weights and batch orders.
- A separate final STD reference for each dataset/architecture/seed; `tau = alpha * Omega(W_STD_ref)` for `alpha = 0.6, 0.8, 1.0`. Do not transfer a MNIST reference or budget to Fashion-MNIST.
- STD, WD, NUC, correctly measured SN, joint ADMM, and **PSGD-M**. Use the specified objectives, bias treatment, regularization grids and energy-matching selection rule.
- Twenty-five epochs, learning rate 0.01, batch size 64, momentum 0.9. Constrained methods use three warmup epochs, five transition epochs and seventeen fixed-budget epochs. Follow the implementation specification's update order exactly.
- PSGD-M projects at the beginning of each active epoch and after every minibatch, preserving its momentum buffer. Do not relabel it as deterministic GD or replace it silently with projection once per epoch.
- Float32 training without AMP/TF32, full float64 spectral calculations, and feasibility tolerance `1e-6 + 1e-5 * max(1, abs(tau))` for the primary outputs. E0 uses the tighter tolerances specified in the documents. Retain raw violations and signed slack.

Budget unique configurations honestly: the E2 core contains **540 runs = 180 constrained + 30 STD + 300 WD/NUC calibration candidates + 30 SN**. Each penalty candidate is trained once and reused across alpha values through selection; do not retrain the winner or triple SN runs. Required E4 adds **60 runs and ten 75-epoch continuations** from epoch 25 to 100: all joint-ADMM 9L configurations at alpha 0.8, for both datasets and all five seeds. Thus the required allocation is **600 runs of 25 epochs, ten continuations of 75 epochs, and 36 five-epoch pilot runs**, plus E0/E1 and analysis/evaluation costs. E3 reuses E2; E5/E6 do not create additional independent training replicates. Optional experiments have separate counts.

## 6. Freeze selection and preserve provenance

Before opening test results, freeze the protocol version, manifest, data splits and hashes, initial tensors, STD references and tau values, hyperparameters, numerical policy, validation-based selection, failure policy and model variants. Use a configuration-derived run ID and a separate attempt ID; deduplicate reused configurations. A continuation or a repeat on another device is not another independent seed.

Save unrounded metrics, objective components, phase labels, current/final budget violations, spectra, effective-weight hashes, and resumable checkpoints including W, biases, Z, u, rho, optimizer/momentum, sampler and RNG states. For SN include parametrization state and an effective snapshot. Record the actual environment, commit and changes. Keep failed attempts and relevant warnings.

Classify numerical, infrastructure, resource and implementation failures using the protocol's predefined rules. Do not drop low-accuracy seeds, replace failed results with averages, retry until accuracy improves, or count duplicate configurations as new evidence. Include STD and WD in feasibility tables.

## 7. Deliver results that support only admissible claims

Deliver E0–E6 with raw per-seed results, complete calibration curves, paired differences, sample standard deviations, missing-pair counts, failure categories, cost accounting and the planned tables/figures. Sample SD is not a confidence interval; five seeds provide limited precision. Epochs and images are not independent training replicates.

Use `CLAIMS_AND_EVIDENCE_MATRIX.md` to connect each conclusion to evidence and assumptions. If ADMM wins, ties, loses, fails at depth, costs more, or produces a feasible Z while W is infeasible, report that outcome explicitly. Small finite residuals are not a proof of asymptotic convergence. Do not claim compression unless a factorized/truncated artifact is actually produced and its storage, latency and predictive effects are measured. Lower effective rank alone is insufficient.

Do not silently alter definitions, tau, datasets, model selection, horizon, seeds or tolerances to fit outcomes. Record a minimal counterexample or failure case and consult Carlos about theoretical discrepancies. Protocol amendments must have a version, reason and affected-run list, preserving prior evidence.

## 8. Required deliverables and first response

**First milestone:** a proposed common code structure, a finding-by-finding response to the historical code audit, reviewable new implementation, completed E0/E1 evidence, a deduplicated inventory, the actual environment record, and the pilot-based compute proposal for author agreement.

**Second milestone, after that agreement:** the frozen manifest and full numerical package for the agreed runs, including raw results, failed attempts, checkpoints, hashes, environment, scripts, reproducible tables/figures and a report mapped to the admissible claims. Mark commands **Implemented and verified** only when actually executed successfully. Keep **Proposed interface — not yet implemented** labels elsewhere. Do not claim the manuscript is submission-ready while required experiments or provenance are missing.

In your first response, confirm which attachments and execution capabilities you actually have, state the current repository revision if inspected, identify any blocking information, and present the implementation/E0/E1/pilot sequence. Explicitly confirm that both MNIST and Fashion-MNIST remain in the plan. Start the authorized inspection and implementation when you have the necessary access; if you cannot run commands, provide concrete steps and wait for the collaborator's real outputs before reporting results.

