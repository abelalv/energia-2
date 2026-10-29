# Experimental handoff — start here

Prepared on 13 October 2026. Protocol: **RAM-2026-10-13-v1**. Language: English.

This independent package supplies the mathematical specification, experimental protocol, audit trail and reusable AI prompt for the collaborator who will implement and run the experiments for **Graph Energy Bounds and Global Nuclear-Norm Budgets in Fully Connected Neural Networks**. It does not require access to the earlier conversation or to any particular AI product.

**Both MNIST and Fashion-MNIST are included in the proposed primary benchmark.** Neither new benchmark has been executed in the preparation of this package. The historical training notebook has not been repaired or run as part of the handoff. A new training implementation remains to be written and validated.

## How to use the package

1. Extract the ZIP into a normal working folder.
2. Read this file and `AI_ASSISTANT_PROMPT.md`.
3. Give your chosen AI assistant the prompt, the five technical Markdown documents below, and the revised English manuscript. The `.txt` version of the prompt is identical and is convenient for copying into a chat.
4. If the assistant has project access, let it inspect the extracted folder and a separate checkout of the historical repository. If it only accepts uploads, attach the files directly. If it cannot open ZIP files, extract the files first. For limited attachment sizes, provide the documents in the listed reading order and obtain confirmation that they were read before implementation.
5. Follow the gates: new implementation → E0/E1 → bounded pilot → measured compute agreement → frozen main benchmark → analysis and reproducibility package.

The prompt works with assistants that can execute code and with chat-only assistants. A chat-only assistant must supply code and commands for you to run and assess the returned logs. No assistant should claim an execution result merely from reading code.

## Included documents and reference material

| File or folder | Purpose |
|---|---|
| `AI_ASSISTANT_PROMPT.md` / `.txt` | Complete assignment to copy into any AI tool, with responsibilities, scope, gates and deliverables |
| `MATHEMATICAL_IMPLEMENTATION_SPECIFICATION.md` | Dimensions, graph/surrogate distinction, projection, spectral normalization, objectives, exact and experimental updates, tolerances, certificates and interfaces |
| `EXPERIMENTAL_PROTOCOL.md` | E0–E6, both datasets, fixed configurations, calibration, pilot, counts, failure policies, statistical reporting and proposed commands |
| `CLAIMS_AND_EVIDENCE_MATRIX.md` | AF01–AF15, assumptions, required evidence and acceptable wording for favorable or unfavorable outcomes |
| `MATHEMATICAL_AUDIT.md` | M01–M28, original claims, corrections and mapping to the revised mathematical results |
| `HISTORICAL_CODE_AUDIT.md` | C01–C27, source locations, observed defects, unconfirmed suspicions and historical provenance |
| `manuscript/` | Revised English manuscript PDF and editable LaTeX sources, including the proofs in the main body |
| `mathematical_checks/` | Small reference calculations, recorded execution evidence and an English description of their limits; no neural-network training code |
| `historical_reference/` | Original English manuscript and source material for tracing the audit locations; not the specification to implement |
| `HANDOFF_VERIFICATION.md` | Checks actually performed when preparing this English handoff |
| `SHA256SUMS.json` | File-integrity manifest for this package, excluding the manifest itself |

All new handoff documentation is in English. The manuscript and historical source identifiers keep their existing filenames so citations and audit coordinates remain traceable. Statements about the AI used in preparing the supplied manuscript describe that earlier work; they do not prescribe the collaborator's choice of assistant.

## What has and has not been done

The mathematical revision supplies explicit proofs, an audit of the historical sources, and 227 recorded small CPU/float64 numerical checks. These checks exercise mathematical formulas and counterexamples. They do not replace E0/E1 for the new implementation, establish its GPU correctness, validate a future autograd path, or constitute a benchmark result.

The `ram_experiments` commands in the protocol are **proposed interfaces to implement**. There is no ready-made training package of that name in this ZIP. The ZIP contains no datasets, trained checkpoints or newly generated neural-network results. Required datasets and the historical repository must be obtained separately following the protocol and their distribution terms.

The historical repository is [cero1979/Ensayos-Energy-Graph](https://github.com/cero1979/Ensayos-Energy-Graph). Its audited commit is `186e27400f94d551432759fa4830e0c114a2aa69`. The audit gives the notebook hash and cell/line coordinates. Inspect this reference without executing notebook cells; compare any later revision explicitly. The historical notebook itself is not bundled here.

## Planned benchmark and compute gate

| Item | Prespecified scope |
|---|---|
| Datasets | MNIST **and Fashion-MNIST**, with separate training-only normalization, shared method splits, and dataset-specific STD references and tau values |
| Main split per dataset | 50,000 train / 10,000 validation from official training data; official 10,000 test examples reserved |
| Architectures | Original 3L, 6L and 9L MLPs, with the exact widths in the protocol |
| Seeds | 42, 123, 456, 789, 1024 |
| Methods | STD, WD, NUC, SN, joint ADMM and PSGD-M |
| Budget factors | 0.6, 0.8, 1.0 times the matched final STD surrogate |
| E2 core | 540 unique runs, including all WD/NUC calibration candidates |
| Required E4 | 60 additional runs and ten continuations from epoch 25 to 100 |
| Main allocation | 600 runs of 25 epochs plus ten 75-epoch continuations |
| Prior pilot | 36 short runs of five epochs, covering both datasets, all architectures and methods, seed 42 |

E3 reuses E2 results; E5/E6 add analysis and measurement rather than independent training replicates. Optional experiments have separate counts. The protocol specifies deduplication, reuse of candidate checkpoints and full calibration costs.

The five-epoch pilot retains the prescribed schedule, so it covers warmup and partial transition. Fixed-budget-phase costs must be labeled as estimates unless separately measured and documented; the pilot does not reach epoch 9.

The collaborator must first pass E0/E1 and measure the pilot. The authors then agree on the measured compute allocation before the main battery. Hardware and runtime are not assumed. A reduced exploratory dataset size, if needed for cost, requires the documented pre-test amendment and new provenance; it does not silently remove Fashion-MNIST or turn a reduced run into the full benchmark.

## Scientific and execution boundaries

- Preserve the historical notebook, CSV files, figures, checkpoints and other people's changes. Develop new common modules with device profiles.
- Keep `Omega(W) = 2 * sum ||W_l||_*` distinct from the full adjacency energy, and include trainable biases in the objective and full-space diagnostics.
- Use actual final W as the primary ADMM model; label Z and projected W separately.
- Freeze validation-based selection and failure rules before test access. Do not tune on the test set or discard unfavorable seeds.
- Report ADMM wins, ties, losses, infeasibility, failures and costs equally. The mathematical results do not require predictive superiority.
- Do not present finite residuals as a convergence theorem, thresholded ranks as rigorous algebraic bounds, or effective rank as measured deployment compression.
- Keep implemented/verified commands distinct from proposals. Log actual execution evidence.

The complete technical documents govern these details. If a discrepancy is discovered, record it and resolve it with Carlos before running affected configurations. No change to the scientific protocol is intended by this English translation or by the change to a general-purpose AI prompt.

## Expected first return from the collaborator

Provide the proposed module structure, response to all historical code findings, reviewable implementation, E0/E1 reports, deduplicated run inventory, actual environment and hardware, and pilot-derived time/memory/cost estimates. After compute agreement, return the frozen manifest, reproducible numerical package and an evidence-linked scientific report. The revised article still needs those results and author confirmations before editorial submission.
