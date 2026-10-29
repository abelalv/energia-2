# English handoff verification

Date: 13 October 2026. This record concerns the preparation of the handoff, not the future neural-network experiments.

## Documentation and translation checks performed

- The full implementation specification, experimental protocol, claims matrix, mathematical audit and historical code audit were translated into English. The generic AI prompt and start guide were written for a collaborator without access to the earlier conversation or a particular AI application.
- All **186 mathematical expressions** and both text pseudocode blocks in the implementation specification match the Spanish source exactly. Numeric settings and interfaces were checked during translation.
- All **M01–M28**, **C01–C27** and **AF01–AF15** identifiers are present. Original source coordinates, result mappings and distinctions between observed defects and pending execution were retained.
- All **E0–E6** stages and the eight proposed CLI commands are present. The commands retain their unimplemented status; no `ram_experiments` package was created or claimed to exist.
- The Markdown and plain-text prompt copies are byte-identical. The prompt has no provider-specific application, tool or local-machine dependency. It describes both code-capable and chat-only assistance.
- **MNIST and Fashion-MNIST are explicitly included in the pilot and primary benchmark.** The prompt preserves separate dataset normalization, matched STD references, five seeds, architecture widths, methods, counts and pre-test selection rules.
- Independent document reviews compared the start guide and prompt with the detailed specification and protocol. The ten mandatory continuations are now stated unambiguously as all joint-ADMM 9L runs at alpha 0.8, both datasets and all five seeds; they are not selected by performance.
- New handoff prose contains no machine-specific absolute paths. Historical filenames and provenance coordinates are intentionally retained and explained.

## Clarifications made without changing the scientific protocol

1. AF06 explicitly says that its E4 layer-metric reference reuses existing prescribed measurements; it does not add a new training experiment.
2. The E0 penalty counterexample now spells out `F_lambda = f + lambda * max(Omega - tau, 0)`, already defined in the mathematical specification and manuscript.
3. The five-epoch pilot retains the three-epoch warmup and five-epoch transition schedule. Consequently, it does not measure the fixed-budget phase starting at epoch 9. The handoff requires fixed-phase extrapolations to be labeled as estimates and any additional technical timing measurements to be separately declared and costed. No schedule, main run count or training outcome was changed.
4. Historical notebook experiment numbering is explicitly distinguished from the redesigned E0–E6 protocol.

## Actual numerical and integrity checks

The portable copy of the mathematical-check script was executed during handoff preparation. It completed successfully with **227 passed checks and zero training runs**, using CPU/float64, seed 20261013, Python 3.11.8, NumPy 1.26.4 and SciPy 1.17.1. Its actual outputs are `mathematical_checks/current_results.json` and `current_execution.log`; observed versions are recorded separately. Only its docstring filename and output JSON filename were adapted from the original script. This rerun is not completion of the future E0/E1 implementation-validation suites.

The revised English PDF still has 12 pages. All copied manuscript and historical source files were verified byte-for-byte against the completed revision; no manuscript content or layout was changed in this packaging task. The six Spanish source documents and all 64 files in the original manuscript directory remain unchanged.

The historical notebook was not executed or modified. A read-only provenance recheck confirmed the three pinned repository hashes documented in the historical code audit. No dataset was loaded, no training ran, and no new predictive accuracy, runtime or compression result was generated.

`verification_results.json` stores the programmatic check counts. `SOURCE_PROVENANCE.json` records source-document hashes and the two filename-only script adaptations. `SHA256SUMS.json` lists every packaged file except itself. The ZIP is checked for archive integrity and each member is compared with its listed hash when the package is created.

## Remaining work

The collaborator must implement the new training code, validate E0/E1, run the bounded pilot, agree the measured compute allocation, freeze the manifest and selection, execute the agreed benchmark and return complete evidence including failures. Hardware compatibility, GPU behavior, new training results, deployment compression and manuscript submission readiness are not established by this handoff.
