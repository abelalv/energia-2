# Small mathematical checks

This folder contains reference calculations supporting the mathematical revision. It contains no neural-network training implementation and does not load the historical notebook or datasets.

| File | Meaning |
|---|---|
| `verify_mathematics.py` | Standalone small CPU/float64 checks using NumPy and SciPy |
| `recorded_results.json` | Unchanged output from the earlier mathematical revision |
| `recorded_execution.log` | Unchanged recorded console output from that revision |
| `current_results.json` | Output of the actual rerun while preparing this English handoff |
| `current_execution.log` | Actual console output of that rerun |
| `observed_environment.json` | Environment actually observed for the handoff rerun |

Both recorded executions report 227 passed checks and zero neural-network training runs. The random seed is 20261013. The handoff rerun used Python 3.11.8, NumPy 1.26.4 and SciPy 1.17.1 on CPU in float64. This is an observed environment, not a Linux/CUDA lockfile or proof of compatibility with every Python installation.

The script was copied from the mathematical revision. Only the script name in its docstring and the generated JSON filename were adapted: it now writes `current_results.json` beside itself instead of the original `resultados.json`. The computational statements are unchanged. `../SOURCE_PROVENANCE.json` records both script hashes and those edits.

The revised manuscript refers to this original script as `comprobaciones/verificar_matematica.py`. That is its earlier location; the equivalent script in this handoff is `mathematical_checks/verify_mathematics.py`.

## Running the checks again

Use a Python interpreter with compatible NumPy and SciPy installed. From the package root, the portable invocation pattern is:

```bash
python3 mathematical_checks/verify_mathematics.py
```

Select the interpreter that you have actually checked on your machine; the command above does not install dependencies. A successful run prints a JSON summary and writes `current_results.json`. It replaces that current output file, so preserve it under a run-specific name if you need to retain multiple executions. It does not overwrite `recorded_results.json`. Any failed assertion terminates execution with an error; inspect the failing input and calculation instead of suppressing the error.

## Scope and limits

The checks cover graph-energy inequalities, joint projection, an independent SLSQP oracle for the singular-value subproblem, spectral clipping, QG examples, residual feasibility, a full-space stationarity identity, normal-cone samples, dual updates, penalty counterexamples, polynomial errors and ReLU rescaling.

These are finite numerical checks with tolerances. They neither replace the manuscript's proofs nor provide rigorous enclosures of floating-point rounding error. The future E0 suite must additionally validate the collaborator's actual implementation, an independent matrix-level oracle, NUC autograd, effective SN weights, failure handling, checkpoint resumption and the chosen devices. The future E1 protocol also contains its own prescribed analytical cases. Passing this script is not evidence that E0/E1 or the MNIST/Fashion-MNIST benchmark has been completed.
