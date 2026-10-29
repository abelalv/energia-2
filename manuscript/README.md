# Revised English manuscript

`manuscrito_RAM.pdf` is the 12-page mathematical working revision prepared on 13 October 2026 for *Results in Applied Mathematics*. Its contents and the supplied source files are copied unchanged from that revision. The important mathematical proofs are in the main text.

The LaTeX entry point is `manuscrito_RAM.tex`. The folder includes `sections/`, `reference.bib`, the Elsevier CAS class/style, bibliography styles, template thumbnails and a separate highlights file. Stable proposition labels connect the manuscript to the handoff's implementation specification and claims matrix.

The following build command was verified for the source revision with TeX Live 2020. It is supplied as the invocation to check on your own TeX installation, not as a claim that your machine already has the dependencies:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error manuscrito_RAM.tex
```

Run it from this `manuscript` folder if you need to edit or rebuild the article. Recompilation is not required to read the PDF or implement the experimental protocol. The original build produced no errors or undefined references; its harmless remaining notices concerned template anchors, bibliography spacing and unverified conference pagination. Missing page numbers were not invented.

The mathematical-check path printed in the PDF is the earlier source location. In this handoff, use `../mathematical_checks/verify_mathematics.py`; its README records the filename-only adaptation. The manuscript's statement about the AI used during its preparation is historical disclosure, not a requirement to use that product for the experiments.

This manuscript does not report newly completed MNIST or Fashion-MNIST experiments. Its prospective evaluation still requires the collaborator's work. Author declarations, the complete postal affiliation, reproducible experimental materials and the final empirical conclusions must be settled before submission. Historical claims retained for audit are not automatically valid conclusions of the revision.
