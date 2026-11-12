# LazyQCQP

Materials for **Tangent Excess and Persistent Difference-of-Convex Regularization for Nonconvex QCQPs**, by François Akoa, MAATAI-LAB, France.

This release accompanies manuscript **R24**, dated 11 November 2026. The main paper has 33 pages and the supplement has 65 pages.

## Paper and supplementary material

- [Main paper](manuscript/Akoa_LazyQCQP_20261111_R24.pdf)
- [Supplement](manuscript/Akoa_LazyQCQP_20261111_R24_Supplement.pdf)
- [LaTeX source](manuscript/)
- [Bibliography](manuscript/LazyQCQP_references.bib)
- [Abstract](ABSTRACT.txt)

The paper studies the tangent curvature added by componentwise DC regularization. A QCQP construction attains the lower bound, and the local factor is κ/(1 + κ). LazyQCQP reuses DC corrections and retains the signed curvature of quadratic equalities. The numerical study includes 4,078 QCQPs, the controlled ASU bound-removal comparison, and the 12-instance reuse comparison.

## Reproduce the reported analyses

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 reproduction/run_reproduction.py --out reproduced
```

This regenerates the benchmark tables, performance profiles and the ASU/persistence summaries from the frozen observation records. It also checks the generated statistics against the archived results. The command does not launch optimization runs. See [reproduction/README.md](reproduction/README.md) for the individual commands and the scope of the recorded certificates.

## Compile the paper

With a TeX Live installation providing `pdflatex`, `amsmath`, `booktabs`, `graphicx`, `xr-hyper` and the other packages used by the sources:

```bash
cd manuscript
bash build.sh
```

The Springer document class and bibliography style are included. Both documents contain their reference lists; the matching `.bib` file is supplied separately.

## Solver and campaigns

See [code/README.md](code/README.md) for the native source profiles, build commands, independent KKT evaluator and campaign scripts. The code documentation identifies the source used by each controlled experiment and its dependencies. The archived observation records are in [analyses/](analyses/).

The [data inventory](DATA.md) identifies all 4,078 cases and the nine original QCQPs included here. It also provides a collector for the complete frozen inputs on the author's TRX installation.

## Verify this release

```bash
python3 verify_bundle.py
```

The manifest covers the distributed files. Validation reports record the paper build, source checks, and offline analysis reproduction. The source/data notes distinguish regeneration from the archived scalar records and evaluation from original problem coefficients and endpoint vectors.

## Citation

Use [CITATION.cff](CITATION.cff). The citation identifies the accompanying R24 preprint.

Repository: https://github.com/francoisakoa/lazyqcqp

Contact: francois.akoa@maatai-lab.com
