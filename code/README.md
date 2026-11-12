# Native solver and experiment sources

This directory contains the native LazyQCQP sources used by the manuscript,
the independent original-QCQP KKT evaluator, frozen experiment contracts,
and the campaign runners. Source profiles are stored as one complete base and
small overlays to avoid repeating identical files.

## Verify the sources and the evaluator

From the repository root:

```bash
python3 code/check_sources.py --out reproduced/code_checks
```

This reconstructs all six source profiles and checks every SHA256. It also runs
13 mathematical checks of the independent evaluator and re-audits two small
archived endpoints. It does not launch an optimizer.

The validated profiles are `mumps` (823 files), `ma27` (823), `asu` (824),
`persistence` (824), `refinement` (825), and `factorial_observer` (825).
The last profile is the historical instrumentation source supplied with the
MA27 campaign runner. `provenance/` retains the archived source/build receipts.
ASU and persistence profiles exactly match the source hashes in their completed
experiment build records. The MUMPS and MA27 profiles preserve their respective
historical source differences.

## Build a native solver

The compiler requires C++17, Boost.Multiprecision headers, BLAS/LAPACK, MUMPS,
and METIS. MA27 runs additionally require the external licensed HSL library.
Ipopt is required only for its comparison runs. Third-party libraries and
executables are not redistributed here.

For the supplied helper, install CMake and Boost headers through the operating
system package manager, then provide CasADi's native dependency directory:

```bash
# Debian/Ubuntu, when these dependencies are not installed:
sudo apt-get install build-essential cmake libboost-dev
python3 -m pip install casadi==3.7.2

python3 code/build_native.py --profile mumps --out reproduced/native_mumps
```

Alternatively pass `--dependency-root /path/to/casadi` and
`--boost-include /path/containing/boost`. The helper records the commands and
uses the external MUMPS, METIS and OpenBLAS libraries from that dependency root.
Other source profiles can be selected with `--profile`.
For a system-library build, first reconstruct the source, then configure its
unchanged CMake project with the appropriate dependency paths:

```bash
python3 code/prepare_profile.py --profile mumps --out reproduced/source_mumps
cmake -S reproduced/source_mumps -B reproduced/build_mumps \
  -DCMAKE_BUILD_TYPE=Release -DLAZYIPDCA_BUILD_TESTS=OFF \
  -DLAZYIPDCA_WITH_IPOPT=OFF -DLAZYIPDCA_WITH_MUMPS=ON
cmake --build reproduced/build_mumps --target lazyipdca_qcqp_cli -j2
```

A fresh full native build was not performed while assembling this release:
CMake and Boost headers were absent from the packaging environment. The
compiler and the external MUMPS/METIS runtime were detected. The dependency
check stopped before compilation; its receipt is in
`validation/LOCAL_NATIVE_BUILD_CHECK.json`. The source checks, 13 evaluator
tests and two endpoint re-audits passed. These checks are separate from the
archived campaign build and numerical records.

## Restore an experiment runner

```bash
python3 code/prepare_runner.py --runner ma27_main --out reproduced/runner_main
python3 code/prepare_runner.py --runner aligned_refinement --out reproduced/runner_aligned
python3 code/prepare_runner.py --runner asu --out reproduced/runner_asu
python3 code/prepare_runner.py --runner persistence --out reproduced/runner_persistence
```

These commands restore source directories and the compressed metadata file;
they do not start computations. The restored runners retain the archived
scripts and relative layout. Their original instructions describe the frozen
installation paths, source hashes and dependency checks. Full campaigns use
the original frozen coefficient/start files and, for MA27, the licensed HSL
installation. Restore those inputs before launching a campaign.

`runners/ma27_main/default_relaunch/CONTRAT_4078.json.gz` contains the frozen
4,078-case contract, input/start hashes and case dimensions. The ASU contract
contains 503 cases; the persistence cohort contains 12. The native source also
contains CUTEst and PGLIB export scripts, synthetic QCQP fixtures and historical
build scripts. The original full 4,078-case coefficient/endpoints archive is
external to this release. Scalar observation extracts used for the paper's
tables are included in the repository's `analyses/` directory.

## Audit an original QCQP endpoint

```bash
python3 code/audit/AUDIT_ORIGINAL.py \
  --input code/audit/nonconvex.qcq \
  --endpoint code/audit/nonconvex_reference/endpoint.json \
  --out reproduced/audit_nonconvex.json
```

The auditor uses the original coefficients and bound multipliers, independently
of the solver's reported status. A strict certificate requires primal, dual and
complementarity residuals at most `1e-6`. It is a first-order KKT certificate.

No new software or data license is assigned by this package. See the repository's
rights information for authorship and third-party material.
