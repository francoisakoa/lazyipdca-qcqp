# R10 actual controller regression

The 25 assertions are preserved from the R10 full-step controller validation.
Each run extracts the production parser, two-attempt controller, nominal
fraction gate, repair line budget, GR derivative, accepted-state staging and
commit, and precommit deadline. It uses the production barrier path and
Gill–Robinson merit implementation. Only original QCQP storage is mocked.

The checks cover a legacy merit rejection repaired by the full consistent
bound path, frozen merit parameters, accepted legacy four-vector identity,
disabled repair, the historical always mode, rejected-step transactionality,
deadlines, a full-step rejection whose half-step would pass, and a nominal
fraction below one. The two latter cases must decline the on-failure repair.

Run with a C++17 compiler and Python standard library:

```sh
python tests/r10_bound_controller/run_actual_r10_control_test.py \
  --source . --output /tmp/r10-bound-controller --cxx g++
```

The script writes generated test inputs, the executable and RESULT.json into
`--output`; it never modifies production source or calls an optimizer. The
receipt includes the exact production source hashes and compile command.
This is a controller/path regression; independent original KKT certification
is covered by the separate endpoint and certificate regressions.
