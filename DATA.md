# Benchmark data

The [data inventory](data/README.md) describes the 4,078 benchmark cases, their input/start hashes and the available original inputs. The [machine-readable manifest](data/BENCHMARK_CASES.json.gz) agrees with the frozen main-campaign observations for all 4,078 cases and 28 dataset labels.

Nine original benchmark QCQPs are included in `data/bundled_frozen/`. The complete coefficient corpus and full endpoint vectors are external to this release. The supplied [collector](data/collect_frozen_inputs.py) can retrieve exact matching frozen inputs from the author's preserved TRX installation, with a SHA256 check for every case.

The [analysis commands](reproduction/README.md) regenerate the reported tables and figures from the included archived observation records. The [native code instructions](code/README.md) describe compilation, the independent original-QCQP evaluator and restoration of the campaign runners.
