# Transformer VM Benchmarks

## CRYPTO-001 Native Arithmetic

The native operations were compared with the retained legacy lowering by
executing the same `(i32, i32) -> i32` function with operands `0x81` and `7`.
Counts include identical constant/local setup and termination tokens. Run
`uv run pytest transformer_vm/tests/test_crypto_primitives.py -k reduce_reference_trace_tokens`
to reproduce the comparison and enforce that every native path remains smaller.

| Operation | Native tokens | Lowered tokens | Reduction |
| --- | ---: | ---: | ---: |
| `i32.and` | 36 | 2,429 | 98.5% |
| `i32.or` | 36 | 2,905 | 98.8% |
| `i32.xor` | 36 | 2,938 | 98.8% |
| `i32.shl` | 36 | 509 | 92.9% |
| `i32.shr_s` | 36 | 8,782 | 99.6% |
| `i32.shr_u` | 36 | 8,530 | 99.6% |
| `i32.rotl` | 36 | 740 | 95.1% |
| `i32.rotr` | 36 | 2,347 | 98.5% |

These are deterministic reference-trace token counts, not wall-clock runtime
measurements. Lowered shift costs depend strongly on operand values.

## CRYPTO-001 Universal Model Build

Command: `uv run wasm-build --save-weights=model.bin`

| Metric | Result |
| --- | ---: |
| Graph operations | 542 |
| Graph dimensions | 651 |
| MILP rows / columns / nonzeros | 166,610 / 65,771 / 401,155 |
| MILP solve time | 181.37 s |
| MILP nodes / LP iterations | 2,577 / 426,748 |
| Layers / phases | 10 / 40 |
| `d_model` / heads / `d_ffn` | 340 / 170 / 280 |
| Vocabulary | 924 |
| Parameters | 8,108,320 |
| Saved weight size | 64,882,935 bytes (61.88 MiB) |

The reported time is the HiGHS MILP solve time from the command output, not
total wall-clock build time. The solver reached the optimal objective of 170.

## CRYPTO-001 Universal Model Execution

Command: `uv run wasm-run`

The standalone C++ engine loaded the newly generated universal model with
98% sparse output-head weights (5,363 of 314,160 nonzero). Results below are
from the same run; Sudoku was still running when these measurements were
recorded.

| Workload | Status | Tokens | Operations | Time | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| addition | PASS | 4,362 | 718 | 6.08 s | 717 tok/s |
| collatz | PASS | 44,332 | 8,961 | 63.36 s | 700 tok/s |
| fibonacci | PASS | 9,037 | 884 | 12.73 s | 710 tok/s |
| hello | PASS | 1,034 | 149 | 1.43 s | 724 tok/s |
| min_cost_matching | PASS | 177,606 | 36,510 | 257.06 s | 691 tok/s |

These are single-run observations. Hardware, operating-system version, and
power state were not captured, so the throughput values should not be treated
as cross-machine comparisons.
