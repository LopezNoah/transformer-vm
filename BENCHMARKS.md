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

## PERF-004 Trace Expansion

`wasm-trace-profile` measures each source crypto opcode under the fixed
`(0x81, 7)` workload, comparing the native VM primitive with its legacy
base-op lowering. The `test_perf004_trace_expansion.py` regression test limits
each native opcode to 36 tokens and requires every legacy lowering to generate
at least 10x as many tokens.

```bash
uv run wasm-trace-profile
uv run pytest transformer_vm/tests/test_perf004_trace_expansion.py
```

## Universal Model Build Comparison

Baseline command: `uv run wasm-run` with no existing `model.bin`.

CRYPTO-001 command: `uv run wasm-build --save-weights=model.bin`.

| Metric | Baseline | CRYPTO-001 | Change |
| --- | ---: | ---: | ---: |
| Graph operations | 159 | 542 | 3.41x |
| Graph dimensions | 186 | 651 | 3.50x |
| MILP rows | 31,047 | 166,610 | 5.37x |
| MILP columns | 12,698 | 65,771 | 5.18x |
| MILP nonzeros | 76,093 | 401,155 | 5.27x |
| MILP solve time | 3.75 s | 181.37 s | 48.37x |
| MILP nodes | 28 | 2,577 | 92.04x |
| LP iterations | 11,275 | 426,748 | 37.85x |
| Layers / phases | 7 / 28 | 10 / 40 | 1.43x / 1.43x |
| `d_model` | 38 | 340 | 8.95x |
| Heads | 19 | 170 | 8.95x |
| `d_ffn` | 48 | 280 | 5.83x |
| Vocabulary | 915 | 924 | 1.01x |
| Parameters | Not reported | 8,108,320 | N/A |
| Saved weight size | 1,194,466 bytes | 64,882,935 bytes | 54.32x |

The reported times are HiGHS MILP solve times, not total wall-clock build
times. Both runs reached their optimal objective. These results predate the
formulation improvements below and are retained as the historical CRYPTO-001
comparison.

## MILP Formulation Optimization

The scheduler was optimized by computing ASAP/ALAP phase bounds, restricting
time-indexed binaries to feasible layer windows, replacing global big-M values
with expression-specific bounds, enforcing `death[d]` as the exact last
consumer phase, transitively reducing dependency and consumer constraints, and
omitting unused final-boundary alive indicators. HiGHS presolve and symmetry
detection are enabled, while the time limit, thread count, and relative MIP gap
are configurable.

The optimized full capability profile was benchmarked with 10 layers and 12
HiGHS worker threads and compared with the historical result above. Both
formulations reached the same optimal `d_model=340` objective.

| Metric | Previous formulation | Optimized formulation | Change |
| --- | ---: | ---: | ---: |
| MILP rows | 166,610 | 34,042 | 79.6% fewer |
| MILP columns | 65,771 | 14,264 | 78.3% fewer |
| MILP nonzeros | 401,155 | 84,430 | 79.0% fewer |
| MILP solve time | 181.37 s | 66.27 s | 2.74x faster |
| MILP nodes | 2,577 | 1,905 | 26.1% fewer |
| LP iterations | 426,748 | 178,004 | 58.3% fewer |
| `d_model` | 340 | 340 | Unchanged |

The optimized base profile contained 2,847 columns and 6,613 rows and solved
to `d_model=38` in 1.97 seconds with one branch-and-bound node.

The optimized full-profile plan reproduced the complete 1,034-token `hello`
reference trace. The non-slow test suite also passed all 91 selected tests.

Run the scheduler with the same thread count using:

```bash
uv run python -m transformer_vm.scheduler.milp --threads 12
```

## Universal Model Execution Comparison

Command: `uv run wasm-run`

Both runs used the standalone C++ engine. The baseline output head was 85%
sparse (5,348 of 34,770 nonzero); the CRYPTO-001 output head was 98% sparse
(5,363 of 314,160 nonzero).

| Workload | Tokens | Operations | Baseline | CRYPTO-001 | Throughput slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| addition | 4,362 | 718 | 0.13 s, 34,428 tok/s | 6.08 s, 717 tok/s | 48.0x |
| collatz | 44,332 | 8,961 | 1.10 s, 40,337 tok/s | 63.36 s, 700 tok/s | 57.6x |
| fibonacci | 9,037 | 884 | 0.20 s, 44,374 tok/s | 12.73 s, 710 tok/s | 62.5x |
| hello | 1,034 | 149 | 0.02 s, 45,762 tok/s | 1.43 s, 724 tok/s | 63.2x |
| min_cost_matching | 177,606 | 36,510 | 4.70 s, 37,819 tok/s | 257.06 s, 691 tok/s | 54.7x |

Across these five completed workloads, elapsed execution time increased from
6.15 seconds to 340.66 seconds, a 55.4x slowdown. The CRYPTO-001 Sudoku result
was not available when its run was recorded.

The baseline Sudoku result was 980,596 tokens and 207,335 operations in 27.91
seconds (35,134 tok/s). The complete six-workload baseline generated 1,216,967
tokens and 254,557 operations in 34.06 seconds (35,732 tok/s and 7,474
WASM-ops/s). Its time breakdown was:

| Stage | Time | Share |
| --- | ---: | ---: |
| Projection | 11.780 s | 34.6% |
| Hull attention | 19.144 s | 56.2% |
| Output head | 2.903 s | 8.5% |
| Miscellaneous | 0.232 s | 0.7% |

These are single-run observations. Hardware, operating-system version, and
power state were not captured, so the throughput values should not be treated
as cross-machine comparisons.

## PERF-001 Sparse Execution

`wasm-build --save-weights=model.bin` now emits versioned CSR projections for
the embedding, every attention and FFN projection, and the output head. The
C++ runtime also accepts the prior dense artifacts. Rebuild the model and run
the same workload with the C++ runner to compare sparse execution on a target
machine; record operating system, CPU, BLAS configuration, and power state.

The C++ engine accepts `--dense` to materialize the same CSR artifact and run
the original dense projection loops. This isolates projection representation
while retaining the same model, cache, program, and reference comparison.

| Workload | Projection mode | Tokens | Time | Throughput | Result |
| --- | --- | ---: | ---: | ---: | --- |
| collatz | dense | 44,332 | 72.36 s | 613 tok/s | PASS |
| collatz | sparse | 44,332 | 1.93 s | 22,954 tok/s | PASS |

These single macOS runs used the full `model.bin`, Accelerate, and identical
Collatz input. Sparse projection execution was 37.4x faster. Repeat with
`wasm-run --dense transformer_vm/data/collatz.txt` and without `--dense` on
each target machine before treating this as a regression threshold.
