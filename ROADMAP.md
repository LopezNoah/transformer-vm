# Transformer VM Improvement Roadmap

This document tracks improvements to the analytically constructed Transformer
WASM VM. Check off tasks only after their acceptance criteria and relevant tests
are complete.

## Working Conventions

- [ ] Keep each change focused on one task ID when practical.
- [ ] Add or update tests for every behavioral change.
- [ ] Record benchmark changes when a task affects token count, runtime, or memory.
- [ ] Update the supported-WASM documentation when execution semantics change.
- [ ] Do not treat project-generated reference traces as the only correctness oracle.

## P0: Execution Correctness

### COR-001: Fix bitwise lowering

- [x] Implement exact variable-operand `i32.and` semantics.
- [x] Implement exact variable-operand `i32.or` semantics.
- [x] Implement exact variable-operand `i32.xor` semantics.
- [x] Add edge-case tests for zero, all-one, alternating-bit, and high-bit values.
- [x] Differentially test results against an independent WASM runtime.

Relevant code: `transformer_vm/compilation/lower.py`

### COR-002: Fix shift and rotate semantics

- [x] Implement exact `i32.shl`, `i32.shr_u`, and `i32.shr_s` behavior.
- [x] Mask variable shift counts according to WASM semantics.
- [x] Verify signed right shift preserves the sign bit.
- [x] Implement and test `i32.rotl` and `i32.rotr`.
- [x] Test shift counts at 0, 1, 7, 8, 15, 16, 31, 32, 33, and large values.

Relevant code: `transformer_vm/compilation/lower.py`

### COR-003: Enforce lowering validation

- [x] Invoke `check_basic_only()` after lowering.
- [x] Reject unsupported instructions before model execution.
- [x] Include the lowering stress fixture in automated tests.
- [x] Add malformed and unsupported-instruction test cases.

Relevant code: `transformer_vm/compilation/lower.py`,
`transformer_vm/tests/fixtures/lowering_test.c`

### COR-004: Require complete execution

- [x] Require generated output to exactly match reference length.
- [x] Require every successful execution to emit `halt`.
- [x] Treat generation-limit exhaustion as a failure.
- [x] Prevent truncated reference traces from being saved as authoritative.
- [x] Apply the same rules in Python, graph, and C++ execution paths.

Relevant code: `transformer_vm/runner.py`, `transformer_vm/evaluator.py`,
`transformer_vm/wasm/reference.py`, `transformer_vm/model/transformer.cpp`

### COR-005: Build a differential conformance suite

- [x] Compare the reference interpreter with an independent WASM engine.
- [x] Compare graph evaluation with PyTorch standard attention.
- [x] Compare PyTorch standard attention with hull attention.
- [x] Compare Python inference with the standalone C++ runtime.
- [x] Cover arithmetic boundaries, control flow, memory, locals, calls, and traps.
- [x] Run a focused conformance subset in CI.

## P0: Input and Artifact Security

### SEC-001: Harden WASM decoding

- [ ] Validate section and function-body lengths before reading or allocating.
- [ ] Bound LEB128 length and reject malformed encodings.
- [ ] Bound program size, nesting depth, function count, local count, and memory size.
- [ ] Add decoder fuzz tests with a persisted regression corpus.

Relevant code: `transformer_vm/compilation/decoder.py`

### SEC-002: Version and validate model files

- [ ] Add a model-file magic value and format version.
- [ ] Add an integrity checksum.
- [ ] Validate dimensions and tensor lengths before allocation.
- [ ] Reject unsupported or truncated model files with clear errors.
- [ ] Test Python and C++ loader parity with corrupt files.

Relevant code: `transformer_vm/model/weights.py`,
`transformer_vm/model/transformer.cpp`

### SEC-003: Bound external processes and execution

- [ ] Add compiler and inference subprocess timeouts.
- [ ] Add configurable token, memory, call-depth, and program-size limits.
- [ ] Report the limit responsible for termination.
- [ ] Run C++ tests under ASan and UBSan in CI.
- [ ] Add dependency and static-analysis checks to CI.

## P1: Cryptography Feature-Set

Cryptography in this project means correctly executing cryptographic WASM
workloads. It does not imply that the runtime is constant-time or suitable for
handling production secrets.

### CRYPTO-001: Add crypto arithmetic primitives

Depends on: `COR-001`, `COR-002`, `COR-005`

- [x] Add tested add-with-carry and subtract-with-borrow helpers.
- [x] Add efficient native XOR, AND, OR, shift, and rotate VM operations.
- [x] Add explicit little-endian and big-endian load/store helpers.
- [x] Add constant-selection functionality without relying on Boolean-only bitwise lowering.
- [x] Verify exact 32-bit modular overflow behavior.
- [x] Measure token-count changes against lowered implementations.

### CRYPTO-002: Support binary-safe inputs and outputs

- [ ] Define a length-delimited binary input representation.
- [ ] Add CLI support for hex input.
- [ ] Add CLI support for binary-file input.
- [ ] Preserve embedded null bytes and non-UTF-8 data.
- [ ] Add deterministic hex output suitable for known-answer comparisons.
- [ ] Document key, nonce, message, and ciphertext encoding conventions.

Relevant code: `transformer_vm/compilation/compile_wasm.py`,
`transformer_vm/runner.py`

### CRYPTO-003: Implement SHA-256

Depends on: `CRYPTO-001`, `CRYPTO-002`

- [ ] Add a portable C/WASM SHA-256 implementation to the examples.
- [ ] Add NIST or RFC known-answer vectors with source citations.
- [ ] Test empty, short, block-boundary, and multi-block messages.
- [ ] Verify exact digest output across every runtime backend.
- [ ] Record tokens per byte, throughput, and peak cache size.
- [ ] Add at least one SHA-256 vector to CI.

Target files: `transformer_vm/examples/sha256.c`,
`transformer_vm/examples/manifest.yaml`, `transformer_vm/tests/`

### CRYPTO-004: Implement ChaCha20

Depends on: `CRYPTO-001`, `CRYPTO-002`

- [ ] Add a portable C/WASM ChaCha20 implementation.
- [ ] Add RFC 8439 block-function and stream-cipher vectors.
- [ ] Test counter progression and multi-block messages.
- [ ] Test all-zero and high-bit key, nonce, and input values.
- [ ] Verify output across every runtime backend.
- [ ] Record tokens per byte, throughput, and peak cache size.

### CRYPTO-005: Implement HMAC and HKDF

Depends on: `CRYPTO-003`

- [ ] Add HMAC-SHA256 with RFC known-answer vectors.
- [ ] Test keys below, equal to, and above the hash block size.
- [ ] Add HKDF-SHA256 extract and expand operations.
- [ ] Add RFC 5869 known-answer vectors.
- [ ] Verify output across every runtime backend.

### CRYPTO-006: Implement authenticated encryption

Depends on: `CRYPTO-004`

- [ ] Add Poly1305 with RFC known-answer vectors.
- [ ] Add ChaCha20-Poly1305 with RFC 8439 vectors.
- [ ] Test invalid tags, modified ciphertext, and modified associated data.
- [ ] Ensure failed authentication does not emit unauthenticated plaintext.
- [ ] Verify positive and negative cases across every runtime backend.

### CRYPTO-007: Evaluate AES support

Depends on: `CRYPTO-001`, `CRYPTO-005`

- [ ] Prototype AES-128 key expansion and block encryption.
- [ ] Measure the cost of byte substitution and finite-field operations.
- [ ] Decide whether native lookup or finite-field primitives are required.
- [ ] Add NIST AES-128 known-answer vectors if performance is acceptable.
- [ ] Add AES-GCM only after AES and authentication primitives are validated.

### CRYPTO-008: Define crypto security boundaries

- [ ] Document that execution is not guaranteed to be constant-time.
- [ ] Document potential leakage through traces, branches, timing, and memory access.
- [ ] Prohibit claims of production cryptographic security without an audit.
- [ ] Add a threat-model section covering untrusted WASM and secret inputs.
- [ ] Document provenance and licensing for every implementation and test vector.

### CRYPTO-009: Create a crypto benchmark report

- [ ] Report correctness against external known-answer vectors.
- [ ] Report generated tokens and tokens per input byte.
- [ ] Report runtime throughput and peak memory/cache use.
- [ ] Compare universal and specialized models.
- [ ] Track benchmark results over time in a reproducible format.

## P1: WASM Capability

### WASM-001: Complete i32 semantics

- [x] Implement divide-by-zero and overflow traps.
- [x] Implement exact signed and unsigned remainder semantics.
- [x] Implement memory bounds traps.
- [x] Verify sign-extension operations.
- [x] Publish an opcode and semantics conformance matrix.

### WASM-002: Add i64 support

- [ ] Define the byte-level i64 stack and memory representation.
- [ ] Implement i64 arithmetic, comparisons, shifts, and rotates.
- [ ] Add conversion operations between i32 and i64.
- [ ] Add differential tests against an independent WASM engine.
- [ ] Re-evaluate SHA-512 and public-key cryptography after completion.

### WASM-003: Expand runtime support

- [ ] Support indirect calls and table validation.
- [ ] Support dynamic memory growth with configured limits.
- [ ] Improve import validation and supported host functions.
- [ ] Define behavior for unsupported SIMD, exception, and reference instructions.
- [ ] Remove or make configurable the 64-locals-per-frame limitation.

## P1: Numerical Robustness

### NUM-001: Make hard attention comparisons robust

- [ ] Document required tie-breaking semantics for every attention lookup.
- [ ] Replace exact floating-point equality where possible.
- [ ] Use integer, rational, or cross-product comparisons where practical.
- [ ] Test large positions and adversarial key/query configurations.
- [ ] Verify standard-cache and hull-cache parity over generated traces.

Relevant code: `transformer_vm/attention/`, `transformer_vm/graph/core.py`

## P2: Runtime Performance

### PERF-001: Expand sparse execution

- [x] Measure sparsity for every projection and layer.
- [x] Define a versioned sparse tensor serialization format.
- [x] Load and execute sparse projections in Python and C++.
- [x] Skip inactive heads and layers where correctness permits.
- [ ] Benchmark against current dense execution on macOS and Linux.

### PERF-002: Fuse inference operations

- [ ] Prototype fused QKV projection.
- [ ] Fuse ReGLU, output projection, erasure, and residual updates where practical.
- [ ] Add vectorized Linux kernels instead of scalar fallback loops.
- [ ] Preserve exact backend parity tests.

### PERF-003: Add batching and parallel execution

- [ ] Support batches larger than one in Python generation.
- [ ] Support multiple independent programs in the C++ runtime.
- [ ] Define cache ownership and limits per program.
- [ ] Benchmark latency and throughput scaling.

### PERF-004: Reduce trace expansion

- [x] Measure token expansion by source WASM opcode.
- [x] Add native VM operations for the most expensive common lowerings.
- [x] Compare trace length before and after each new primitive.
- [x] Set performance regression thresholds for crypto workloads.

### PERF-005: Scale program specialization

- [ ] Profile MILP and lookup costs by program size.
- [ ] Replace program-size-dependent FFN lookup with static dispatch or tables.
- [ ] Test specialization beyond the `hello` example.
- [ ] Benchmark universal versus specialized crypto workloads.

## P2: Tests and Reproducibility

### TEST-001: Expand backend coverage

- [ ] Test every example in `transformer_vm/examples/manifest.yaml`.
- [ ] Test model save/load round trips.
- [ ] Exercise the standalone C++ runtime in CI.
- [ ] Add standard-attention versus hull-attention parity tests.
- [ ] Add specialization tests for nontrivial programs.
- [ ] Establish and enforce a meaningful coverage threshold.

### TEST-002: Version generated artifacts

- [ ] Store source hashes and compiler versions with generated artifacts.
- [ ] Store compiler flags and runtime semantic version.
- [ ] Invalidate outputs when any relevant input changes.
- [ ] Move generated data to a configurable cache or build directory.
- [ ] Make benchmark and test-case generation reproducible from a clean checkout.

### TEST-003: Improve workload manifests

- [ ] Support multiple named inputs per program.
- [ ] Record expected outputs and external reference provenance.
- [ ] Add explicit fast, integration, and benchmark categories.
- [ ] Add checksums for source inputs and expected outputs.

## P2: Documentation and Developer Experience

### DOC-001: Correct current documentation

- [ ] Replace or qualify claims that graph/model execution uses exact arithmetic.
- [ ] Replace the invalid `wasm-eval --regen` documentation with `wasm-reference`.
- [ ] Update the supported-opcode list to match actual lowering behavior.
- [ ] Document generation limits and failure modes.

Relevant file: `README.md`

### DOC-002: Add architecture and format documentation

- [ ] Document the graph-to-transformer construction pipeline.
- [ ] Document universal and specialized execution modes.
- [ ] Document the model binary format and compatibility policy.
- [ ] Document the program token format.
- [ ] Document hard-attention assumptions and tie-breaking behavior.

### DOC-003: Add project governance documents

- [ ] Add `SECURITY.md` with reporting and support expectations.
- [ ] Add a threat model for untrusted model and WASM inputs.
- [ ] Add a model card describing capabilities and limitations.
- [ ] Add a reproducible benchmark guide.

## Future Research

### RESEARCH-001: Public-key cryptography feasibility

Depends on: `WASM-002`, `CRYPTO-009`

- [ ] Benchmark multiprecision addition, multiplication, and modular reduction.
- [ ] Evaluate whether RSA is practical within trace and memory limits.
- [ ] Evaluate constant-structure elliptic-curve field arithmetic.
- [ ] Do not schedule production implementations until feasibility is demonstrated.

### RESEARCH-002: Learned model extensions

- [ ] Decide whether learned components are desirable alongside exact VM semantics.
- [ ] Define a trace dataset format and leakage-resistant splits.
- [ ] Define a teacher-forced objective and checkpoint format.
- [ ] Require regression tests proving learned changes do not break VM semantics.
