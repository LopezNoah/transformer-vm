"""Regression coverage for PERF-004 trace expansion."""

from transformer_vm.compilation.trace_profile import (
    MAX_NATIVE_CRYPTO_TOKENS,
    MIN_CRYPTO_TRACE_REDUCTION,
    measure_crypto_trace_expansion,
)


def test_crypto_workload_trace_expansion_thresholds():
    results = measure_crypto_trace_expansion()

    assert len(results) == 8
    for result in results:
        assert result.native_tokens <= MAX_NATIVE_CRYPTO_TOKENS, result.opcode
        assert result.lowered_tokens >= result.native_tokens * MIN_CRYPTO_TRACE_REDUCTION, result.opcode
