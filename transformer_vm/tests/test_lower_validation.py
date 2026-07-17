"""Compiler boundary tests for post-lowering instruction validation."""

import shutil

import pytest

from transformer_vm.compilation.compile_wasm import compile_c_to_wasm, compile_wasm_to_prefix
from transformer_vm.wasm.interpreter import CRYPTO_OPCODES


def _wasm_with_memory_grow() -> bytes:
    """Build a minimal module containing a decoded but unsupported instruction."""
    body = bytes([0, 0x40, 0, 0x0B])
    code = bytes([1, len(body)]) + body
    return (
        b"\x00asm\x01\x00\x00\x00"
        + b"\x01\x04\x01\x60\x00\x00"
        + b"\x03\x02\x01\x00"
        + bytes([10, len(code)])
        + code
    )


def test_lowering_stress_fixture_compiles_to_basic_instructions(tmp_path):
    """The stress fixture passes the compiler's post-lowering validation."""
    fixture = tmp_path / "lowering_test.c"
    shutil.copyfile(
        "transformer_vm/tests/fixtures/lowering_test.c",
        fixture,
    )

    wasm_path = compile_c_to_wasm(str(fixture))
    prefix, _ = compile_wasm_to_prefix(wasm_path)
    base_prefix, _ = compile_wasm_to_prefix(wasm_path, profile="base")

    assert prefix.startswith("{\n")
    assert CRYPTO_OPCODES.isdisjoint(base_prefix.split())


def test_compiler_rejects_unsupported_instruction_after_lowering(tmp_path):
    """Decoded instructions outside the basic subset cannot reach model input."""
    wasm_path = tmp_path / "unsupported.wasm"
    wasm_path.write_bytes(_wasm_with_memory_grow())

    with pytest.raises(ValueError, match=r"Function 0.*memory.grow \(1\)"):
        compile_wasm_to_prefix(str(wasm_path))


def test_compiler_rejects_malformed_wasm_before_lowering(tmp_path):
    """Malformed modules fail before the lowering and model-execution stages."""
    wasm_path = tmp_path / "malformed.wasm"
    wasm_path.write_bytes(b"\x00asm")

    with pytest.raises(ValueError, match="File too short"):
        compile_wasm_to_prefix(str(wasm_path))
