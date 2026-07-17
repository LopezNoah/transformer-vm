"""WASM-002 i64 pair legalization conformance tests."""

import json
import shutil
import subprocess

import pytest

from transformer_vm.compilation.compile_wasm import compile_wasm_to_prefix
from transformer_vm.wasm.reference import load_program_from_string, run


def _u32(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _s64(value):
    value &= (1 << 64) - 1
    if value >= 1 << 63:
        value -= 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        done = (value == 0 and not byte & 0x40) or (value == -1 and byte & 0x40)
        out.append(byte | (0x00 if done else 0x80))
        if done:
            return bytes(out)


def _section(section_id, payload):
    return bytes([section_id]) + _u32(len(payload)) + payload


def _i64_binary_module(left, right, opcode=0x7C):
    """Build a small module that stores then outputs a binary i64 result."""
    types = b"\x02\x60\x01\x7f\x00\x60\x00\x00"
    imports = b"\x01\x03env\x0boutput_byte\x00\x00"
    functions = b"\x01\x01"
    memory = b"\x01\x00\x01"
    exports = b"\x01\x07compute\x00\x01"
    instructions = b"\x00\x41\x00\x42" + _s64(left) + b"\x42" + _s64(right) + bytes([opcode]) + b"\x37\x03\x00"
    instructions += b"\x41\x04\x28\x02\x00\x10\x00\x41\x00\x28\x02\x00\x10\x00\x0b"
    code = b"\x01" + _u32(len(instructions)) + instructions
    return b"\x00asm\x01\x00\x00\x00" + _section(1, types) + _section(2, imports) + _section(3, functions) + _section(5, memory) + _section(7, exports) + _section(10, code)


def _node_output(wasm_path):
    script = """
const fs = require('fs');
const output = [];
(async () => {
  const {instance} = await WebAssembly.instantiate(fs.readFileSync(process.argv[1]),
    {env: {output_byte: value => output.push(value & 255)}});
  instance.exports.compute();
  process.stdout.write(JSON.stringify(output));
})();
"""
    completed = subprocess.run(["node", "-e", script, str(wasm_path)], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
@pytest.mark.parametrize("left,right", [(0xFFFFFFFF, 1), (0x7FFFFFFF00000000, 0x100000000)])
def test_i64_add_is_legalized_and_matches_v8(tmp_path, left, right):
    wasm_path = tmp_path / "i64-add.wasm"
    wasm_path.write_bytes(_i64_binary_module(left, right))

    prefix, _ = compile_wasm_to_prefix(str(wasm_path))
    program = load_program_from_string(prefix)
    *_, output, halted, trapped = run(program)

    assert halted and not trapped
    assert list(output.encode("latin-1")) == _node_output(wasm_path)
    assert "i64." not in prefix


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
@pytest.mark.parametrize("opcode", [0x86, 0x87, 0x88, 0x89, 0x8A])
@pytest.mark.parametrize("count", [0, 1, 31, 32, 33, 63, 64, 65])
def test_i64_shifts_and_rotates_match_v8(tmp_path, opcode, count):
    wasm_path = tmp_path / "i64-shift.wasm"
    wasm_path.write_bytes(_i64_binary_module(0x8123456789ABCDEF, count, opcode))

    prefix, _ = compile_wasm_to_prefix(str(wasm_path))
    *_, output, halted, trapped = run(load_program_from_string(prefix), max_tokens=200_000)

    assert halted and not trapped
    assert list(output.encode("latin-1")) == _node_output(wasm_path)
