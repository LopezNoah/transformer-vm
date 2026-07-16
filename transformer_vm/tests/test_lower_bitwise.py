"""Conformance tests for variable-operand i32 bitwise lowering."""

import base64
import json
import shutil
import subprocess

import pytest

from transformer_vm.compilation.decoder import (
    OP_ELSE,
    OP_END,
    OP_I32_ADD,
    OP_I32_AND,
    OP_I32_CONST,
    OP_I32_EQZ,
    OP_I32_GE_U,
    OP_I32_LOAD,
    OP_I32_LOAD8_U,
    OP_I32_OR,
    OP_I32_ROTL,
    OP_I32_ROTR,
    OP_I32_SHL,
    OP_I32_SHR_S,
    OP_I32_SHR_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_SUB,
    OP_I32_XOR,
    OP_IF,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_RETURN,
    decode,
)
from transformer_vm.compilation.lower import check_basic_only, lower_hard_ops

MASK32 = 0xFFFFFFFF
EDGE_VALUES = [0, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 0x80000000]
OPCODES = {"and": OP_I32_AND, "or": OP_I32_OR, "xor": OP_I32_XOR}
SHIFT_OPCODES = {
    "shl": OP_I32_SHL,
    "shr_u": OP_I32_SHR_U,
    "shr_s": OP_I32_SHR_S,
    "rotl": OP_I32_ROTL,
    "rotr": OP_I32_ROTR,
}
SHIFT_COUNTS = [0, 1, 7, 8, 15, 16, 31, 32, 33, 255, 0xFFFFFFFF]


def _wasm_bitop(opcode: int) -> bytes:
    """Build a minimal `(i32, i32) -> i32` WASM module for a single bit op."""
    body = bytes([0, OP_LOCAL_GET, 0, OP_LOCAL_GET, 1, opcode, OP_END])
    code = bytes([1, len(body)]) + body
    return b"\x00asm\x01\x00\x00\x00" + b"\x01\x07\x01\x60\x02\x7f\x7f\x01\x7f" + b"\x03\x02\x01\x00" + b"\x07\x0b\x01\x07compute\x00\x00" + bytes([10, len(code)]) + code


def _wasm_const_shift(opcode: int, count: int) -> bytes:
    """Build a two-parameter module whose shift count is an i32.const."""
    body = bytes([0, OP_LOCAL_GET, 0, OP_I32_CONST, count, opcode, OP_END])
    code = bytes([1, len(body)]) + body
    return b"\x00asm\x01\x00\x00\x00" + b"\x01\x07\x01\x60\x02\x7f\x7f\x01\x7f" + b"\x03\x02\x01\x00" + b"\x07\x0b\x01\x07compute\x00\x00" + bytes([10, len(code)]) + code


def _node_results(
    modules: dict[str, bytes], left_inputs=EDGE_VALUES, right_inputs=None
) -> dict[str, list[int]]:
    right_inputs = left_inputs if right_inputs is None else right_inputs
    payload = {name: base64.b64encode(module).decode("ascii") for name, module in modules.items()}
    script = """
const [modules, leftInputs, rightInputs] = JSON.parse(process.argv[1]);
(async () => {
  const output = {};
  for (const [name, encoded] of Object.entries(modules)) {
    const {instance} = await WebAssembly.instantiate(Buffer.from(encoded, 'base64'));
    output[name] = leftInputs.flatMap(a => rightInputs.map(b => instance.exports.compute(a, b) >>> 0));
  }
  process.stdout.write(JSON.stringify(output));
})();
"""
    result = subprocess.run(
        ["node", "-e", script, json.dumps([payload, left_inputs, right_inputs])],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _uleb(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def _sleb(value: int) -> bytes:
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        done = (value == 0 and not byte & 0x40) or (value == -1 and byte & 0x40)
        encoded.append(byte | (0x80 if not done else 0))
        if done:
            return bytes(encoded)


def _lowered_wasm(func) -> bytes:
    """Encode a lowered `(i32, i32) -> i32` function for an independent runtime."""
    memarg_ops = {OP_I32_LOAD, OP_I32_LOAD8_U, OP_I32_STORE, OP_I32_STORE8}
    block_ops = {OP_IF}
    instruction_bytes = bytearray()
    for instruction in func.instructions:
        instruction_bytes.append(instruction.opcode)
        if instruction.opcode == OP_I32_CONST:
            instruction_bytes.extend(_sleb(instruction.immediates[0]))
        elif instruction.opcode in memarg_ops:
            instruction_bytes.extend(_uleb(instruction.immediates[0]))
            instruction_bytes.extend(_uleb(instruction.immediates[1]))
        elif instruction.opcode in block_ops:
            instruction_bytes.append(instruction.immediates[0])
        elif instruction.immediates:
            instruction_bytes.extend(_uleb(instruction.immediates[0]))

    locals_ = b"\x00" if not func.num_locals else b"\x01" + _uleb(func.num_locals) + b"\x7f"
    body = locals_ + bytes(instruction_bytes)
    code = b"\x01" + _uleb(len(body)) + body
    # A one-page memory is required by byte-based lowering helpers.
    return (
        b"\x00asm\x01\x00\x00\x00"
        + b"\x01\x07\x01\x60\x02\x7f\x7f\x01\x7f"
        + b"\x03\x02\x01\x00"
        + b"\x05\x03\x01\x00\x01"
        + b"\x07\x0b\x01\x07compute\x00\x00"
        + b"\x0a"
        + _uleb(len(code))
        + code
    )


def _matching_control(instructions, start: int) -> tuple[int | None, int]:
    depth = 1
    else_index = None
    for index in range(start, len(instructions)):
        opcode = instructions[index].opcode
        if opcode == OP_IF:
            depth += 1
        elif opcode == OP_END:
            depth -= 1
            if depth == 0:
                return else_index, index
        elif opcode == OP_ELSE and depth == 1:
            else_index = index
    raise AssertionError("unterminated control instruction")


def _run_lowered(func, left: int, right: int) -> int:
    """Execute the basic instruction subset emitted by variable bitwise lowering."""
    locals_ = [left & MASK32, right & MASK32] + [0] * func.num_locals
    memory = bytearray(8)
    stack: list[int] = []
    pc = 0
    instructions = func.instructions
    while pc < len(instructions):
        instruction = instructions[pc]
        opcode = instruction.opcode
        if opcode == OP_I32_CONST:
            stack.append(instruction.immediates[0] & MASK32)
        elif opcode == OP_LOCAL_GET:
            stack.append(locals_[instruction.immediates[0]])
        elif opcode == OP_LOCAL_SET:
            locals_[instruction.immediates[0]] = stack.pop()
        elif opcode == OP_I32_ADD:
            stack.append((stack.pop() + stack.pop()) & MASK32)
        elif opcode == OP_I32_SUB:
            rhs, lhs = stack.pop(), stack.pop()
            stack.append((lhs - rhs) & MASK32)
        elif opcode == OP_I32_GE_U:
            rhs, lhs = stack.pop(), stack.pop()
            stack.append(int(lhs >= rhs))
        elif opcode == OP_I32_EQZ:
            stack.append(int(stack.pop() == 0))
        elif opcode == OP_I32_STORE:
            value, address = stack.pop(), stack.pop()
            offset = instruction.immediates[1]
            memory[address + offset : address + offset + 4] = value.to_bytes(4, "little")
        elif opcode == OP_I32_STORE8:
            value, address = stack.pop(), stack.pop()
            memory[address + instruction.immediates[1]] = value & 0xFF
        elif opcode == OP_I32_LOAD:
            address = stack.pop() + instruction.immediates[1]
            stack.append(int.from_bytes(memory[address : address + 4], "little"))
        elif opcode == OP_I32_LOAD8_U:
            stack.append(memory[stack.pop() + instruction.immediates[1]])
        elif opcode == OP_IF:
            else_index, end_index = _matching_control(instructions, pc + 1)
            if not stack.pop():
                pc = (else_index if else_index is not None else end_index) + 1
                continue
        elif opcode == OP_ELSE:
            _, end_index = _matching_control(instructions, pc + 1)
            pc = end_index + 1
            continue
        elif opcode == OP_END:
            pass
        elif opcode == OP_RETURN:
            return stack.pop()
        else:
            raise AssertionError(f"unexpected lowered opcode: {opcode:#x}")
        pc += 1
    assert stack, "lowered function produced no result"
    return stack.pop()


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
def test_variable_bitops_match_node_wasm_runtime():
    """Lowered results match an independent WebAssembly runtime at bit boundaries."""
    expected = _node_results({name: _wasm_bitop(opcode) for name, opcode in OPCODES.items()})
    for name, opcode in OPCODES.items():
        original = decode(_wasm_bitop(opcode)).functions[0]
        lowered = lower_hard_ops(original, num_params=2, native_crypto=False)
        assert check_basic_only(lowered) == {}
        actual = [_run_lowered(lowered, left, right) for left in EDGE_VALUES for right in EDGE_VALUES]
        assert actual == expected[name]


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
def test_shift_and_rotate_lowering_matches_node_wasm_runtime():
    """Shift counts use WASM's low five bits and signed shifts retain their sign."""
    left_inputs = [0, 1, 0x7F, 0x80, 0xFF]
    originals = {name: _wasm_bitop(opcode) for name, opcode in SHIFT_OPCODES.items()}
    expected = _node_results(originals, left_inputs, SHIFT_COUNTS)
    for name, original_wasm in originals.items():
        original = decode(original_wasm).functions[0]
        lowered = lower_hard_ops(original, num_params=2, native_crypto=False)
        assert check_basic_only(lowered) == {}
        actual = _node_results({name: _lowered_wasm(lowered)}, left_inputs, SHIFT_COUNTS)[name]
        assert actual == expected[name]


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
def test_signed_right_shift_const_preserves_high_bit():
    """The byte-aligned signed expansion sign-extends negative i32 values."""
    original_wasm = _wasm_const_shift(OP_I32_SHR_S, 8)
    original = decode(original_wasm).functions[0]
    lowered = lower_hard_ops(original, num_params=2, native_crypto=False)
    assert check_basic_only(lowered) == {}
    inputs = [0x80000000, 0xFFFFFFFF]
    expected = _node_results({"shr_s": original_wasm}, inputs)
    actual = _node_results({"shr_s": _lowered_wasm(lowered)}, inputs)
    assert actual == expected
