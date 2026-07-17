"""Correctness and trace-size coverage for CRYPTO-001 primitives."""

import json
import shutil
import subprocess

import pytest

from transformer_vm.compilation.compile_wasm import (
    compile_c_to_wasm,
    compile_function,
    compile_wasm_to_prefix,
)
from transformer_vm.compilation.decoder import (
    OP_END,
    OP_I32_AND,
    OP_I32_MUL,
    OP_I32_OR,
    OP_I32_ROTL,
    OP_I32_ROTR,
    OP_I32_SHL,
    OP_I32_SHR_S,
    OP_I32_SHR_U,
    OP_I32_XOR,
    OP_LOCAL_GET,
    FuncBody,
    WasmInstr,
    WasmModule,
)
from transformer_vm.compilation.lower import lower_hard_ops
from transformer_vm.wasm.reference import load_program_from_string, run

MASK32 = 0xFFFFFFFF
NATIVE_OPS = {
    "i32.and": OP_I32_AND,
    "i32.or": OP_I32_OR,
    "i32.xor": OP_I32_XOR,
    "i32.shl": OP_I32_SHL,
    "i32.shr_s": OP_I32_SHR_S,
    "i32.shr_u": OP_I32_SHR_U,
    "i32.rotl": OP_I32_ROTL,
    "i32.rotr": OP_I32_ROTR,
}


def _native_program(opcode, left, right):
    return [
        ("i32.const", left),
        ("i32.const", right),
        (next(name for name, value in NATIVE_OPS.items() if value == opcode), 0),
        ("halt", 0),
    ]


def _expected(name, left, right):
    count = right & 31
    if name == "i32.and":
        return left & right
    if name == "i32.or":
        return left | right
    if name == "i32.xor":
        return left ^ right
    if name == "i32.shl":
        return (left << count) & MASK32
    if name == "i32.shr_u":
        return left >> count
    if name == "i32.shr_s":
        signed = left - (1 << 32) if left & (1 << 31) else left
        return (signed >> count) & MASK32
    if name == "i32.rotl":
        return ((left << count) | (left >> ((32 - count) & 31))) & MASK32
    return ((left >> count) | (left << ((32 - count) & 31))) & MASK32


@pytest.mark.parametrize("name,opcode", NATIVE_OPS.items())
@pytest.mark.parametrize(
    "left,right",
    [(0, 0), (MASK32, 1), (0x80000000, 31), (0xAAAAAAAA, 32), (0x12345678, 255)],
)
def test_native_crypto_ops_are_exact_32_bit(name, opcode, left, right):
    instructions, tokens, _output, halted, trapped, trace = run(
        _native_program(opcode, left, right), trace=True
    )
    result = int.from_bytes(bytes(int(token.rstrip("'"), 16) for token in trace[10:14]), "little")

    assert result == _expected(name, left, right)
    assert (instructions, tokens, halted, trapped) == (4, 16, True, False)


@pytest.fixture(scope="module")
def crypto_graph_runtime():
    from transformer_vm.evaluator import Runtime
    from transformer_vm.wasm.interpreter import WASMMachine

    runtime = Runtime(use_hull=False, program_graph=WASMMachine(profile="full").build())
    yield runtime
    runtime.destroy()


@pytest.mark.slow
def test_native_crypto_ops_execute_in_graph(crypto_graph_runtime):
    left = 0x89ABCDEF
    for right in [0, 1, 7, 8, 15, 16, 31, 32, 33, 255]:
        for name, opcode in NATIVE_OPS.items():
            program = _native_program(opcode, left, right)
            prefix = ["{"]
            for op, immediate in program:
                prefix.extend(
                    [op, *[f"{byte:02x}" for byte in immediate.to_bytes(4, "little")]]
                )
            prefix.append("}")
            expected = run(program, trace=True)[-1]

            crypto_graph_runtime.reset()
            values = None
            for token in prefix:
                values = crypto_graph_runtime.step(token)
            actual = []
            for _ in range(len(expected)):
                token = crypto_graph_runtime.predict_next(values)
                actual.append(token)
                if token == "halt":
                    break
                values = crypto_graph_runtime.step(token)

            assert actual == expected, (name, right)


def test_lowering_preserves_native_crypto_ops_by_default():
    body = FuncBody(
        locals=[],
        num_locals=0,
        instructions=[
            WasmInstr(OP_LOCAL_GET, (0,)),
            WasmInstr(OP_LOCAL_GET, (1,)),
            WasmInstr(OP_I32_XOR),
            WasmInstr(OP_END),
        ],
    )

    native = lower_hard_ops(body, num_params=2)
    lowered = lower_hard_ops(body, num_params=2, native_crypto=False)

    assert [instruction.opcode for instruction in native.instructions] == [
        OP_LOCAL_GET,
        OP_LOCAL_GET,
        OP_I32_XOR,
        OP_END,
    ]
    assert len(lowered.instructions) > len(native.instructions)


def test_lowering_preserves_native_crypto_when_other_ops_are_lowered():
    body = FuncBody(
        locals=[],
        num_locals=0,
        instructions=[
            WasmInstr(OP_LOCAL_GET, (0,)),
            WasmInstr(OP_LOCAL_GET, (1,)),
            WasmInstr(OP_I32_XOR),
            WasmInstr(OP_LOCAL_GET, (0,)),
            WasmInstr(OP_I32_MUL),
            WasmInstr(OP_END),
        ],
    )

    lowered = lower_hard_ops(body, num_params=2)
    opcodes = [instruction.opcode for instruction in lowered.instructions]

    assert OP_I32_XOR in opcodes
    assert OP_I32_MUL not in opcodes


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
def test_runtime_crypto_helpers_match_independent_wasm(tmp_path):
    source = tmp_path / "crypto_helpers.c"
    source.write_text(
        r"""
static void out32(unsigned int value) {
    putchar(value); putchar(value >> 8); putchar(value >> 16); putchar(value >> 24);
}

void compute(const char *input) {
    unsigned int carry, borrow;
    unsigned char little[4], big[4];
    unsigned int value;
    value = tvm_addcarry_u32(0xffffffffu, 1u, 0u, &carry); out32(value); out32(carry);
    value = tvm_addcarry_u32(0xffffffffu, 0u, 1u, &carry); out32(value); out32(carry);
    value = tvm_addcarry_u32(0x7fffffffu, 1u, 0u, &carry); out32(value); out32(carry);
    value = tvm_subborrow_u32(0u, 1u, 0u, &borrow); out32(value); out32(borrow);
    value = tvm_subborrow_u32(0u, 0xffffffffu, 1u, &borrow); out32(value); out32(borrow);
    value = tvm_subborrow_u32(0x80000000u, 1u, 0u, &borrow); out32(value); out32(borrow);
    tvm_store32_le(little, 0x89abcdefu);
    tvm_store32_be(big, 0x01234567u);
    putchar(little[0]); putchar(little[1]); putchar(little[2]); putchar(little[3]);
    putchar(big[0]); putchar(big[1]); putchar(big[2]); putchar(big[3]);
    out32(tvm_load32_le(little)); out32(tvm_load32_be(big));
    out32(tvm_select_u32(0xaaaaaaaau, 0x55555555u, input[0] != 0));
    out32(tvm_select_u32(0xaaaaaaaau, 0x55555555u, 0u));
}
"""
    )
    wasm_path = compile_c_to_wasm(str(source))
    prefix, input_base = compile_wasm_to_prefix(wasm_path)

    script = r"""
const fs = require('fs');
(async () => {
  const output = [];
  const {instance} = await WebAssembly.instantiate(fs.readFileSync(process.argv[1]), {
    env: {output_byte: value => output.push(value & 255)},
  });
  const memory = new Uint8Array(instance.exports.memory.buffer);
  memory[instance.exports.__heap_base.value] = 1;
  instance.exports.compute(instance.exports.__heap_base.value);
  process.stdout.write(JSON.stringify(output));
})();
"""
    node_output = json.loads(
        subprocess.run(
            ["node", "-e", script, str(wasm_path)], check=True, capture_output=True, text=True
        ).stdout
    )
    _instructions, _tokens, output, halted, trapped = run(
        load_program_from_string(prefix), "x", input_base=input_base
    )

    arithmetic_words = [
        0,
        1,
        0,
        1,
        0x80000000,
        0,
        0xFFFFFFFF,
        1,
        0,
        1,
        0x7FFFFFFF,
        0,
    ]
    expected = b"".join(word.to_bytes(4, "little") for word in arithmetic_words)
    expected += bytes.fromhex("efcdab8901234567")
    expected += b"".join(
        word.to_bytes(4, "little")
        for word in [0x89ABCDEF, 0x01234567, 0xAAAAAAAA, 0x55555555]
    )
    assert bytes(node_output) == expected
    assert output == expected
    assert halted and not trapped


@pytest.mark.parametrize("name,opcode", NATIVE_OPS.items())
def test_native_ops_reduce_reference_trace_tokens(name, opcode):
    body = FuncBody(
        locals=[],
        num_locals=0,
        instructions=[
            WasmInstr(OP_LOCAL_GET, (0,)),
            WasmInstr(OP_LOCAL_GET, (1,)),
            WasmInstr(opcode),
            WasmInstr(OP_END),
        ],
    )
    native = lower_hard_ops(body, num_params=2)
    lowered = lower_hard_ops(body, num_params=2, native_crypto=False)
    module = WasmModule()

    def execute(compiled_body):
        compiled = compile_function(compiled_body, module)
        program = [
            ("i32.const", 0x81),
            ("local.set", 0),
            ("i32.const", 7),
            ("local.set", 1),
        ]
        for index, (op, immediate) in enumerate(compiled):
            value = int.from_bytes(bytes(immediate), "little")
            if op in {"br", "br_if"}:
                value = (value - index - 1) & MASK32
            program.append((op, value))
        return run(program)[1]

    native_tokens = execute(native)
    lowered_tokens = execute(lowered)
    assert native_tokens == 36
    assert lowered_tokens > native_tokens, name
