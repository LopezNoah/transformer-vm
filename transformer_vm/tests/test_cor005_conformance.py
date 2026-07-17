"""Focused differential conformance coverage for all execution backends."""

import base64
import json
import os
import platform
import shutil
import subprocess

import pytest
import torch

from transformer_vm._paths import DATA_DIR
from transformer_vm.attention import HullKVCache, StandardKVCache
from transformer_vm.compilation.compile_wasm import compile_c_to_wasm, compile_wasm_to_prefix
from transformer_vm.model.weights import build_model, save_weights
from transformer_vm.wasm.reference import load_program_from_string, run

pytestmark = [pytest.mark.slow, pytest.mark.conformance]


def _run_node_wasm(wasm_path, input_str):
    """Execute the project's compute ABI through Node's independent V8 engine."""
    script = r"""
const fs = require('fs');
const bytes = fs.readFileSync(process.argv[1]);
const payload = Buffer.from(process.argv[2]);
const input = Buffer.alloc(4 + payload.length + 1);
input.writeUInt32LE(payload.length, 0);
payload.copy(input, 4);
const output = [];
(async () => {
  try {
    const {instance} = await WebAssembly.instantiate(bytes, {
      env: {output_byte: value => output.push(value & 255)},
    });
    const memory = new Uint8Array(instance.exports.memory.buffer);
    memory.set(input, instance.exports.__heap_base.value);
    instance.exports.compute(instance.exports.__heap_base.value + 4);
    process.stdout.write(JSON.stringify({status: 'ok', output: Buffer.from(output).toString('base64')}));
  } catch (error) {
    process.stdout.write(JSON.stringify({status: 'trap', output: Buffer.from(output).toString('base64')}));
  }
})();
"""
    result = subprocess.run(
        ["node", "-e", script, str(wasm_path), input_str],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return payload["status"], base64.b64decode(payload["output"])


def _generate_model_tokens(model, all_tokens, tok_to_idx_map, input_tokens, cache_class, max_new_tokens):
    token_ids = [tok_to_idx_map[token] for token in input_tokens]
    generated = model.generate_with_cache(
        torch.tensor([token_ids], dtype=torch.long),
        max_new_tokens=max_new_tokens,
        cache_class=cache_class,
    )
    return [all_tokens[token_id] for token_id in generated[0].tolist()]


def _generate_graph_tokens(input_tokens, max_new_tokens):
    from transformer_vm.evaluator import Runtime

    runtime = Runtime(use_hull=False)
    try:
        vals = None
        for token in input_tokens:
            vals = runtime.step(token)
        generated = list(input_tokens)
        for _ in range(max_new_tokens):
            token = runtime.predict_next(vals)
            generated.append(token)
            if token in {"halt", "trap"}:
                return generated
            vals = runtime.step(token)
        raise AssertionError("graph execution exhausted its generation limit")
    finally:
        runtime.destroy()


def _compile_cpp(binary_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = "transformer_vm/model/transformer.cpp"
    command = [compiler, "-std=c++17", "-O3", "-I", "transformer_vm/attention", source, "-o", str(binary_path)]
    if platform.system() == "Darwin":
        command[3:3] = ["-framework", "Accelerate"]
    subprocess.run(command, check=True)


def _read_cpp_trace(path):
    tokens = []
    for line in path.read_bytes().splitlines():
        length, token = line.split(b":", 1)
        assert len(token) == int(length)
        tokens.append(token.decode())
    return tokens


def _read_tokens(path):
    with open(path) as file:
        return file.read().split()


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
@pytest.mark.parametrize("input_str", ["0", "1", "2147483647"])
def test_reference_matches_node_for_arithmetic_control_memory_locals_and_calls(tmp_path, input_str):
    fixture = tmp_path / "conformance.c"
    shutil.copyfile("transformer_vm/tests/fixtures/conformance.c", fixture)
    wasm_path = compile_c_to_wasm(str(fixture))
    prefix, input_base = compile_wasm_to_prefix(wasm_path)

    node_status, node_output = _run_node_wasm(wasm_path, input_str)
    _instructions, _tokens, reference_output, halted, trapped = run(
        load_program_from_string(prefix), input_str, input_base=input_base
    )

    assert node_status == "ok"
    assert halted
    assert not trapped
    assert reference_output == node_output


@pytest.mark.skipif(shutil.which("node") is None, reason="requires the Node WebAssembly runtime")
def test_reference_matches_node_trap(tmp_path):
    fixture = tmp_path / "trap.c"
    fixture.write_text("void compute(const char *input) { (void)input; __builtin_trap(); }\n")
    wasm_path = compile_c_to_wasm(str(fixture))
    prefix, input_base = compile_wasm_to_prefix(wasm_path)

    node_status, node_output = _run_node_wasm(wasm_path, "")
    _instructions, _tokens, reference_output, halted, trapped = run(
        load_program_from_string(prefix), input_base=input_base
    )

    assert node_status == "trap"
    assert node_output == b""
    assert reference_output == b""
    assert not halted
    assert trapped


@pytest.fixture(scope="module")
def universal_model():
    model, all_tokens, tok_to_idx_map, _ = build_model(plan_path=None)
    return model, all_tokens, tok_to_idx_map


def test_graph_matches_pytorch_standard_attention(universal_model):
    model, all_tokens, tok_to_idx_map = universal_model
    program_path = os.path.join(DATA_DIR, "hello.txt")
    ref_path = os.path.join(DATA_DIR, "hello_ref.txt")
    input_tokens = _read_tokens(program_path)
    reference_tokens = _read_tokens(ref_path)
    max_new_tokens = len(reference_tokens) - len(input_tokens) + 1

    graph_tokens = _generate_graph_tokens(input_tokens, max_new_tokens)
    standard_tokens = _generate_model_tokens(
        model, all_tokens, tok_to_idx_map, input_tokens, StandardKVCache, max_new_tokens
    )

    assert graph_tokens == standard_tokens == reference_tokens


def test_pytorch_standard_attention_matches_hull(universal_model):
    model, all_tokens, tok_to_idx_map = universal_model
    input_tokens = _read_tokens(os.path.join(DATA_DIR, "hello.txt"))
    reference_tokens = _read_tokens(os.path.join(DATA_DIR, "hello_ref.txt"))
    max_new_tokens = len(reference_tokens) - len(input_tokens) + 1

    standard_tokens = _generate_model_tokens(
        model, all_tokens, tok_to_idx_map, input_tokens, StandardKVCache, max_new_tokens
    )
    hull_tokens = _generate_model_tokens(
        model, all_tokens, tok_to_idx_map, input_tokens, HullKVCache, max_new_tokens
    )

    assert standard_tokens == hull_tokens == reference_tokens


def test_python_hull_inference_matches_cpp(universal_model, tmp_path):
    model, all_tokens, tok_to_idx_map = universal_model
    program_path = os.path.join(DATA_DIR, "hello.txt")
    input_tokens = _read_tokens(program_path)
    reference_tokens = _read_tokens(os.path.join(DATA_DIR, "hello_ref.txt"))
    max_new_tokens = len(reference_tokens) - len(input_tokens) + 1
    python_tokens = _generate_model_tokens(
        model, all_tokens, tok_to_idx_map, input_tokens, HullKVCache, max_new_tokens
    )

    weights_path = tmp_path / "model.bin"
    binary_path = tmp_path / "transformer"
    trace_path = tmp_path / "trace.txt"
    save_weights(model, all_tokens, weights_path)
    _compile_cpp(binary_path)
    subprocess.run(
        [str(binary_path), str(weights_path), f"--trace-file={trace_path}", program_path],
        check=True,
    )

    assert _read_cpp_trace(trace_path) == python_tokens
