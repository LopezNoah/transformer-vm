"""Capability profile selection and graph isolation tests."""

import pytest

from transformer_vm.graph.core import InputDimension, ProgramGraph, persist, reset_graph
from transformer_vm.runner import required_profile
from transformer_vm.specialize import parse_program
from transformer_vm.wasm.interpreter import CRYPTO_OPCODES, WASMMachine


def test_base_graph_excludes_native_crypto_vocabulary_and_circuitry():
    base = WASMMachine(profile="base").build()
    full = WASMMachine(profile="full").build()

    assert CRYPTO_OPCODES.isdisjoint(base.input_tokens)
    assert full.input_tokens.keys() >= CRYPTO_OPCODES
    assert len(base.all_dims) < len(full.all_dims)
    assert len(base.all_lookups) < len(full.all_lookups)


def test_base_specialization_rejects_native_crypto_program():
    program = [
        {"opcode": "i32.and", "bytes": [0, 0, 0, 0]},
        {"opcode": "halt", "bytes": [0, 0, 0, 0]},
    ]

    with pytest.raises(ValueError, match="i32.and"):
        WASMMachine(program=program, profile="base").build()


def test_program_graph_prunes_unreachable_registered_operations():
    reset_graph()
    source = InputDimension("source")
    reachable = persist(source, name="reachable")
    orphan = next(iter(persist(source + 1, name="orphan").terms))

    graph = ProgramGraph({"source": source * 1}, {"result": reachable})

    assert next(iter(reachable.terms)) in graph.all_dims
    assert orphan not in graph.all_dims


def test_required_profile_is_derived_from_program_opcodes(tmp_path):
    base_program = tmp_path / "base.txt"
    base_program.write_text("{ i32.const 00 00 00 00 halt 00 00 00 00 }\n")
    crypto_program = tmp_path / "crypto.txt"
    crypto_program.write_text("{ i32.xor 00 00 00 00 halt 00 00 00 00 }\n")

    assert required_profile(base_program) == "base"
    assert required_profile(crypto_program) == "full"
    assert {instruction["opcode"] for instruction in parse_program(base_program)} == {
        "i32.const",
        "halt",
    }
