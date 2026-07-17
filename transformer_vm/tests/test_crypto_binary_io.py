"""CRYPTO-002 binary input framing and deterministic output coverage."""

from transformer_vm.compilation.compile_wasm import format_input_section, format_spec_input
from transformer_vm.runner import _output_hex
from transformer_vm.wasm.reference import _extract_input, run


def test_binary_input_section_preserves_nuls_and_non_utf8_bytes():
    payload = bytes.fromhex("0001ff804100")
    tokens = ["{"] + ["}"] + format_input_section(payload).split()

    assert _extract_input(tokens) == payload
    assert format_spec_input(payload).split() == [
        "start",
        "06",
        "00",
        "00",
        "00",
        "00",
        "01",
        "ff",
        "80",
        "A",
        "00",
        "00",
        "commit(+0,sts=0,bt=0)",
    ]


def test_reference_output_is_bytes_and_has_deterministic_hex():
    program = [
        ("i32.const", 0),
        ("output", 0),
        ("i32.const", 0x80),
        ("output", 0),
        ("i32.const", 0xFF),
        ("output", 0),
        ("halt", 0),
    ]

    _instructions, _tokens, output, halted, trapped = run(program)

    assert output == b"\x00\x80\xff"
    assert output.hex() == "0080ff"
    assert _output_hex(["out(00)", "out(80)", "out(ff)"]) == "0080ff"
    assert halted and not trapped
