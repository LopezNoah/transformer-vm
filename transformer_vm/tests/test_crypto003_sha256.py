"""Known-answer coverage for the CRYPTO-003 SHA-256 example."""

import shutil
from pathlib import Path

import pytest

from transformer_vm.compilation.compile_wasm import compile_c_to_wasm, compile_wasm_to_prefix
from transformer_vm.wasm.reference import load_program_from_string, run


@pytest.fixture(scope="module")
def sha256_program(examples_dir, tmp_path_factory):
    source = tmp_path_factory.mktemp("sha256") / "sha256.c"
    shutil.copyfile(Path(examples_dir) / "sha256.c", source)
    wasm_path = compile_c_to_wasm(str(source))
    prefix, input_base = compile_wasm_to_prefix(wasm_path, profile="full")
    return load_program_from_string(prefix), input_base


# "abc" and the 448-bit message are SHA-256 vectors from RFC 6234 section 8.
# The other lengths exercise both sides of the padding and block boundaries.
@pytest.mark.parametrize(
    ("message", "digest"),
    [
        (b"", b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            b"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
        (b"a" * 55, b"9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"),
        (b"a" * 64, b"ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"),
    ],
    ids=["empty", "short", "rfc-448-bit", "padding-boundary", "full-block"],
)
def test_sha256_known_answers(sha256_program, message, digest):
    program, input_base = sha256_program
    _, _, output, halted, trapped = run(
        program, message, input_base=input_base, max_tokens=5_000_000
    )

    assert output == digest + b"\n"
    assert halted and not trapped
