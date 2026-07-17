"""WASM-001 boundary behavior for integer and memory instructions."""

import pytest

from transformer_vm.wasm.reference import MEMORY_BYTES, run


def _run(*program):
    return run(list(program), trace=True)


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        ("i32.div_u", 7, 3, 2),
        ("i32.div_s", -7, 3, -2),
        ("i32.div_s", 7, -3, -2),
        ("i32.rem_u", 7, 3, 1),
        ("i32.rem_s", -7, 3, -1),
        ("i32.rem_s", 7, -3, 1),
    ],
)
def test_division_and_remainder_semantics(op, left, right, expected):
    *_, halted, trapped, trace = _run(
        ("i32.const", left),
        ("i32.const", right),
        (op, 0),
        ("halt", 0),
    )

    assert halted and not trapped
    assert trace[-6:-2] == [f"{(expected & 0xFFFFFFFF) >> (8 * i) & 0xFF:02x}" for i in range(4)]


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        ("i32.div_u", 1, 0),
        ("i32.rem_u", 1, 0),
        ("i32.div_s", -2147483648, -1),
    ],
)
def test_division_traps(op, left, right):
    *_, halted, trapped, trace = _run(
        ("i32.const", left),
        ("i32.const", right),
        (op, 0),
    )

    assert not halted and trapped
    assert trace[-1] == "trap"


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("i32.extend8_s", 0x80, 0xFFFFFF80),
        ("i32.extend16_s", 0x8000, 0xFFFF8000),
    ],
)
def test_sign_extensions(op, value, expected):
    *_, halted, trapped, trace = _run(("i32.const", value), (op, 0), ("halt", 0))

    assert halted and not trapped
    assert trace[-6:-2] == [f"{expected >> (8 * i) & 0xFF:02x}" for i in range(4)]


@pytest.mark.parametrize("op", ["i32.load", "i32.load8_u", "i32.load16_s"])
def test_out_of_bounds_load_traps(op):
    *_, halted, trapped, trace = _run(("i32.const", MEMORY_BYTES), (op, 0))

    assert not halted and trapped
    assert trace[-1] == "trap"


def test_out_of_bounds_store_traps():
    *_, halted, trapped, trace = _run(
        ("i32.const", MEMORY_BYTES),
        ("i32.const", 0),
        ("i32.store", 0),
    )

    assert not halted and trapped
    assert trace[-1] == "trap"
