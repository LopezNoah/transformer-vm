"""Measure reference-trace expansion caused by WASM instruction lowering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from transformer_vm.wasm.reference import run

from .compile_wasm import compile_function
from .decoder import (
    OP_END,
    OP_I32_AND,
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
from .lower import lower_hard_ops

MASK32 = 0xFFFFFFFF
CRYPTO_OPERANDS = (0x81, 7)
MAX_NATIVE_CRYPTO_TOKENS = 36
MIN_CRYPTO_TRACE_REDUCTION = 10

CRYPTO_OPCODES = {
    "i32.and": OP_I32_AND,
    "i32.or": OP_I32_OR,
    "i32.xor": OP_I32_XOR,
    "i32.shl": OP_I32_SHL,
    "i32.shr_s": OP_I32_SHR_S,
    "i32.shr_u": OP_I32_SHR_U,
    "i32.rotl": OP_I32_ROTL,
    "i32.rotr": OP_I32_ROTR,
}


@dataclass(frozen=True)
class TraceExpansion:
    """Trace-token cost of one source WASM opcode for the crypto workload."""

    opcode: str
    native_tokens: int
    lowered_tokens: int

    @property
    def reduction(self) -> float:
        return 1 - self.native_tokens / self.lowered_tokens


def _trace_tokens(opcode: int, native_crypto: bool) -> int:
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
    compiled = compile_function(
        lower_hard_ops(body, num_params=2, native_crypto=native_crypto), WasmModule()
    )
    program = [
        ("i32.const", CRYPTO_OPERANDS[0]),
        ("local.set", 0),
        ("i32.const", CRYPTO_OPERANDS[1]),
        ("local.set", 1),
    ]
    for index, (op, immediate) in enumerate(compiled):
        value = int.from_bytes(bytes(immediate), "little")
        if op in {"br", "br_if"}:
            value = (value - index - 1) & MASK32
        program.append((op, value))
    return run(program)[1]


def measure_crypto_trace_expansion() -> list[TraceExpansion]:
    """Measure each native crypto opcode against its legacy base-op lowering."""
    return [
        TraceExpansion(
            name,
            native_tokens=_trace_tokens(opcode, native_crypto=True),
            lowered_tokens=_trace_tokens(opcode, native_crypto=False),
        )
        for name, opcode in CRYPTO_OPCODES.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("opcode        native  lowered  reduction")
    for result in measure_crypto_trace_expansion():
        print(
            f"{result.opcode:<13} {result.native_tokens:>6} {result.lowered_tokens:>8}"
            f" {result.reduction:>9.1%}"
        )


if __name__ == "__main__":
    main()
