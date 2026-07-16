"""Completion requirements shared by execution and reference paths."""

import pytest

from transformer_vm.evaluator import run_program
from transformer_vm.runner import run_model_program
from transformer_vm.wasm.reference import generate_ref, run


class _GeneratedTokens:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __getitem__(self, index):
        assert index == 0
        return self

    def tolist(self):
        return self.token_ids


class _Model:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def generate_with_cache(self, _input, **_kwargs):
        return _GeneratedTokens(self.token_ids)


def _run_model(tmp_path, predicted, reference=None):
    program = tmp_path / "program.txt"
    program.write_text("{ }\n")
    ref_path = None
    if reference is not None:
        ref_path = tmp_path / "program_ref.txt"
        ref_path.write_text(" ".join(reference) + "\n")
    tokens = ["{", "}", "halt", "extra"]
    model = _Model([tokens.index(token) for token in predicted])
    return run_model_program(model, tokens, {token: i for i, token in enumerate(tokens)}, program, ref_path)


def test_python_runner_requires_exact_reference_length_and_halt(tmp_path):
    assert _run_model(tmp_path, ["{", "}", "halt"], ["{", "}", "halt"])[0]
    assert not _run_model(tmp_path, ["{", "}"], ["{", "}", "halt"])[0]
    assert not _run_model(tmp_path, ["{", "}", "halt", "extra"], ["{", "}", "halt"])[0]
    assert not _run_model(tmp_path, ["{", "}"])[0]


def test_graph_runner_requires_halt(monkeypatch, tmp_path):
    class Runtime:
        def __init__(self, use_hull):
            self.generated = iter(["halt"])

        def step(self, _token):
            return object()

        def predict_next(self, _vals):
            return next(self.generated)

        def destroy(self):
            pass

    program = tmp_path / "program.txt"
    ref_path = tmp_path / "program_ref.txt"
    program.write_text("{ }\n")
    ref_path.write_text("{ } halt\n")
    monkeypatch.setattr("transformer_vm.evaluator.Runtime", Runtime)

    assert run_program(program, ref_path)

    class NonHaltingRuntime(Runtime):
        def __init__(self, use_hull):
            self.generated = iter(["not_halt"] * 50_000)

    monkeypatch.setattr("transformer_vm.evaluator.Runtime", NonHaltingRuntime)
    assert not run_program(program, ref_path)


def test_reference_run_reports_limit_exhaustion():
    _instructions, token_count, _output, halted, trapped = run([("i32.const", 0)], max_tokens=5)

    assert token_count == 5
    assert not halted
    assert not trapped


def test_reference_generator_does_not_write_incomplete_trace(tmp_path, monkeypatch):
    program = tmp_path / "program.txt"
    ref_path = tmp_path / "program_ref.txt"
    program.write_text("{\ni32.const 00 00 00 00\n}\n")
    monkeypatch.setattr(
        "transformer_vm.wasm.reference.run",
        lambda *_args, **_kwargs: (1, 5, "", False, False, ["00", "00", "00", "00"]),
    )

    with pytest.raises(RuntimeError, match="did not emit halt"):
        generate_ref(program, ref_path)

    assert not ref_path.exists()
