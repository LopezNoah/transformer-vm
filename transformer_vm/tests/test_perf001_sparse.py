"""PERF-001 sparse artifact and projection coverage."""

import torch

from transformer_vm.model.transformer import VanillaTransformer
from transformer_vm.model.weights import load_vocabulary, load_weights, save_weights


def test_sparse_artifact_round_trip_executes_all_projections(tmp_path):
    torch.manual_seed(0)
    model = VanillaTransformer(vocab=4, d_model=4, n_heads=2, n_layers=1, d_ffn=3)
    model.attn_erase = [[]]
    model.ffn_erase = [[]]
    model.head_tiebreak = [[0, 0]]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter[parameter.abs() < 0.4] = 0

    path = tmp_path / "sparse.bin"
    tokens = ["a", "b", "halt", "trap"]
    save_weights(model, tokens, path)
    loaded, loaded_tokens, _ = load_weights(path)

    assert loaded_tokens == tokens
    assert load_vocabulary(path) == tokens
    assert set(loaded.sparse_projections) == {
        "embedding",
        "attn.0.qkv",
        "attn.0.out",
        "ffn.0.in",
        "ffn.0.out",
        "head",
    }
    assert all(projection.layout == torch.sparse_csr for projection in loaded.sparse_projections.values())

    x = torch.randn(4)
    assert torch.allclose(
        model.attn[0].in_proj_weight @ x,
        loaded._project("attn.0.qkv", loaded.attn[0].in_proj_weight, x),
    )
    assert torch.allclose(
        model.attn[0].out_proj.weight @ x,
        loaded._project("attn.0.out", loaded.attn[0].out_proj.weight, x),
    )
    assert torch.allclose(
        model.ff_in[0].weight @ x,
        loaded._project("ffn.0.in", loaded.ff_in[0].weight, x),
    )
    assert torch.allclose(
        model.ff_out[0].weight @ x[:3],
        loaded._project("ffn.0.out", loaded.ff_out[0].weight, x[:3]),
    )
    assert torch.allclose(model.head.weight @ x, loaded._project("head", loaded.head.weight, x))
    assert torch.allclose(model.tok.weight[1], loaded._embedding(1))


def test_sparse_loader_marks_empty_layers_inactive(tmp_path):
    model = VanillaTransformer(vocab=2, d_model=4, n_heads=2, n_layers=1, d_ffn=2)
    model.attn_erase = [[]]
    model.ffn_erase = [[]]
    model.head_tiebreak = [[0, 0]]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    path = tmp_path / "empty-layer.bin"
    save_weights(model, ["halt", "trap"], path)
    loaded, _, _ = load_weights(path)
    assert loaded.inactive_layers == (0,)
    assert loaded.generate_with_cache(torch.tensor([[1]]), max_new_tokens=1).tolist() == [[1, 0]]
