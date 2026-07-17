"""O(n) reference hard-attention KV cache."""

import torch


class StandardKVCache:
    """Brute-force hard-attention cache.

    Every lookup selects all maximum-score keys. Ties return their mean by
    default; heads configured with ``latest=True`` return the latest maximum.
    """

    def __init__(self, n_layers, n_heads):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self._keys = [[] for _ in range(n_layers)]
        self._vals = [[] for _ in range(n_layers)]
        self._latest = [[False] * n_heads for _ in range(n_layers)]

    def clear(self):
        """Reset all cached keys and values."""
        self._keys = [[] for _ in range(self.n_layers)]
        self._vals = [[] for _ in range(self.n_layers)]

    def set_tiebreak(self, layer, head, latest):
        """Set a head's tie policy: latest winner or mean of all winners."""
        self._latest[layer][head] = latest

    def layer_step(self, layer, keys, queries, values):
        """Append KV pair and compute hard-attention output for one layer."""
        self._keys[layer].append(keys.clone())
        self._vals[layer].append(values.clone())

        K = torch.stack(self._keys[layer]).reshape(-1, self.n_heads, keys.shape[0] // self.n_heads)
        V = torch.stack(self._vals[layer]).reshape(-1, self.n_heads, keys.shape[0] // self.n_heads)
        Q = queries.reshape(self.n_heads, -1)

        scores = torch.einsum("thi,hi->th", K, Q)
        max_scores = scores.max(dim=0, keepdim=True).values
        winners = scores >= max_scores
        if self._latest[layer].count(True) == 0:
            weights = winners / winners.sum(dim=0, keepdim=True)
            out = torch.einsum("th,thi->hi", weights, V)
        else:
            out = torch.empty_like(V[0])
            for head, latest in enumerate(self._latest[layer]):
                if latest:
                    out[head] = V[winners[:, head].nonzero()[-1, 0], head]
                else:
                    out[head] = V[winners[:, head], head].mean(dim=0)
        return out.flatten()
