"""Optional kikuchipy adapter; importing this module does not import kikuchipy."""
import math
import numpy as np

from . import index_patterns


class _PatternRows:
    """Flatten navigation axes while computing only requested lazy row blocks."""

    def __init__(self, data):
        self.shape = (math.prod(data.shape[:-2]), *data.shape[-2:])
        self.data = data.reshape(self.shape)

    def __getitem__(self, rows):
        block = self.data[rows]
        if hasattr(block, "compute"):
            block = block.compute()
        return np.asarray(block)


def dictionary_indexing_gpu(signal, dictionary, *, metric="ncc", keep_n=20,
                            pattern_batch=128, dictionary_batch=4096,
                            signal_mask=None, device="cuda"):
    """Match EBSD/LazyEBSD signals using the batched CUDA matching core.

    Returns IndexingResult (host scores and zero-based simulation indices),
    NOT a CrystalMap. Rows follow C-order signal.data navigation order.
    NCC subtracts each pattern's mean after masking; NDP only L2-normalizes.
    True signal_mask pixels are excluded, as in kikuchipy.
    CPU mode is explicit and intended for verification. No CPU fallback.
    """
    if metric not in ("ncc", "ndp"):
        raise ValueError("metric must be 'ncc' or 'ndp'")
    if not isinstance(keep_n, int) or isinstance(keep_n, bool) or keep_n < 1:
        raise ValueError("keep_n must be a positive integer")
    p, d = signal.data, dictionary.data
    if len(p.shape) < 3 or len(d.shape) != 3:
        raise ValueError("Expected navigation axes plus two detector axes; dictionary must have one navigation axis")
    if tuple(p.shape[-2:]) != tuple(d.shape[-2:]):
        raise ValueError("Experimental and dictionary detector shapes must match")
    if math.prod(p.shape[:-2]) == 0 or d.shape[0] == 0:
        raise ValueError("Experimental patterns and dictionary must be nonempty")
    mask = None
    if signal_mask is not None:
        excluded = np.asarray(signal_mask)
        if excluded.dtype != np.bool_ or excluded.shape != p.shape[-2:]:
            raise ValueError("signal_mask must be boolean and match the detector shape")
        mask = ~excluded
    return index_patterns(
        _PatternRows(p), _PatternRows(d), top_k=min(keep_n, d.shape[0]),
        pattern_batch=pattern_batch, dictionary_batch=dictionary_batch,
        center=metric == "ncc", mask=mask, device=device,
    )
