"""EMsoft-style normalized dictionary matching on NVIDIA GPUs."""
from dataclasses import dataclass
import numpy as np
from .simulation import detector_rays, generate_patterns


@dataclass
class IndexingResult:
    indices: np.ndarray
    scores: np.ndarray
    orientations: np.ndarray | None = None


def index_patterns(patterns, dictionary, *, orientations=None, top_k=1,
                   pattern_batch=128, dictionary_batch=4096, device="cuda",
                   mask=None, center=False):
    """Match (N, pixels) or (N, height, width) array-like pattern sources.

    Sources must expose shape and support row slicing (NumPy, memmap, HDF5
    datasets). Results are host arrays, with zero-based dictionary indices.
    Inputs must already have consistent detector geometry and preprocessing.
    center=False preserves EMsoft's uncentered normalized inner product.
    Equal scores are ordered by ascending dictionary index.
    """
    if device == "cuda":
        try:
            import cupy as xp
        except ImportError as exc:
            raise RuntimeError("Install the CUDA extra matching your NVIDIA driver") from exc
        if xp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No NVIDIA CUDA GPU available")
        host = xp.asnumpy
    elif device == "cpu":
        xp, host = np, np.asarray
    else:
        raise ValueError("device must be 'cuda' or 'cpu'")
    if len(patterns.shape) < 2 or len(dictionary.shape) < 2:
        raise ValueError("Sources need a pattern axis and pixel axes")
    n, m = patterns.shape[0], dictionary.shape[0]
    pixels = int(np.prod(patterns.shape[1:]))
    if pixels != int(np.prod(dictionary.shape[1:])) or pixels == 0:
        raise ValueError("Pattern pixel counts must match and be nonzero")
    for name, value in (("top_k", top_k), ("pattern_batch", pattern_batch),
                        ("dictionary_batch", dictionary_batch)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > m:
        raise ValueError("top_k exceeds dictionary size")
    angles = None if orientations is None else np.asarray(orientations)
    if angles is not None and (angles.ndim != 2 or angles.shape[0] != m):
        raise ValueError("orientations must have one row per dictionary pattern")
    selection = None
    if mask is not None:
        selection = np.asarray(mask, dtype=bool).reshape(-1)
        if selection.size != pixels or not selection.any():
            raise ValueError("mask must select at least one pixel and match pattern size")
        selection = xp.asarray(selection)

    def prepare(source, start, stop):
        a = xp.asarray(source[start:stop], dtype=xp.float32).reshape(stop-start, pixels)
        if selection is not None:
            a = a[:, selection]
        if not bool(xp.isfinite(a).all()):
            raise ValueError("Patterns contain nonfinite pixels")
        if center:
            a = a - a.mean(axis=1, keepdims=True)
        norm = xp.linalg.norm(a, axis=1, keepdims=True)
        if bool((norm == 0).any()) or not bool(xp.isfinite(norm).all()):
            raise ValueError("Pattern has zero or nonfinite norm after preprocessing")
        return a / norm

    indices = np.empty((n, top_k), dtype=np.int64)
    scores = np.empty((n, top_k), dtype=np.float32)
    for start in range(0, n, pattern_batch):
        stop = min(start + pattern_batch, n)
        experimental = prepare(patterns, start, stop)
        best_s = xp.empty((stop-start, 0), dtype=xp.float32)
        best_i = xp.empty((stop-start, 0), dtype=xp.int64)
        for ds in range(0, m, dictionary_batch):
            de = min(ds + dictionary_batch, m)
            simulated = prepare(dictionary, ds, de)
            block = experimental @ simulated.T
            ids = xp.broadcast_to(xp.arange(ds, de), block.shape)
            candidates_s = xp.concatenate((best_s, block), axis=1)
            candidates_i = xp.concatenate((best_i, ids), axis=1)
            # Sort the secondary key first, then stably sort descending scores.
            # CuPy 13 argsort is stable by default; lexsort has no axis argument.
            by_index = xp.argsort(candidates_i, axis=1)
            sorted_scores = xp.take_along_axis(candidates_s, by_index, axis=1)
            if device == "cpu":
                by_score = xp.argsort(-sorted_scores, axis=1, kind="stable")
            else:
                by_score = xp.argsort(-sorted_scores, axis=1)
            order = xp.take_along_axis(by_index, by_score[:, :top_k], axis=1)
            best_s = xp.take_along_axis(candidates_s, order, axis=1)
            best_i = xp.take_along_axis(candidates_i, order, axis=1)
        indices[start:stop], scores[start:stop] = host(best_i), host(best_s)
    return IndexingResult(indices, scores, None if angles is None else angles[indices])
