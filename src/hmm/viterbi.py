"""Log-space Viterbi decoder for map-matching.

Hidden states at time step t are the candidate road arcs generated for GPS
observation t. Each step can have a different number of candidates, so the
decoder works with ragged inputs:

    emission_logp[t]      : shape (n_t,)          log P(observation_t | candidate j)
    transition_logp[t]    : shape (n_{t-1}, n_t)  log P(candidate j at t | candidate i at t-1)
                            (transition_logp[0] is ignored)

Robustness rules
----------------
* An observation with no candidates (n_t == 0) receives no arc (index -1) and
  the chain restarts at the next observation that has candidates.
* If every transition into step t is impossible (all -inf), the route is
  treated as broken: the best path up to t-1 is frozen and decoding restarts
  at t using emissions only. Break positions are reported so that route
  continuity can be analysed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

NEG_INF = -np.inf


@dataclass
class ViterbiResult:
    path: list[int]                      # candidate index per observation (-1 = no candidate)
    log_prob: float                      # sum of segment log-probabilities
    breaks: list[int] = field(default_factory=list)  # steps where a new segment started


def _backtrack(backptr: list[np.ndarray], last_best: int, start: int, end: int) -> list[int]:
    """Recover the best state sequence for steps start..end (inclusive)."""
    seq = [last_best]
    for t in range(end, start, -1):
        seq.append(int(backptr[t][seq[-1]]))
    seq.reverse()
    return seq


def viterbi(
    emission_logp: Sequence[np.ndarray],
    transition_logp: Sequence[np.ndarray | None],
) -> ViterbiResult:
    """Decode the most probable candidate sequence.

    Parameters
    ----------
    emission_logp
        List of length T; element t is a 1-D array of log emission scores.
    transition_logp
        List of length T; element t (t >= 1) is a 2-D array of log transition
        scores from candidates at t-1 to candidates at t. Element 0 may be None.
    """
    T = len(emission_logp)
    if len(transition_logp) != T:
        raise ValueError("emission_logp and transition_logp must have the same length")

    path = [-1] * T
    breaks: list[int] = []
    total = 0.0

    delta: np.ndarray | None = None      # running scores for the current segment
    backptr: list[np.ndarray] = [np.empty(0, dtype=int)] * T
    seg_start = -1                       # first step of the current segment
    prev_t = -1                          # last step that had candidates

    def close_segment(end_t: int) -> None:
        nonlocal total
        if delta is None or seg_start < 0:
            return
        best_last = int(np.argmax(delta))
        total += float(delta[best_last])
        seq = _backtrack(backptr, best_last, seg_start, end_t)
        for offset, idx in enumerate(seq):
            path[seg_start + offset] = idx

    for t in range(T):
        emis = np.asarray(emission_logp[t], dtype=float)
        n_t = emis.shape[0]

        if n_t == 0:                     # missing candidates: close the segment
            close_segment(prev_t)
            delta, seg_start, prev_t = None, -1, -1
            continue

        if delta is None:                # start of a new segment
            if t > 0:
                breaks.append(t)
            delta = emis.copy()
            backptr[t] = np.full(n_t, -1, dtype=int)
            seg_start, prev_t = t, t
            continue

        trans = np.asarray(transition_logp[t], dtype=float)
        if trans.shape != (delta.shape[0], n_t):
            raise ValueError(
                f"transition_logp[{t}] has shape {trans.shape}, expected {(delta.shape[0], n_t)}"
            )

        scores = delta[:, None] + trans  # (n_{t-1}, n_t)
        best_prev = np.argmax(scores, axis=0)
        best_score = scores[best_prev, np.arange(n_t)]

        if np.all(np.isneginf(best_score)):   # disconnected: restart here
            close_segment(prev_t)
            breaks.append(t)
            delta = emis.copy()
            backptr[t] = np.full(n_t, -1, dtype=int)
            seg_start, prev_t = t, t
            continue

        delta = best_score + emis
        backptr[t] = best_prev
        prev_t = t

    close_segment(prev_t)
    return ViterbiResult(path=path, log_prob=total, breaks=breaks)


def log(p: np.ndarray | list | float) -> np.ndarray:
    """Natural log that maps probability 0 to -inf without warnings."""
    arr = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(arr)
