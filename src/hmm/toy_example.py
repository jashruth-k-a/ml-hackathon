"""Hand-checkable Viterbi example (Member 3 - Decoder, first task).

Three GPS observations, two candidate arcs each:

    obs 0 : A, B      emissions 0.6, 0.4
    obs 1 : C, D      emissions 0.3, 0.7
    obs 2 : E, F      emissions 0.8, 0.2

Picking the best arc independently gives A -> D -> E, but A and D are NOT
connected in the road network (transition probability 0). Viterbi must find
the best *connected* route instead.

Each scenario lists hand-computed path probabilities, and the result is also
cross-checked against brute-force enumeration of every path.

Run from the project root:
    python -m src.hmm.toy_example
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from src.hmm.viterbi import log, viterbi

CANDS = [["A", "B"], ["C", "D"], ["E", "F"]]
EMIS = [[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]]
# Transition into obs 2 is the same in every scenario: from C or D -> E 0.9, F 0.1
T_12 = [[0.9, 0.1], [0.9, 0.1]]

SCENARIOS = [
    {
        "name": "1. Greedy route is disconnected (A->D impossible)",
        "T_01": [[0.9, 0.0], [0.5, 0.5]],   # rows: from A, B ; cols: to C, D
        "hand": "ACE = .6*.9*.3*.9*.8 = 0.11664 ; BDE = .4*.5*.7*.9*.8 = 0.10080 ; ADE = 0",
        "expected": ["A", "C", "E"],
    },
    {
        "name": "2. Connect A->D: Viterbi should now follow the greedy route",
        "T_01": [[0.5, 0.5], [0.5, 0.5]],
        "hand": "ADE = .6*.5*.7*.9*.8 = 0.15120 ; BDE = 0.10080 ; ACE = 0.06480",
        "expected": ["A", "D", "E"],
    },
    {
        "name": "3. Weaken A->C, force B->D: path switches to start at B",
        "T_01": [[0.2, 0.0], [0.0, 1.0]],
        "hand": "BDE = .4*1.*.7*.9*.8 = 0.20160 ; ACE = .6*.2*.3*.9*.8 = 0.02592",
        "expected": ["B", "D", "E"],
    },
]


def build(t01):
    emis = [log(e) for e in EMIS]
    trans = [None, log(t01), log(T_12)]
    return emis, trans


def greedy(emis_p):
    return [CANDS[t][int(np.argmax(e))] for t, e in enumerate(emis_p)]


def brute_force(t01):
    best, best_p = None, -1.0
    for idx in itertools.product(range(2), repeat=3):
        p = EMIS[0][idx[0]]
        p *= t01[idx[0]][idx[1]] * EMIS[1][idx[1]]
        p *= T_12[idx[1]][idx[2]] * EMIS[2][idx[2]]
        if p > best_p:
            best, best_p = idx, p
    return [CANDS[t][i] for t, i in enumerate(best)], best_p


def run() -> str:
    lines: list[str] = []
    all_ok = True
    lines.append(f"Greedy (emission-only) route: {'-'.join(greedy(EMIS))}\n")

    for sc in SCENARIOS:
        emis, trans = build(sc["T_01"])
        res = viterbi(emis, trans)
        decoded = [CANDS[t][i] for t, i in enumerate(res.path)]
        bf_path, bf_p = brute_force(sc["T_01"])
        ok = decoded == sc["expected"] == bf_path and np.isclose(np.exp(res.log_prob), bf_p)
        all_ok &= ok
        lines += [
            sc["name"],
            f"   hand calc : {sc['hand']}",
            f"   expected  : {'-'.join(sc['expected'])}",
            f"   decoded   : {'-'.join(decoded)}   P = {np.exp(res.log_prob):.5f}",
            f"   brute     : {'-'.join(bf_path)}   P = {bf_p:.5f}",
            f"   result    : {'PASS' if ok else 'FAIL'}",
            "",
        ]

    # Scenario 4: every transition into obs 2 impossible -> route break.
    emis = [log(e) for e in EMIS]
    trans = [None, log(SCENARIOS[0]["T_01"]), log([[0.0, 0.0], [0.0, 0.0]])]
    res = viterbi(emis, trans)
    decoded = [CANDS[t][i] for t, i in enumerate(res.path)]
    expected = ["A", "C", "E"]   # segment 1: AC = .6*.9*.3 = .162 > BD = .14 ; segment 2: E (.8)
    ok = decoded == expected and res.breaks == [2]
    all_ok &= ok
    lines += [
        "4. No valid transition into obs 2 -> decoder must split the route",
        "   hand calc : seg1 AC = .162, BD = .140, BC = .060 ; seg2 E = .8 > F = .2",
        f"   expected  : {'-'.join(expected)}  breaks=[2]",
        f"   decoded   : {'-'.join(decoded)}  breaks={res.breaks}",
        f"   result    : {'PASS' if ok else 'FAIL'}",
        "",
    ]

    # Scenario 5: obs 1 has no candidates at all.
    emis = [log(EMIS[0]), np.array([]), log(EMIS[2])]
    trans = [None, np.empty((2, 0)), np.empty((0, 2))]
    res = viterbi(emis, trans)
    ok = res.path == [0, -1, 0] and res.breaks == [2]
    all_ok &= ok
    lines += [
        "5. Obs 1 has no candidates -> no arc for obs 1, chain restarts at obs 2",
        "   expected  : path idx [0, -1, 0]  breaks=[2]",
        f"   decoded   : path idx {res.path}  breaks={res.breaks}",
        f"   result    : {'PASS' if ok else 'FAIL'}",
        "",
    ]

    lines.append("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return "\n".join(lines)


if __name__ == "__main__":
    report = run()
    print(report)
    out = Path(__file__).resolve().parents[2] / "results" / "hmm" / "toy_example.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out}")
