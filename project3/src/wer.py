# src/wer.py

import numpy as np

def edit_distance(ref, hyp):
    D = np.zeros((len(ref)+1, len(hyp)+1))

    for i in range(len(ref)+1):
        D[i, 0] = i
    for j in range(len(hyp)+1):
        D[0, j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            D[i, j] = min(
                D[i-1, j] + 1,
                D[i, j-1] + 1,
                D[i-1, j-1] + cost
            )
    return D[-1, -1]


def word_error_rate(ref, hyp):
    return edit_distance(ref, hyp) / len(ref)
