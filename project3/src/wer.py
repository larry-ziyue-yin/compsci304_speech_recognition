# src/wer.py

import numpy as np

def edit_distance(ref, hyp):
    """
    Calculate edit distance (Levenshtein distance) between two sequences.
    ref and hyp should be lists or strings.
    """
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
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    For digit recognition, each digit is treated as a word.
    
    Args:
        ref: reference string (e.g., "123")
        hyp: hypothesis string (e.g., "124")
    
    Returns:
        WER as a float (edit_distance / num_words_in_ref)
    """
    # Convert strings to lists of digits (each digit is a word)
    ref_list = list(ref) if isinstance(ref, str) else ref
    hyp_list = list(hyp) if isinstance(hyp, str) else hyp
    
    if len(ref_list) == 0:
        # If reference is empty, WER is 1.0 if hyp is non-empty, 0.0 if hyp is also empty
        return 1.0 if len(hyp_list) > 0 else 0.0
    
    edit_dist = edit_distance(ref_list, hyp_list)
    return edit_dist / len(ref_list)
