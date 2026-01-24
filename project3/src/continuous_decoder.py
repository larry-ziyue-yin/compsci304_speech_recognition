# src/continuous_decoder.py

import numpy as np

LOG_ZERO = -1e10

# =====TODO: implement Viterbi decoding=====
def viterbi_decode(hmm, observations):
    """
    Perform Viterbi decoding for a CompositeHMM.
    Support non-emitting states and insertion penalties.
    """
    T = len(observations)
    N = hmm.transition.shape[0]

    # =====TODO: initialize Viterbi score and backpointer matrices =====
    V = np.full((N, T), LOG_ZERO)
    B = np.zeros((N, T), dtype=int)

    # =====TODO: initialize start state =====
    V[hmm.start_state, 0] = 0.0

    # =====TODO: main Viterbi loop over frames and states =====
    for t in range(T):
        for s in range(N):
            # - compute scores from previous states
            # - update backpointer B
            # - add emission probability if state is emitting
            pass
    # =====TODO: end main loop =====

    # =====TODO: backtrace the best path =====
    best_final = np.argmax(V[:, -1])
    path = [best_final]
    for t in range(T - 1, 0, -1):
        # follow backpointers
        pass
    # =====TODO: end backtrace =====

    return path[::-1]
# =====TODO: end viterbi_decode =====


# =====TODO: implement state sequence → digit sequence conversion=====
def states_to_digits(path, hmm):
    """
    Convert Viterbi state sequence to digit sequence.
    
    """
    digits = []
    prev = None

    for s in path:
        # - check if s is emitting and get corresponding digit
        pass

    return digits
# =====TODO: end states_to_digits =====
