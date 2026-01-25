# src/continuous_decoder.py

from collections import deque
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
    emit0 = hmm.emitting_start

    # =====TODO: initialize Viterbi score and backpointer matrices =====
    V = np.full((N, T), LOG_ZERO, dtype=float)
    B = np.zeros((N, T), dtype=int)

    # Precompute predecessor lists for speed
    preds = [np.where(hmm.transition[:, s] > LOG_ZERO / 2)[0] for s in range(N)]

    # Precompute epsilon outgoing edges: edges that go to NON-emitting states (dest < emit0)
    eps_out = [[] for _ in range(N)]
    for i in range(N):
        js = np.where((hmm.transition[i, :] > LOG_ZERO / 2) & (np.arange(N) < emit0))[0]
        for j in js:
            eps_out[i].append((int(j), float(hmm.transition[i, j])))

    # =====TODO: initialize start state =====
    prev = np.full(N, LOG_ZERO, dtype=float)
    prev[hmm.start_state] = 0.0

    # =====TODO: main Viterbi loop over frames and states =====
    for t in range(T):
        cur = np.full(N, LOG_ZERO, dtype=float)
        bp = np.zeros(N, dtype=int)

        # 1) Update EMITTING states: consume observation[t]
        for s in range(emit0, N):
            p = preds[s]
            if p.size == 0:
                continue
            scores = prev[p] + hmm.transition[p, s]
            k = int(np.argmax(scores))
            best = float(scores[k])
            if best <= LOG_ZERO / 2:
                continue
            cur[s] = best + float(hmm.emission_logprob(observations[t], s))
            bp[s] = int(p[k])

        # 2) Epsilon-closure within the SAME frame t:
        #    propagate scores via transitions that land in NON-emitting states (no observation consumed)
        scores = cur.copy()
        bp_all = bp.copy()

        inq = np.zeros(N, dtype=bool)
        q = deque()

        # Start relaxation from all currently reachable states (mostly emitting)
        for i in range(N):
            if scores[i] > LOG_ZERO / 2 and eps_out[i]:
                q.append(i)
                inq[i] = True

        while q:
            i = q.popleft()
            inq[i] = False
            si = scores[i]
            if si <= LOG_ZERO / 2:
                continue
            for j, lp in eps_out[i]:
                cand = si + lp
                if cand > scores[j] + 1e-12:
                    scores[j] = cand
                    bp_all[j] = i
                    if (not inq[j]) and eps_out[j]:
                        q.append(j)
                        inq[j] = True

        V[:, t] = scores
        B[:, t] = bp_all
        prev = scores  # for next frame

    # =====TODO: backtrace the best path (one state per frame) =====
    path = [0] * T

    # choose best EMITTING state at last frame (otherwise non-emitting backtrace needs extra bookkeeping)
    if emit0 < N:
        state = int(np.argmax(V[emit0:, -1]) + emit0)
    else:
        state = int(np.argmax(V[:, -1]))

    for t in range(T - 1, 0, -1):
        path[t] = state
        pred = int(B[state, t])  # predecessor at time t-1 (may be non-emitting)

        # Ensure we land on an EMITTING state for frame t-1:
        st = pred
        steps = 0
        while st < emit0 and steps < N:
            st = int(B[st, t - 1])  # unwind epsilon chain inside time (t-1)
            steps += 1

        if st < emit0 and emit0 < N:
            st = int(np.argmax(V[emit0:, t - 1]) + emit0)

        state = st

    path[0] = state
    # =====TODO: end backtrace =====

    return path
# =====TODO: end viterbi_decode =====


# =====TODO: implement state sequence → digit sequence conversion=====
def states_to_digits(path, hmm):
    """
    Convert Viterbi state sequence to digit sequence.
    Return a string (so main_continues can compare with ref directly).
    """
    digits = []
    prev = None

    for s in path:
        if s >= hmm.emitting_start:
            d = hmm.state_digit[s - hmm.emitting_start]
            if d != prev:
                digits.append(d)
                prev = d

    return "".join(digits)
# =====TODO: end states_to_digits =====