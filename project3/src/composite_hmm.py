# src/composite_hmm.py

import numpy as np
from scipy.stats import multivariate_normal

LOG_ZERO = -1e10

class CompositeHMM:
    def __init__(self):
        self.transition = None        # full transition matrix
        self.states = []              # emitting states only
        self.state_digit = []         # which digit each emitting state belongs to
        self.start_state = None
        self.final_state = None
        self.emitting_start = None    # index of first emitting state

    # =====TODO: implement build method=====
    def build(self, grammar, digit_models, insertion_penalty):
        """
        Insert digit HMMs into the grammar graph.
        """
        # =====TODO: initialize data structures =====
        n_grammar = grammar.n_states
        self.states = []
        self.state_digit = []
        
        # sparse transitions first (list of dict: row -> {col: logp})
        trans_rows = [dict() for _ in range(n_grammar)]
        next_state = n_grammar  # next free global state index for emitting states

        def _set_arc(i, j, logp):
            """Keep the best (max) logp if multiple arcs happen to overlap."""
            prev = trans_rows[i].get(j, LOG_ZERO)
            if logp > prev:
                trans_rows[i][j] = logp

        def _log(p):
            if p is None or p <= 0.0:
                return LOG_ZERO
            return float(np.log(p))

        def _get_model(digit_label):
            # digit_models may be dict keyed by '0'..'9' or list indexed by int
            if isinstance(digit_models, dict):
                return digit_models[str(digit_label)]
            return digit_models[int(digit_label)]

        # =====TODO: loop over grammar edges =====
        # if edge has a digit:
        #   - insert digit HMM states
        #   - connect grammar non-emitting state to digit HMM entry
        #   - connect digit HMM exit to next grammar state
        #   - register emitting states and corresponding digits
        # else:
        #   - handle non-emitting state edge with insertion penalty
        for (src, dst, label) in grammar.edges:
            if label is None:
                # non-emitting grammar edge (e.g., loop-back), apply insertion penalty in log domain
                _set_arc(src, dst, _log(float(insertion_penalty)))
                continue

            # edge has a digit: insert a *copy* of the digit HMM
            digit = str(label)
            model = _get_model(digit)

            n_states = int(model.n_states)
            offset = next_state  # first global index of this inserted HMM

            # extend sparse rows for newly added emitting states
            trans_rows.extend([dict() for _ in range(n_states)])

            # connect grammar state -> digit HMM entry using start_prob
            # (Proj2 typically has start_prob[0]=1, others 0, but we support general case)
            for i in range(n_states):
                sp = float(model.start_prob[i]) if hasattr(model, "start_prob") else (1.0 if i == 0 else 0.0)
                if sp > 0.0:
                    _set_arc(src, offset + i, _log(sp))

            # internal digit-HMM transitions
            tmat = model.transition_matrix
            for i in range(n_states):
                for j in range(n_states):
                    p = float(tmat[i, j])
                    if p > 0.0:
                        _set_arc(offset + i, offset + j, _log(p))

            # connect digit HMM exit -> next grammar state
            # Only allow exiting from the final (absorbing) state.
            last = n_states - 1
            _set_arc(offset + last, dst, 0.0)

            # register emitting states and their digit labels (for states_to_digits)
            for i in range(n_states):
                self.states.append(model.gaussians[i])  # dict (single Gaussian) or list (GMM)
                self.state_digit.append(digit)

            next_state += n_states
        # =====TODO: end loop =====

        # =====TODO: build dense transition matrix =====
        N = next_state
        dense = np.full((N, N), LOG_ZERO, dtype=float)
        for i, row in enumerate(trans_rows):
            for j, logp in row.items():
                dense[i, j] = float(logp)
        self.transition = dense

        # =====TODO: set start_state, final_state, emitting_start =====
        self.start_state = int(grammar.start_state)
        self.final_state = int(grammar.end_state)
        self.emitting_start = int(n_grammar)
        # =====TODO: end build =====

    # =====TODO: implement emission_logprob=====
    def emission_logprob(self, obs, state):
        """
        Compute log probability of observation given a state.
        """
        # if state is emitting:
        #   - handle single Gaussian or GMM
        state = int(state)
        if self.emitting_start is None or state < self.emitting_start:
            return 0.0

        idx = state - self.emitting_start
        spec = self.states[idx]

        # helper: stable logsumexp
        def _logsumexp(logvals):
            m = np.max(logvals)
            if np.isneginf(m) or m <= LOG_ZERO / 2:
                return LOG_ZERO
            return float(m + np.log(np.sum(np.exp(logvals - m))))

        # Single Gaussian: spec is a dict with 'mean' and 'cov'
        if isinstance(spec, dict):
            mean = spec["mean"]
            cov = spec["cov"]
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            try:
                return float(multivariate_normal.logpdf(obs, mean=mean, cov=cov, allow_singular=True))
            except Exception:
                return LOG_ZERO

        # GMM: spec is a list of {'mean','cov','weight'}
        if isinstance(spec, list):
            log_terms = []
            for mix in spec:
                mean = mix["mean"]
                cov = mix["cov"]
                w = float(mix.get("weight", 0.0))
                if w <= 0.0:
                    continue
                cov = cov + np.eye(cov.shape[0]) * 1e-6
                try:
                    lp = float(multivariate_normal.logpdf(obs, mean=mean, cov=cov, allow_singular=True))
                except Exception:
                    lp = LOG_ZERO
                log_terms.append(np.log(w) + lp)

            if not log_terms:
                return LOG_ZERO
            return _logsumexp(np.array(log_terms, dtype=float))

        # unexpected format
        return LOG_ZERO
    # =====TODO: end emission_logprob =====