# src/composite_hmm.py

import numpy as np
from scipy.stats import multivariate_normal


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

        # =====TODO: loop over grammar edges =====
        # if edge has a digit:
        #   - insert digit HMM states
        #   - connect grammar non-emitting state to digit HMM entry
        #   - connect digit HMM exit to next grammar state
        #   - register emitting states and corresponding digits
        # else:
        #   - handle non-emitting state edge with insertion penalty
        # =====TODO: end loop =====

        # =====TODO: build dense transition matrix =====

        # =====TODO: set start_state, final_state, emitting_start =====
        pass
    # =====TODO: end build =====

    # =====TODO: implement emission_logprob=====
    def emission_logprob(self, obs, state):
        """
        Compute log probability of observation given a state.
        """
        # if state is emitting:
        #   - handle single Gaussian or GMM
        pass
    # =====TODO: end emission_logprob =====
