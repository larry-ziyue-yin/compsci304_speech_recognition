# src/grammar.py

# =====TODO: implement DigitLoopGrammar=====
class DigitLoopGrammar:
    """
    Grammar for continuous digit sequences:
        start(0) --digit--> end(1)
        end(1) --loop--> start(0)
        
    """

    def __init__(self, digits=range(10)):
        # =====TODO: set number of states, start and end states =====
        self.n_states = 2
        self.start_state = 0
        self.end_state = 1

        # =====TODO: initialize edges list =====
        self.edges = []

        # =====TODO: add edges for each digit =====
        # For each digit, create an edge from start to end with the digit label
        for digit in digits:
            self.edges.append((self.start_state, self.end_state, digit))

        # =====TODO: add loop-back edge (non-emitting) from end to start =====
        self.edges.append((self.end_state, self.start_state, None))
# =====TODO: end DigitLoopGrammar =====