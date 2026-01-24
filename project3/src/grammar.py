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
        self.n_states = None
        self.start_state = None
        self.end_state = None

        # =====TODO: initialize edges list =====
        self.edges = []

        # =====TODO: add edges for each digit =====
        # For each digit, create an edge from start to end with the digit label
        pass

        # =====TODO: add loop-back edge (non-emitting) from end to start =====
        pass
# =====TODO: end DigitLoopGrammar =====
