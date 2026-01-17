import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class GaussianHMM:
    def __init__(self, n_states=5, n_mixtures=1):
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.transition_matrix = None
        self.start_prob = None
        self.gaussians = [] 
        
    def initialize(self, n_features):
        """
        Initialize HMM parameters for a left-to-right Gaussian HMM.
        """
        self.n_features = n_features 
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        # =============TODO: Initialize transition matrix =============
        # Create a (n_states x n_states) transition matrix for a left-to-right HMM.
        # - From state i, transitions are allowed only to:
        #     * itself (self-loop)
        #     * the next state (i+1)
        # - The last state should be an absorbing state
        #
        # Hint:
        #   transition_matrix[i, i]     = 
        #   transition_matrix[i, i + 1] = 
        # =============TODO: Initialize transition matrix =============
        
        self.start_prob = np.zeros(self.n_states)
        # =============TODO: Initialize start probability =============
        # The HMM should always start from the first state.
        #
        # Hint:
        #   start_prob[0] = 
        # =============TODO: Initialize start probability =============
        
        self.gaussians = []
        # =============TODO: Initialize emission distributions =============
        # -------------------------
        # For each state:
        # - If n_mixtures == 1:
        #     initialize a single Gaussian with:
        #       mean = zero vector
        #       covariance = identity matrix
        #
        # - If n_mixtures > 1:
        #     initialize a GMM with:
        #       equal mixture weights
        #       each component having zero mean and identity covariance
        # =============TODO: Initialize emission distributions =============
    
    def compute_emission_prob(self, observation, state_idx):
        """
        Compute emission probability P(o | state).
        """
        if self.n_mixtures == 1:
            gaussian = self.gaussians[state_idx]
            mean = gaussian['mean']
            cov = gaussian['cov']
            
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            
            try:
                prob = multivariate_normal.pdf(observation, mean=mean, cov=cov)
            except:
                prob = 1e-100
            return prob
        else:
            # GMM
            gmm = self.gaussians[state_idx]
            prob = 0.0
            for mixture in gmm:
                mean = mixture['mean']
                cov = mixture['cov']
                weight = mixture['weight']
                
                cov = cov + np.eye(cov.shape[0]) * 1e-6
                
                try:
                    prob += weight * multivariate_normal.pdf(observation, mean=mean, cov=cov)
                except:
                    prob += 1e-100
            return prob
    
    def viterbi_decode(self, observations):
        """
        Perform Viterbi decoding for left-to-right HMM.
        """
        T = len(observations)
        N = self.n_states

        viterbi = np.full((N, T), -np.inf)
        backpointer = np.zeros((N, T), dtype=int)

        # =============TODO  Viterbi initialization=============
        # -------------------------
        # Initialize viterbi[:, 0] using:
        #   log(start_prob[state]) + log(emission_prob)
        #
        # Only states with non-zero start probability should be initialized.
        # =============TODO  Viterbi initialization=============

        #  =============TODO Viterbi recursion with left-to-right constraintn=============
        # -------------------------
        # For each time t and state s:
        # - The previous state can only be:
        #     * s     (self-loop)
        #     * s - 1 (left-to-right transition)
        #
        # - Choose the best previous state using max()
        # - Store both:
        #     * best score
        #     * backpointer
        #  =============TODO Viterbi recursion with left-to-right constraintn=============

        best_path = np.zeros(T, dtype=int)
        #  =============TODO Backtracking=============
        # - Find the best final state
        # - Backtrack using backpointer to recover the best state sequence
        #  =============TODO Backtracking=============
        

        best_score = np.max(viterbi[:, -1])
        return best_path, best_score

    