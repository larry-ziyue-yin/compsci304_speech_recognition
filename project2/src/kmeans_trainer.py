import numpy as np
from sklearn.cluster import KMeans
from .hmm_model import GaussianHMM

class SegmentalKMeansTrainer:
    def __init__(self, n_states=5, n_mixtures=1, max_iter=20):
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.max_iter = max_iter
        self.models = {}

    def train(self, train_data):
        for digit, features_list in train_data.items():
            print(f"Training digit {digit}")
            hmm = GaussianHMM(self.n_states, self.n_mixtures)
            hmm.initialize(features_list[0].shape[1])
            self._train_digit(hmm, features_list)
            self.models[digit] = hmm
        return self.models

    def _train_digit(self, hmm, features_list):
        for _ in range(self.max_iter):
            all_segments = [[] for _ in range(hmm.n_states)]

            # Viterbi alignment
            # ============= TODO: Viterbi alignment =============
            # For each training utterance:
            #   - Decode the best state sequence using Viterbi
            #   - Assign each frame to its corresponding state
            #
            # Hint:
            #   states, _ = hmm.viterbi_decode(features)
            # ============= TODO: Viterbi alignment =============
            for features in features_list:
                states, _ = hmm.viterbi_decode(features)
                for t, state in enumerate(states):
                    all_segments[state].append(features[t])

            # Re-estimation
            for s in range(hmm.n_states):
                if len(all_segments[s]) == 0:
                    continue

                data = np.array(all_segments[s])

                # ============= TODO: Single Gaussian update =============
                # If using a single Gaussian:
                #   - Estimate mean
                #   - Estimate diagonal covariance
                #
                # Hint:
                #   mean = np.mean(data, axis=0)
                #   cov  = np.diag(np.var(data, axis=0)) + epsilon * I
                # ============= TODO: Single Gaussian update =============
                if hmm.n_mixtures == 1:
                    mean = np.mean(data, axis=0)
                    var = np.var(data, axis=0)
                    cov = np.diag(var) + 1e-6 * np.eye(data.shape[1])
                    hmm.gaussians[s] = [{'mean': mean, 'cov': cov, 'weight': 1.0 / hmm.n_mixtures}]

                else:
                    # ============= TODO: GMM update using KMeans =============
                    # If using multiple mixtures:
                    #   - Cluster data with KMeans
                    #   - Estimate mean, covariance, and weight for each cluster
                    #   - Normalize mixture weights
                    # ============= TODO: GMM update using KMeans =============
                    n_samples = data.shape[0]
                    n_clusters = hmm.n_mixtures
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    labels = kmeans.fit_predict(data)
                    mixtures = []
                    for k in range(n_clusters):
                        cluster_data = data[labels == k]
                        if cluster_data.shape[0] == 0:
                            continue

                        mean = np.mean(cluster_data, axis=0)
                        var = np.var(cluster_data, axis=0)
                        cov = np.diag(var) + 1e-6 * np.eye(data.shape[1])
                        weight = float(cluster_data.shape[0]) / float(n_samples)

                        mixtures.append({'mean': mean, 'cov': cov, 'weight': weight})

                    if len(mixtures) == 0:
                        # Fallback to a single Gaussian if clustering fails
                        mean = np.mean(data, axis=0)
                        var = np.var(data, axis=0)
                        cov = np.diag(var) + 1e-6 * np.eye(data.shape[1])
                        mixtures = [{'mean': mean, 'cov': cov, 'weight': 1.0 / hmm.n_mixtures}]

                    # Pad mixtures if we had fewer clusters than requested
                    while len(mixtures) < hmm.n_mixtures:
                        base = mixtures[len(mixtures) % len(mixtures)]
                        mixtures.append(
                            {
                                'mean': np.array(base['mean'], copy=True),
                                'cov': np.array(base['cov'], copy=True),
                                'weight': 1.0 / hmm.n_mixtures,
                            }
                        )

                    # Normalize weights
                    w_sum = sum(max(m['weight'], 0.0) for m in mixtures)
                    if w_sum <= 0:
                        for m in mixtures:
                            m['weight'] = 1.0 / len(mixtures)
                    else:
                        for m in mixtures:
                            m['weight'] = float(m['weight']) / w_sum

                    hmm.gaussians[s] = mixtures

    def recognize(self, test_features):
        predictions = []
        
        for features in test_features:
            best_score = -float('inf')
            best_digit = None
            
            for digit, hmm in self.models.items():
                _, score = hmm.viterbi_decode(features)
                
                if score > best_score:
                    best_score = score
                    best_digit = digit
            
            predictions.append(best_digit)
        
        return predictions
    
    def evaluate(self, test_data):
        total_correct = 0
        total_samples = 0
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        for true_digit, features_list in test_data.items():
            predictions = self.recognize(features_list)
            
            for pred in predictions:
                true_idx = int(true_digit)
                pred_idx = int(pred)
                
                confusion_matrix[true_idx, pred_idx] += 1
                
                if pred == true_digit:
                    total_correct += 1
                
                total_samples += 1
        
        accuracy = total_correct / total_samples * 100
        
        return accuracy, confusion_matrix