import os
import numpy as np

class FeatureExtractor:
    def __init__(self, n_mfcc=13, n_fft=512, hop_length=160, sr=16000):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.cepstra_dim = n_mfcc
        self.delta_dim = n_mfcc
        self.double_delta_dim = n_mfcc
        self.total_dim = 39  # 13 MFCC + 13 delta + 13 double delta
        
    def extract_features(self, audio_path, remove_silence=True):
        """
        extract audio features: MFCC + delta + double delta
        """
        # TODO:  All feature vectors must be 39-dimensional features (cepstra, delta cepstra, and double delta cepstra) obtained using the code you wrote for Project 1. 
        # Mean subtraction and variance normalization must be performed.
        features = None
        return features

    
    def extract_all_features(self, data_dir, save_dir=None):
        """
        extract all features and save in files
        """
        features_dict = {}
        
        for digit in range(10):
            digit_features = []
            
            audio_files = [f for f in os.listdir(data_dir) if f.startswith(f"digit_{digit}_")]
            
            for audio_file in audio_files:
                audio_path = os.path.join(data_dir, audio_file)
                features = self.extract_features(audio_path)
                digit_features.append(features)
            
            features_dict[str(digit)] = digit_features
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f"digit_{digit}_features.npy"), 
        np.array(digit_features, dtype=object), allow_pickle=True)
        
        return features_dict