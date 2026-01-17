import os
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import librosa

from .project1 import (
    pre_emphasis,
    framing,
    windowing,
    power_spectrum,
    mel_filterbank,
    mel_log_spectrum,
    compute_mfcc,
    compute_delta,
    mean_variance_normalization
)

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

    def extract_features(self, audio_path, remove_silence=False):
        """
        extract audio features: MFCC + delta + double delta
        """
        # TODO:  All feature vectors must be 39-dimensional features (cepstra, delta cepstra, and double delta cepstra) obtained using the code you wrote for Project 1. 
        # Mean subtraction and variance normalization must be performed.
        signal, sample_rate = librosa.load(audio_path, sr=self.sr)
        signal = signal.astype(np.float32)

        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        if sample_rate != self.sr:
            gcd = math.gcd(sample_rate, self.sr)
            up = self.sr // gcd
            down = sample_rate // gcd
            signal = resample_poly(signal, up, down).astype(np.float32)
            sample_rate = self.sr

        signal = pre_emphasis(signal)

        frame_length_ms = 25
        frame_shift_ms = int(round(self.hop_length / sample_rate * 1000.0))
        frames = framing(
            signal,
            sample_rate,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms
        )

        if remove_silence and frames.size > 0:
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            threshold = np.percentile(rms, 25)
            keep_mask = rms > max(threshold, 1e-6)
            if np.any(keep_mask):
                frames = frames[keep_mask]

        win_frames = windowing(frames)
        power_spec = power_spectrum(win_frames, n_fft=self.n_fft)
        mel_fb = mel_filterbank(sample_rate, n_fft=self.n_fft, num_filters=40)
        mel_log = mel_log_spectrum(power_spec, mel_fb)
        mfcc = compute_mfcc(mel_log, num_ceps=self.n_mfcc)

        delta = compute_delta(mfcc)
        double_delta = compute_delta(delta)

        mfcc = mean_variance_normalization(mfcc)
        delta = mean_variance_normalization(delta)
        double_delta = mean_variance_normalization(double_delta)

        features = np.concatenate([mfcc, delta, double_delta], axis=1).astype(np.float32)
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