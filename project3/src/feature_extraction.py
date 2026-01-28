import os
import numpy as np
import librosa
from python_speech_features import mfcc, delta


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
        Compute 39-D features: MFCC + delta + double-delta.
        """
        # TODO:
        # All feature vectors must be 39-dimensional features (cepstra, delta cepstra, and double delta cepstra)
        # obtained using the code you wrote for Project 1.
        # Mean subtraction and variance normalization must be performed.

        wav, sr = librosa.load(audio_path, sr=16000)

        # Optional: drop silent portions
        if remove_silence:
            segs = librosa.effects.split(wav, top_db=25)
            if len(segs) > 0:
                kept = np.concatenate([wav[a:b] for a, b in segs])
                if kept.size > 0:
                    wav = kept

        # MFCC extraction (fixed parameters as in the original code)
        c = mfcc(
            wav,
            samplerate=sr,
            winlen=0.025,
            winstep=0.01,
            numcep=13,
            nfilt=40,
            nfft=512,
            preemph=0.97,
        )

        d1 = delta(c, 2)
        d2 = delta(d1, 2)

        # normalize each block independently (same as original)
        c = self._normalize_features(c)
        d1 = self._normalize_features(d1)
        d2 = self._normalize_features(d2)

        # concatenate into 39-D
        return np.hstack([c, d1, d2])

    def _normalize_features(self, feats):
        """
        Mean subtraction + variance normalization (per dimension).
        """
        mu = np.mean(feats, axis=0)
        sigma = np.std(feats, axis=0) + 1e-6  # prevent divide-by-zero
        return (feats - mu) / sigma

    def extract_all_features(self, data_dir, save_dir=None):
        """
        Extract features for all digit files in a directory.
        Optionally save per-digit .npy files.
        """
        all_feats = {}

        for d in range(10):
            per_digit = []
            files = [name for name in os.listdir(data_dir) if name.startswith(f"digit_{d}_")]

            for name in files:
                path = os.path.join(data_dir, name)
                per_digit.append(self.extract_features(path))

            all_feats[str(d)] = per_digit

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"digit_{d}_features.npy")
                np.save(out_path, np.array(per_digit, dtype=object), allow_pickle=True)

        return all_feats