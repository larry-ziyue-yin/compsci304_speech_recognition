import os
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fftpack import dct

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
        
        self.winlen_s = 0.025   # 25 ms
        self.winstep_s = 0.01   # 10 ms
        self.preemph = 0.97
        self.nfilt = 40
        self.low_freq = 133.33
        self.high_freq = 6855.4976
        self.eps = 1e-10

    def _pre_emphasis(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - self.preemph * x[:-1]
        return y

    def _framing(self, x: np.ndarray, sr: int) -> np.ndarray:
        frame_len = int(round(sr * self.winlen_s))
        frame_shift = int(round(sr * self.winstep_s))
        frame_len = max(1, frame_len)
        frame_shift = max(1, frame_shift)

        # robust num_frames: at least 1 frame even if signal shorter than a frame
        if len(x) <= frame_len:
            num_frames = 1
        else:
            num_frames = 1 + int(np.floor((len(x) - frame_len) / frame_shift))

        pad_len = (num_frames - 1) * frame_shift + frame_len - len(x)
        pad_len = max(0, pad_len)
        x_pad = np.pad(x, (0, pad_len), mode="constant")

        indices = np.zeros((num_frames, frame_len), dtype=np.int64)
        for i in range(num_frames):
            start = i * frame_shift
            indices[i] = np.arange(start, start + frame_len)

        frames = x_pad[indices]
        return frames

    def _hamming(self, frames: np.ndarray) -> np.ndarray:
        L = frames.shape[1]
        if L <= 1:
            return frames
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(L) / (L - 1))
        return frames * w

    def _power_spectrum(self, frames: np.ndarray) -> np.ndarray:
        fft_result = np.fft.rfft(frames, n=self.n_fft, axis=1)
        mag = np.abs(fft_result)
        power = mag ** 2
        return power

    def _hz_to_mel(self, hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank(self, sr: int) -> np.ndarray:
        nyq = sr / 2.0
        low = float(self.low_freq)
        high = float(min(self.high_freq, nyq))

        low_mel = self._hz_to_mel(low)
        high_mel = self._hz_to_mel(high)

        mel_points = np.linspace(low_mel, high_mel, self.nfilt + 2)
        hz_points = self._mel_to_hz(mel_points)

        bins = np.round((self.n_fft + 1) * hz_points / sr).astype(int)
        bins = np.clip(bins, 0, self.n_fft // 2)

        fbank = np.zeros((self.nfilt, self.n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, self.nfilt + 1):
            left = bins[m - 1]
            center = bins[m]
            right = bins[m + 1]

            if center < left:
                center = left
            if right < center:
                right = center

            # rising
            for k in range(left, center):
                denom = (center - left)
                if denom != 0:
                    fbank[m - 1, k] = (k - left) / denom
            # falling
            for k in range(center, right):
                denom = (right - center)
                if denom != 0:
                    fbank[m - 1, k] = (right - k) / denom

        return fbank

    def _compute_delta(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        # same as your Project1 compute_delta
        T, D = features.shape
        delta = np.zeros_like(features, dtype=np.float32)
        padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
        denom = 2.0 * sum(n ** 2 for n in range(1, N + 1))

        for t in range(T):
            num = np.zeros(D, dtype=np.float32)
            for n in range(1, N + 1):
                num += n * (padded[t + N + n] - padded[t + N - n])
            delta[t] = num / (denom + self.eps)

        return delta

    def _cmvn(self, features: np.ndarray) -> np.ndarray:
        # same spirit as your Project1 mean_variance_normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + self.eps)

    def extract_features(self, audio_path, remove_silence=True):
        """
        extract audio features: MFCC + delta + double delta
        """
        # TODO:  All feature vectors must be 39-dimensional features (cepstra, delta cepstra, and double delta cepstra) obtained using the code you wrote for Project 1. 
        # Mean subtraction and variance normalization must be performed.
        # 1) load audio (and resample to self.sr if needed)
        sr, sig = wavfile.read(audio_path)
        sig = sig.astype(np.float32)

        # stereo -> mono
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1)

        # resample if needed
        if sr != self.sr:
            g = math.gcd(int(sr), int(self.sr))
            up = self.sr // g
            down = sr // g
            sig = resample_poly(sig, up, down).astype(np.float32)
            sr = self.sr

        # 2) pre-emphasis
        sig = self._pre_emphasis(sig)

        # 3) framing + windowing
        frames = self._framing(sig, sr)
        win_frames = self._hamming(frames)

        # (optional) simple silence removal based on frame log-energy
        if remove_silence:
            frame_energy = np.sum(win_frames ** 2, axis=1) + self.eps
            loge = np.log(frame_energy)
            thr = np.median(loge) - 0.5 * np.std(loge)
            keep = loge > thr
            # keep at least a few frames to avoid over-pruning
            if np.sum(keep) >= 5:
                win_frames = win_frames[keep]

        # 4) power spectrum
        power = self._power_spectrum(win_frames)  # (T, n_fft//2+1)

        # 5) mel log spectrum
        mel_fb = self._mel_filterbank(sr)  # (40, n_fft//2+1)
        mel_power = np.dot(power, mel_fb.T)
        mel_log = np.log(mel_power + self.eps)

        # 6) MFCC: DCT + keep first 13
        mfcc_full = dct(mel_log, axis=1, norm="ortho")
        mfcc = mfcc_full[:, : self.n_mfcc].astype(np.float32)

        # 7) delta + double-delta (N=2)
        d1 = self._compute_delta(mfcc, N=2)
        d2 = self._compute_delta(d1, N=2)

        # 8) stack to 39-dim + CMVN
        features = np.hstack([mfcc, d1, d2]).astype(np.float32)
        features = self._cmvn(features)

        # safety check
        if features.shape[1] != self.total_dim:
            raise ValueError(f"Expected {self.total_dim}-D features, got {features.shape[1]}")
        
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