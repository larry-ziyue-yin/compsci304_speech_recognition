import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import scipy.signal
import scipy.io.wavfile as wav
import os


def pre_emphasis(signal, coeff=0.97):
    """
    Apply pre-emphasis filter to the signal.
    Formula: y[n] = x[n] - coeff * x[n-1]
    The first sample remains unchanged.

    Parameters:
    -----------
    signal : numpy array
        Input audio signal
    coeff : float
        Pre-emphasis coefficient
    
    Returns:
    --------
    emphasized_signal : numpy array
        Pre-emphasized signal
    """
    # TODO 1: Implement pre-emphasis filter.
    return np.array([0.0])  # REPLACE THIS LINE


def framing(signal, sample_rate, frame_length_ms=25, frame_shift_ms=10):
    """
    Split continuous audio signal into overlapping frames.
    
    Parameters:
    -----------
    signal : numpy array
        Input audio signal
    sample_rate : int
        Sampling rate in Hz
    frame_length_ms : int
        Frame length in milliseconds
    frame_shift_ms : int
        Frame shift in milliseconds
    
    Returns:
    --------
    frames : numpy array of shape (num_frames, frame_length_samples)
        Extracted frames
    """
    # TODO 2: Calculate frame length and frame shift in samples
    frame_len = 0  # REPLACE THIS
    frame_shift = 0  # REPLACE THIS
    
    # TODO 3: Calculate total number of frames
    num_frames = 0  # REPLACE THIS
    
    # TODO 4: Pad signal to fit integer number of frames
    pad_len = 0  # REPLACE THIS
    pad_signal = np.zeros(pad_len)  # REPLACE THIS
    
    # TODO 5: Create frame indices matrix
    indices = np.zeros((num_frames, frame_len), dtype=int)  # REPLACE THIS
    
    # TODO 6: Extract frames using indices
    frames = np.zeros((num_frames, frame_len))  # REPLACE THIS
    
    return frames


def windowing(frames):
    """
    Apply Hamming window to each frame.
    
    Parameters:
    -----------
    frames : numpy array of shape (num_frames, frame_length)
        Extracted frames
    
    Returns:
    --------
    windowed_frames : numpy array
        Windowed frames
    """
    # TODO 7: Create Hamming window and apply to each frame (refers to PPT slide 32 to write Hamming window)
    windowed = np.zeros_like(frames)  # REPLACE THIS
    
    return windowed


def power_spectrum(frames, n_fft=512):
    """
    Compute power spectrum for each frame.
    
    Parameters:
    -----------
    frames : numpy array of shape (num_frames, frame_length)
        Windowed frames
    n_fft : int
        FFT size (should be power of 2)
    
    Returns:
    --------
    power_spec : numpy array of shape (num_frames, n_fft//2 + 1)
        Power spectrum
    """
    # TODO 8: Compute real FFT for each frame
    mag = np.zeros((frames.shape[0], n_fft//2 + 1))  # REPLACE THIS
    
    # TODO 9: Compute power spectrum
    power = np.zeros_like(mag)  # REPLACE THIS
    
    return power


def hz_to_mel(hz):
    """
    Convert frequency from Hz to Mel scale.
    
    Parameters:
    -----------
    hz : float or numpy array
        Frequency in Hz
    
    Returns:
    --------
    mel : float or numpy array
        Frequency in Mel scale
    """
    # TODO 10: Implement Hz to Mel conversion formula (refers to PPT slide 58)
    # Formula: mel = 2595 * log10(1 + hz/700)
    mel = 0  # REPLACE THIS
    
    return mel


def mel_to_hz(mel):
    """
    Convert frequency from Mel scale to Hz.
    
    Parameters:
    -----------
    mel : float or numpy array
        Frequency in Mel scale
    
    Returns:
    --------
    hz : float or numpy array
        Frequency in Hz
    """
    # STUDENT TODO 11: Implement Mel to Hz conversion formula
    # Inverse of the above formula
    hz = 0  # REPLACE THIS
    
    return hz


def mel_filterbank(sample_rate, n_fft, num_filters=40,
                   low_freq=133.33, high_freq=6855.4976):
    """
    Create Mel filter bank.
    
    Parameters:
    -----------
    sample_rate : int
        Sampling rate in Hz
    n_fft : int
        FFT size
    num_filters : int
        Number of Mel filters
    low_freq : float
        Lowest frequency in Hz
    high_freq : float
        Highest frequency in Hz
    
    Returns:
    --------
    fbank : numpy array of shape (num_filters, n_fft//2 + 1)
        Mel filter bank
    """
    # TODO 12: Convert frequency limits to Mel scale
    low_mel = 0  # REPLACE THIS
    high_mel = 0  # REPLACE THIS
    
    # TODO 13: Create equally spaced points in Mel scale
    mel_points = np.zeros(num_filters + 2)  # REPLACE THIS
    
    # TODO 14: Convert back to Hz
    hz_points = np.zeros(num_filters + 2)  # REPLACE THIS
    
    # TODO 15: Convert Hz points to FFT bin indices
    bins = np.zeros(num_filters + 2, dtype=int)  # REPLACE THIS
    
    # TODO 16: Initialize filter bank matrix
    fbank = np.zeros((num_filters, n_fft//2 + 1))
    
    # TODO 17: Create triangular filters
    for m in range(1, num_filters + 1):
        left = bins[m-1]
        center = bins[m]
        right = bins[m+1]
        
        # Rising slope
        for k in range(left, center):
            fbank[m-1, k] = 0  # REPLACE THIS
        
        # Falling slope
        for k in range(center, right):
            fbank[m-1, k] = 0  # REPLACE THIS
    
    return fbank


def mel_log_spectrum(power_spec, mel_fb):
    """
    Compute Mel-log spectrum.
    
    Parameters:
    -----------
    power_spec : numpy array of shape (num_frames, n_fft//2 + 1)
        Power spectrum
    mel_fb : numpy array of shape (num_filters, n_fft//2 + 1)
        Mel filter bank
    
    Returns:
    --------
    mel_log : numpy array of shape (num_frames, num_filters)
        Mel-log spectrum
    """
    # TODO 18: Apply Mel filters to power spectrum
    mel_power = np.zeros((power_spec.shape[0], mel_fb.shape[0]))  # REPLACE THIS
    
    # TODO 19: Take logarithm (add small value to avoid log(0))
    mel_log = np.zeros_like(mel_power)  # REPLACE THIS
    
    return mel_log


def compute_mfcc(mel_log, num_ceps=13):
    """
    Compute MFCC from Mel-log spectrum.
    
    Parameters:
    -----------
    mel_log : numpy array of shape (num_frames, num_filters)
        Mel-log spectrum
    num_ceps : int
        Number of cepstral coefficients to keep
    
    Returns:
    --------
    mfcc : numpy array of shape (num_frames, num_ceps)
        MFCC coefficients
    """
    # TODO 20: Apply DCT (Discrete Cosine Transform)
    mfcc_full = np.zeros_like(mel_log)  # REPLACE THIS
    
    # TODO 21: Keep only first num_ceps coefficients
    mfcc = np.zeros((mel_log.shape[0], num_ceps))  # REPLACE THIS
    
    return mfcc


def idct_reconstruct_logspec(mfcc, n_fft=128):
    """
    Reconstruct log spectrum from MFCC using IDCT.
    
    Parameters:
    -----------
    mfcc : numpy array of shape (num_frames, num_ceps)
        MFCC coefficients
    n_fft : int
        Size for IDCT (should be >= num_ceps)
    
    Returns:
    --------
    recon_logspec : numpy array of shape (num_frames, n_fft)
        Reconstructed log spectrum
    """
    num_frames, num_ceps = mfcc.shape
    
    # TODO 22: Zero-pad MFCC to n_fft length
    padded = np.zeros((num_frames, n_fft))  # REPLACE THIS
    
    # TODO 23: Apply Inverse DCT (IDCT)
    recon = np.zeros_like(padded)  # REPLACE THIS
    
    return recon



def compute_delta(features, N=2):
    """
    Compute delta (first derivative) features.
    
    Parameters:
    -----------
    features : numpy array of shape (num_frames, num_features)
        Static features
    N : int
        Window size for derivative computation
    
    Returns:
    --------
    delta : numpy array
        Delta features
    """
    num_frames, num_features = features.shape
    delta = np.zeros_like(features)
    
    # TODO 24: Implement delta feature computation
    
    return delta


def mean_variance_normalization(features):
    """
    Apply mean and variance normalization.
    
    Parameters:
    -----------
    features : numpy array
        Input features
    
    Returns:
    --------
    normalized_features : numpy array
        Normalized features
    """
    # TODO 25: Calculate mean and std along feature dimension
    mean_val = 0  # REPLACE THIS
    std_val = 0  # REPLACE THIS
    
    # TODO 26: Apply normalization (avoid division by zero)
    normalized = np.zeros_like(features)  # REPLACE THIS
    
    return normalized


def compute_features(wav_path, num_mels=40, compute_deltas=False,
                     normalize=False, out_dir="./output"):
    """
    Complete MFCC feature extraction pipeline.
    
    Parameters:
    -----------
    wav_path : str
        Path to WAV file
    num_mels : int
        Number of Mel filters
    compute_deltas : bool
        Whether to compute delta and double-delta features
    normalize : bool
        Whether to apply mean-variance normalization
    out_dir : str
        Output directory for visualizations
    
    Returns:
    --------
    features : dict
        Dictionary containing all computed features
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load audio file
    sample_rate, signal = wav.read(wav_path)
    signal = signal.astype(np.float32)
    
    # Step 1: Pre-emphasis
    signal = pre_emphasis(signal)
    
    # Step 2: Framing
    frames = framing(signal, sample_rate)
    
    # Step 3: Windowing
    win_frames = windowing(frames)
    
    # Step 4: Power spectrum
    power_spec = power_spectrum(win_frames, n_fft=512)

    # Step 5: Mel filter bank
    mel_fb = mel_filterbank(sample_rate, n_fft=512, num_filters=num_mels)
    
    # Step 6: Mel-log spectrum
    mel_log = mel_log_spectrum(power_spec, mel_fb)
    
    # Step 7: MFCC
    mfcc = compute_mfcc(mel_log, num_ceps=13)
    
    # Step 8: Reconstruct log spectrum for visualization
    recon_logspec = idct_reconstruct_logspec(mfcc, n_fft=128)
    
    # Step 9: Delta features
    delta = None
    double_delta = None
    if compute_deltas:
        delta = compute_delta(mfcc)
        double_delta = compute_delta(delta)
    
    # Step 10: Normalization
    if normalize:
        mfcc = mean_variance_normalization(mfcc)
        if compute_deltas:
            delta = mean_variance_normalization(delta)
            double_delta = mean_variance_normalization(double_delta)
    
    # TODO 27: Visualization 
    # 1.Mel-Log Spectrum 
    # 2.Log Spectrum Reconstructed from MFCC 
    
    return



# Main execution block
if __name__ == "__main__":
    # Test with a sample audio file
    test_audio = "asset/test.wav"  # Replace with actual audio file
    
    # Test with different filter numbers
    for num_filters in [40, 30, 25]:
        print(f"Testing with {num_filters} Mel filters\n")
        
        features = compute_features(
            wav_path=test_audio,
            num_mels=num_filters,
            compute_deltas=True,
            normalize=True,
            out_dir=f"./output_{num_filters}_filters"
        )