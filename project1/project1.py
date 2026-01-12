import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import scipy.signal
import scipy.io.wavfile as wav
import os


def pre_emphasis(signal, coeff=0.97):
    """
    原理：语音谱通常有“高频能量天然更低”的 spectral tilt，预加重相当于做一个简单差分，提升高频、让谱更“平”，后续建模更稳。
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
    if signal.size == 0:
        return signal.astype(np.float32) # `.astype(np.float32)` converts the array to float32 type
    
    signal = signal.astype(np.float32, copy=False)
    emphasized_signal = np.empty_like(signal)
    emphasized_signal[0] = signal[0]
    emphasized_signal[1:] = signal[1:] - coeff * signal[:-1]
    
    return emphasized_signal


def framing(signal, sample_rate, frame_length_ms=25, frame_shift_ms=10):
    """
    原理：语音短时近似平稳，所以用 20/25ms 的窗分段处理，帧移通常 10ms（即重叠）。
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
    signal = signal.astype(np.float32, copy=False) # By converting to float32 type, we can avoid potential issues with integer overflow
    
    # TODO 2: Calculate frame length and frame shift in samples
    frame_len = int(round(sample_rate * frame_length_ms / 1000.0))  # frame_len = sr * 25ms
    frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))  # frame_shift = sr * 10ms
    frame_len = max(1, frame_len) # ensures that the frame length is at least 1 sample
    frame_shift = max(1, frame_shift) # ensures that the frame shift is at least 1 sample
    
    # TODO 3: Calculate total number of frames
    sig_len = len(signal)
    if sig_len <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((sig_len - frame_len) / frame_shift))
    
    # TODO 4: Pad signal to fit integer number of frames
    target_len = (num_frames - 1) * frame_shift + frame_len
    pad_len = target_len - sig_len
    if pad_len > 0:
        pad_signal = np.pad(signal, (0, pad_len), mode="constant")  # signal前不做padding，signal后做`pad_len`个0的padding。cosntant表示用常数0填充。
    else:
        pad_signal = signal
    
    # TODO 5: Create frame indices matrix
    indices = np.zeros((num_frames, frame_len), dtype=int)
    for i in range(num_frames):
        start_idx = i * frame_shift
        indices[i] = np.arange(start_idx, start_idx + frame_len)  # indices[i] 是第 i 帧对应的采样点下标区间 [i*shift, i*shift+frame_len)
    # 这样可以用一次 numpy 索引把所有帧都取出来（pad_signal[indices]），逻辑清晰，后续 window/FFT 都能批处理。
    
    # TODO 6: Extract frames using indices
    frames = pad_signal[indices]  # 得到 shape (num_frames, frame_len) 的矩阵
    # 后面的 windowing、功率谱、Mel 滤波器组等都按“每一帧一行”来算，矩阵化最方便。
    
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
    M = frames.shape[1]
    n = np.arange(M)
    hamming_window = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (M-1))
    windowed = frames * hamming_window[None, :]
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
    spectrum = np.fft.rfft(frames, n=n_fft)
    mag = np.abs(spectrum)
    
    # TODO 9: Compute power spectrum
    power = mag ** 2
    
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
    mel = 2595 * np.log10(1 + hz/700)
    
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
    hz = 700 * (10 ** (mel/2595) - 1)
    
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
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    
    # TODO 13: Create equally spaced points in Mel scale
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
    # TODO 14: Convert back to Hz
    hz_points = mel_to_hz(mel_points)
    
    # TODO 15: Convert Hz points to FFT bin indices
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # TODO 16: Initialize filter bank matrix
    fbank = np.zeros((num_filters, n_fft//2 + 1))
    
    # TODO 17: Create triangular filters
    for m in range(1, num_filters + 1):
        left = bins[m-1]
        center = bins[m]
        right = bins[m+1]
        
        # Rising slope
        for k in range(left, center):
            if center - left == 0:
                fbank[m-1, k] = 0
            else:
                fbank[m-1, k] = (k - left) / (center - left)
        
        # Falling slope
        for k in range(center, right):
            if right - center == 0:
                fbank[m-1, k] = 0
            else:
                fbank[m-1, k] = (right - k) / (right - center)
    
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
    mel_power = power_spec @ mel_fb.T
    
    # TODO 19: Take logarithm (add small value to avoid log(0))
    mel_log = np.log(mel_power + 1e-6)
    
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
    mfcc_full = dct(mel_log, type=2, norm='ortho')
    
    # TODO 21: Keep only first num_ceps coefficients
    mfcc = mfcc_full[:, :num_ceps]
    
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
    padded = np.zeros((num_frames, n_fft))
    padded[:, :num_ceps] = mfcc
    
    # TODO 23: Apply Inverse DCT (IDCT)
    recon = idct(padded, axis=1, norm='ortho')
    
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
    denom = 2.0 * sum((n ** 2) for n in range(1, N + 1))
    if denom == 0:
        return delta

    padded = np.pad(features, ((N, N), (0, 0)), mode='edge').astype(np.float32, copy=False)

    for t in range(num_frames):
        acc = np.zeros((num_features,), dtype=np.float32)
        for n in range(1, N + 1):
            acc += n * (padded[t + N + n] - padded[t + N - n])
        delta[t] = acc / denom

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
    mean_val = np.mean(features, axis=0)
    std_val = np.std(features, axis=0)
    
    # TODO 26: Apply normalization (avoid division by zero)
    normalized = (features - mean_val) / (std_val + 1e-6)
    
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
    
    # ----- paths -----
    mel_img_path = os.path.join(out_dir, "mel_log_spectrum.png")
    recon_img_path = os.path.join(out_dir, "recon_logspec_from_mfcc.png")
    combo_img_path = os.path.join(out_dir, f"spectrograms_{num_mels}_filters.png")

    # ----- axis extents (match reference) -----
    duration_s = len(signal) / float(sample_rate)  # x-axis in seconds

    # IMPORTANT: these are exactly the handout-specified band edges used in mel_filterbank defaults
    low_freq = 133.33
    high_freq = 6855.4976

    # Right plot: show 0..Nyquist as "Frequency (Hz)" like the reference
    nyquist = sample_rate / 2.0

    # ----- 1) Single plots (optional, but keep) -----
    # Mel-log spectrum (use Hz on y-axis)
    plt.figure(figsize=(8, 4))
    im0 = plt.imshow(
        mel_log.T,
        origin="lower",
        aspect="auto",
        extent=[0, duration_s, low_freq, high_freq],
        cmap="viridis",
    )
    plt.title(f"Mel-Log Spectrum\n({num_mels} filters)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    c0 = plt.colorbar(im0)
    c0.set_label("log(magnitude)")
    plt.tight_layout()
    plt.savefig(mel_img_path)
    plt.close()

    # IDCT-derived log spectrum (map bins to 0..Nyquist for visualization)
    plt.figure(figsize=(8, 4))
    im1 = plt.imshow(
        recon_logspec.T,
        origin="lower",
        aspect="auto",
        extent=[0, duration_s, 0, nyquist],
        cmap="magma",
    )
    plt.title("Log Spectrum from MFCC\n(13 coeffs via IDCT)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    c1 = plt.colorbar(im1)
    c1.set_label("log(magnitude)")
    plt.tight_layout()
    plt.savefig(recon_img_path)
    plt.close()

    # ----- 2) Combined figure (match reference image layout) -----
    fig, axes = plt.subplots(1, 2, figsize=(20, 5), constrained_layout=True)

    # Left: Mel-Log Spectrum
    imL = axes[0].imshow(
        mel_log.T,
        origin="lower",
        aspect="auto",
        extent=[0, duration_s, low_freq, high_freq],
        cmap="viridis",
    )
    axes[0].set_title(f"Mel-Log Spectrum\n({num_mels} filters)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    # make ticks look like the reference (6 evenly spaced ticks)
    axes[0].set_yticks(np.linspace(low_freq, high_freq, 6))

    cbL = fig.colorbar(imL, ax=axes[0])
    cbL.set_label("log(magnitude)")

    # Right: Log Spectrum from MFCC (via IDCT)
    imR = axes[1].imshow(
        recon_logspec.T,
        origin="lower",
        aspect="auto",
        extent=[0, duration_s, 0, nyquist],
        cmap="magma",
    )
    axes[1].set_title("Log Spectrum from MFCC\n(13 coeffs via IDCT)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_yticks(np.linspace(0, nyquist, 6))

    cbR = fig.colorbar(imR, ax=axes[1])
    cbR.set_label("log(magnitude)")

    fig.savefig(combo_img_path)
    plt.close(fig)

    # ----- Save features -----
    features = {
        "sample_rate": sample_rate,
        "num_mels": num_mels,
        "mfcc": mfcc,
        "delta": delta,
        "double_delta": double_delta,
        "mel_log": mel_log,
        "recon_logspec": recon_logspec,
        "power_spec": power_spec,
        "mel_fb": mel_fb,
        "mel_img_path": mel_img_path,
        "recon_img_path": recon_img_path,
        "combo_img_path": combo_img_path,
    }

    np.savez(
        os.path.join(out_dir, "features.npz"),
        mfcc=mfcc,
        mel_log=mel_log,
        recon_logspec=recon_logspec,
        delta=(delta if delta is not None else np.array([])),
        double_delta=(double_delta if double_delta is not None else np.array([])),
    )

    return features



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