import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import scipy.signal
import scipy.io.wavfile as wav
import os
import re


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
    emphasized_signal = np.zeros_like(signal)
    emphasized_signal[0] = signal[0]
    emphasized_signal[1:] = signal[1:] - coeff * signal[:-1]
    return emphasized_signal


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
    frame_len = int(round(sample_rate * frame_length_ms / 1000.0))
    frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))
    
    # TODO 3: Calculate total number of frames
    num_frames = 1 + (len(signal) - frame_len) // frame_shift
    
    # TODO 4: Pad signal to fit integer number of frames
    pad_len = (num_frames - 1) * frame_shift + frame_len - len(signal)
    pad_len = max(0, pad_len)  # Ensure pad_len is non-negative
    pad_signal = np.pad(signal, (0, pad_len), mode='constant')
    
    # TODO 5: Create frame indices matrix
    indices = np.zeros((num_frames, frame_len), dtype=int)
    for i in range(num_frames):
        start_idx = i * frame_shift
        indices[i] = np.arange(start_idx, start_idx + frame_len)
    
    # TODO 6: Extract frames using indices
    frames = pad_signal[indices]
    
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
    frame_len = frames.shape[1]
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_len) / (frame_len - 1))
    windowed = frames * hamming_window
    
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
    fft_result = np.fft.rfft(frames, n=n_fft, axis=1)
    mag = np.abs(fft_result)
    
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
    mel = 2595 * np.log10(1 + hz / 700.0)
    
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
    hz = 700 * (10 ** (mel / 2595.0) - 1)
    
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
    bins = np.round((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # TODO 16: Initialize filter bank matrix
    fbank = np.zeros((num_filters, n_fft//2 + 1))
    
    # TODO 17: Create triangular filters
    for m in range(1, num_filters + 1):
        left = bins[m-1]
        center = bins[m]
        right = bins[m+1]
        
        # Rising slope
        for k in range(left, center):
            if center != left:
                fbank[m-1, k] = (k - left) / (center - left)
        
        # Falling slope
        for k in range(center, right):
            if right != center:
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
    mel_power = np.dot(power_spec, mel_fb.T)
    
    # TODO 19: Take logarithm (add small value to avoid log(0))
    eps = 1e-10
    mel_log = np.log(mel_power + eps)
    
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
    mfcc_full = dct(mel_log, axis=1, norm='ortho')
    
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
    padded = np.pad(features, ((N, N), (0, 0)), mode="edge")
    denom = 2 * sum(n ** 2 for n in range(1, N + 1))
    
    for t in range(num_frames):
        num = np.zeros(num_features)
        for n in range(1, N + 1):
            num += n * (padded[t + N + n] - padded[t + N - n])
        delta[t] = num / denom
    
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
    eps = 1e-10
    normalized = (features - mean_val) / (std_val + eps)
    
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
    
    # Convert stereo to mono if needed
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
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

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    # Remove timestamp pattern: _YYYYMMDD_HHMMSS
    base_name_without_timestamp = re.sub(r'_\d{8}_\d{6}$', '', base_name)
    # Add filter count to base name
    base_name_with_filters = f"{base_name_without_timestamp}_{num_mels}_filters"

    mel_img_path = os.path.join(out_dir, f"{base_name_with_filters}_mel_log_spectrum.png")
    recon_img_path = os.path.join(out_dir, f"{base_name_with_filters}_recon_logspec_from_mfcc.png")
    combo_img_path = os.path.join(out_dir, f"{base_name_with_filters}_combined_spectrum.png")

    # Calculate actual duration covered by frames
    # Frames: frame_length_ms=25, frame_shift_ms=10
    frame_len_samples = int(round(sample_rate * 25 / 1000.0))
    frame_shift_samples = int(round(sample_rate * 10 / 1000.0))
    num_frames = mel_log.shape[0]  # This is the number of frames we have
    # Last frame ends at: (num_frames-1) * frame_shift + frame_len
    frames_duration_s = ((num_frames - 1) * frame_shift_samples + frame_len_samples) / float(sample_rate)

    low_freq = 133.33
    high_freq = 6855.4976
    nyquist = sample_rate / 2.0

    # Mel-log spectrum (Hz on y-axis)
    plt.figure(figsize=(8, 4))
    im0 = plt.imshow(
        mel_log.T,
        origin="lower",
        aspect="auto",
        extent=[0, frames_duration_s, low_freq, high_freq],
        cmap="viridis",
        interpolation='nearest',  # Disable interpolation to show actual resolution differences
        vmin=-5,
        vmax=22.5
    )
    plt.title(f"Mel-Log Spectrum\n({num_mels} filters)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    c0 = plt.colorbar(im0)
    c0.set_label("Log Mel-band Energy")
    plt.tight_layout()
    plt.savefig(mel_img_path, dpi=200)
    plt.close()

    # IDCT-derived log spectrum (map bins to 0..Nyquist for visualization)
    plt.figure(figsize=(8, 4))
    im1 = plt.imshow(
        recon_logspec.T,
        origin="lower",
        aspect="auto",
        extent=[0, frames_duration_s, 0, nyquist],
        cmap="magma",
        interpolation='nearest',  # Disable interpolation for consistency
        vmin=-5,
        vmax=12
    )
    plt.title("Log Spectrum from MFCC\n(13 coeffs via IDCT)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    c1 = plt.colorbar(im1)
    c1.set_label("Log-spectrum (from MFCC)")
    plt.tight_layout()
    plt.savefig(recon_img_path, dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), constrained_layout=True)

    # Left: Mel-Log Spectrum
    imL = axes[0].imshow(
        mel_log.T,
        origin="lower",
        aspect="auto",
        extent=[0, frames_duration_s, low_freq, high_freq],
        cmap="viridis",
        interpolation='nearest',  # Disable interpolation to show actual resolution differences
        vmin=-5,
        vmax=22.5
    )
    axes[0].set_title(f"Mel-Log Spectrum\n({num_mels} filters)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_yticks(np.linspace(low_freq, high_freq, 6))
    cbL = fig.colorbar(imL, ax=axes[0])
    cbL.set_label("Log Mel-band Energy")

    # Right: Log Spectrum from MFCC (via IDCT)
    imR = axes[1].imshow(
        recon_logspec.T,
        origin="lower",
        aspect="auto",
        extent=[0, frames_duration_s, 0, nyquist],
        cmap="magma",
        interpolation='nearest',  # Disable interpolation for consistency
        vmin=-5,
        vmax=12
    )
    axes[1].set_title("Log Spectrum from MFCC\n(13 coeffs via IDCT)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_yticks(np.linspace(0, nyquist, 6))
    cbR = fig.colorbar(imR, ax=axes[1])
    cbR.set_label("Log-spectrum (from MFCC)")

    fig.savefig(combo_img_path, dpi=200)
    plt.close(fig)
    
    # Persist features to disk
    np.save(os.path.join(out_dir, "mfcc.npy"), mfcc)
    np.save(os.path.join(out_dir, "mel_log.npy"), mel_log)
    np.save(os.path.join(out_dir, "recon_logspec.npy"), recon_logspec)
    if delta is not None:
        np.save(os.path.join(out_dir, "delta.npy"), delta)
        np.save(os.path.join(out_dir, "double_delta.npy"), double_delta)
    # Pack all features in a single archive for convenience
    np.savez(
        os.path.join(out_dir, "features.npz"),
        mfcc=mfcc,
        mel_log=mel_log,
        recon_logspec=recon_logspec,
        power_spec=power_spec,
        delta=delta,
        double_delta=double_delta
    )
    
    # Return features dictionary
    features_dict = {
        'mfcc': mfcc,
        'mel_log': mel_log,
        'recon_logspec': recon_logspec,
        'power_spec': power_spec,
        'delta': delta,
        'double_delta': double_delta
    }
    
    return features_dict



# Main execution block
if __name__ == "__main__":
    # Test with a sample audio file
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Compute and plot features from audio file")
    parser.add_argument("--audio", type=str, help="Path to input audio file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory containing audio files")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # Determine which mode to use: single file or batch processing
    if args.data_dir:
        # Batch processing mode: find all .wav files recursively
        if not os.path.isdir(args.data_dir):
            parser.error(f"Data directory does not exist: {args.data_dir}")
        
        # Find all .wav files recursively
        wav_files = []
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        if not wav_files:
            print(f"No .wav files found in {args.data_dir}")
            exit(0)
        
        print(f"Found {len(wav_files)} audio files. Processing...\n")
        
        # Process each file
        for idx, test_audio in enumerate(wav_files, 1):
            print(f"[{idx}/{len(wav_files)}] Processing: {test_audio}")
            
            base = os.path.splitext(os.path.basename(test_audio))[0]
            parent = os.path.basename(os.path.dirname(test_audio))
            
            # Extract base name without timestamp (e.g., zero_01_20260110_150527 -> zero_01)
            # Remove timestamp pattern: _YYYYMMDD_HHMMSS
            base_without_timestamp = re.sub(r'_\d{8}_\d{6}$', '', base)

            # Test with different filter numbers
            for num_filters in [40, 30, 25]:
                print(f"  Testing with {num_filters} Mel filters")
                
                features = compute_features(
                    wav_path=test_audio,
                    num_mels=num_filters,
                    compute_deltas=True,
                    normalize=True,
                    out_dir=f"./outputs/{parent}/{base_without_timestamp}/{num_filters}_filters"
                )
            
            print(f"  Completed: {test_audio}\n")
        
        print(f"All {len(wav_files)} files processed successfully!")
        
    elif args.audio:
        # Single file processing mode (original behavior)
        test_audio = args.audio
        
        if not os.path.isfile(test_audio):
            parser.error(f"Audio file does not exist: {test_audio}")
        
        base = os.path.splitext(os.path.basename(test_audio))[0]
        parent = os.path.basename(os.path.dirname(test_audio))
        
        # Extract base name without timestamp (e.g., zero_01_20260110_150527 -> zero_01)
        # Remove timestamp pattern: _YYYYMMDD_HHMMSS
        base_without_timestamp = re.sub(r'_\d{8}_\d{6}$', '', base)

        # Test with different filter numbers
        for num_filters in [40, 30, 25]:
            print(f"Testing with {num_filters} Mel filters\n")
            
            features = compute_features(
                wav_path=test_audio,
                num_mels=num_filters,
                compute_deltas=True,
                normalize=True,
                out_dir=f"./outputs/{parent}/{base_without_timestamp}/{num_filters}_filters"
            )
    else:
        parser.error("Either --audio or --data_dir must be provided")
