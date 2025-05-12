import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import os
import argparse

def get_bits(text):
    """
    Convert text to binary string (8-bit ASCII per character).
    
    Args:
        text (str): Input text.
    
    Returns:
        str: Binary string.
    """
    return ''.join(format(ord(c), '08b') for c in text)

def mixer(L, bits, lower, upper):
    """
    Create a mixing signal for embedding bits.
    
    Args:
        L (int): Frame length.
        bits (str): Binary string.
        lower (float): Value for bit 0.
        upper (float): Value for bit 1.
    
    Returns:
        np.ndarray: Mixing signal.
    """
    N = len(bits)
    indices = np.arange(N * L) // L
    bits_array = np.array([int(b) for b in bits])
    m_sig = np.where(bits_array[indices] == 1, upper, lower)
    window = np.hanning(L).repeat(N)
    return m_sig * window

def echo_embed(signal, text, d0=200, d1=300, alpha=0.6, L=8192):
    """
    Embed a text message into an audio signal using echo steganography.
    
    Args:
        signal (np.ndarray): Preprocessed audio signal (2D: N x channels).
        text (str): Text to embed.
        d0 (int): Delay for bit 0 (samples).
        d1 (int): Delay for bit 1 (samples).
        alpha (float): Echo amplitude.
        L (int): Frame length.
    
    Returns:
        np.ndarray: Encoded audio signal.
    
    Raises:
        ValueError: If signal is invalid or too short.
    """
    if not isinstance(signal, np.ndarray) or signal.ndim != 2:
        raise ValueError("Signal must be a 2D NumPy array (N x channels)!")
    if L <= max(d0, d1):
        raise ValueError("Frame length L must be greater than delays d0 and d1!")
    
    s_ch = signal.shape[1]
    s_len = signal.shape[0]
    if s_len < L * 8:
        raise ValueError("Audio signal is too short to embed any message!")
    
    bits = get_bits(text)
    nframe = s_len // L
    max_bits = nframe - (nframe % 8)
    if len(bits) > max_bits:
        print(f"Warning: Message truncated to {max_bits // 8} characters.")
        bits = bits[:max_bits]
    bits = bits + '0' * (max_bits - len(bits))
    
    k0 = np.concatenate([np.zeros(d0), [alpha]])
    k1 = np.concatenate([np.zeros(d1), [alpha]])
    
    echo_zro = np.zeros_like(signal)
    echo_one = np.zeros_like(signal)
    for ch in range(s_ch):
        echo_zro[:, ch] = lfilter(k0, 1, signal[:, ch])
        echo_one[:, ch] = lfilter(k1, 1, signal[:, ch])
    
    max_echo = np.max(np.abs(echo_zro + echo_one))
    if max_echo > 0.5:
        scale = 0.5 / max_echo
        echo_zro *= scale
        echo_one *= scale
    
    m_sig = mixer(L, bits, 0, 1)
    mix = np.tile(m_sig, (s_ch, 1)).T
    out = signal[:max_bits * L, :] + echo_zro[:max_bits * L, :] * (1 - mix) + echo_one[:max_bits * L, :] * mix
    
    if max_bits * L < s_len:
        out = np.vstack([out, signal[max_bits * L:, :]])
    
    max_val = np.max(np.abs(out))
    if max_val > 1:
        out = out / max_val
    return out

def audiosave(out_signal, sample_rate, filename):
    """
    Save audio signal as a 16-bit WAV file.
    
    Args:
        out_signal (np.ndarray): Audio signal to save.
        sample_rate (int): Sample rate in Hz.
        filename (str): Output WAV file path.
    """
    if np.any(np.abs(out_signal) > 1):
        print("Warning: Clipping detected, signal normalized.")
        out_signal = np.clip(out_signal, -1, 1)
    out_signal_int16 = np.int16(out_signal * 32767)
    wavfile.write(filename, sample_rate, out_signal_int16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed text in preprocessed audio using echo steganography.")
    parser.add_argument("input_npy", help="Input preprocessed .npy file")
    parser.add_argument("output_wav", help="Output WAV file")
    parser.add_argument("text", help="Text to embed")
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate")
    args = parser.parse_args()
    
    try:
        signal = np.load(args.input_npy)
        out_signal = echo_embed(signal, args.text, d0=200, d1=300, alpha=0.6, L=8192)
        audiosave(out_signal, args.rate, args.output_wav)
        print(f"Signal with hidden message saved to {args.output_wav}")
    except Exception as e:
        print(f"Error: {e}")