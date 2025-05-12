import numpy as np
from scipy.io import wavfile
import os
import argparse

def preprocess_audio(input_wav, output_npy):
    """
    Preprocess a WAV file by normalizing and saving as a NumPy array.
    
    Args:
        input_wav (str): Path to input WAV file.
        output_npy (str): Path to output .npy file.
    
    Returns:
        int: Sample rate of the audio.
    
    Raises:
        ValueError: If file is invalid or not 16-bit WAV.
    """
    if not os.path.exists(input_wav):
        raise ValueError(f"File {input_wav} does not exist!")
    
    try:
        sample_rate, data = wavfile.read(input_wav)
    except ValueError as e:
        raise ValueError(f"Invalid WAV file: {e}")
    
    if data.dtype != np.int16:
        raise ValueError("Only 16-bit WAV files are supported!")
    
    # Normalize to [-1, 1]
    data = data.astype(np.float32) / 32768.0
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    
    # Handle mono/stereo
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    # Save as .npy
    np.save(output_npy, data)
    print(f"Preprocessed audio saved to {output_npy}")
    return sample_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio for echo steganography.")
    parser.add_argument("input_wav", help="Input WAV file")
    parser.add_argument("output_npy", help="Output .npy file")
    args = parser.parse_args()
    
    try:
        preprocess_audio(args.input_wav, args.output_npy)
    except Exception as e:
        print(f"Error: {e}")