import numpy as np
from scipy.io import wavfile
import argparse

def generate_sine_wave(duration=2.0, sample_rate=44100, frequency=440.0, amplitude=0.5, stereo=False):
    """
    Generate a sine wave audio signal and save it as a 16-bit WAV file.
    
    Args:
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sample rate in Hz (default: 44100).
        frequency (float): Frequency of the sine wave in Hz (default: 440.0, A4 note).
        amplitude (float): Amplitude of the signal (0 to 1, default: 0.5).
        stereo (bool): If True, generate stereo signal (default: False).
    
    Returns:
        tuple: (sample_rate, signal) where signal is a NumPy array.
    """
    # Calculate number of samples
    num_samples = int(duration * sample_rate)
    
    # Generate time points
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Generate sine wave
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to stereo if requested
    if stereo:
        signal = np.stack([signal, signal], axis=1)  # Duplicate for left and right channels
    
    return sample_rate, signal

def save_wav(signal, sample_rate, filename):
    """
    Save signal as a 16-bit WAV file.
    
    Args:
        signal (np.ndarray): Audio signal (mono or stereo).
        sample_rate (int): Sample rate in Hz.
        filename (str): Output WAV file path.
    """
    # Clip and convert to 16-bit
    signal = np.clip(signal, -1, 1)
    signal_int16 = np.int16(signal * 32767)
    
    # Save to WAV file
    wavfile.write(filename, sample_rate, signal_int16)
    print("Saved audio to {}".format(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sine wave audio file.")
    parser.add_argument("output_file", help="Output WAV file path")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds (default: 2.0)")
    parser.add_argument("--frequency", type=float, default=440.0, help="Sine wave frequency in Hz (default: 440.0)")
    parser.add_argument("--amplitude", type=float, default=0.5, help="Amplitude (0 to 1, default: 0.5)")
    parser.add_argument("--stereo", action="store_true", help="Generate stereo signal")
    args = parser.parse_args()
    
    try:
        sample_rate, signal = generate_sine_wave(
            duration=args.duration,
            sample_rate=44100,
            frequency=args.frequency,
            amplitude=args.amplitude,
            stereo=args.stereo
        )
        save_wav(signal, sample_rate, args.output_file)
    except Exception as e:
        print("Error: {}".format(e))
