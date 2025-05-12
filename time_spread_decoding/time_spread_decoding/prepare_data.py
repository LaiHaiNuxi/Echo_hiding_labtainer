import numpy as np
from scipy.io import wavfile

def prepare_data(input_file, output_file):
    sample_rate, data = wavfile.read(input_file)
    if data.dtype != np.int16:
        raise ValueError("Only 16-bit WAV files are supported!")
    data = data.astype(np.float32) / 32768.0
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    np.save(output_file, data)
    print(f"Normalized data saved to {output_file}")

if __name__ == "__main__":
    prepare_data("output.wav", "decoded_preprocessed.npy")