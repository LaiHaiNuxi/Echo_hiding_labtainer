#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.io import wavfile

# Tham số cố định
FRAME_LENGTH = 4096
OVERLAP = 0.5
REP_CODE = True
NUM_REPS = 3
MAX_EFFECTIVE_NBIT = 40

def fix(xs):
    return np.floor(xs) if xs >= 0 else np.ceil(xs)

def task2_compute_params(host_signal_file, output_file):
    # Đọc file âm thanh
    _, host_signal = wavfile.read(host_signal_file)
    signal_len = len(host_signal)

    # Tính toán tham số
    frame_shift = int(FRAME_LENGTH * (1 - OVERLAP))
    overlap_length = int(FRAME_LENGTH * OVERLAP)
    embed_nbit = fix((signal_len - overlap_length) / frame_shift)

    if REP_CODE:
        effective_nbit = np.floor(embed_nbit / NUM_REPS)
        effective_nbit = min(effective_nbit, MAX_EFFECTIVE_NBIT)
        embed_nbit = effective_nbit * NUM_REPS
    else:
        effective_nbit = embed_nbit

    # Chuyển sang kiểu int
    frame_shift = int(frame_shift)
    effective_nbit = int(effective_nbit)
    embed_nbit = int(embed_nbit)

    # Lưu tham số
    with open(output_file, 'w') as f:
        f.write(f"frame_shift={frame_shift}\n")
        f.write(f"embed_nbit={embed_nbit}\n")
        f.write(f"effective_nbit={effective_nbit}\n")

    print(f"frame_shift = {frame_shift}")
    print(f"embed_nbit = {embed_nbit}")
    print(f"effective_nbit = {effective_nbit}")

def main():
    parser = argparse.ArgumentParser(description="Compute embedding parameters.")
    parser.add_argument("--host_signal_file", type=str, default="bass_half.wav", help="Input audio file")
    parser.add_argument("--output_file", type=str, default="embed_params.dat", help="Output file for parameters")
    args = parser.parse_args()

    task2_compute_params(args.host_signal_file, args.output_file)

if __name__ == '__main__':
    main()