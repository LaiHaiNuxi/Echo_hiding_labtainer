#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.io import wavfile

# Tham số cố định
FRAME_LENGTH = 4096
OVERLAP = 0.5
NEGATIVE_DELAY = 4
LOG_FLOOR = 0.00001

def task4_detect(watermark_signal_file, secret_key_file, params_file, output_file, signal_type):
    # Đọc file âm thanh đã nhúng
    _, eval_signal = wavfile.read(watermark_signal_file)

    # Đọc tham số
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            params[key] = int(value)
    frame_shift = params['frame_shift']
    embed_nbit = params['embed_nbit']

    # Đọc secret key
    with open(secret_key_file, 'r') as f:
        secret_key = np.array([int(line.strip()) for line in f])

    # Giới hạn embed_nbit
    embed_nbit = min(embed_nbit, len(secret_key))

    # Định nghĩa các delay
    delay11, delay10 = 100, 110  # Cho secret key = 1
    delay01, delay00 = 120, 130  # Cho secret key = 0

    # Phát hiện bit
    detected_bits = np.zeros(embed_nbit)
    pointer = 0
    for i in range(embed_nbit):
        wmarked_frame = eval_signal[pointer: pointer + FRAME_LENGTH]
        ceps = np.fft.ifft(np.log(np.square(np.fft.fft(wmarked_frame)) + LOG_FLOOR)).real

        if secret_key[i] == 1:
            if signal_type == 'signal2':
                detected_bits[i] = 1 if (ceps[delay11] - ceps[delay11 + NEGATIVE_DELAY]) > (ceps[delay10] - ceps[delay10 + NEGATIVE_DELAY]) else 0
            else:
                detected_bits[i] = 1 if ceps[delay11] > ceps[delay10] else 0
        else:
            if signal_type == 'signal2':
                detected_bits[i] = 1 if (ceps[delay01] - ceps[delay01 + NEGATIVE_DELAY]) > (ceps[delay00] - ceps[delay00 + NEGATIVE_DELAY]) else 0
            else:
                detected_bits[i] = 1 if ceps[delay01] > ceps[delay00] else 0

        pointer += frame_shift

    # Lưu kết quả phát hiện
    with open(output_file, 'w') as f:
        for bit in detected_bits:
            f.write(f"{int(bit)}\n")
    print(f"Detected bits saved to {output_file}")
    print(f"Detected bits: {detected_bits}")
    

def main():
    parser = argparse.ArgumentParser(description="Detect watermark from audio.")
    parser.add_argument("--watermark_signal_file", type=str, required=True, help="Input watermarked audio file")
    parser.add_argument("--secret_key_file", type=str, default="secret_key.dat", help="Secret key file")
    parser.add_argument("--params_file", type=str, default="embed_params.dat", help="Parameters file")
    parser.add_argument("--output_file", type=str, default="detected_bits.dat", help="Output file for detected bits")
    parser.add_argument("--signal_type", type=str, choices=['signal1', 'signal2', 'signal3'], required=True, help="Type of signal (signal1, signal2, signal3)")
    args = parser.parse_args()

    task4_detect(
        args.watermark_signal_file,
        args.secret_key_file,
        args.params_file,
        args.output_file,
        args.signal_type
    )

if __name__ == '__main__':
    main()