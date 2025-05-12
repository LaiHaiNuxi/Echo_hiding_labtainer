#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import windows

# Tham số cố định
FRAME_LENGTH = 4096
CONTROL_STRENGTH = 0.2
OVERLAP = 0.5

def task3_embed_positive(host_signal_file, watermark_extended_file, secret_key_file, params_file, output_file):
    # Đọc file âm thanh
    sr, host_signal = wavfile.read(host_signal_file)
    signal_len = len(host_signal)

    # Đọc tham số
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            params[key] = int(value)
    frame_shift = params['frame_shift']
    embed_nbit = params['embed_nbit']

    # Đọc watermark và secret key
    with open(watermark_extended_file, 'r') as f:
        wmark_extended = np.array([int(line.strip()) for line in f])
    with open(secret_key_file, 'r') as f:
        secret_key_extended = np.array([int(line.strip()) for line in f])

    # Giới hạn embed_nbit
    embed_nbit = min(embed_nbit, len(wmark_extended), len(secret_key_extended))

    # Định nghĩa các delay
    delay11, delay10 = 100, 110  # Cho secret key = 1
    delay01, delay00 = 120, 130  # Cho secret key = 0

    # Khởi tạo tín hiệu nhúng
    echoed_signal = np.zeros(frame_shift * embed_nbit)
    prev = np.zeros(FRAME_LENGTH)

    pointer = 0
    for i in range(embed_nbit):
        frame = host_signal[pointer: pointer + FRAME_LENGTH]

        # Chọn delay
        if secret_key_extended[i] == 1:
            delay = delay11 if wmark_extended[i] == 1 else delay10
        else:
            delay = delay01 if wmark_extended[i] == 1 else delay00

        # Tạo positive echo
        echo = CONTROL_STRENGTH * np.concatenate((np.zeros(delay), frame[:FRAME_LENGTH - delay]))

        # Tạo frame đã nhúng
        echoed_frame = frame + echo

        # Áp dụng Hann window
        overlap_length = int(FRAME_LENGTH * OVERLAP)
        echoed_frame = echoed_frame * windows.hann(FRAME_LENGTH)
        echoed_signal[frame_shift * i: frame_shift * (i+1)] = np.concatenate(
            (prev[frame_shift:FRAME_LENGTH] + echoed_frame[:overlap_length], echoed_frame[overlap_length:frame_shift]))
        prev = echoed_frame

        pointer += frame_shift

    # Nối phần còn lại
    echoed_signal = np.concatenate((echoed_signal, host_signal[len(echoed_signal):]))

    # Lưu file âm thanh
    wavfile.write(output_file, sr, echoed_signal.astype(np.int16))

def main():
    parser = argparse.ArgumentParser(description="Embed watermark using positive echo.")
    parser.add_argument("--host_signal_file", type=str, default="bass_half.wav", help="Input audio file")
    parser.add_argument("--watermark_extended_file", type=str, default="watermark_extended.dat", help="Extended watermark file")
    parser.add_argument("--secret_key_file", type=str, default="secret_key.dat", help="Secret key file")
    parser.add_argument("--params_file", type=str, default="embed_params.dat", help="Parameters file")
    parser.add_argument("--output_file", type=str, default="wmed_signal1.wav", help="Output watermarked audio file")
    args = parser.parse_args()

    task3_embed_positive(
        args.host_signal_file,
        args.watermark_extended_file,
        args.secret_key_file,
        args.params_file,
        args.output_file
    )

    print(f"Watermarked by positive echo saved to {args.output_file}")

if __name__ == '__main__':
    main()