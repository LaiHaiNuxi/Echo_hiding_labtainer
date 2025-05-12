#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.io import wavfile

# Tham số cố định
REP_CODE = True
NUM_REPS = 3

def task5_evaluate(host_signal_file, watermark_signal_file, watermark_original_file, detected_bits_file, params_file):
    # Đọc file âm thanh
    _, host_signal = wavfile.read(host_signal_file)
    _, eval_signal = wavfile.read(watermark_signal_file)

    # Đọc tham số
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            params[key] = int(value)
    effective_nbit = params['effective_nbit']

    # Đọc watermark gốc
    with open(watermark_original_file, 'r') as f:
        wmark_original = np.array([int(line.strip()) for line in f])

    # Đọc bit phát hiện
    with open(detected_bits_file, 'r') as f:
        detected_bits = np.array([int(line.strip()) for line in f])

    # Xử lý repetition coding
    if REP_CODE:
        wmark_recovered = np.zeros(effective_nbit)
        count = 0
        for i in range(effective_nbit):
            ave = np.sum(detected_bits[count:count + NUM_REPS]) / NUM_REPS
            wmark_recovered[i] = 1 if ave >= 0.5 else 0
            count += NUM_REPS
    else:
        wmark_recovered = detected_bits

    # Tính BER
    ber = np.sum(np.abs(wmark_recovered - wmark_original)) / effective_nbit * 100

    # Tính SNR
    snr = 10 * np.log10(np.sum(np.square(host_signal.astype(np.float32))) /
                        np.sum(np.square(host_signal.astype(np.float32) - eval_signal.astype(np.float32))))

    # Lưu kết quả
    print(f"BER: {ber:.2f}%")
    print(f"SNR: {snr:.2f} dB")
    
    # with open(output_file, 'w') as f:
    #     f.write(f"BER={ber:.2f}%\n")
    #     f.write(f"SNR={snr:.2f}dB\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate watermarking performance (BER and SNR).")
    parser.add_argument("--host_signal_file", type=str, default="bass_half.wav", help="Original audio file")
    parser.add_argument("--watermark_signal_file", type=str, required=True, help="Watermarked audio file")
    parser.add_argument("--watermark_original_file", type=str, default="watermark_ori.dat", help="Original watermark file")
    parser.add_argument("--detected_bits_file", type=str, required=True, help="Detected bits file")
    parser.add_argument("--params_file", type=str, default="embed_params.dat", help="Parameters file")
    # parser.add_argument("--output_file", type=str, default="results.txt", help="Output file for results")
    args = parser.parse_args()

    task5_evaluate(
        args.host_signal_file,
        args.watermark_signal_file,
        args.watermark_original_file,
        args.detected_bits_file,
        args.params_file,
        # args.output_file
    )

if __name__ == '__main__':
    main()