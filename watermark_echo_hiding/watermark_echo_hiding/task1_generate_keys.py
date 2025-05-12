#!/usr/bin/env python3

import argparse
import numpy as np

# Tham số cố định
EFFECTIVE_NBIT = 40
NUM_REPS = 3
REP_CODE = True

def task1_generate_keys(watermark_original_file, watermark_extended_file, secret_key_file):
    # Tạo watermark ngẫu nhiên
    wmark_original = np.random.randint(2, size=EFFECTIVE_NBIT)

    # Lưu watermark gốc
    with open(watermark_original_file, 'w') as f:
        for d in wmark_original:
            f.write("%d\n" % d)

    # Mở rộng watermark
    wmark_extended = np.repeat(wmark_original, NUM_REPS) if REP_CODE else wmark_original

    # Lưu watermark mở rộng
    with open(watermark_extended_file, 'w') as f:
        for d in wmark_extended:
            f.write("%d\n" % d)

    # Tạo secret key ngẫu nhiên
    secret_key = np.random.randint(2, size=EFFECTIVE_NBIT)

    # Mở rộng secret key
    secret_key_extended = np.repeat(secret_key, NUM_REPS) if REP_CODE else secret_key

    # Lưu secret key
    with open(secret_key_file, 'w') as f:
        for d in secret_key_extended:
            f.write("%d\n" % d)
    print("Watermark and secret key generated successfully.")

def main():
    parser = argparse.ArgumentParser(description="Generate watermark and secret key.")
    parser.add_argument("--watermark_original_file", type=str, default="watermark_ori.dat", help="Output file for original watermark")
    parser.add_argument("--watermark_extended_file", type=str, default="watermark_extended.dat", help="Output file for extended watermark")
    parser.add_argument("--secret_key_file", type=str, default="secret_key.dat", help="Output file for secret key")
    args = parser.parse_args()

    task1_generate_keys(
        args.watermark_original_file,
        args.watermark_extended_file,
        args.secret_key_file
    )

if __name__ == '__main__':
    main()