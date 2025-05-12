import numpy as np

def calculate_ber(original, decoded):
    min_len = min(len(original), len(decoded))
    errors = sum(o != d for o, d in zip(original[:min_len], decoded[:min_len]))
    ber = errors / min_len * 100
    return ber

def calculate_nc(original, decoded):
    if len(original) == 0 or len(decoded) == 0:
        return 0.0
    min_len = min(len(original), len(decoded))
    original = original[:min_len]
    decoded = decoded[:min_len]
    s1 = sum(ord(o) * ord(d) for o, d in zip(original, decoded))
    s2 = np.sqrt(sum(ord(o)**2 for o in original) * sum(ord(d)**2 for d in decoded))
    return s1 / s2 if s2 != 0 else 0.0

if __name__ == "__main__":
    original_message = "HELLO"
    with open("decoded_message.txt", 'r') as f:
        decoded_message = f.read().strip()
    ber = calculate_ber(original_message, decoded_message)
    nc = calculate_nc(original_message, decoded_message)
    print(f"Original message: {original_message}")
    print(f"Decoded message: {decoded_message}")
    print(f"Bit Error Rate (BER): {ber:.2f}%")
    print(f"Normalized Correlation (NC): {nc:.4f}")





