import numpy as np

def decode_message(input_file, output_file, delay=100):
    data = np.load(input_file)
    binary = ""
    for i in range(0, len(data) - delay, delay):
        segment = data[i:i + delay]
        echo_energy = np.sum(np.abs(segment))
        binary += '1' if echo_energy > 0.1 else '0'
    message = ""
    for i in range(0, len(binary), 8):
        byte = binary[i:i + 8]
        if len(byte) == 8:
            message += chr(int(byte, 2))
    with open(output_file, 'w') as f:
        f.write(message)
    print(f"Decoded message saved to {output_file}")

if __name__ == "__main__":
    decode_message("decoded_preprocessed.npy", "decoded_message.txt")