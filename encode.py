import numpy as np

vocab = list("0123456789abcdef") + ["X"]
char_to_index = {char: idx for idx, char in enumerate(vocab)}
PAD_CHAR = "X"
MAX_LENGTH = 256


def hex_to_onehot(hex_str):
    hex_str = hex_str.lower()

    if len(hex_str) > MAX_LENGTH:
        hex_str = hex_str[:MAX_LENGTH]
    else:
        hex_str = hex_str.ljust(MAX_LENGTH, PAD_CHAR)

    indices = [char_to_index[char] for char in hex_str]
    one_hot_matrix = np.eye(len(vocab))[indices]

    return one_hot_matrix


def process_pcap_file(pcap_path, output_path=None):
    from scapy.all import rdpcap, TCP

    packets = rdpcap(pcap_path)
    results = []
    for pkt in packets:
        if pkt.haslayer(TCP) and len(bytes(pkt[TCP].payload)) > 0:
            payload = bytes(pkt[TCP].payload)
            hex_str = payload.hex()
            one_hot = hex_to_onehot(hex_str)
            results.append(one_hot)

    results = np.array(results)

    if output_path:
        np.save(output_path, results)

    return results