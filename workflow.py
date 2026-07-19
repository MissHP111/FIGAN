import binascii
import torch

from config import N, NOISE_DIM, DEVICE


def phase1_heuristic_preprocessing(pcap_path):
    from heuristic import extract_protocol_identifier, extract_length_field
    from scapy.all import rdpcap

    raw_packets = rdpcap(pcap_path)
    protocol_ids = extract_protocol_identifier(raw_packets)
    length_field = extract_length_field(raw_packets)
    template = {'protocol_id': protocol_ids, 'length_field': length_field}

    return template


def phase2_generative_modeling(pcap_path=None, onehot_path=None):
    from encode import process_pcap_file
    from train import train

    if pcap_path is None:
        pcap_path = INPUT_PATH
    if onehot_path is None:
        onehot_path = INPUT_PATH_ONEHOT

    process_pcap_file(pcap_path, onehot_path)

    G = train(onehot_data_path=onehot_path)

    return G


def phase3_closed_loop_verification(G, template, n_gen, raw_packets=None):
    from reconstruct import decode_vocab_ids_to_hex, fix_packet, interaction_verify
    from evaluate import TIAR, DGD, DGB

    D_aug = []
    calibrated_packets = []

    while len(D_aug) < n_gen:
        z = torch.randn(1, NOISE_DIM).to(DEVICE)
        logits = G(z)
        tokens = torch.argmax(logits, dim=-1).cpu().numpy()

        hex_strs = decode_vocab_ids_to_hex(tokens)
        for hex_str in hex_strs:
            if len(hex_str) % 2 != 0:
                hex_str = hex_str[:-1]
            try:
                P_raw = binascii.unhexlify(hex_str)
            except binascii.Error:
                continue

            P_calib = fix_packet(P_raw, template)

            if P_calib is not None:
                calibrated_packets.append(P_calib)

                if interaction_verify(P_calib):
                    D_aug.append(P_calib)

            if len(D_aug) >= n_gen:
                break

    tiar = TIAR(calibrated_packets)
    dgd = DGD(calibrated_packets)
    print(f"  TIAR = {tiar:.2f}%")
    print(f"  DGD  = {dgd:.2f}%")

    if raw_packets is not None:
        dgb = DGB(calibrated_packets, raw_packets)
        print(f"  DGB  = {dgb:.2f}%")

    return D_aug


def run(pcap_path, onehot_path=None, n_gen=None):
    from scapy.all import rdpcap, TCP

    if n_gen is None:
        from config import N
        n_gen = N

    # PHASE 1
    template = phase1_heuristic_preprocessing(pcap_path)

    # PHASE 2
    G = phase2_generative_modeling(pcap_path, onehot_path)

    # PHASE 3
    raw_packets = [bytes(pkt[TCP].payload) for pkt in rdpcap(pcap_path)
                   if pkt.haslayer(TCP) and len(bytes(pkt[TCP].payload)) > 0]
    D_aug = phase3_closed_loop_verification(G, template, n_gen, raw_packets)

    return D_aug


if __name__ == "__main__":
    augmented = run(
        pcap_path="toy.pcap",
        onehot_path="toy_onehot.npy",
    )
