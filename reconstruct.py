import torch


def load_generator(checkpoint_path='./epoch/99generator.pth'):
    from model import Generator
    from config import DEVICE
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def generate_packets(n_samples=100, checkpoint_path='./epoch/99generator.pth'):
    from config import NOISE_DIM
    G = load_generator(checkpoint_path)
    z = torch.randn(n_samples, NOISE_DIM).to(DEVICE)
    logits = G(z)
    tokens = torch.argmax(logits, dim=-1).cpu().numpy()
    return decode_vocab_ids_to_hex(tokens)


def decode_vocab_ids_to_hex(vocab_ids):
    id2char = [f"{i:x}" for i in range(16)] + ['00']
    packets = []
    for sample in vocab_ids:
        hex_str = ''
        for i in range(0, len(sample) - 1, 2):
            a, b = sample[i], sample[i + 1]
            if a == 16 or b == 16:
                break
            hex_str += id2char[a] + id2char[b]
        packets.append(hex_str)
    return packets


def fix_packet(packet, template=None):
    if template is None:
        return packet

    p = bytearray(packet)

    for start, end, value in template.get('protocol_id', []):
        p[start:end + 1] = value

    length_info = template.get('length_field')
    if length_info:
        start = length_info['start']
        end = length_info['end']
        O = length_info['offset']

        V = int.from_bytes(p[start:end + 1], byteorder='big')
        L_exp = V + int(round(O))

        if L_exp < length_info.get('L_min', 1):
            return None

        if len(p) < L_exp:
            p.extend(b'\x00' * (L_exp - len(p)))
        elif len(p) > L_exp:
            p = p[:L_exp]

    return bytes(p)


def interaction_verify(packet):
    """Interaction-based Verification: replay the packet to the target ICS
    device or simulation and check the response.

    A valid response indicates the packet passed the device's internal protocol
    stack parsing. Packets with no response or error codes should be filtered out.

    NOTE: This is a placeholder that always returns True. Replace this function
    with your own implementation that:
        1. Encapsulates the packet with appropriate transport headers
        2. Transmits it to the target device / simulation environment
        3. Monitors the device feedback (response packet or error code)
        4. Returns True if a valid response is received, False otherwise

    Returns:
        bool: True if the packet is accepted by the device, False otherwise.
    """
    return True
