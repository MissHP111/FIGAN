import binascii
import os

import torch
import torch.nn.functional as F


def gumbel_softmax_sample(logits, tau=0.5):
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    y = (logits + gumbel_noise) / tau
    return F.softmax(y, dim=-1)


def toSave(packets, outputFilePath, encoding='utf-8'):
    with open(outputFilePath, 'w', encoding=encoding) as f:
        for pkt in packets:
            f.write(pkt + '\n')


def read_halfbyte_txt_to_byte_packets(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")

    packets = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            hex_str = line.strip()
            if not hex_str:
                continue
            if len(hex_str) % 2 != 0:
                hex_str = hex_str[:-1]
            try:
                packet = binascii.unhexlify(hex_str)
                packets.append(packet)
            except binascii.Error as e:
                print(f"[ERROR] Line {line_num}: {e}")
    return packets
