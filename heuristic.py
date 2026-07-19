import os
from collections import defaultdict

import numpy as np
from scapy.all import rdpcap, TCP
from scipy.stats import pearsonr


def partition_by_port(packets):
    """Partition packets by destination port for protocol purity verification.

    When the mined protocol identifier holds for only a subset of the trace
    (implying mixed protocols), the destination port should be used as a
    secondary filter to partition the dataset into homogeneous subsets.

    NOTE: This is a placeholder that returns a single random partition.
    Replace this function with a proper port-based partitioning implementation
    for your specific dataset
    """
    mid = len(packets) // 2
    return [packets[:mid], packets[mid:]]


def extract_protocol_identifier(packets):
    candidate_bytes = []
    for packet in packets:
        if packet.haslayer(TCP):
            payload = bytes(packet[TCP].payload)
            if len(payload) >= 4:
                candidate_bytes.append(payload[:4])

    byte_positions = {i: [] for i in range(4)}
    for b in candidate_bytes:
        for i in range(4):
            byte_positions[i].append(b[i])

    stats = {}
    for pos in range(4):
        value_counts = defaultdict(int)
        for byte in byte_positions[pos]:
            value_counts[byte] += 1
        total = len(byte_positions[pos])
        stats[pos] = {k: v / total for k, v in value_counts.items()}

    static_fields = {}
    for pos in range(4):
        for value, freq in stats[pos].items():
            if freq >= 0.95:
                static_fields[pos] = value

    continuous_static = []
    i = 0
    while i < 4:
        if i in static_fields:
            j = i
            while j + 1 in static_fields:
                j += 1
            continuous_static.append((i, j))
            i = j + 1
        else:
            i += 1

    protocol_identifiers = []
    for start, end in continuous_static:
        pi_values = [b[start:end + 1] for b in candidate_bytes]
        if len(set(pi_values)) / len(pi_values) < 0.05:
            protocol_identifiers.append((start, end, pi_values[0]))
        else:
            # Mined rule does not hold globally; partition by port and retry
            partitions = partition_by_port(packets)
            for sub_packets in partitions:
                sub_ids = extract_protocol_identifier(sub_packets)
                protocol_identifiers.extend(sub_ids)

    return protocol_identifiers


def extract_length_field(packets):
    length_groups = defaultdict(list)
    for packet in packets:
        if packet.haslayer(TCP):
            payload = bytes(packet[TCP].payload)
            length_groups[len(payload)].append(payload)

    group_items = list(length_groups.items())
    if len(group_items) < 2:
        return None

    L_min = min(len(p) for group in length_groups.values() for p in group)
    header_len = L_min

    candidate_positions = []
    for pos in range(header_len):
        group_static = True
        for _, payloads in group_items:
            values = [p[pos] for p in payloads if len(p) > pos]
            if len(set(values)) > 1:
                group_static = False
                break

        if group_static:
            all_values = [p[pos] for _, payloads in group_items for p in payloads if len(p) > pos]
            if len(set(all_values)) > 1:
                candidate_positions.append(pos)

    if not candidate_positions:
        return None

    merged_candidates = []
    i = 0
    while i < len(candidate_positions):
        j = i
        while j + 1 < len(candidate_positions) and candidate_positions[j + 1] == candidate_positions[j] + 1:
            j += 1
        field_size = j - i + 1
        if field_size in [1, 2, 4]:
            merged_candidates.append({
                'start': candidate_positions[i],
                'end': candidate_positions[j],
                'field_size': field_size
            })
        i = j + 1

    best_result = None
    best_correlation = -1
    best_offset = None

    for candidate in merged_candidates:
        start = candidate['start']
        end = candidate['end']
        field_size = candidate['field_size']

        for endian in ['big', 'little']:
            delta_T = []
            delta_L = []

            for i in range(len(group_items)):
                len_i, payloads_i = group_items[i]
                for j in range(i + 1, len(group_items)):
                    len_j, payloads_j = group_items[j]

                    dt = len_i - len_j

                    L_i = [int.from_bytes(p[start:end + 1], byteorder=endian)
                           for p in payloads_i if len(p) > end]
                    L_j = [int.from_bytes(p[start:end + 1], byteorder=endian)
                           for p in payloads_j if len(p) > end]

                    if not L_i or not L_j:
                        continue

                    dl = np.mean(L_i) - np.mean(L_j)
                    delta_T.append(dt)
                    delta_L.append(dl)

            if len(delta_T) > 1:
                r, _ = pearsonr(delta_T, delta_L)
                if r > 0.95 and r > best_correlation:
                    # Estimate offset O = mean(L_phy - V)
                    all_offsets = []
                    for group in length_groups.values():
                        for p in group:
                            if len(p) > end:
                                V = int.from_bytes(p[start:end + 1], byteorder=endian)
                                all_offsets.append(len(p) - V)
                    O = np.mean(all_offsets) if all_offsets else 0

                    # Linear consistency verification: L_phy = V + O
                    valid_count = 0
                    total_count = 0
                    for group in length_groups.values():
                        for p in group:
                            if len(p) > end:
                                V = int.from_bytes(p[start:end + 1], byteorder=endian)
                                if abs(len(p) - (V + O)) < 1e-6:
                                    valid_count += 1
                                total_count += 1

                    if total_count > 0 and valid_count / total_count >= 0.95:
                        best_result = {'start': start, 'end': end, 'offset': O, 'L_min': L_min}
                        best_correlation = r
                        best_offset = O

    return best_result
