# The current TIAR/DGD/DGB implementation is a simplified Modbus TCP version.
# When targeting a different ICS protocol, replace the entire TIAR function
# (including its internal validation logic and constants) with your own
# protocol-specific acceptance test.

from config import NUM_FUNC, HEADER_LEN


def is_valid(packet_bytes):
    """Simplified Modbus TCP validation (length field only).
    Replace with your own protocol-specific validation when targeting a different ICS protocol.
    """
    if len(packet_bytes) < 8:
        return False
    length_field = int.from_bytes(packet_bytes[4:6], byteorder='big')
    expected_total_len = length_field + 6
    return expected_total_len == len(packet_bytes)


def TIAR(packets):
    cnt = sum(1 for pck in packets if is_valid(pck))
    return cnt / len(packets) * 100


def DGD(packets):
    """Data-Generating Diversity: ratio of unique function codes to total known codes.

    The function-code byte offset (currently 7) is Modbus-specific. Replace
    it with the correct offset for your target protocol.
    """
    function_bytes = set()
    for pck in packets:
        if len(pck) > 7:
            function_bytes.add(pck[7])
    return len(function_bytes) / NUM_FUNC * 100


def DGB(packets_gen, packets_orig):
    positions_gen = [set() for _ in range(HEADER_LEN)]
    positions_orig = [set() for _ in range(HEADER_LEN)]

    for i in range(HEADER_LEN):
        for pck in packets_gen:
            if len(pck) > i:
                positions_gen[i].add(pck[i])
        for pck in packets_orig:
            if len(pck) > i:
                positions_orig[i].add(pck[i])

    temp_dgb = []
    for i in range(HEADER_LEN):
        v_orig = len(positions_orig[i])
        v_gen = len(positions_gen[i])
        if v_orig == 0:
            continue
        gap = (v_gen - v_orig) / v_orig
        temp_dgb.append(gap)

    if not temp_dgb:
        return 0.0
    return sum(temp_dgb) / len(temp_dgb) * 100
