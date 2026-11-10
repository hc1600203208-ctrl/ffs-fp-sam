"""SUD1 datagrams shared by the satellite renderer and ground applications."""
import struct
import time
import zlib

HEADER = struct.Struct("<4sBBHIHHIQI")
CHUNK_SIZE = 1152
MAX_MESSAGE_SIZE = 8 * 1024 * 1024
CONTROL, ACK, STEREO_PNG, TRUTH, RESULT = range(1, 6)


def encode(kind, message_id, stamp_ns, payload):
    if kind not in range(1, 6) or len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError("Invalid SUD1 message")
    count = max(1, (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE)
    for part in range(count):
        chunk = payload[part * CHUNK_SIZE:(part + 1) * CHUNK_SIZE]
        yield HEADER.pack(b"SUD1", 1, kind, 0, message_id, part, count,
                          len(payload), stamp_ns, zlib.crc32(chunk)) + chunk


class Reassembler:
    def __init__(self):
        self.pending = {}

    def push(self, packet, peer):
        if len(packet) < HEADER.size:
            return None
        magic, version, kind, reserved, message_id, part, count, total, stamp_ns, crc = HEADER.unpack_from(packet)
        payload = packet[HEADER.size:]
        if (magic != b"SUD1" or version != 1 or kind not in range(1, 6) or reserved or
                total > MAX_MESSAGE_SIZE or not count or part >= count or
                count != max(1, (total + CHUNK_SIZE - 1) // CHUNK_SIZE) or
                len(payload) != min(CHUNK_SIZE, total - part * CHUNK_SIZE) or
                zlib.crc32(payload) != crc):
            return None
        key = (peer, kind, message_id)
        current = self.pending.get(key)
        if current is None or current[0:2] != (total, stamp_ns) or len(current[2]) != count:
            current = (total, stamp_ns, [None] * count, time.monotonic())
            self.pending[key] = current
        current[2][part] = payload
        if any(chunk is None for chunk in current[2]):
            return None
        del self.pending[key]
        return kind, message_id, stamp_ns, b"".join(current[2])

    def expire(self):
        now = time.monotonic()
        self.pending = {key: value for key, value in self.pending.items() if now - value[3] < 3}
