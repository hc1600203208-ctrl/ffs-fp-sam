"""Transport-independent SCT1 control packet encoding for the ground app."""
import math
import struct


HEADER = struct.Struct("<4sBBHII")


def _text(value):
    encoded = value.encode("utf-8")
    if not 0 < len(encoded) <= 4096 or b"\0" in encoded:
        raise ValueError("SCT1 strings must be 1..4096 UTF-8 bytes without NUL")
    return struct.pack("<H", len(encoded)) + encoded


def command(kind, request_id, payload=b""):
    if kind not in (1, 2, 3) or not 0 <= request_id <= 0xFFFFFFFF:
        raise ValueError("Invalid SCT1 command")
    return HEADER.pack(b"SCT1", kind, 0, 0, request_id, len(payload)) + payload


def configuration(width, height, fps, bytes_per_second, amplitudes, frequencies,
                  calibration_path, objects):
    if not (1 <= width <= 4096 and 1 <= height <= 4096 and
            0 < fps <= 1000 and math.isfinite(fps) and
            0 < bytes_per_second <= 0x7FFFFFFF and 1 <= len(objects) <= 32):
        raise ValueError("Invalid SCT1 rendering parameters")
    if len(amplitudes) != 6 or len(frequencies) != 6:
        raise ValueError("SCT1 requires six amplitudes and six frequencies")
    values = [*amplitudes, *frequencies]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SCT1 motion values must be finite")
    payload = bytearray(struct.pack("<IIdIHH", width, height, fps, bytes_per_second,
                                    len(objects), 0))
    payload.extend(struct.pack("<12d", *values))
    payload.extend(_text(calibration_path))
    for name, mesh_path, pose in objects:
        if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
            raise ValueError("Each SCT1 object needs six finite pose values")
        payload.extend(_text(name))
        payload.extend(_text(mesh_path))
        payload.extend(struct.pack("<6d", *pose))
    return bytes(payload)


def parse_ack(data):
    if len(data) < 19:
        raise ValueError("Short SCT1 acknowledgement")
    magic, kind, flags, reserved, request_id, length = HEADER.unpack_from(data)
    if magic != b"SCT1" or kind != 0x80 or flags or reserved or length != len(data) - 16:
        raise ValueError("Invalid SCT1 acknowledgement header")
    status, text_length = struct.unpack_from("<BH", data, 16)
    if length != 3 + text_length:
        raise ValueError("Invalid SCT1 acknowledgement length")
    return request_id, status, data[19:].decode("utf-8")
