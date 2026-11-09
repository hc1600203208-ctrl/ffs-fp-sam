import struct
import unittest

from sct_protocol import command, configuration, parse_ack


class SctProtocolTest(unittest.TestCase):
    def test_configuration_layout(self):
        payload = configuration(640, 480, 8.0, 10_000_000,
                                "cal.txt",
                                [("target", "mesh.obj", [0.0] * 6, [0.1] * 6, [0.2] * 6),
                                 ("other", "mesh.obj", [1.0] * 6, [0.3] * 6, [0.4] * 6)])
        packet = command(1, 42, payload)
        self.assertEqual(packet[:16], struct.pack("<4sBBHII", b"SCT1", 1, 0, 0, 42, len(payload)))
        self.assertEqual(struct.unpack_from("<IIdIHH", packet, 16),
                         (640, 480, 8.0, 10_000_000, 2, 1))
        self.assertEqual(packet[16 + 24:16 + 26], b"\x07\x00")
        self.assertEqual(packet[16 + 26:16 + 33], b"cal.txt")
        first = 16 + 33 + 2 + len("target") + 2 + len("mesh.obj")
        self.assertEqual(struct.unpack_from("<18d", packet, first),
                         (0.0,) * 6 + (0.1,) * 6 + (0.2,) * 6)
        second = first + 18 * 8 + 2 + len("other") + 2 + len("mesh.obj")
        self.assertEqual(struct.unpack_from("<18d", packet, second),
                         (1.0,) * 6 + (0.3,) * 6 + (0.4,) * 6)

    def test_ack_and_invalid_fields(self):
        ack = struct.pack("<4sBBHII", b"SCT1", 0x80, 0, 0, 42, 5) + b"\x00\x02\x00ok"
        self.assertEqual(parse_ack(ack), (42, 0, "ok"))
        with self.assertRaises(ValueError):
            parse_ack(ack[:-1])
        with self.assertRaises(ValueError):
            configuration(640, 480, 8.0, 10_000_000,
                          "", [("target", "mesh.obj", [0.0] * 6, [0.1] * 6, [0.2] * 6)])


if __name__ == "__main__":
    unittest.main()
