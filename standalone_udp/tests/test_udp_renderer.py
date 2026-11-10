"""Run with: PYTHONPATH=standalone_udp/app:app python3 -m unittest standalone_udp.tests.test_udp_renderer"""
import socket
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from sct_protocol import command, configuration, parse_ack
from udp_wire import ACK, CONTROL, STEREO_PNG, TRUTH, Reassembler, encode


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path("/tmp/standalone_udp_build/satellite_renderer_udp")


class RendererIntegrationTest(unittest.TestCase):
    def test_no_ros_udp_control_stereo_truth(self):
        if not BINARY.is_file():
            self.skipTest("Build satellite_renderer_udp before running the integration test")
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            (assets / "calibration.txt").write_bytes((ROOT / "ffs/64091.txt").read_bytes())
            mesh = assets / "mesh.obj"
            mesh.write_text("v -0.04 -0.04 0\nv 0.04 -0.04 0\nv 0 0.04 0\nf 1 2 3\n")
            satellite = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            satellite.bind(("127.0.0.1", 0))
            satellite.settimeout(0.5)
            port = 50500
            process = subprocess.Popen([str(BINARY), str(port), directory],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            reassembler = Reassembler()
            try:
                def receive_until(predicate, timeout=8):
                    deadline = time.monotonic() + timeout
                    messages = []
                    while time.monotonic() < deadline:
                        try:
                            packet, peer = satellite.recvfrom(2048)
                        except socket.timeout:
                            continue
                        message = reassembler.push(packet, peer)
                        if message is not None:
                            messages.append(message)
                            if predicate(message, messages):
                                return messages
                    self.fail(f"UDP message timeout; received {[(m[0], m[1]) for m in messages]}")

                objects = [("first", "mesh.obj", [0, 0, .8, 0, 0, 0], [0] * 6, [0] * 6),
                           ("second", "mesh.obj", [.1, 0, .8, 0, 0, 0], [.1, 0, 0, 0, 0, 0],
                            [.25, 0, 0, 0, 0, 0])]
                payload = configuration(640, 480, 2.0, 30_000_000, "calibration.txt", objects)
                for request_id, kind, body in ((1, 1, payload), (2, 2, b"")):
                    for _ in range(3):
                        for packet in encode(CONTROL, request_id, 0, command(kind, request_id, body)):
                            satellite.sendto(packet, ("127.0.0.1", port))
                        try:
                            messages = receive_until(lambda m, _: m[0] == ACK and m[1] == request_id, 2)
                            break
                        except AssertionError:
                            continue
                    else:
                        self.fail(f"No ACK for command {kind}")
                    self.assertEqual(parse_ack(messages[-1][3])[1], 0)
                messages = receive_until(lambda m, ms: any(x[0] == STEREO_PNG for x in ms) and
                                         any(x[0] == TRUTH for x in ms), 8)
                frame = next(m for m in messages if m[0] == STEREO_PNG)
                truth = next(m for m in messages if m[0] == TRUTH and m[1] == frame[1])
                self.assertEqual(frame[2], truth[2])
                self.assertEqual(struct.unpack_from("<H", truth[3])[0], 2)
                image = cv2.imdecode(np.frombuffer(frame[3], np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape, (480, 1280, 3))
                self.assertGreater(np.count_nonzero(image), 0)
                self.assertAlmostEqual(struct.unpack_from("<d", truth[3], 2)[0], 0)
                # A lost ACK can make the App retry START. The renderer must
                # acknowledge that request without resetting its frame counter.
                for packet in encode(CONTROL, 2, 0, command(2, 2)):
                    satellite.sendto(packet, ("127.0.0.1", port))
                duplicate_ack = receive_until(lambda m, _: m[0] == ACK and m[1] == 2, 3)
                self.assertEqual(parse_ack(duplicate_ack[-1][3])[1], 0)
                later = receive_until(lambda m, _: m[0] == STEREO_PNG, 5)[-1]
                self.assertGreater(later[1], frame[1])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                process.stderr.close()
                satellite.close()


if __name__ == "__main__":
    unittest.main()
