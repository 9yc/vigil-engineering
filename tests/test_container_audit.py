import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import zlib


class TestContainerAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.binary = artifact_dir / "container_audit"
        if not cls.binary.is_file():
            raise RuntimeError(f"Missing compiled binary at {cls.binary}")
        cls.tmp_root = Path(os.environ["TMPDIR"])
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _build_sample_archive(self, td: Path) -> Path:
        (td / "weights.bin").write_bytes(struct.pack("<6f", 1.5, -2.0, 3.25, 0.0, 8.5, -4.75))
        (td / "indices.bin").write_bytes(struct.pack("<3q", 10, 20, 30))
        (td / "mask.bin").write_bytes(bytes([1, 0, 1, 1, 0]))

        manifest = {
            "metadata": [
                {"key": "producer", "type": "string", "value": "unit_f_packer"},
                {"key": "epoch", "type": "int64", "value": -42},
                {"key": "loss_scale", "type": "float64", "value": 0.125},
                {"key": "verified", "type": "bool", "value": True},
            ],
            "tensors": [
                {
                    "name": "layer0.weight",
                    "dtype": "f32",
                    "dims": [2, 3],
                    "alignment": 64,
                    "path": "weights.bin",
                },
                {
                    "name": "layer0.indices",
                    "dtype": "i64",
                    "dims": [3],
                    "alignment": 64,
                    "path": "indices.bin",
                },
                {
                    "name": "layer0.mask",
                    "dtype": "u8",
                    "dims": [5],
                    "alignment": 256,
                    "path": "mask.bin",
                },
            ],
        }
        manifest_path = td / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        archive_path = td / "bundle.tcnt"
        res = subprocess.run(
            [
                str(self.binary),
                "pack",
                "--manifest",
                str(manifest_path),
                "--output",
                str(archive_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        return archive_path

    def _audit_bytes(self, td: Path, name: str, raw: bytes):
        target = td / name
        target.write_bytes(raw)
        res = subprocess.run(
            [str(self.binary), "audit", "--input", str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        return res, json.loads(res.stdout)

    def _locate_tensor_descriptor_tail(self, raw: bytes, tensor_name: str) -> int:
        needle = struct.pack("<H", len(tensor_name)) + tensor_name.encode("utf-8")
        idx = raw.find(needle)
        self.assertNotEqual(idx, -1)
        pos = idx + len(needle)
        rank = raw[pos + 1]
        return pos + 2 + 8 * rank

    def test_pack_64b_alignment_metadata_roundtrip_and_clobber_guard(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td_str:
            td = Path(td_str)
            archive = self._build_sample_archive(td)

            dup = subprocess.run(
                [
                    str(self.binary),
                    "pack",
                    "--manifest",
                    str(td / "manifest.json"),
                    "--output",
                    str(archive),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(dup.returncode, 0)

            ins = subprocess.run(
                [str(self.binary), "inspect", "--input", str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ins.returncode, 0, msg=ins.stderr)
            info = json.loads(ins.stdout)

            meta_by_key = {m["key"]: (m["type"], m["value"]) for m in info["metadata"]}
            self.assertEqual(meta_by_key["producer"], ("string", "unit_f_packer"))
            self.assertEqual(meta_by_key["epoch"], ("int64", -42))
            self.assertAlmostEqual(meta_by_key["loss_scale"][1], 0.125, places=6)
            self.assertEqual(meta_by_key["verified"], ("bool", True))

            tensors = info["tensors"]
            self.assertEqual(len(tensors), 3)
            self.assertEqual(tensors[0]["payload_offset"] % 64, 0)
            self.assertEqual(tensors[1]["payload_offset"] % 64, 0)
            self.assertEqual(tensors[2]["payload_offset"] % 256, 0)

            expected_crc0 = zlib.crc32((td / "weights.bin").read_bytes()) & 0xFFFFFFFF
            self.assertEqual(tensors[0]["crc32"], expected_crc0)

            aud = subprocess.run(
                [str(self.binary), "audit", "--input", str(archive)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(aud.returncode, 0, msg=aud.stderr)
            aud_json = json.loads(aud.stdout)
            self.assertTrue(aud_json["valid"])
            self.assertEqual(aud_json["findings"], [])

    def test_detects_nonzero_padding_unaligned_offset_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td_str:
            td = Path(td_str)
            archive = self._build_sample_archive(td)
            orig = archive.read_bytes()
            info = json.loads(
                subprocess.run(
                    [str(self.binary), "inspect", "--input", str(archive)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            t0, t1 = info["tensors"][0], info["tensors"][1]

            pad_pos = t0["payload_offset"] + t0["payload_size"]
            self.assertLess(pad_pos, t1["payload_offset"])
            mut_pad = bytearray(orig)
            mut_pad[pad_pos] = 0xAB
            res_pad, rep_pad = self._audit_bytes(td, "bad_pad.tcnt", bytes(mut_pad))
            self.assertEqual(res_pad.returncode, 2)
            self.assertIn("NON_ZERO_PADDING", {f["code"] for f in rep_pad["findings"]})

            tail1 = self._locate_tensor_descriptor_tail(orig, "layer0.indices")
            mut_unaligned = bytearray(orig)
            struct.pack_into("<Q", mut_unaligned, tail1 + 8, t1["payload_offset"] - 8)
            res_ual, rep_ual = self._audit_bytes(td, "unaligned.tcnt", bytes(mut_unaligned))
            self.assertEqual(res_ual.returncode, 2)
            self.assertIn("UNALIGNED_PAYLOAD", {f["code"] for f in rep_ual["findings"]})

            mut_ov = bytearray(orig)
            struct.pack_into("<Q", mut_ov, tail1 + 8, t0["payload_offset"])
            res_ov, rep_ov = self._audit_bytes(td, "overlap.tcnt", bytes(mut_ov))
            self.assertEqual(res_ov.returncode, 2)
            self.assertIn("OVERLAPPING_SPAN", {f["code"] for f in rep_ov["findings"]})

    def test_detects_shape_overflow_crc_corruption_and_eof_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as td_str:
            td = Path(td_str)
            archive = self._build_sample_archive(td)
            orig = archive.read_bytes()
            info = json.loads(
                subprocess.run(
                    [str(self.binary), "inspect", "--input", str(archive)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )

            needle = struct.pack("<H", len("layer0.weight")) + b"layer0.weight"
            dims_pos = orig.find(needle) + len(needle) + 2
            mut_overflow = bytearray(orig)
            struct.pack_into("<QQ", mut_overflow, dims_pos, 1 << 34, 1 << 34)
            res_of, rep_of = self._audit_bytes(td, "overflow.tcnt", bytes(mut_overflow))
            self.assertEqual(res_of.returncode, 2)
            self.assertIn("SHAPE_OVERFLOW", {f["code"] for f in rep_of["findings"]})

            mut_crc = bytearray(orig)
            mut_crc[info["tensors"][0]["payload_offset"] + 3] ^= 0x01
            res_crc, rep_crc = self._audit_bytes(td, "corrupt_crc.tcnt", bytes(mut_crc))
            self.assertEqual(res_crc.returncode, 2)
            self.assertIn("CRC32_MISMATCH", {f["code"] for f in rep_crc["findings"]})

            res_eof, rep_eof = self._audit_bytes(td, "trailing.tcnt", orig + b"\x00")
            self.assertEqual(res_eof.returncode, 2)
            self.assertIn("EOF_MISMATCH", {f["code"] for f in rep_eof["findings"]})


if __name__ == "__main__":
    unittest.main()
