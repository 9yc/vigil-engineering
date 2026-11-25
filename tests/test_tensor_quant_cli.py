import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


class TestTensorQuantCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact_dir = Path(os.environ["ARTIFACT_DIR"])
        cls.bin_path = artifact_dir / "tensor_quant_cli"
        if not cls.bin_path.is_file():
            raise RuntimeError(f"Missing binary at {cls.bin_path}")
        cls.tmp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.work_dir = tempfile.TemporaryDirectory(dir=self.tmp_root)
        self.root = Path(self.work_dir.name)

    def tearDown(self):
        self.work_dir.cleanup()

    def run_cli(self, args, expect_ok=True):
        proc = subprocess.run(
            [str(self.bin_path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_ok and proc.returncode != 0:
            self.fail(f"CLI failed ({proc.returncode}): {proc.stderr}")
        if not expect_ok and proc.returncode == 0:
            self.fail(f"Expected CLI failure, but succeeded: {proc.stdout}")
        return proc

    def write_raw_f32(self, name, floats):
        path = self.root / name
        path.write_bytes(struct.pack(f"<{len(floats)}f", *floats))
        return path

    def read_raw_f32(self, path, count):
        raw = path.read_bytes()
        self.assertEqual(len(raw), count * 4)
        return list(struct.unpack(f"<{count}f", raw))

    def test_fp16_subnormals_ties_infinities_and_nans(self):
        special_values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            65504.0,
            -65504.0,
            100000.0,
            -100000.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            2.0 ** -14,
            2.0 ** -24,
            -(2.0 ** -24),
            0.5 * (2.0 ** -24),
            0.75 * (2.0 ** -24),
            1.0 + (2.0 ** -11),
            1.0 + (2.0 ** -10) + (2.0 ** -11),
        ]
        src = self.write_raw_f32("fp16_special.bin", special_values)
        q_out = self.root / "fp16.qtns"
        dq_out = self.root / "fp16_roundtrip.bin"

        self.run_cli([
            "quantize",
            "--input", str(src),
            "--output", str(q_out),
            "--dtype", "f16",
            "--rows", "3",
            "--cols", "6",
        ])
        insp = json.loads(self.run_cli(["inspect", "--input", str(q_out)]).stdout)
        self.assertEqual(insp["dtype"], "f16")
        self.assertEqual(insp["nan_count"], 1)
        self.assertEqual(insp["inf_count"], 4)

        self.run_cli([
            "dequantize",
            "--input", str(q_out),
            "--output", str(dq_out),
            "--raw-f32",
        ])
        got = self.read_raw_f32(dq_out, len(special_values))

        self.assertEqual(math.copysign(1.0, got[0]), 1.0)
        self.assertEqual(math.copysign(1.0, got[1]), -1.0)
        self.assertEqual(got[4], 65504.0)
        self.assertEqual(got[5], -65504.0)
        self.assertTrue(math.isinf(got[6]) and got[6] > 0)
        self.assertTrue(math.isinf(got[7]) and got[7] < 0)
        self.assertTrue(math.isinf(got[8]) and got[8] > 0)
        self.assertTrue(math.isinf(got[9]) and got[9] < 0)
        self.assertTrue(math.isnan(got[10]))
        self.assertEqual(got[11], 2.0 ** -14)
        self.assertEqual(got[12], 2.0 ** -24)
        self.assertEqual(got[13], -(2.0 ** -24))
        self.assertEqual(got[14], 0.0)
        self.assertEqual(got[15], 2.0 ** -24)
        self.assertEqual(got[16], 1.0)
        self.assertEqual(got[17], 1.0 + 2.0 * (2.0 ** -10))

    def test_bf16_round_to_nearest_even_and_nan_preservation(self):
        quiet_unlikely_nan = struct.unpack("<f", struct.pack("<I", 0x7F800001))[0]
        tie_even = struct.unpack("<f", struct.pack("<I", 0x3F808000))[0]
        tie_odd = struct.unpack("<f", struct.pack("<I", 0x3F818000))[0]
        vals = [tie_even, tie_odd, -3.5, 42.25, float("-inf"), quiet_unlikely_nan]

        src = self.write_raw_f32("bf16_in.bin", vals)
        q_out = self.root / "bf16.qtns"
        dq_out = self.root / "bf16_out.bin"

        self.run_cli([
            "quantize",
            "--input", str(src),
            "--output", str(q_out),
            "--dtype", "bf16",
            "--rows", "2",
            "--cols", "3",
        ])
        self.run_cli([
            "dequantize",
            "--input", str(q_out),
            "--output", str(dq_out),
            "--raw-f32",
        ])
        got = self.read_raw_f32(dq_out, len(vals))
        self.assertEqual(struct.unpack("<I", struct.pack("<f", got[0]))[0], 0x3F800000)
        self.assertEqual(struct.unpack("<I", struct.pack("<f", got[1]))[0], 0x3F820000)
        self.assertEqual(got[2], -3.5)
        self.assertEqual(got[3], 42.25)
        self.assertTrue(math.isinf(got[4]) and got[4] < 0)
        self.assertTrue(math.isnan(got[5]))

    def test_int8_and_int4_sqnr_and_verify_pipeline(self):
        rng = random.Random(1337)
        rows, cols = 16, 64
        gaussian = [rng.gauss(0.0, 1.2) for _ in range(rows * cols)]
        shifted = [rng.uniform(2.0, 5.0) for _ in range(rows * cols)]

        src_sym = self.write_raw_f32("sym_src.bin", gaussian)
        src_asym = self.write_raw_f32("asym_src.bin", shifted)

        cases = [
            ("q8_sym", src_sym, "f32", 38.0),
            ("q8_asym", src_asym, "f16", 42.0),
            ("q4_sym", src_sym, "f32", 16.0),
            ("q4_asym", src_asym, "f16", 20.0),
        ]
        for dtype, ref_file, scale_dt, min_sqnr in cases:
            with self.subTest(dtype=dtype):
                q_path = self.root / f"{dtype}.qtns"
                rep_path = self.root / f"{dtype}_rep.json"
                self.run_cli([
                    "quantize",
                    "--input", str(ref_file),
                    "--output", str(q_path),
                    "--dtype", dtype,
                    "--rows", str(rows),
                    "--cols", str(cols),
                    "--group-size", "32",
                    "--scale-dtype", scale_dt,
                    "--report", str(rep_path),
                ])
                q_rep = json.loads(rep_path.read_text(encoding="utf-8"))
                self.assertGreater(q_rep["metrics"]["sqnr_db"], min_sqnr)
                self.assertGreater(q_rep["metrics"]["cosine_similarity"], 0.98)
                self.assertEqual(len(q_rep["metrics"]["top_outliers"]), 5)

                ver_proc = self.run_cli([
                    "verify",
                    "--input", str(q_path),
                    "--reference", str(ref_file),
                    "--min-sqnr-db", str(min_sqnr),
                    "--top-k", "3",
                ])
                v_rep = json.loads(ver_proc.stdout)
                self.assertAlmostEqual(
                    v_rep["metrics"]["sqnr_db"],
                    q_rep["metrics"]["sqnr_db"],
                    places=5,
                )
                self.assertEqual(len(v_rep["metrics"]["top_outliers"]), 3)

                if dtype.startswith("q4"):
                    self.assertGreater(q_rep["compression_ratio"], 6.0)

        self.run_cli([
            "verify",
            "--input", str(self.root / "q4_sym.qtns"),
            "--reference", str(src_sym),
            "--min-sqnr-db", "60.0",
        ], expect_ok=False)

    def test_corruption_truncation_nan_quant_and_no_clobber(self):
        vals = [float(i) * 0.25 for i in range(64)]
        src = self.write_raw_f32("clean.bin", vals)
        q_path = self.root / "clean_q8.qtns"
        self.run_cli([
            "quantize",
            "--input", str(src),
            "--output", str(q_path),
            "--dtype", "q8_sym",
            "--rows", "8",
            "--cols", "8",
            "--group-size", "16",
        ])

        self.run_cli([
            "quantize",
            "--input", str(src),
            "--output", str(q_path),
            "--dtype", "q8_sym",
            "--rows", "8",
            "--cols", "8",
        ], expect_ok=False)

        raw_bytes = bytearray(q_path.read_bytes())

        trunc = self.root / "truncated.qtns"
        trunc.write_bytes(bytes(raw_bytes[:-3]))
        self.run_cli(["inspect", "--input", str(trunc)], expect_ok=False)

        bad_magic = self.root / "bad_magic.qtns"
        mut = bytearray(raw_bytes)
        mut[0] = ord("X")
        bad_magic.write_bytes(bytes(mut))
        self.run_cli(["dequantize", "--input", str(bad_magic), "--output", str(self.root / "x.bin")], expect_ok=False)

        bad_scale = self.root / "bad_scale.qtns"
        mut_scale = bytearray(raw_bytes)
        mut_scale[32:36] = struct.pack("<I", 0x7FC00000)
        bad_scale.write_bytes(bytes(mut_scale))
        self.run_cli(["inspect", "--input", str(bad_scale)], expect_ok=False)

        nan_src = self.write_raw_f32("nan_src.bin", [1.0, 2.0, float("nan"), 4.0])
        self.run_cli([
            "quantize",
            "--input", str(nan_src),
            "--output", str(self.root / "nan_q8.qtns"),
            "--dtype", "q8_sym",
            "--rows", "2",
            "--cols", "2",
        ], expect_ok=False)


if __name__ == "__main__":
    unittest.main()
