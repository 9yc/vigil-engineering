import json
import math
import random
import struct
import unittest

from tools.tensor_quant import (
    apply_rope_interleaved,
    build_rope_cache,
    build_safetensors_header,
    dequantize_symmetric_int4,
    dequantize_symmetric_int8,
    parse_safetensors_header,
    quantization_error_metrics,
    quantize_symmetric_int4,
    quantize_symmetric_int8,
)


class QuantizationTests(unittest.TestCase):
    def test_int8_per_row_and_groupwise_sqnr(self):
        rng = random.Random(42)
        rows, cols = 8, 64
        weights = [rng.gauss(0.0, 0.75) for _ in range(rows * cols)]

        per_row = quantize_symmetric_int8(weights, rows, cols)
        rec_row = dequantize_symmetric_int8(per_row)
        metrics_row = quantization_error_metrics(weights, rec_row)
        self.assertEqual(len(per_row.packed), rows * cols)
        self.assertEqual(len(per_row.scales), rows)
        self.assertGreater(metrics_row["sqnr_db"], 38.0)

        grouped = quantize_symmetric_int8(weights, rows, cols, group_size=16)
        rec_grouped = dequantize_symmetric_int8(grouped)
        metrics_grouped = quantization_error_metrics(weights, rec_grouped)
        self.assertEqual(len(grouped.scales), (rows * cols) // 16)
        self.assertGreaterEqual(metrics_grouped["sqnr_db"], metrics_row["sqnr_db"])

    def test_int4_packing_extremes_zeros_and_sqnr(self):
        # Exact representable multiples of scale = 0.5 => [-3.5, -0.5, 0.0, 3.5]
        exact = [-3.5, -0.5, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0]
        q4 = quantize_symmetric_int4(exact, 2, 4, group_size=4)
        self.assertEqual(len(q4.packed), 4)
        rec = dequantize_symmetric_int4(q4)
        for a, b in zip(exact, rec):
            self.assertAlmostEqual(a, b, places=6)

        rng = random.Random(99)
        weights = [rng.uniform(-1.2, 1.2) for _ in range(4 * 32)]
        q_rand = quantize_symmetric_int4(weights, 4, 32, group_size=16)
        metrics = quantization_error_metrics(weights, dequantize_symmetric_int4(q_rand))
        self.assertGreater(metrics["sqnr_db"], 16.0)
        self.assertLessEqual(metrics["max_abs_error"], (1.2 / 7.0) * 0.501)

    def test_quantizer_rejects_invalid_shapes_and_values(self):
        with self.assertRaises(ValueError):
            quantize_symmetric_int8([1.0, 2.0, 3.0], 2, 2)
        with self.assertRaises(ValueError):
            quantize_symmetric_int8([1.0, float("nan"), 0.0, 0.0], 2, 2)
        with self.assertRaises(ValueError):
            quantize_symmetric_int8([1.0] * 6, 2, 3, group_size=2)
        with self.assertRaises(ValueError):
            quantize_symmetric_int4([1.0] * 6, 2, 3)
        with self.assertRaises(ValueError):
            quantize_symmetric_int4([1.0] * 6, 1, 6, group_size=3)


class SafeTensorsHeaderTests(unittest.TestCase):
    def test_roundtrip_header_and_offset_accounting(self):
        prefix, payload_bytes = build_safetensors_header(
            [
                ("embed.weight", "BF16", [16, 8]),
                ("norm.weight", "F32", [8]),
                ("scalar_step", "I64", []),
            ],
            metadata={"format": "pt"},
        )
        self.assertEqual(payload_bytes, 16 * 8 * 2 + 8 * 4 + 8)
        parsed = parse_safetensors_header(
            prefix, total_file_size=len(prefix) + payload_bytes
        )
        self.assertEqual(parsed["payload_bytes"], payload_bytes)
        self.assertEqual(parsed["metadata"], {"format": "pt"})
        self.assertEqual(parsed["tensors"]["embed.weight"]["data_offsets"], (0, 256))
        self.assertEqual(parsed["tensors"]["norm.weight"]["data_offsets"], (256, 288))
        self.assertEqual(parsed["tensors"]["scalar_step"]["data_offsets"], (288, 296))

    def test_corrupt_overlapping_or_mismatched_headers_fail(self):
        bad_dicts = [
            {"w": {"dtype": "F16", "shape": [4], "data_offsets": [0, 10]}},
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
             "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}},
            {"a": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}},
            {"a": {"dtype": "FP8_UNKNOWN", "shape": [2], "data_offsets": [0, 2]}},
            {"__metadata__": {"bad": 123}},
        ]
        for obj in bad_dicts:
            raw = json.dumps(obj).encode("utf-8")
            framed = struct.pack("<Q", len(raw)) + raw
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                parse_safetensors_header(framed)

        prefix, payload_bytes = build_safetensors_header([("w", "F32", [4])])
        with self.assertRaises(ValueError):
            parse_safetensors_header(prefix, total_file_size=len(prefix) + payload_bytes + 4)
        with self.assertRaises(ValueError):
            parse_safetensors_header(prefix[:6])
        with self.assertRaises(ValueError):
            parse_safetensors_header(prefix[:-2])


class RopeCacheTests(unittest.TestCase):
    def test_rope_norm_preservation_and_relative_dot_product(self):
        cos_table, sin_table = build_rope_cache(16, 8, base=10000.0)
        self.assertEqual(cos_table[0], (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(sin_table[0], (0.0, 0.0, 0.0, 0.0))

        q = [0.5, -1.2, 0.8, 0.3, -0.4, 1.1, 0.2, -0.7]
        k = [-0.3, 0.9, 0.4, -0.6, 1.0, 0.2, -0.5, 0.8]
        orig_norm_sq = sum(x * x for x in q)
        q_pos5 = apply_rope_interleaved(q, cos_table[5], sin_table[5])
        rot_norm_sq = sum(x * x for x in q_pos5)
        self.assertAlmostEqual(orig_norm_sq, rot_norm_sq, places=12)

        # Relative position invariance: <R(m)q, R(n)k> == <R(m+d)q, R(n+d)k>
        k_pos2 = apply_rope_interleaved(k, cos_table[2], sin_table[2])
        q_pos9 = apply_rope_interleaved(q, cos_table[9], sin_table[9])
        k_pos6 = apply_rope_interleaved(k, cos_table[6], sin_table[6])
        dot_a = math.fsum(a * b for a, b in zip(q_pos5, k_pos2))
        dot_b = math.fsum(a * b for a, b in zip(q_pos9, k_pos6))
        self.assertAlmostEqual(dot_a, dot_b, places=12)

    def test_rope_validation(self):
        for args, kwargs in (
            ((0, 8), {}),
            ((8, 7), {}),
            ((8, 8), {"base": 1.0}),
            ((8, 8), {"scaling_factor": 0.0}),
        ):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                build_rope_cache(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
