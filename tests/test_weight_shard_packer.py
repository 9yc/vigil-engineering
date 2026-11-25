from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from tools import weight_shard_packer as wsp


def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def pack_f16(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def pack_bf16(values: list[float]) -> bytes:
    u16: list[int] = []
    for v in values:
        bits32 = struct.unpack("<I", struct.pack("<f", v))[0]
        u16.append((bits32 >> 16) & 0xFFFF)
    return struct.pack(f"<{len(u16)}H", *u16)


def pack_i8(values: list[int]) -> bytes:
    return array("b", values).tobytes()


def build_synthetic_sharded_checkpoint(model_dir: Path, seed: int = 1234) -> dict[str, list[float]]:
    rng = random.Random(seed)
    vocab, hidden, kv_dim, inter = 24, 16, 8, 32
    emb_vals = [rng.uniform(-1.5, 1.5) for _ in range(vocab * hidden)]
    lm_vals = [rng.uniform(-1.0, 1.0) for _ in range(vocab * hidden)]
    norm_vals = [rng.uniform(0.8, 1.2) for _ in range(hidden)]

    shard1_tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.embed_tokens.weight": ("BF16", [vocab, hidden], pack_bf16(emb_vals)),
        "model.norm.weight": ("F32", [hidden], pack_f32(norm_vals)),
    }
    shard2_tensors: dict[str, tuple[str, list[int], bytes]] = {
        "lm_head.weight": ("F16", [vocab, hidden], pack_f16(lm_vals)),
        "aux.quant_marker": ("I8", [8], pack_i8([-127, -64, -1, 0, 1, 32, 99, 127])),
    }

    ref_floats: dict[str, list[float]] = {
        "model.norm.weight": [wsp.f32_round(v) for v in norm_vals],
    }

    for layer_idx in range(2):
        target = shard1_tensors if layer_idx == 0 else shard2_tensors
        dtype_cycle = ("F32", "BF16", "F16", "F32", "F16", "BF16", "F32")
        proj_specs = (
            ("self_attn.q_proj", hidden, hidden),
            ("self_attn.k_proj", kv_dim, hidden),
            ("self_attn.v_proj", kv_dim, hidden),
            ("self_attn.o_proj", hidden, hidden),
            ("mlp.gate_proj", inter, hidden),
            ("mlp.up_proj", inter, hidden),
            ("mlp.down_proj", hidden, inter),
        )
        for (sub_name, rows, cols), dt in zip(proj_specs, dtype_cycle):
            tname = f"model.layers.{layer_idx}.{sub_name}.weight"
            vals = [rng.uniform(-0.75, 0.75) for _ in range(rows * cols)]
            if dt == "F32":
                raw = pack_f32(vals)
            elif dt == "F16":
                raw = pack_f16(vals)
            else:
                raw = pack_bf16(vals)
            target[tname] = (dt, [rows, cols], raw)

    s1_name = "model-00001-of-00002.safetensors"
    s2_name = "model-00002-of-00002.safetensors"
    wsp.write_safetensors_file(model_dir / s1_name, shard1_tensors, metadata={"format": "pt"})
    wsp.write_safetensors_file(model_dir / s2_name, shard2_tensors, metadata={"format": "pt"})

    weight_map: dict[str, str] = {}
    for k in shard1_tensors:
        weight_map[k] = s1_name
    for k in shard2_tensors:
        weight_map[k] = s2_name
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_shards": 2}, "weight_map": weight_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    return ref_floats


class TestWeightShardPacker(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmp_env = os.environ.get("TMPDIR")
        cls.tmp_root = Path(tmp_env) if tmp_env else Path(tempfile.gettempdir())
        cls.tmp_root.mkdir(parents=True, exist_ok=True)
        art_env = os.environ.get("ARTIFACT_DIR")
        cls.ref_cli = (Path(art_env) / "transformer_ref_cli") if art_env else None
        cls.packer_script = Path(__file__).resolve().parent.parent / "tools" / "weight_shard_packer.py"

    def test_ieee754_f16_bf16_i8_and_sharded_safetensors_reader(self) -> None:
        special_u16 = [
            0x0000,  # +0.0
            0x8000,  # -0.0
            0x0001,  # smallest positive subnormal
            0x03FF,  # largest subnormal
            0x0400,  # smallest positive normal
            0x3C00,  # 1.0
            0xC000,  # -2.0
            0x7BFF,  # max normal 65504.0
            0x7C00,  # +inf
            0xFC00,  # -inf
            0x7E00,  # qNaN
        ]
        for u16 in special_u16:
            got = wsp.ieee754_f16_bits_to_f32(u16)
            ref = struct.unpack("<e", struct.pack("<H", u16))[0]
            if math.isnan(ref):
                self.assertTrue(math.isnan(got))
            else:
                self.assertEqual(got, ref)
                self.assertEqual(math.copysign(1.0, got), math.copysign(1.0, ref))

        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            mdir = Path(td) / "sharded_model"
            ref = build_synthetic_sharded_checkpoint(mdir)
            arc = wsp.SafeTensorArchive(mdir)

            self.assertIn("model.embed_tokens.weight", arc.tensor_names())
            self.assertIn("lm_head.weight", arc.tensor_names())
            n_shape, n_vals = arc.read_f32("model.norm.weight")
            self.assertEqual(n_shape, (16,))
            self.assertEqual(n_vals, ref["model.norm.weight"])

            i8_shape, i8_vals = arc.read_i8("aux.quant_marker")
            self.assertEqual(i8_shape, (8,))
            self.assertEqual(i8_vals, [-127, -64, -1, 0, 1, 32, 99, 127])

    def test_emb1_and_lmh1_export_and_cpp_cross_compatibility(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            mdir = tdir / "model"
            build_synthetic_sharded_checkpoint(mdir, seed=99)
            arc = wsp.SafeTensorArchive(mdir)

            emb_out = tdir / "tok_emb.emb1"
            emb_stats = wsp.export_emb1(arc, "model.embed_tokens.weight", emb_out)
            self.assertEqual(emb_stats["magic"], "EMB1")
            self.assertEqual(emb_stats["vocab_size"], 24)
            self.assertEqual(emb_stats["hidden_size"], 16)
            self.assertGreater(emb_stats["cos_min"], 0.999)
            self.assertLess(emb_stats["max_abs_err"], 0.02)

            emb_raw = emb_out.read_bytes()
            magic, ver, vocab, hidden, stride, flags, r0, r1 = struct.unpack("<4s7I", emb_raw[:32])
            self.assertEqual((magic, ver, vocab, hidden, stride, flags, r0, r1), (b"EMB1", 1, 24, 16, 20, 1, 0, 0))
            self.assertEqual(hashlib.sha256(emb_raw).hexdigest(), emb_stats["sha256"])

            lmh_out = tdir / "lm_head.lmh1"
            lmh_stats = wsp.export_lmh1(
                arc,
                lmh_out,
                lm_head_tensor="lm_head.weight",
                norm_tensor="model.norm.weight",
                rms_eps=1e-6,
            )
            self.assertFalse(lmh_stats["untied_from_embeddings"])
            self.assertTrue(lmh_stats["has_norm_weight"])
            lmh_raw = lmh_out.read_bytes()
            hdr = struct.unpack("<4s15I", lmh_raw[:64])
            self.assertEqual(hdr[0], b"LMH1")
            self.assertEqual(hdr[1:5], (1, 24, 16, 1))
            self.assertEqual(hdr[5] % 64, 0)
            self.assertEqual(hdr[6] % 64, 0)
            self.assertEqual(hdr[7] % 64, 0)

            tied_dir = tdir / "tied_model"
            wsp.write_safetensors_file(
                tied_dir / "model.safetensors",
                {
                    "model.embed_tokens.weight": ("BF16", [12, 8], pack_bf16([0.1 * (i + 1) for i in range(96)])),
                },
            )
            tied_arc = wsp.SafeTensorArchive(tied_dir)
            tied_lmh = tdir / "tied.lmh1"
            tied_stats = wsp.export_lmh1(tied_arc, tied_lmh)
            self.assertTrue(tied_stats["untied_from_embeddings"])

            if self.ref_cli is not None and self.ref_cli.is_file():
                emb_json = tdir / "cpp_emb_lookup.json"
                proc_emb = subprocess.run(
                    [str(self.ref_cli), "lookup-emb", "--emb", str(emb_out), "--tokens", "0,5,23", "--output", str(emb_json)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc_emb.returncode, 0, msg=proc_emb.stderr)
                parsed_emb = json.loads(emb_json.read_text(encoding="utf-8"))
                self.assertEqual(len(parsed_emb["rows"]), 3)
                self.assertEqual(len(parsed_emb["rows"][0]), 16)

                lmh_json = tdir / "cpp_lmh_run.json"
                proc_lmh = subprocess.run(
                    [str(self.ref_cli), "run-lmhead", "--lmhead", str(lmh_out), "--synthetic-input", "--threads", "2", "--output", str(lmh_json)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc_lmh.returncode, 0, msg=proc_lmh.stderr)
                parsed_lmh = json.loads(lmh_json.read_text(encoding="utf-8"))
                self.assertEqual(parsed_lmh["vocab_size"], 24)

    def test_region_shard_splitting_alignment_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            mdir = tdir / "model"
            build_synthetic_sharded_checkpoint(mdir, seed=2026)
            arc = wsp.SafeTensorArchive(mdir)

            reg_dir = tdir / "regions"
            max_bytes = 2400
            manifest = wsp.pack_region_shards(
                arc, reg_dir, align=64, max_shard_bytes=max_bytes, storage_format="w8"
            )
            self.assertEqual(manifest["total_tensors"], 14)
            self.assertGreaterEqual(manifest["total_shards"], 3)

            for s_meta in manifest["shards"]:
                s_path = reg_dir / s_meta["file"]
                self.assertTrue(s_path.is_file())
                self.assertLessEqual(s_path.stat().st_size, max_bytes)
            for t_meta in manifest["tensors"]:
                self.assertEqual(t_meta["offset"] % 64, 0)

            manifest_path = reg_dir / "shard_manifest.json"
            clean_audit = wsp.audit_shard_manifest(manifest_path)
            self.assertTrue(clean_audit["ok"], msg=str(clean_audit["errors"]))
            self.assertEqual(clean_audit["checked_tensors"], 14)

            first_tensor = manifest["tensors"][0]
            shard0_path = reg_dir / first_tensor["shard_file"]
            orig_bytes = shard0_path.read_bytes()
            tampered = bytearray(orig_bytes)
            tampered[first_tensor["offset"]] ^= 0xFF
            shard0_path.write_bytes(bytes(tampered))

            bad_audit = wsp.audit_shard_manifest(manifest_path)
            self.assertFalse(bad_audit["ok"])
            self.assertTrue(any("SHA-256 mismatch" in err for err in bad_audit["errors"]))

            shard0_path.write_bytes(orig_bytes)
            self.assertTrue(wsp.audit_shard_manifest(manifest_path)["ok"])

    def test_cli_entrypoints_and_non_destructive_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            mdir = tdir / "model"
            build_synthetic_sharded_checkpoint(mdir, seed=555)
            env = dict(os.environ)

            emb_path = tdir / "cli.emb1"
            emb_stats = tdir / "cli_emb_stats.json"
            r1 = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "export-emb", "--model", str(mdir), "--output", str(emb_path), "--stats-output", str(emb_stats),
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertEqual(r1.returncode, 0, msg=r1.stderr)
            saved_bytes = emb_path.read_bytes()

            r_clobber = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "export-emb", "--model", str(mdir), "--output", str(emb_path),
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertNotEqual(r_clobber.returncode, 0)
            self.assertEqual(emb_path.read_bytes(), saved_bytes)

            r_over = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "export-emb", "--model", str(mdir), "--output", str(emb_path), "--overwrite",
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertEqual(r_over.returncode, 0, msg=r_over.stderr)

            reg_dir = tdir / "cli_regions"
            r_pack = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "pack-regions", "--model", str(mdir), "--output", str(reg_dir), "--align", "64", "--max-shard-bytes", "3000",
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertEqual(r_pack.returncode, 0, msg=r_pack.stderr)

            audit_out = tdir / "audit_out.json"
            r_aud = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "audit-manifest", "--manifest", str(reg_dir / "shard_manifest.json"), "--output", str(audit_out),
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertEqual(r_aud.returncode, 0, msg=r_aud.stderr)
            self.assertTrue(json.loads(audit_out.read_text(encoding="utf-8"))["ok"])

            lmh_cli_path = tdir / "cli_head.lmh1"
            r_lmh = subprocess.run(
                [
                    sys.executable, "-B", str(self.packer_script),
                    "export-lmhead", "--model", str(mdir), "--norm-tensor", "model.norm.weight", "--output", str(lmh_cli_path),
                ],
                capture_output=True, text=True, env=env, check=False,
            )
            self.assertEqual(r_lmh.returncode, 0, msg=r_lmh.stderr)
            self.assertEqual(lmh_cli_path.read_bytes()[:4], b"LMH1")

    def test_corrupt_safetensors_and_oversized_tensor_rejection(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            bad_st = tdir / "corrupt.safetensors"
            bad_st.write_bytes(struct.pack("<Q", 999999) + b"{}")
            with self.assertRaises(ValueError):
                wsp.SafeTensorArchive(bad_st)

            tiny_model = tdir / "tiny.safetensors"
            wsp.write_safetensors_file(
                tiny_model,
                {"model.layers.0.self_attn.q_proj.weight": ("F32", [16, 16], pack_f32([0.1] * 256))},
            )
            arc = wsp.SafeTensorArchive(tiny_model)
            with self.assertRaises(ValueError):
                wsp.pack_region_shards(arc, tdir / "tiny_out", align=64, max_shard_bytes=128)


if __name__ == "__main__":
    unittest.main()
