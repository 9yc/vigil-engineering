import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest


def f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def pack_f32_list(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


def py_rms_norm(x: list[float], w: list[float], eps: float) -> list[float]:
    inv_rms = f32(1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps))
    return [f32(f32(v * inv_rms) * w_i) for v, w_i in zip(x, w)]


def py_quantize_row_i8(row: list[float]) -> tuple[list[int], float]:
    scale = f32(max(max(abs(v) for v in row) / 127.0, 1e-12))
    return [max(-127, min(127, int(round(f32(v / scale))))) for v in row], scale


def py_fake_quant_w8(mat: list[float], rows: int, cols: int) -> list[float]:
    out: list[float] = []
    for r in range(rows):
        codes, scale = py_quantize_row_i8(mat[r * cols : (r + 1) * cols])
        out.extend(f32(float(c) * scale) for c in codes)
    return out


def py_fake_quant_a16(vec: list[float]) -> list[float]:
    s_tok = f32(max(max(abs(v) for v in vec) / 32767.0, 1e-12))
    return [f32(float(max(-32767, min(32767, int(round(f32(v / s_tok)))))) * s_tok) for v in vec]


def py_linear(x: list[float], w: list[float], out_dim: int, in_dim: int, w8a16: bool) -> list[float]:
    x_use = py_fake_quant_a16(x) if w8a16 else x
    return [f32(sum(w[o * in_dim + i] * x_use[i] for i in range(in_dim))) for o in range(out_dim)]


def py_apply_rope(vec: list[float], n_heads: int, head_dim: int, pos: int, theta: float) -> list[float]:
    res = list(vec)
    half = head_dim // 2
    for j in range(half):
        angle = pos * (theta ** (-2.0 * j / head_dim))
        c, s = f32(math.cos(angle)), f32(math.sin(angle))
        for h in range(n_heads):
            off = h * head_dim
            x0, x1 = res[off + j], res[off + half + j]
            res[off + j] = f32(x0 * c - x1 * s)
            res[off + half + j] = f32(x1 * c + x0 * s)
    return res


def py_block_sequence_ref(
    seq_in: list[list[float]], weights: dict[str, list[float]], hidden: int,
    n_heads: int, n_kv_heads: int, head_dim: int, inter: int, eps: float, theta: float, w8a16: bool,
) -> list[list[float]]:
    qd, kvd = n_heads * head_dim, n_kv_heads * head_dim
    wq = py_fake_quant_w8(weights["wq"], qd, hidden) if w8a16 else weights["wq"]
    wk = py_fake_quant_w8(weights["wk"], kvd, hidden) if w8a16 else weights["wk"]
    wv = py_fake_quant_w8(weights["wv"], kvd, hidden) if w8a16 else weights["wv"]
    wo = py_fake_quant_w8(weights["wo"], hidden, qd) if w8a16 else weights["wo"]
    wg = py_fake_quant_w8(weights["wg"], inter, hidden) if w8a16 else weights["wg"]
    wu = py_fake_quant_w8(weights["wu"], inter, hidden) if w8a16 else weights["wu"]
    wd = py_fake_quant_w8(weights["wd"], hidden, inter) if w8a16 else weights["wd"]

    k_cache: list[list[float]] = []
    v_cache: list[list[float]] = []
    outputs: list[list[float]] = []
    group = n_heads // n_kv_heads
    inv_sqrt_d = f32(1.0 / math.sqrt(head_dim))

    for pos, x in enumerate(seq_in):
        xn = py_rms_norm(x, weights["rms_attn"], eps)
        q = py_apply_rope(py_linear(xn, wq, qd, hidden, w8a16), n_heads, head_dim, pos, theta)
        k = py_apply_rope(py_linear(xn, wk, kvd, hidden, w8a16), n_kv_heads, head_dim, pos, theta)
        v = py_linear(xn, wv, kvd, hidden, w8a16)
        k_cache.append(k)
        v_cache.append(v)

        attn_cat = [0.0] * qd
        for qh in range(n_heads):
            kvh = qh // group
            q_vec = q[qh * head_dim : (qh + 1) * head_dim]
            scores = [
                sum(a * b for a, b in zip(q_vec, k_cache[t][kvh * head_dim : (kvh + 1) * head_dim])) * inv_sqrt_d
                for t in range(pos + 1)
            ]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            denom = sum(exps)
            for t, e in enumerate(exps):
                prob = f32(e / denom)
                v_vec = v_cache[t][kvh * head_dim : (kvh + 1) * head_dim]
                for d in range(head_dim):
                    attn_cat[qh * head_dim + d] = f32(attn_cat[qh * head_dim + d] + f32(prob * v_vec[d]))

        o_proj = py_linear(attn_cat, wo, hidden, qd, w8a16)
        h_mid = [f32(a + b) for a, b in zip(x, o_proj)]
        mlp_in = py_rms_norm(h_mid, weights["rms_mlp"], eps)
        gate = py_linear(mlp_in, wg, inter, hidden, w8a16)
        up = py_linear(mlp_in, wu, inter, hidden, w8a16)
        act = [f32((g / (1.0 + math.exp(-g))) * u) for g, u in zip(gate, up)]
        down = py_linear(act, wd, hidden, inter, w8a16)
        outputs.append([f32(a + b) for a, b in zip(h_mid, down)])
    return outputs


class TestTransformerRefCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli_bin = Path(os.environ["ARTIFACT_DIR"]) / "transformer_ref_cli"
        if not cls.cli_bin.is_file():
            raise RuntimeError(f"Compiled binary not found at {cls.cli_bin}")
        cls.tmp_root = Path(os.environ["TMPDIR"])
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run([str(self.cli_bin), *args], capture_output=True, text=True, check=False)
        if check and proc.returncode != 0:
            self.fail(f"CLI failed ({proc.returncode}): {' '.join(args)}\nstderr: {proc.stderr}")
        return proc

    def test_emb1_layout_pread_parity_and_bf16_lookup(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            rng = random.Random(101)
            vocab, hidden = 19, 24
            raw_vals = [f32(rng.uniform(-2.5, 2.5)) for _ in range(vocab * hidden)]
            raw_path, emb1_path, stats_path = tdir / "emb_f32.bin", tdir / "emb.emb1", tdir / "stats.json"
            raw_path.write_bytes(pack_f32_list(raw_vals))

            self._run("pack-emb", "--input", str(raw_path), "--vocab", str(vocab),
                      "--hidden", str(hidden), "--output", str(emb1_path), "--stats-json", str(stats_path))

            blob = emb1_path.read_bytes()
            self.assertEqual(len(blob), 32 + vocab * (4 + hidden))
            magic, ver, v_hdr, h_hdr, stride, flags, r0, r1 = struct.unpack("<4s7I", blob[:32])
            self.assertEqual(magic, b"EMB1")
            self.assertEqual((ver, v_hdr, h_hdr, stride, flags, r0, r1), (1, vocab, hidden, 4 + hidden, 1, 0, 0))
            self.assertGreater(json.loads(stats_path.read_text(encoding="utf-8"))["cos_min"], 0.999)

            out_json = tdir / "lookup.json"
            query_ids = [18, 0, 7, 12]
            self._run("lookup-emb", "--emb", str(emb1_path),
                      "--tokens", ",".join(str(i) for i in query_ids), "--output", str(out_json))
            res = json.loads(out_json.read_text(encoding="utf-8"))
            for idx, tok in enumerate(query_ids):
                codes, scale = py_quantize_row_i8(raw_vals[tok * hidden : (tok + 1) * hidden])
                expected = [f32(float(c) * scale) for c in codes]
                for got_v, exp_v in zip(res["rows"][idx], expected):
                    self.assertAlmostEqual(got_v, exp_v, places=6)

            bf16_u16, expected_bf16 = [], []
            for v in raw_vals[: 4 * hidden]:
                top16 = (struct.unpack("<I", struct.pack("<f", v))[0] >> 16) & 0xFFFF
                bf16_u16.append(top16)
                expected_bf16.append(struct.unpack("<f", struct.pack("<I", top16 << 16))[0])
            bf16_path, bf16_out = tdir / "raw_bf16.bin", tdir / "bf16_lookup.json"
            bf16_path.write_bytes(struct.pack(f"<{len(bf16_u16)}H", *bf16_u16))
            self._run("lookup-emb", "--emb", str(bf16_path), "--format", "bf16",
                      "--hidden", str(hidden), "--tokens", "3,1", "--output", str(bf16_out))
            bf16_res = json.loads(bf16_out.read_text(encoding="utf-8"))
            self.assertEqual(bf16_res["rows"][0], expected_bf16[3 * hidden : 4 * hidden])
            self.assertEqual(bf16_res["rows"][1], expected_bf16[1 * hidden : 2 * hidden])

    def test_lmh1_multithreaded_bit_exact_and_sampler_invariants(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            rng = random.Random(202)
            vocab, hidden = 97, 32
            w_vals = [f32(rng.uniform(-1.2, 1.2)) for _ in range(vocab * hidden)]
            norm_vals = [f32(rng.uniform(0.7, 1.3)) for _ in range(hidden)]
            x_vals = [f32(rng.uniform(-0.8, 0.8)) for _ in range(hidden)]

            w_path, norm_path, x_path, lmh_path = tdir / "w.bin", tdir / "n.bin", tdir / "x.bin", tdir / "h.lmh1"
            w_path.write_bytes(pack_f32_list(w_vals))
            norm_path.write_bytes(pack_f32_list(norm_vals))
            x_path.write_bytes(pack_f32_list(x_vals))
            self._run("pack-lmhead", "--input", str(w_path), "--norm-weight", str(norm_path),
                      "--vocab", str(vocab), "--hidden", str(hidden), "--output", str(lmh_path))

            out_t1, out_t4 = tdir / "run_t1.json", tdir / "run_t4.json"
            common = ["--lmhead", str(lmh_path), "--hidden-in", str(x_path),
                      "--top-k", "12", "--top-p", "0.75", "--temp", "0.8", "--seed", "777", "--draws", "16"]
            self._run("run-lmhead", *common, "--threads", "1", "--output", str(out_t1))
            self._run("run-lmhead", *common, "--threads", "4", "--output", str(out_t4))

            r1 = json.loads(out_t1.read_text(encoding="utf-8"))
            r4 = json.loads(out_t4.read_text(encoding="utf-8"))
            self.assertEqual((r1["argmax"], r1["logits"], r1["samples"]), (r4["argmax"], r4["logits"], r4["samples"]))

            x_norm = py_rms_norm(x_vals, norm_vals, 1e-6)
            ref_logits = []
            for r in range(vocab):
                codes, scale = py_quantize_row_i8(w_vals[r * hidden : (r + 1) * hidden])
                ref_logits.append(f32(sum(c * xv for c, xv in zip(codes, x_norm)) * scale))
            for got_l, exp_l in zip(r1["logits"], ref_logits):
                self.assertAlmostEqual(got_l, exp_l, places=5)
            self.assertEqual(r1["argmax"], max(range(vocab), key=lambda i: ref_logits[i]))

            dist = r1["distribution"]
            self.assertTrue(1 <= len(dist) <= 12)
            probs = [item["prob"] for item in dist]
            self.assertAlmostEqual(sum(probs), 1.0, places=9)
            self.assertEqual(probs, sorted(probs, reverse=True))
            self.assertTrue(all(tok in {item["token_id"] for item in dist} for tok in r1["samples"]))

    def test_gqa_swiglu_block_parity_and_non_destructive_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            rng = random.Random(303)
            hidden, n_heads, n_kv_heads, head_dim, inter, seq_len = 16, 4, 2, 4, 32, 5
            qd, kvd = n_heads * head_dim, n_kv_heads * head_dim
            weights = {
                "rms_attn": [f32(rng.uniform(0.8, 1.2)) for _ in range(hidden)],
                "rms_mlp": [f32(rng.uniform(0.8, 1.2)) for _ in range(hidden)],
                "wq": [f32(rng.uniform(-0.3, 0.3)) for _ in range(qd * hidden)],
                "wk": [f32(rng.uniform(-0.3, 0.3)) for _ in range(kvd * hidden)],
                "wv": [f32(rng.uniform(-0.3, 0.3)) for _ in range(kvd * hidden)],
                "wo": [f32(rng.uniform(-0.3, 0.3)) for _ in range(hidden * qd)],
                "wg": [f32(rng.uniform(-0.3, 0.3)) for _ in range(inter * hidden)],
                "wu": [f32(rng.uniform(-0.3, 0.3)) for _ in range(inter * hidden)],
                "wd": [f32(rng.uniform(-0.3, 0.3)) for _ in range(hidden * inter)],
            }
            flat_w = (
                weights["rms_attn"] + weights["rms_mlp"] + weights["wq"] + weights["wk"]
                + weights["wv"] + weights["wo"] + weights["wg"] + weights["wu"] + weights["wd"]
            )
            seq_in = [[f32(rng.uniform(-1.0, 1.0)) for _ in range(hidden)] for _ in range(seq_len)]
            w_path, x_path = tdir / "blk_w.bin", tdir / "blk_x.bin"
            w_path.write_bytes(pack_f32_list(flat_w))
            x_path.write_bytes(pack_f32_list([v for row in seq_in for v in row]))

            for w8a16 in (False, True):
                out_path = tdir / f"verify_{int(w8a16)}.json"
                args = ["verify-block", "--weights", str(w_path), "--input", str(x_path),
                        "--hidden", str(hidden), "--heads", str(n_heads), "--kv-heads", str(n_kv_heads),
                        "--head-dim", str(head_dim), "--intermediate", str(inter),
                        "--seq-len", str(seq_len), "--output", str(out_path)]
                if w8a16:
                    args.append("--w8a16")
                self._run(*args)
                got = json.loads(out_path.read_text(encoding="utf-8"))
                ref = py_block_sequence_ref(
                    seq_in, weights, hidden, n_heads, n_kv_heads, head_dim, inter, 1e-6, 10000.0, w8a16
                )
                for t in range(seq_len):
                    for g_v, r_v in zip(got["outputs"][t], ref[t]):
                        self.assertAlmostEqual(g_v, r_v, places=5)

            existing = out_path.read_bytes()
            dup = self._run("verify-block", "--output", str(out_path), "--seq-len", "2", check=False)
            self.assertNotEqual(dup.returncode, 0)
            self.assertEqual(out_path.read_bytes(), existing)

    def test_corrupt_header_and_truncated_file_rejection(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(self.tmp_root)) as td:
            tdir = Path(td)
            good_emb = tdir / "good.emb1"
            self._run("pack-emb", "--synthetic", "--vocab", "8", "--hidden", "16", "--output", str(good_emb))

            bad_magic = tdir / "bad_magic.emb1"
            raw = bytearray(good_emb.read_bytes())
            raw[:4] = b"NOPE"
            bad_magic.write_bytes(bytes(raw))
            self.assertNotEqual(self._run("lookup-emb", "--emb", str(bad_magic), "--tokens", "0", check=False).returncode, 0)

            trunc_emb = tdir / "trunc.emb1"
            trunc_emb.write_bytes(good_emb.read_bytes()[:-5])
            self.assertNotEqual(self._run("lookup-emb", "--emb", str(trunc_emb), "--tokens", "0", check=False).returncode, 0)

            good_lmh = tdir / "good.lmh1"
            self._run("pack-lmhead", "--synthetic", "--vocab", "16", "--hidden", "16", "--output", str(good_lmh))
            trunc_lmh = tdir / "trunc.lmh1"
            trunc_lmh.write_bytes(good_lmh.read_bytes()[:80])
            self.assertNotEqual(
                self._run("run-lmhead", "--lmhead", str(trunc_lmh), "--synthetic-input", check=False).returncode, 0
            )


if __name__ == "__main__":
    unittest.main()
