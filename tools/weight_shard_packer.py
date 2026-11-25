#!/usr/bin/env python3
"""Pure-stdlib SafeTensors reader, EMB1/LMH1 int8 exporters, and aligned region shard packer."""
from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

EMB1_MAGIC = b"EMB1"
EMB1_HEADER_BYTES = 32
EMB1_VERSION = 1
EMB1_FLAG_ROW_SYM_I8 = 1

LMH1_MAGIC = b"LMH1"
LMH1_HEADER_BYTES = 64
LMH1_VERSION = 1
LMH1_FLAG_HAS_NORM = 1

RSH1_MAGIC = b"RSH1"
RSH1_HEADER_BYTES = 32
RSH1_VERSION = 1

MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1}
LAYER_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

_F16_LUT: list[float] | None = None
_BF16_LUT: list[float] | None = None


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


def f32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ieee754_f16_bits_to_f32(u16: int) -> float:
    """Exact IEEE-754 binary16 -> binary32 conversion without external libraries."""
    sign = (u16 >> 15) & 0x1
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x03FF
    if exp == 0:
        if mant == 0:
            bits32 = sign << 31
        else:
            exp32 = 127 - 14
            while (mant & 0x0400) == 0:
                mant <<= 1
                exp32 -= 1
            bits32 = (sign << 31) | (exp32 << 23) | ((mant & 0x03FF) << 13)
    elif exp == 31:
        bits32 = (sign << 31) | (0xFF << 23) | (mant << 13)
    else:
        bits32 = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", bits32))[0]


def _get_f16_lut() -> list[float]:
    global _F16_LUT
    if _F16_LUT is None:
        _F16_LUT = [ieee754_f16_bits_to_f32(i) for i in range(65536)]
    return _F16_LUT


def _get_bf16_lut() -> list[float]:
    global _BF16_LUT
    if _BF16_LUT is None:
        raw = b"".join(struct.pack("<I", i << 16) for i in range(65536))
        _BF16_LUT = list(struct.unpack("<65536f", raw))
    return _BF16_LUT


def ensure_non_destructive(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing path without --overwrite: {path}")


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard_path: Path
    data_begin: int
    data_end: int

    @property
    def numel(self) -> int:
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod


def write_safetensors_file(
    path: Path,
    tensors: dict[str, tuple[str, list[int] | tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Write a SafeTensors binary file from `(dtype, shape, raw_le_bytes)` entries."""
    ensure_non_destructive(path, overwrite)
    header_obj: dict[str, Any] = {"__metadata__": dict(metadata)} if metadata is not None else {}
    offset = 0
    payload_parts: list[bytes] = []
    for name, (dtype, shape, raw_bytes) in tensors.items():
        if dtype not in DTYPE_ITEMSIZE:
            raise ValueError(f"Unsupported SafeTensors dtype: {dtype}")
        numel = 1
        for dim in shape:
            if dim < 0:
                raise ValueError(f"Negative dimension in tensor {name}: {shape}")
            numel *= int(dim)
        expected_bytes = numel * DTYPE_ITEMSIZE[dtype]
        if len(raw_bytes) != expected_bytes:
            raise ValueError(f"Byte length mismatch for {name}: {len(raw_bytes)} != {expected_bytes}")
        header_obj[name] = {
            "dtype": dtype,
            "shape": [int(d) for d in shape],
            "data_offsets": [offset, offset + expected_bytes],
        }
        payload_parts.append(raw_bytes)
        offset += expected_bytes

    hdr_json = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    pad = (-len(hdr_json)) % 8
    if pad:
        hdr_json += b" " * pad

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(struct.pack("<Q", len(hdr_json)))
        fp.write(hdr_json)
        for part in payload_parts:
            fp.write(part)


class SafeTensorArchive:
    """Pure-stdlib reader for single-file or sharded SafeTensors checkpoints."""

    def __init__(self, model_path: str | Path) -> None:
        root = Path(model_path)
        if not root.exists():
            raise FileNotFoundError(f"Model path does not exist: {root}")

        shard_files: list[Path] = []
        expected_map: dict[str, str] | None = None
        if root.is_file():
            shard_files.append(root)
        else:
            index_path = root / "model.safetensors.index.json"
            single_path = root / "model.safetensors"
            if index_path.is_file():
                idx_data = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = idx_data.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"Invalid weight_map in {index_path}")
                expected_map = {str(k): str(v) for k, v in weight_map.items()}
                seen_rel: list[str] = []
                for rel in expected_map.values():
                    if rel not in seen_rel:
                        seen_rel.append(rel)
                for rel in seen_rel:
                    sp = (root / rel).resolve()
                    if not sp.is_file():
                        raise FileNotFoundError(f"Missing SafeTensors shard referenced by index: {sp}")
                    shard_files.append(sp)
            elif single_path.is_file():
                shard_files.append(single_path)
            else:
                raise FileNotFoundError(f"No model.safetensors or index found in {root}")

        self._entries: dict[str, TensorEntry] = {}
        for sp in shard_files:
            self._load_shard_header(sp, root, expected_map)

        if expected_map is not None:
            missing = [name for name in expected_map if name not in self._entries]
            if missing:
                raise ValueError(f"SafeTensors index references missing tensors: {missing[:5]}")

    def _load_shard_header(
        self, shard_path: Path, root_dir: Path, expected_map: dict[str, str] | None
    ) -> None:
        file_size = shard_path.stat().st_size
        if file_size < 8:
            raise ValueError(f"Truncated SafeTensors file (<8 bytes): {shard_path}")
        with shard_path.open("rb") as fp:
            hdr_len = struct.unpack("<Q", fp.read(8))[0]
            if hdr_len == 0 or hdr_len > MAX_SAFETENSORS_HEADER_BYTES or 8 + hdr_len > file_size:
                raise ValueError(f"Invalid SafeTensors header length {hdr_len} in {shard_path}")
            raw_hdr = fp.read(hdr_len)

        try:
            parsed = json.loads(raw_hdr.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Malformed SafeTensors JSON header in {shard_path}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"SafeTensors header root must be an object in {shard_path}")

        data_start = 8 + hdr_len
        data_limit = file_size - data_start
        for name, meta in parsed.items():
            if name == "__metadata__":
                continue
            if expected_map is not None:
                mapped_rel = expected_map.get(name)
                if mapped_rel is None or (root_dir / mapped_rel).resolve() != shard_path.resolve():
                    continue
            if not isinstance(meta, dict):
                raise ValueError(f"Invalid metadata entry for tensor {name}")
            dtype = str(meta.get("dtype", ""))
            if dtype not in DTYPE_ITEMSIZE:
                raise ValueError(f"Unsupported tensor dtype '{dtype}' for {name}")
            raw_shape, offsets = meta.get("shape"), meta.get("data_offsets")
            if not isinstance(raw_shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"Invalid shape or data_offsets for {name}")
            shape = tuple(int(d) for d in raw_shape)
            numel = 1
            for dim in shape:
                if dim < 0:
                    raise ValueError(f"Negative dimension in tensor {name}: {shape}")
                numel *= dim
            begin, end = int(offsets[0]), int(offsets[1])
            if begin < 0 or end < begin or end > data_limit or (end - begin) != numel * DTYPE_ITEMSIZE[dtype]:
                raise ValueError(f"Out-of-bounds or mismatched data_offsets for {name}")
            self._entries[name] = TensorEntry(
                name=name, dtype=dtype, shape=shape, shard_path=shard_path,
                data_begin=data_start + begin, data_end=data_start + end,
            )

    def tensor_names(self) -> list[str]:
        return sorted(self._entries.keys())

    def has_tensor(self, name: str) -> bool:
        return name in self._entries

    def entry(self, name: str) -> TensorEntry:
        if name not in self._entries:
            raise KeyError(f"Tensor not found in archive: {name}")
        return self._entries[name]

    def read_raw_bytes(self, name: str) -> tuple[TensorEntry, bytes]:
        ent = self.entry(name)
        nbytes = ent.data_end - ent.data_begin
        with ent.shard_path.open("rb") as fp:
            fp.seek(ent.data_begin)
            raw = fp.read(nbytes)
        if len(raw) != nbytes:
            raise ValueError(f"Truncated read for tensor {name}")
        return ent, raw

    def read_f32(self, name: str) -> tuple[tuple[int, ...], list[float]]:
        ent, raw = self.read_raw_bytes(name)
        if ent.dtype == "F32":
            arr = array("f")
            arr.frombytes(raw)
            if sys.byteorder != "little":
                arr.byteswap()
            return ent.shape, arr.tolist()
        if ent.dtype in ("F16", "BF16"):
            u16 = array("H")
            u16.frombytes(raw)
            if sys.byteorder != "little":
                u16.byteswap()
            lut = _get_f16_lut() if ent.dtype == "F16" else _get_bf16_lut()
            return ent.shape, [lut[v] for v in u16]
        if ent.dtype == "I8":
            i8 = array("b")
            i8.frombytes(raw)
            return ent.shape, [float(v) for v in i8]
        raise ValueError(f"Unsupported dtype {ent.dtype}")

    def read_i8(self, name: str) -> tuple[tuple[int, ...], list[int]]:
        ent, raw = self.read_raw_bytes(name)
        if ent.dtype != "I8":
            raise ValueError(f"Tensor {name} has dtype {ent.dtype}, expected I8")
        i8 = array("b")
        i8.frombytes(raw)
        return ent.shape, i8.tolist()


def quantize_matrix_sym_i8(
    values: list[float], rows: int, cols: int
) -> tuple[bytes, list[float], dict[str, float]]:
    """Quantize `[rows, cols]` float32 matrix to per-row symmetric int8 + exact audit stats."""
    if len(values) != rows * cols or rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid matrix dimensions [{rows}, {cols}] for len={len(values)}")

    all_codes = array("b")
    scales: list[float] = []
    cos_sum, cos_min, max_abs_err, valid_rows = 0.0, 1.0, 0.0, 0

    for r in range(rows):
        base = r * cols
        row = [f32_round(values[base + c]) for c in range(cols)]
        scale = f32_round(max(max(abs(v) for v in row) / 127.0, 1e-12))
        scales.append(scale)

        dot, norm_orig, norm_deq = 0.0, 0.0, 0.0
        for orig_v in row:
            q = max(-127, min(127, int(round(f32_round(orig_v / scale)))))
            all_codes.append(q)
            deq_v = f32_round(float(q) * scale)
            err = abs(orig_v - deq_v)
            if err > max_abs_err:
                max_abs_err = err
            dot += orig_v * deq_v
            norm_orig += orig_v * orig_v
            norm_deq += deq_v * deq_v

        denom = math.sqrt(norm_orig) * math.sqrt(norm_deq)
        if denom > 0.0:
            cos_val = max(-1.0, min(1.0, dot / denom))
            if valid_rows == 0 or cos_val < cos_min:
                cos_min = cos_val
            cos_sum += cos_val
            valid_rows += 1

    return all_codes.tobytes(), scales, {
        "cos_min": cos_min if valid_rows > 0 else 1.0,
        "cos_mean": (cos_sum / valid_rows) if valid_rows > 0 else 1.0,
        "max_abs_err": max_abs_err,
    }


def export_emb1(
    archive: SafeTensorArchive, tensor_name: str, output_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    out = Path(output_path)
    ensure_non_destructive(out, overwrite)
    shape, values = archive.read_f32(tensor_name)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Embedding tensor {tensor_name} must be 2D, got {shape}")
    vocab_size, hidden_size = shape
    codes_bytes, scales, audit = quantize_matrix_sym_i8(values, vocab_size, hidden_size)
    row_stride = 4 + hidden_size
    header = struct.pack(
        "<4s7I", EMB1_MAGIC, EMB1_VERSION, vocab_size, hidden_size, row_stride, EMB1_FLAG_ROW_SYM_I8, 0, 0
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with out.open("wb") as fp:
        fp.write(header)
        hasher.update(header)
        for r in range(vocab_size):
            row_blob = struct.pack("<f", scales[r]) + codes_bytes[r * hidden_size : (r + 1) * hidden_size]
            fp.write(row_blob)
            hasher.update(row_blob)

    return {
        "magic": "EMB1", "source_tensor": tensor_name, "vocab_size": vocab_size,
        "hidden_size": hidden_size, "row_stride": row_stride,
        "file_bytes": EMB1_HEADER_BYTES + vocab_size * row_stride,
        "sha256": hasher.hexdigest(), **audit,
    }


def export_lmh1(
    archive: SafeTensorArchive, output_path: str | Path, *,
    lm_head_tensor: str = "lm_head.weight", fallback_emb_tensor: str = "model.embed_tokens.weight",
    allow_tied_fallback: bool = True, norm_tensor: str | None = None,
    rms_eps: float = 1e-6, overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(output_path)
    ensure_non_destructive(out, overwrite)
    untied_from_emb = False
    if archive.has_tensor(lm_head_tensor):
        chosen_tensor = lm_head_tensor
    elif allow_tied_fallback and archive.has_tensor(fallback_emb_tensor):
        chosen_tensor = fallback_emb_tensor
        untied_from_emb = True
    else:
        raise KeyError(f"Neither {lm_head_tensor} nor fallback {fallback_emb_tensor} found")

    shape, values = archive.read_f32(chosen_tensor)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"LM-head tensor {chosen_tensor} must be 2D, got {shape}")
    vocab_size, hidden_size = shape
    codes_bytes, scales, audit = quantize_matrix_sym_i8(values, vocab_size, hidden_size)

    norm_vals: list[float] | None = None
    if norm_tensor is not None:
        n_shape, n_data = archive.read_f32(norm_tensor)
        if len(n_shape) != 1 or n_shape[0] != hidden_size:
            raise ValueError(f"Norm tensor {norm_tensor} shape {n_shape} != [{hidden_size}]")
        norm_vals = [f32_round(v) for v in n_data]

    scale_off = LMH1_HEADER_BYTES
    scale_bytes = struct.pack(f"<{vocab_size}f", *scales)
    codes_off = align_up(scale_off + len(scale_bytes), 64)
    norm_off = align_up(codes_off + len(codes_bytes), 64) if norm_vals is not None else 0
    flags = LMH1_FLAG_HAS_NORM if norm_vals is not None else 0
    header = struct.pack(
        "<4s15I", LMH1_MAGIC, LMH1_VERSION, vocab_size, hidden_size, flags,
        scale_off, codes_off, norm_off, f32_to_bits(rms_eps), 0, 0, 0, 0, 0, 0, 0,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with out.open("wb") as fp:
        pad1 = b"\x00" * (codes_off - (scale_off + len(scale_bytes)))
        for chunk in (header, scale_bytes, pad1, codes_bytes):
            fp.write(chunk)
            hasher.update(chunk)
        if norm_vals is not None:
            pad2 = b"\x00" * (norm_off - (codes_off + len(codes_bytes)))
            norm_bytes = struct.pack(f"<{hidden_size}f", *norm_vals)
            for chunk in (pad2, norm_bytes):
                fp.write(chunk)
                hasher.update(chunk)

    return {
        "magic": "LMH1", "source_tensor": chosen_tensor, "untied_from_embeddings": untied_from_emb,
        "has_norm_weight": norm_vals is not None, "vocab_size": vocab_size, "hidden_size": hidden_size,
        "scale_offset": scale_off, "codes_offset": codes_off, "norm_offset": norm_off,
        "file_bytes": out.stat().st_size, "sha256": hasher.hexdigest(), **audit,
    }


def _parse_layer_projection(tensor_name: str) -> tuple[int, str] | None:
    parts = tensor_name.split(".")
    if not parts or parts[-1] != "weight":
        return None
    layer_idx: int | None = None
    for i, token in enumerate(parts[:-1]):
        if token == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            break
    if layer_idx is None or parts[-2] not in LAYER_PROJECTIONS:
        return None
    return layer_idx, parts[-2]


def pack_region_shards(
    archive: SafeTensorArchive, output_dir: str | Path, *,
    align: int = 64, max_shard_bytes: int = 64 * 1024 * 1024,
    storage_format: str = "w8", overwrite: bool = False,
) -> dict[str, Any]:
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    header_span = align_up(RSH1_HEADER_BYTES, align)
    if max_shard_bytes <= header_span:
        raise ValueError(f"max_shard_bytes ({max_shard_bytes}) too small for shard header")
    if storage_format not in ("w8", "f32"):
        raise ValueError(f"Unsupported storage_format '{storage_format}'")

    out_dir = Path(output_dir)
    manifest_path = out_dir / "shard_manifest.json"
    ensure_non_destructive(manifest_path, overwrite)

    proj_rank = {name: idx for idx, name in enumerate(LAYER_PROJECTIONS)}
    discovered: list[tuple[int, int, str, str]] = []
    for tname in archive.tensor_names():
        parsed = _parse_layer_projection(tname)
        if parsed is not None:
            discovered.append((parsed[0], proj_rank[parsed[1]], parsed[1], tname))
    if not discovered:
        raise ValueError("No layer projection weights found in archive")
    discovered.sort()
    out_dir.mkdir(parents=True, exist_ok=True)

    shards_meta: list[dict[str, Any]] = []
    tensors_meta: list[dict[str, Any]] = []
    cur_buf = bytearray()
    cur_shard_idx, cur_tensor_count = 0, 0

    def flush_current_shard() -> None:
        nonlocal cur_buf, cur_shard_idx, cur_tensor_count
        if cur_tensor_count == 0:
            return
        total_len = len(cur_buf)
        cur_buf[0:RSH1_HEADER_BYTES] = struct.pack(
            "<4s7I", RSH1_MAGIC, RSH1_VERSION, cur_shard_idx, cur_tensor_count, align, total_len, 0, 0
        )
        shard_name = f"region_shard_{cur_shard_idx:04d}.bin"
        shard_path = out_dir / shard_name
        ensure_non_destructive(shard_path, overwrite)
        shard_bytes = bytes(cur_buf)
        shard_path.write_bytes(shard_bytes)
        shards_meta.append({
            "shard_index": cur_shard_idx, "file": shard_name, "byte_length": total_len,
            "tensor_count": cur_tensor_count, "sha256": hashlib.sha256(shard_bytes).hexdigest(),
        })
        cur_shard_idx += 1
        cur_tensor_count = 0
        cur_buf = bytearray()

    for layer_idx, _rank, proj, tname in discovered:
        shape, vals = archive.read_f32(tname)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Projection {tname} must be 2D, got {shape}")
        rows, cols = shape
        if storage_format == "w8":
            codes_b, scales, audit = quantize_matrix_sym_i8(vals, rows, cols)
            scales_b = struct.pack(f"<{rows}f", *scales)
            blob, scales_len, codes_len, max_err = scales_b + codes_b, len(scales_b), len(codes_b), audit["max_abs_err"]
        else:
            blob = struct.pack(f"<{rows * cols}f", *(f32_round(v) for v in vals))
            scales_len, codes_len, max_err = 0, 0, 0.0

        if header_span + len(blob) > max_shard_bytes:
            raise ValueError(f"Single tensor {tname} exceeds max_shard_bytes={max_shard_bytes}")
        if cur_tensor_count == 0:
            cur_buf = bytearray(b"\x00" * header_span)
        aligned_off = align_up(len(cur_buf), align)
        if cur_tensor_count > 0 and (aligned_off + len(blob)) > max_shard_bytes:
            flush_current_shard()
            cur_buf = bytearray(b"\x00" * header_span)
            aligned_off = header_span
        if aligned_off > len(cur_buf):
            cur_buf.extend(b"\x00" * (aligned_off - len(cur_buf)))
        cur_buf.extend(blob)
        cur_tensor_count += 1

        tensors_meta.append({
            "name": tname, "layer": layer_idx, "proj": proj, "shape": [rows, cols],
            "storage_format": storage_format, "shard_index": cur_shard_idx,
            "shard_file": f"region_shard_{cur_shard_idx:04d}.bin", "offset": aligned_off,
            "byte_length": len(blob), "scales_byte_length": scales_len,
            "codes_byte_length": codes_len, "max_abs_err": max_err,
            "sha256": hashlib.sha256(blob).hexdigest(),
        })

    flush_current_shard()
    manifest = {
        "format": "RSH1_MANIFEST", "version": 1, "align": align,
        "max_shard_bytes": max_shard_bytes, "storage_format": storage_format,
        "num_layers": len({int(t["layer"]) for t in tensors_meta}),
        "total_shards": len(shards_meta), "total_tensors": len(tensors_meta),
        "shards": shards_meta, "tensors": tensors_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit_shard_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify binary shard headers, alignment, size caps, padding, and per-tensor SHA-256 digests."""
    mpath = Path(manifest_path)
    if not mpath.is_file():
        return {"ok": False, "checked_shards": 0, "checked_tensors": 0, "errors": [f"Missing manifest: {mpath}"]}
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "checked_shards": 0, "checked_tensors": 0, "errors": [f"Invalid JSON: {exc}"]}

    errors: list[str] = []
    if manifest.get("format") != "RSH1_MANIFEST" or manifest.get("version") != 1:
        errors.append("Unsupported manifest format or version")
    align = int(manifest.get("align", 0))
    max_shard_bytes = int(manifest.get("max_shard_bytes", 0))
    if align <= 0 or max_shard_bytes <= 0:
        errors.append(f"Invalid align ({align}) or max_shard_bytes ({max_shard_bytes})")
        return {"ok": False, "checked_shards": 0, "checked_tensors": 0, "errors": errors}

    base_dir = mpath.parent
    header_span = align_up(RSH1_HEADER_BYTES, align)
    shards, tensors = manifest.get("shards", []), manifest.get("tensors", [])
    shard_blobs: dict[int, bytes] = {}
    expected_counts: dict[int, int] = {}

    for s_entry in shards:
        s_idx = int(s_entry["shard_index"])
        s_file = base_dir / str(s_entry["file"])
        if not s_file.is_file():
            errors.append(f"Missing shard file: {s_file}")
            continue
        raw = s_file.read_bytes()
        shard_blobs[s_idx] = raw
        expected_counts[s_idx] = 0
        if len(raw) != int(s_entry["byte_length"]):
            errors.append(f"Shard {s_file.name} size {len(raw)} != manifest {s_entry['byte_length']}")
        if len(raw) > max_shard_bytes:
            errors.append(f"Shard {s_file.name} size {len(raw)} > max_shard_bytes {max_shard_bytes}")
        if len(raw) < header_span:
            errors.append(f"Shard {s_file.name} smaller than aligned header ({header_span})")
            continue
        magic, ver, hdr_idx, hdr_tcnt, hdr_align, hdr_total, _r0, _r1 = struct.unpack(
            "<4s7I", raw[:RSH1_HEADER_BYTES]
        )
        if magic != RSH1_MAGIC or ver != RSH1_VERSION:
            errors.append(f"Shard {s_file.name} invalid RSH1 header magic/version")
        if hdr_idx != s_idx or hdr_tcnt != int(s_entry["tensor_count"]) or hdr_align != align or hdr_total != len(raw):
            errors.append(f"Shard {s_file.name} header metadata mismatch")
        if any(b != 0 for b in raw[RSH1_HEADER_BYTES:header_span]):
            errors.append(f"Shard {s_file.name} non-zero header alignment padding")
        if hashlib.sha256(raw).hexdigest() != s_entry.get("sha256"):
            errors.append(f"Shard {s_file.name} SHA-256 mismatch")

    spans_by_shard: dict[int, list[tuple[int, int, str]]] = {idx: [] for idx in shard_blobs}
    checked_tensors = 0
    for t_entry in tensors:
        tname, s_idx = str(t_entry["name"]), int(t_entry["shard_index"])
        off, blen = int(t_entry["offset"]), int(t_entry["byte_length"])
        checked_tensors += 1
        if off % align != 0:
            errors.append(f"Tensor {tname} offset {off} is not {align}-byte aligned")
        if s_idx not in shard_blobs:
            errors.append(f"Tensor {tname} references unavailable shard_index={s_idx}")
            continue
        expected_counts[s_idx] += 1
        raw = shard_blobs[s_idx]
        if off < header_span or blen <= 0 or off + blen > len(raw):
            errors.append(f"Tensor {tname} range [{off}, {off + blen}) out of shard bounds")
            continue
        if hashlib.sha256(raw[off : off + blen]).hexdigest() != t_entry.get("sha256"):
            errors.append(f"Tensor {tname} SHA-256 mismatch")
        spans_by_shard[s_idx].append((off, off + blen, tname))

    for s_entry in shards:
        s_idx = int(s_entry["shard_index"])
        if s_idx not in shard_blobs:
            continue
        if expected_counts[s_idx] != int(s_entry["tensor_count"]):
            errors.append(f"Shard {s_idx} tensor count mismatch")
        cursor = header_span
        raw = shard_blobs[s_idx]
        for begin, end, tname in sorted(spans_by_shard[s_idx]):
            if begin < cursor:
                errors.append(f"Overlapping tensor span at {tname} in shard {s_idx}")
            elif any(b != 0 for b in raw[cursor:begin]):
                errors.append(f"Non-zero inter-tensor padding before {tname} in shard {s_idx}")
            cursor = max(cursor, end)

    return {"ok": len(errors) == 0, "checked_shards": len(shard_blobs), "checked_tensors": checked_tensors, "errors": errors}


def _write_json_output(path_str: str | None, payload: dict[str, Any], overwrite: bool) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if path_str:
        out_p = Path(path_str)
        ensure_non_destructive(out_p, overwrite)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SafeTensors EMB1/LMH1 & region shard packer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_emb = sub.add_parser("export-emb")
    p_emb.add_argument("--model", required=True)
    p_emb.add_argument("--tensor", default="model.embed_tokens.weight")
    p_emb.add_argument("--output", required=True)
    p_emb.add_argument("--stats-output", default=None)
    p_emb.add_argument("--overwrite", action="store_true")

    p_lmh = sub.add_parser("export-lmhead")
    p_lmh.add_argument("--model", required=True)
    p_lmh.add_argument("--tensor", default="lm_head.weight")
    p_lmh.add_argument("--fallback-tensor", default="model.embed_tokens.weight")
    p_lmh.add_argument("--no-tied-fallback", action="store_true")
    p_lmh.add_argument("--norm-tensor", default=None)
    p_lmh.add_argument("--rms-eps", type=float, default=1e-6)
    p_lmh.add_argument("--output", required=True)
    p_lmh.add_argument("--stats-output", default=None)
    p_lmh.add_argument("--overwrite", action="store_true")

    p_reg = sub.add_parser("pack-regions")
    p_reg.add_argument("--model", required=True)
    p_reg.add_argument("--output", required=True)
    p_reg.add_argument("--align", type=int, default=64)
    p_reg.add_argument("--max-shard-bytes", type=int, default=64 * 1024 * 1024)
    p_reg.add_argument("--format", choices=("w8", "f32"), default="w8")
    p_reg.add_argument("--overwrite", action="store_true")

    p_aud = sub.add_parser("audit-manifest")
    p_aud.add_argument("--manifest", required=True)
    p_aud.add_argument("--output", default=None)
    p_aud.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "export-emb":
            stats = export_emb1(SafeTensorArchive(args.model), args.tensor, args.output, overwrite=args.overwrite)
            _write_json_output(args.stats_output, stats, args.overwrite)
            return 0
        if args.cmd == "export-lmhead":
            stats = export_lmh1(
                SafeTensorArchive(args.model), args.output, lm_head_tensor=args.tensor,
                fallback_emb_tensor=args.fallback_tensor, allow_tied_fallback=not args.no_tied_fallback,
                norm_tensor=args.norm_tensor, rms_eps=args.rms_eps, overwrite=args.overwrite,
            )
            _write_json_output(args.stats_output, stats, args.overwrite)
            return 0
        if args.cmd == "pack-regions":
            pack_region_shards(
                SafeTensorArchive(args.model), args.output, align=args.align,
                max_shard_bytes=args.max_shard_bytes, storage_format=args.format, overwrite=args.overwrite,
            )
            return 0
        if args.cmd == "audit-manifest":
            res = audit_shard_manifest(args.manifest)
            _write_json_output(args.output, res, args.overwrite)
            return 0 if res["ok"] else 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
