"""Symmetric INT8/INT4 weight quantization, SafeTensors header validation and RoPE tables.

Implemented using the Python standard library (`array`, `struct`, `json`, `math`)
so tensor packing, SafeTensors offset verification and rotary embedding caches
can be validated without external array dependencies.
"""

from array import array
from dataclasses import dataclass
import json
import math
import struct


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_MAX_HEADER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class QuantizedMatrix:
    bits: int
    rows: int
    cols: int
    group_size: int
    packed: bytes
    scales: tuple


def _validate_matrix_dims(rows, cols, group_size, *, require_even_cols=False):
    if type(rows) is not int or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if type(cols) is not int or cols <= 0:
        raise ValueError("cols must be a positive integer")
    if require_even_cols and cols % 2 != 0:
        raise ValueError("INT4 packing requires an even column count")
    effective_group = cols if group_size is None else group_size
    if type(effective_group) is not int or effective_group <= 0:
        raise ValueError("group_size must be a positive integer")
    if cols % effective_group != 0:
        raise ValueError("cols must be divisible by group_size")
    return effective_group


def _validate_float_buffer(values, expected_len):
    items = list(values)
    if len(items) != expected_len:
        raise ValueError(f"expected {expected_len} elements, got {len(items)}")
    out = [0.0] * expected_len
    for idx, val in enumerate(items):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"element at index {idx} is not a finite number")
        out[idx] = float(val)
    return out


def quantize_symmetric_int8(values, rows, cols, *, group_size=None):
    """Quantize a row-major `[rows, cols]` matrix to symmetric groupwise INT8."""
    gsize = _validate_matrix_dims(rows, cols, group_size)
    flat = _validate_float_buffer(values, rows * cols)
    qbuf = array("b")
    scales = []
    for start in range(0, len(flat), gsize):
        group = flat[start:start + gsize]
        max_abs = max(abs(x) for x in group)
        scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
        scales.append(scale)
        if scale == 0.0:
            qbuf.extend([0] * gsize)
        else:
            inv = 1.0 / scale
            for x in group:
                q = int(round(x * inv))
                qbuf.append(max(-127, min(127, q)))
    return QuantizedMatrix(
        bits=8,
        rows=rows,
        cols=cols,
        group_size=gsize,
        packed=qbuf.tobytes(),
        scales=tuple(scales),
    )


def dequantize_symmetric_int8(qmat):
    """Reconstruct row-major floats from an INT8 QuantizedMatrix."""
    if not isinstance(qmat, QuantizedMatrix) or qmat.bits != 8:
        raise ValueError("expected an 8-bit QuantizedMatrix")
    gsize = _validate_matrix_dims(qmat.rows, qmat.cols, qmat.group_size)
    expected_len = qmat.rows * qmat.cols
    if len(qmat.packed) != expected_len:
        raise ValueError("packed INT8 byte length does not match shape")
    expected_groups = expected_len // gsize
    if len(qmat.scales) != expected_groups:
        raise ValueError("scale count does not match group count")
    qbuf = array("b")
    qbuf.frombytes(qmat.packed)
    out = [0.0] * expected_len
    for g_idx, scale in enumerate(qmat.scales):
        base = g_idx * gsize
        for offset in range(gsize):
            out[base + offset] = qbuf[base + offset] * scale
    return out


def quantize_symmetric_int4(values, rows, cols, *, group_size=None):
    """Quantize `[rows, cols]` to symmetric signed INT4 (`[-7, 7]`), 2 nibbles/byte."""
    gsize = _validate_matrix_dims(rows, cols, group_size, require_even_cols=True)
    if gsize % 2 != 0:
        raise ValueError("INT4 group_size must be even")
    flat = _validate_float_buffer(values, rows * cols)
    packed = bytearray(len(flat) // 2)
    scales = []
    byte_idx = 0
    for start in range(0, len(flat), gsize):
        group = flat[start:start + gsize]
        max_abs = max(abs(x) for x in group)
        scale = max_abs / 7.0 if max_abs > 0.0 else 0.0
        scales.append(scale)
        inv = (1.0 / scale) if scale > 0.0 else 0.0
        for offset in range(0, gsize, 2):
            q0 = max(-7, min(7, int(round(group[offset] * inv)))) if scale > 0.0 else 0
            q1 = max(-7, min(7, int(round(group[offset + 1] * inv)))) if scale > 0.0 else 0
            packed[byte_idx] = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
            byte_idx += 1
    return QuantizedMatrix(
        bits=4,
        rows=rows,
        cols=cols,
        group_size=gsize,
        packed=bytes(packed),
        scales=tuple(scales),
    )


def _sign_extend_int4(nibble):
    return nibble - 16 if nibble >= 8 else nibble


def dequantize_symmetric_int4(qmat):
    """Unpack two's-complement INT4 nibbles and scale back to row-major floats."""
    if not isinstance(qmat, QuantizedMatrix) or qmat.bits != 4:
        raise ValueError("expected a 4-bit QuantizedMatrix")
    gsize = _validate_matrix_dims(
        qmat.rows, qmat.cols, qmat.group_size, require_even_cols=True
    )
    if gsize % 2 != 0:
        raise ValueError("INT4 group_size must be even")
    expected_len = qmat.rows * qmat.cols
    if len(qmat.packed) * 2 != expected_len:
        raise ValueError("packed INT4 byte length does not match shape")
    if len(qmat.scales) != expected_len // gsize:
        raise ValueError("scale count does not match group count")

    out = [0.0] * expected_len
    byte_idx = 0
    for g_idx, scale in enumerate(qmat.scales):
        base = g_idx * gsize
        for offset in range(0, gsize, 2):
            raw_byte = qmat.packed[byte_idx]
            byte_idx += 1
            q0 = _sign_extend_int4(raw_byte & 0x0F)
            q1 = _sign_extend_int4((raw_byte >> 4) & 0x0F)
            out[base + offset] = q0 * scale
            out[base + offset + 1] = q1 * scale
    return out


def quantization_error_metrics(reference, reconstructed):
    """Compute max absolute error, RMSE and SQNR (dB) between two float sequences."""
    ref = list(reference)
    rec = list(reconstructed)
    if not ref or len(ref) != len(rec):
        raise ValueError("reference and reconstructed must be non-empty and equal length")
    sig_sq = []
    err_sq = []
    max_abs = 0.0
    for a, b in zip(ref, rec):
        if any(isinstance(x, bool) or not math.isfinite(x) for x in (a, b)):
            raise ValueError("inputs must contain finite numbers")
        diff = abs(float(a) - float(b))
        if diff > max_abs:
            max_abs = diff
        sig_sq.append(float(a) * float(a))
        err_sq.append(diff * diff)
    mse = math.fsum(err_sq) / len(ref)
    sig_power = math.fsum(sig_sq) / len(ref)
    if mse == 0.0:
        sqnr_db = float("inf") if sig_power > 0.0 else 0.0
    elif sig_power == 0.0:
        sqnr_db = -float("inf")
    else:
        sqnr_db = 10.0 * math.log10(sig_power / mse)
    return {
        "count": len(ref),
        "max_abs_error": max_abs,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "sqnr_db": sqnr_db,
    }


def _reject_json_constant(token):
    raise ValueError(f"invalid JSON constant in SafeTensors header: {token}")


def build_safetensors_header(tensors, metadata=None):
    """Build a framed `[u64 header_len][utf8 json]` SafeTensors prefix."""
    header_obj = {}
    if metadata is not None:
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError("__metadata__ must map strings to strings")
        header_obj["__metadata__"] = dict(metadata)
    offset = 0
    for name, dtype, shape in tensors:
        if not isinstance(name, str) or not name or name == "__metadata__":
            raise ValueError(f"invalid tensor name: {name!r}")
        if name in header_obj:
            raise ValueError(f"duplicate tensor name: {name!r}")
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"unsupported SafeTensors dtype: {dtype!r}")
        dims = list(shape)
        if any(type(d) is not int or d < 0 for d in dims):
            raise ValueError(f"invalid shape for {name!r}: {shape!r}")
        elem_count = math.prod(dims) if dims else 1
        nbytes = elem_count * _SAFETENSORS_DTYPE_BYTES[dtype]
        header_obj[name] = {
            "dtype": dtype,
            "shape": dims,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    encoded = json.dumps(header_obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded, offset


def parse_safetensors_header(raw_prefix, *, total_file_size=None):
    """Parse and validate a SafeTensors binary header prefix and data offset layout."""
    if not isinstance(raw_prefix, (bytes, bytearray, memoryview)):
        raise ValueError("raw_prefix must be bytes-like")
    buf = bytes(raw_prefix)
    if len(buf) < 8:
        raise ValueError("SafeTensors prefix shorter than 8-byte length header")
    (header_len,) = struct.unpack("<Q", buf[:8])
    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"invalid SafeTensors header length: {header_len}")
    if len(buf) < 8 + header_len:
        raise ValueError("SafeTensors prefix truncated before end of JSON header")

    header_json = buf[8:8 + header_len].decode("utf-8")
    parsed = json.loads(header_json, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise ValueError("SafeTensors header root must be a JSON object")

    metadata = {}
    if "__metadata__" in parsed:
        raw_meta = parsed["__metadata__"]
        if not isinstance(raw_meta, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in raw_meta.items()
        ):
            raise ValueError("__metadata__ must be an object of string keys and values")
        metadata = dict(raw_meta)

    tensors = {}
    spans = []
    for name, entry in parsed.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"tensor {name!r} must specify dtype, shape and data_offsets")
        dtype = entry["dtype"]
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype!r} for tensor {name!r}")
        shape = entry["shape"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"invalid shape for tensor {name!r}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(x) is not int or x < 0 for x in offsets)):
            raise ValueError(f"invalid data_offsets for tensor {name!r}")
        begin, end = offsets
        if begin > end:
            raise ValueError(f"descending data_offsets for tensor {name!r}")
        elem_count = math.prod(shape) if shape else 1
        expected_bytes = elem_count * _SAFETENSORS_DTYPE_BYTES[dtype]
        if end - begin != expected_bytes:
            raise ValueError(
                f"offset span {end - begin} != shape byte size {expected_bytes} for {name!r}"
            )
        tensors[name] = {
            "dtype": dtype,
            "shape": tuple(shape),
            "data_offsets": (begin, end),
            "nbytes": expected_bytes,
        }
        spans.append((begin, end, name))

    spans.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            raise ValueError(f"hole or overlap before tensor {name!r} at offset {begin}")
        cursor = end

    data_section_start = 8 + header_len
    if total_file_size is not None:
        if type(total_file_size) is not int or total_file_size < data_section_start:
            raise ValueError("total_file_size smaller than header")
        if total_file_size - data_section_start != cursor:
            raise ValueError("SafeTensors file size does not match end of tensor payload")

    return {
        "header_bytes": header_len,
        "data_section_offset": data_section_start,
        "payload_bytes": cursor,
        "metadata": metadata,
        "tensors": tensors,
    }


def build_rope_cache(seq_len, head_dim, *, base=10000.0, scaling_factor=1.0):
    """Build `(cos_table, sin_table)` of shape `[seq_len][head_dim // 2]`."""
    if type(seq_len) is not int or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")
    if isinstance(base, bool) or not math.isfinite(base) or base <= 1.0:
        raise ValueError("base must be finite and > 1.0")
    if isinstance(scaling_factor, bool) or not math.isfinite(scaling_factor) or scaling_factor <= 0.0:
        raise ValueError("scaling_factor must be finite and positive")

    half = head_dim // 2
    inv_freq = [1.0 / (base ** ((2.0 * idx) / head_dim)) for idx in range(half)]
    cos_table = []
    sin_table = []
    for pos in range(seq_len):
        scaled_pos = pos / scaling_factor
        cos_table.append(tuple(math.cos(scaled_pos * freq) for freq in inv_freq))
        sin_table.append(tuple(math.sin(scaled_pos * freq) for freq in inv_freq))
    return tuple(cos_table), tuple(sin_table)


def apply_rope_interleaved(vector, cos_row, sin_row):
    """Apply rotary embedding to adjacent pairs `(x_{2k}, x_{2k+1})`."""
    vals = _validate_float_buffer(vector, len(vector))
    if len(vals) == 0 or len(vals) % 2 != 0:
        raise ValueError("vector length must be a positive even integer")
    half = len(vals) // 2
    if len(cos_row) != half or len(sin_row) != half:
        raise ValueError("cos/sin row length must equal len(vector) // 2")
    rotated = [0.0] * len(vals)
    for idx in range(half):
        x0 = vals[2 * idx]
        x1 = vals[2 * idx + 1]
        c = cos_row[idx]
        s = sin_row[idx]
        rotated[2 * idx] = x0 * c - x1 * s
        rotated[2 * idx + 1] = x0 * s + x1 * c
    return rotated
