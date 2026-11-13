"""Little-endian fields and UTF-8 strings in a caller-defined frame.

CRC32 detects accidental corruption; it does not authenticate messages.
No application message types or field layouts are defined by this module.
"""

import struct
import zlib


def _crc(data):
    return zlib.crc32(data) & 0xFFFFFFFF


class FrameWriter:
    def __init__(self, magic):
        self.buf = bytearray(magic)

    def put(self, fmt, *values):
        self.buf += struct.pack("<" + fmt, *values)

    def text(self, value):
        raw = value.encode("utf-8")
        if len(raw) > 65535:
            raise ValueError("UTF-8 string exceeds the 16-bit length field")
        self.put("H", len(raw))
        self.buf += raw

    def done(self):
        return bytes(self.buf) + struct.pack("<I", _crc(self.buf))


class FrameReader:
    def __init__(self, data, magic):
        if (len(data) < len(magic) + 4 or data[:len(magic)] != magic
                or struct.unpack("<I", data[-4:])[0] != _crc(data[:-4])):
            raise ValueError("bad magic or CRC")
        self.data, self.off, self.end = data, len(magic), len(data) - 4

    def get(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.off + size > self.end:
            raise ValueError("truncated field")
        out = struct.unpack_from("<" + fmt, self.data, self.off)
        if not out:
            raise ValueError("format must produce at least one value")
        self.off += size
        return out if len(out) > 1 else out[0]

    def text(self):
        size = self.get("H")
        if self.off + size > self.end:
            raise ValueError("truncated string")
        raw = self.data[self.off:self.off + size]
        self.off += size
        return raw.decode("utf-8")

    def finish(self):
        if self.off != self.end:
            raise ValueError("trailing bytes")
