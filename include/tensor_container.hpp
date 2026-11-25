#ifndef WAVE_TENSOR_CONTAINER_HPP_
#define WAVE_TENSOR_CONTAINER_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tensor_container {

inline constexpr std::array<uint8_t, 8> kMagic = {'T', 'C', 'N', 'T', 'B', 'I', 'N', '1'};
inline constexpr uint16_t kVersionMajor = 1;
inline constexpr uint16_t kVersionMinor = 0;

enum class MetaType : uint8_t { kString = 1, kInt64 = 2, kFloat64 = 3, kBool = 4 };
enum class DType : uint8_t {
  kF32 = 1, kF64 = 2, kF16 = 3, kBF16 = 4, kI64 = 5, kI32 = 6, kI16 = 7, kI8 = 8, kU8 = 9
};

inline uint64_t DTypeByteWidth(DType dt) {
  switch (dt) {
    case DType::kF64: case DType::kI64: return 8;
    case DType::kF32: case DType::kI32: return 4;
    case DType::kF16: case DType::kBF16: case DType::kI16: return 2;
    case DType::kI8: case DType::kU8: return 1;
  }
  return 0;
}

inline std::string DTypeName(DType dt) {
  switch (dt) {
    case DType::kF32: return "f32"; case DType::kF64: return "f64";
    case DType::kF16: return "f16"; case DType::kBF16: return "bf16";
    case DType::kI64: return "i64"; case DType::kI32: return "i32";
    case DType::kI16: return "i16"; case DType::kI8: return "i8";
    case DType::kU8: return "u8";
  }
  return "unknown";
}

inline std::optional<DType> ParseDType(const std::string& s) {
  if (s == "f32") return DType::kF32;
  if (s == "f64") return DType::kF64;
  if (s == "f16") return DType::kF16;
  if (s == "bf16") return DType::kBF16;
  if (s == "i64") return DType::kI64;
  if (s == "i32") return DType::kI32;
  if (s == "i16") return DType::kI16;
  if (s == "i8") return DType::kI8;
  if (s == "u8") return DType::kU8;
  return std::nullopt;
}

struct MetadataEntry {
  std::string key;
  MetaType type = MetaType::kString;
  std::variant<std::string, int64_t, double, bool> value;
};

struct TensorRecord {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<uint64_t> dims;
  uint64_t alignment = 64, payload_offset = 0, payload_size = 0;
  uint32_t crc32 = 0;
};

struct TensorInput {
  std::string name;
  DType dtype = DType::kF32;
  std::vector<uint64_t> dims;
  uint64_t alignment = 64;
  std::vector<uint8_t> data;
};

struct AuditFinding { std::string code, message; };

struct AuditReport {
  bool valid = false;
  uint64_t file_bytes = 0, payload_section_offset = 0;
  std::vector<MetadataEntry> metadata;
  std::vector<TensorRecord> tensors;
  std::vector<AuditFinding> findings;
};

inline uint32_t ComputeCrc32(const uint8_t* data, size_t len) noexcept {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint32_t>(data[i]);
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
  }
  return ~crc;
}

inline constexpr bool IsPowerOfTwo(uint64_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

inline bool CheckedShapeBytes(DType dt, const std::vector<uint64_t>& dims,
                              uint64_t* out_bytes) noexcept {
  const uint64_t width = DTypeByteWidth(dt);
  if (width == 0) return false;
  uint64_t elems = 1;
  for (uint64_t d : dims) {
    if (d == 0) { elems = 0; break; }
    if (elems > std::numeric_limits<uint64_t>::max() / d) return false;
    elems *= d;
  }
  if (elems > 0 && elems > std::numeric_limits<uint64_t>::max() / width) return false;
  *out_bytes = elems * width;
  return true;
}

inline bool AlignUp(uint64_t val, uint64_t align, uint64_t* out) noexcept {
  if (!IsPowerOfTwo(align)) return false;
  const uint64_t mask = align - 1;
  if (val > std::numeric_limits<uint64_t>::max() - mask) return false;
  *out = (val + mask) & ~mask;
  return true;
}

namespace detail {

inline void AppendU8(std::vector<uint8_t>* b, uint8_t v) { b->push_back(v); }
inline void AppendU16LE(std::vector<uint8_t>* b, uint16_t v) {
  b->push_back(static_cast<uint8_t>(v & 0xFFu));
  b->push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}
inline void AppendU32LE(std::vector<uint8_t>* b, uint32_t v) {
  for (int s = 0; s < 32; s += 8) b->push_back(static_cast<uint8_t>((v >> s) & 0xFFu));
}
inline void AppendU64LE(std::vector<uint8_t>* b, uint64_t v) {
  for (int s = 0; s < 64; s += 8) b->push_back(static_cast<uint8_t>((v >> s) & 0xFFu));
}
inline void WriteU64LEAt(std::vector<uint8_t>* b, size_t pos, uint64_t v) {
  for (size_t i = 0; i < 8; ++i) (*b)[pos + i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFFu);
}
inline void WriteU32LEAt(std::vector<uint8_t>* b, size_t pos, uint32_t v) {
  for (size_t i = 0; i < 4; ++i) (*b)[pos + i] = static_cast<uint8_t>((v >> (i * 8)) & 0xFFu);
}

class ByteReader {
 public:
  ByteReader(const uint8_t* d, size_t sz) : d_(d), sz_(sz) {}
  size_t Pos() const noexcept { return pos_; }
  size_t Remaining() const noexcept { return sz_ - pos_; }
  bool ReadU8(uint8_t* o) noexcept {
    if (Remaining() < 1) return false;
    *o = d_[pos_++]; return true;
  }
  bool ReadU16(uint16_t* o) noexcept {
    if (Remaining() < 2) return false;
    *o = static_cast<uint16_t>(d_[pos_]) | (static_cast<uint16_t>(d_[pos_ + 1]) << 8);
    pos_ += 2; return true;
  }
  bool ReadU32(uint32_t* o) noexcept {
    if (Remaining() < 4) return false;
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) v |= static_cast<uint32_t>(d_[pos_ + i]) << (i * 8);
    pos_ += 4; *o = v; return true;
  }
  bool ReadU64(uint64_t* o) noexcept {
    if (Remaining() < 8) return false;
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) v |= static_cast<uint64_t>(d_[pos_ + i]) << (i * 8);
    pos_ += 8; *o = v; return true;
  }
  bool ReadString(size_t n, std::string* o) {
    if (Remaining() < n) return false;
    o->assign(reinterpret_cast<const char*>(d_ + pos_), n);
    pos_ += n; return true;
  }

 private:
  const uint8_t* d_ = nullptr;
  size_t sz_ = 0, pos_ = 0;
};

}  // namespace detail

inline std::vector<uint8_t> BuildArchive(const std::vector<MetadataEntry>& metadata,
                                         const std::vector<TensorInput>& tensors) {
  std::vector<uint8_t> out(kMagic.begin(), kMagic.end());
  detail::AppendU16LE(&out, kVersionMajor);
  detail::AppendU16LE(&out, kVersionMinor);
  detail::AppendU32LE(&out, 0u);
  detail::AppendU32LE(&out, static_cast<uint32_t>(metadata.size()));
  detail::AppendU32LE(&out, static_cast<uint32_t>(tensors.size()));
  const size_t sec_patch = out.size();
  detail::AppendU64LE(&out, 0);
  const size_t total_patch = out.size();
  detail::AppendU64LE(&out, 0);

  std::set<std::string> seen_keys;
  for (const MetadataEntry& m : metadata) {
    if (m.key.empty() || m.key.size() > std::numeric_limits<uint16_t>::max() ||
        !seen_keys.insert(m.key).second) {
      throw std::invalid_argument("Invalid or duplicate metadata key: " + m.key);
    }
    detail::AppendU16LE(&out, static_cast<uint16_t>(m.key.size()));
    out.insert(out.end(), m.key.begin(), m.key.end());
    detail::AppendU8(&out, static_cast<uint8_t>(m.type));
    if (m.type == MetaType::kString) {
      const auto& sv = std::get<std::string>(m.value);
      detail::AppendU32LE(&out, static_cast<uint32_t>(sv.size()));
      out.insert(out.end(), sv.begin(), sv.end());
    } else if (m.type == MetaType::kInt64) {
      int64_t iv = std::get<int64_t>(m.value);
      uint64_t bits = 0; std::memcpy(&bits, &iv, 8);
      detail::AppendU64LE(&out, bits);
    } else if (m.type == MetaType::kFloat64) {
      double dv = std::get<double>(m.value);
      uint64_t bits = 0; std::memcpy(&bits, &dv, 8);
      detail::AppendU64LE(&out, bits);
    } else {
      detail::AppendU8(&out, std::get<bool>(m.value) ? 1u : 0u);
    }
  }

  struct Patch { size_t off, sz, crc; };
  std::vector<Patch> patches;
  std::set<std::string> seen_names;
  for (const TensorInput& t : tensors) {
    uint64_t expected = 0;
    if (t.name.empty() || t.name.size() > std::numeric_limits<uint16_t>::max() ||
        !seen_names.insert(t.name).second || t.dims.size() > 32 ||
        !IsPowerOfTwo(t.alignment) || !CheckedShapeBytes(t.dtype, t.dims, &expected) ||
        expected != t.data.size()) {
      throw std::invalid_argument("Invalid tensor specification: " + t.name);
    }
    detail::AppendU16LE(&out, static_cast<uint16_t>(t.name.size()));
    out.insert(out.end(), t.name.begin(), t.name.end());
    detail::AppendU8(&out, static_cast<uint8_t>(t.dtype));
    detail::AppendU8(&out, static_cast<uint8_t>(t.dims.size()));
    for (uint64_t d : t.dims) detail::AppendU64LE(&out, d);
    detail::AppendU64LE(&out, t.alignment);
    const size_t off_p = out.size(); detail::AppendU64LE(&out, 0);
    const size_t sz_p = out.size(); detail::AppendU64LE(&out, 0);
    const size_t crc_p = out.size(); detail::AppendU32LE(&out, 0);
    patches.push_back({off_p, sz_p, crc_p});
  }

  uint64_t payload_sec_off = 0;
  if (!AlignUp(out.size(), tensors.empty() ? 1 : tensors[0].alignment, &payload_sec_off)) {
    throw std::overflow_error("Alignment overflow");
  }
  out.resize(static_cast<size_t>(payload_sec_off), 0u);
  detail::WriteU64LEAt(&out, sec_patch, payload_sec_off);

  for (size_t i = 0; i < tensors.size(); ++i) {
    uint64_t aligned_off = 0;
    if (!AlignUp(out.size(), tensors[i].alignment, &aligned_off)) {
      throw std::overflow_error("Alignment overflow");
    }
    out.resize(static_cast<size_t>(aligned_off), 0u);
    const uint32_t crc = ComputeCrc32(tensors[i].data.data(), tensors[i].data.size());
    out.insert(out.end(), tensors[i].data.begin(), tensors[i].data.end());
    detail::WriteU64LEAt(&out, patches[i].off, aligned_off);
    detail::WriteU64LEAt(&out, patches[i].sz, tensors[i].data.size());
    detail::WriteU32LEAt(&out, patches[i].crc, crc);
  }
  detail::WriteU64LEAt(&out, total_patch, out.size());
  return out;
}

inline AuditReport AuditArchiveBytes(const uint8_t* bytes, size_t len) {
  AuditReport rep;
  rep.file_bytes = len;
  auto err = [&](const std::string& c, const std::string& m) { rep.findings.push_back({c, m}); };

  if (len < 40) { err("TRUNCATED_HEADER", "File shorter than 40B header"); return rep; }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes)) {
    err("BAD_MAGIC", "Container magic mismatch");
    return rep;
  }

  detail::ByteReader rd(bytes + 8, len - 8);
  uint16_t vmaj = 0, vmin = 0;
  uint32_t flags = 0, meta_cnt = 0, tensor_cnt = 0;
  uint64_t payload_sec_off = 0, decl_bytes = 0;
  rd.ReadU16(&vmaj); rd.ReadU16(&vmin); rd.ReadU32(&flags);
  rd.ReadU32(&meta_cnt); rd.ReadU32(&tensor_cnt);
  rd.ReadU64(&payload_sec_off); rd.ReadU64(&decl_bytes);
  rep.payload_section_offset = payload_sec_off;

  if (vmaj != kVersionMajor || vmin > kVersionMinor || flags != 0) {
    err("UNSUPPORTED_VERSION", "Unsupported container version or flags");
  }
  if (decl_bytes != len) err("EOF_MISMATCH", "Header size disagrees with file size");

  std::set<std::string> mkeys;
  for (uint32_t i = 0; i < meta_cnt; ++i) {
    uint16_t klen = 0; uint8_t mtype = 0;
    MetadataEntry e;
    if (!rd.ReadU16(&klen) || klen == 0 || !rd.ReadString(klen, &e.key) || !rd.ReadU8(&mtype)) {
      err("CORRUPT_METADATA", "Truncated metadata record"); return rep;
    }
    if (!mkeys.insert(e.key).second) err("DUPLICATE_META_KEY", "Duplicate key " + e.key);
    e.type = static_cast<MetaType>(mtype);
    if (e.type == MetaType::kString) {
      uint32_t slen = 0; std::string sv;
      if (!rd.ReadU32(&slen) || !rd.ReadString(slen, &sv)) {
        err("CORRUPT_METADATA", "Bad string metadata"); return rep;
      }
      e.value = std::move(sv);
    } else if (e.type == MetaType::kInt64 || e.type == MetaType::kFloat64) {
      uint64_t bits = 0;
      if (!rd.ReadU64(&bits)) { err("CORRUPT_METADATA", "Truncated 64-bit metadata"); return rep; }
      if (e.type == MetaType::kInt64) { int64_t iv = 0; std::memcpy(&iv, &bits, 8); e.value = iv; }
      else { double dv = 0.0; std::memcpy(&dv, &bits, 8); e.value = dv; }
    } else if (e.type == MetaType::kBool) {
      uint8_t bv = 0;
      if (!rd.ReadU8(&bv) || bv > 1) { err("CORRUPT_METADATA", "Invalid bool"); return rep; }
      e.value = (bv == 1);
    } else {
      err("CORRUPT_METADATA", "Unknown metadata type"); return rep;
    }
    rep.metadata.push_back(std::move(e));
  }

  std::set<std::string> tnames;
  for (uint32_t i = 0; i < tensor_cnt; ++i) {
    uint16_t nlen = 0; uint8_t dt_raw = 0, rank = 0;
    TensorRecord tr;
    if (!rd.ReadU16(&nlen) || nlen == 0 || !rd.ReadString(nlen, &tr.name) ||
        !rd.ReadU8(&dt_raw) || !rd.ReadU8(&rank) || rank > 32) {
      err("CORRUPT_DIRECTORY", "Malformed tensor descriptor"); return rep;
    }
    if (!tnames.insert(tr.name).second) err("DUPLICATE_TENSOR", "Duplicate tensor " + tr.name);
    tr.dtype = static_cast<DType>(dt_raw);
    if (DTypeByteWidth(tr.dtype) == 0) err("INVALID_DTYPE", "Bad dtype for " + tr.name);
    tr.dims.resize(rank);
    for (uint8_t d = 0; d < rank; ++d) {
      if (!rd.ReadU64(&tr.dims[d])) { err("CORRUPT_DIRECTORY", "Truncated dims"); return rep; }
    }
    if (!rd.ReadU64(&tr.alignment) || !rd.ReadU64(&tr.payload_offset) ||
        !rd.ReadU64(&tr.payload_size) || !rd.ReadU32(&tr.crc32)) {
      err("CORRUPT_DIRECTORY", "Truncated descriptor fields"); return rep;
    }
    rep.tensors.push_back(std::move(tr));
  }

  const uint64_t dir_end = 8ULL + rd.Pos();
  if (payload_sec_off < dir_end || payload_sec_off > len) {
    err("BAD_PAYLOAD_SECTION", "Invalid payload section offset"); return rep;
  }
  auto verify_zero = [&](uint64_t b, uint64_t e, const std::string& ctx) {
    for (uint64_t p = b; p < e; ++p) {
      if (bytes[p] != 0u) { err("NON_ZERO_PADDING", "Non-zero padding byte (" + ctx + ")"); return; }
    }
  };
  verify_zero(dir_end, payload_sec_off, "header_pad");

  uint64_t cursor = payload_sec_off;
  for (const TensorRecord& tr : rep.tensors) {
    uint64_t exp_sz = 0;
    if (!CheckedShapeBytes(tr.dtype, tr.dims, &exp_sz)) err("SHAPE_OVERFLOW", "Overflow: " + tr.name);
    else if (exp_sz != tr.payload_size) err("SIZE_MISMATCH", "Size mismatch: " + tr.name);
    if (!IsPowerOfTwo(tr.alignment) || (tr.payload_offset % tr.alignment) != 0) {
      err("UNALIGNED_PAYLOAD", "Misaligned payload: " + tr.name);
    }
    if (tr.payload_offset < cursor) err("OVERLAPPING_SPAN", "Overlap at " + tr.name);
    else if (tr.payload_offset <= len) verify_zero(cursor, tr.payload_offset, tr.name);
    if (tr.payload_offset > len || tr.payload_size > len - tr.payload_offset) {
      err("OUT_OF_BOUNDS_PAYLOAD", "Out-of-bounds payload: " + tr.name);
    } else {
      if (ComputeCrc32(bytes + tr.payload_offset, static_cast<size_t>(tr.payload_size)) != tr.crc32) {
        err("CRC32_MISMATCH", "CRC32 mismatch: " + tr.name);
      }
      cursor = std::max(cursor, tr.payload_offset + tr.payload_size);
    }
  }
  if (cursor != len) err("EOF_MISMATCH", "Unindexed bytes at container tail");
  rep.valid = rep.findings.empty();
  return rep;
}

inline AuditReport AuditArchiveFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) throw std::runtime_error("Cannot open container: " + path);
  const std::streamsize sz = in.tellg();
  if (sz < 0) throw std::runtime_error("Cannot stat container: " + path);
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  in.seekg(0, std::ios::beg);
  if (sz > 0 && !in.read(reinterpret_cast<char*>(buf.data()), sz)) {
    throw std::runtime_error("Read failure on container: " + path);
  }
  return AuditArchiveBytes(buf.data(), buf.size());
}

}  // namespace tensor_container

#endif  // WAVE_TENSOR_CONTAINER_HPP_
