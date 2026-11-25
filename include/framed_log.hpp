#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flog {

namespace detail {

constexpr std::array<uint32_t, 256> make_crc32c_table() noexcept {
    std::array<uint32_t, 256> tbl{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int b = 0; b < 8; ++b) c = (c & 1u) ? ((c >> 1) ^ 0x82F63B78u) : (c >> 1);
        tbl[i] = c;
    }
    return tbl;
}

inline constexpr std::array<uint32_t, 256> kCrc32cTable = make_crc32c_table();

}  // namespace detail

inline uint32_t crc32c_update(uint32_t inv_crc, const void* data, size_t len) noexcept {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) inv_crc = detail::kCrc32cTable[(inv_crc ^ p[i]) & 0xFFu] ^ (inv_crc >> 8);
    return inv_crc;
}

inline uint32_t crc32c(const void* data, size_t len) noexcept {
    return ~crc32c_update(0xFFFFFFFFu, data, len);
}

struct SegmentHeader {
    char magic[4] = {'F', 'L', 'G', '1'};
    uint16_t version = 1;
    uint16_t flags = 0;
    uint64_t first_seq = 0;
    uint32_t segment_id = 0;
    uint32_t header_crc = 0;
};
static_assert(sizeof(SegmentHeader) == 24, "SegmentHeader must be 24 bytes");

struct FrameHeader {
    uint32_t frame_magic = 0x46524D31u;
    uint32_t payload_len = 0;
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
    uint16_t record_type = 0;
    uint16_t reserved = 0;
    uint32_t flags = 0;
};
static_assert(sizeof(FrameHeader) == 32, "FrameHeader must be 32 bytes");

struct IndexHeader {
    char magic[4] = {'F', 'I', 'D', 'X'};
    uint16_t version = 1;
    uint16_t stride = 16;
    uint32_t segment_id = 0;
    uint32_t header_crc = 0;
};
static_assert(sizeof(IndexHeader) == 16, "IndexHeader must be 16 bytes");

struct IndexEntry {
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
    uint32_t file_offset = 0;
    uint32_t entry_crc = 0;
};
static_assert(sizeof(IndexEntry) == 24, "IndexEntry must be 24 bytes");

struct LogRecord {
    uint64_t seq = 0;
    uint64_t timestamp_ns = 0;
    uint16_t record_type = 0;
    std::string payload;
};

struct WriterConfig {
    size_t max_segment_bytes = 64 * 1024;
    size_t max_segment_records = 4096;
    uint16_t index_stride = 8;
    uint64_t initial_seq = 1;
};

struct AuditReport {
    bool valid = true;
    size_t segments_checked = 0;
    uint64_t records_verified = 0;
    size_t corrupt_frames = 0;
    bool truncated_tail_detected = false;
    bool recovered_tail = false;
    uint64_t truncated_bytes_removed = 0;
    std::string error_detail;
};

inline std::filesystem::path segment_file_path(const std::filesystem::path& dir, uint32_t id) {
    std::ostringstream os;
    os << "segment_" << std::setw(6) << std::setfill('0') << id << ".flog";
    return dir / os.str();
}

inline std::filesystem::path index_file_path(const std::filesystem::path& dir, uint32_t id) {
    std::ostringstream os;
    os << "segment_" << std::setw(6) << std::setfill('0') << id << ".fidx";
    return dir / os.str();
}

inline uint32_t compute_frame_crc(const FrameHeader& fh, const void* payload) noexcept {
    uint32_t st = crc32c_update(0xFFFFFFFFu, &fh, sizeof(FrameHeader));
    if (fh.payload_len > 0 && payload != nullptr) st = crc32c_update(st, payload, fh.payload_len);
    return ~st;
}

class FramedLogWriter {
public:
    FramedLogWriter(std::filesystem::path log_dir, WriterConfig cfg = {})
        : dir_(std::move(log_dir)), cfg_(cfg), next_seq_(cfg.initial_seq) {
        if (cfg_.max_segment_bytes < sizeof(SegmentHeader) + sizeof(FrameHeader) + 8 ||
            cfg_.max_segment_records == 0 || cfg_.index_stride == 0) {
            throw std::invalid_argument("Invalid FramedLogWriter limits");
        }
        std::filesystem::create_directories(dir_);
    }

    ~FramedLogWriter() {
        if (seg_out_.is_open()) seg_out_.close();
        if (idx_out_.is_open()) idx_out_.close();
    }

    uint64_t append(uint64_t timestamp_ns, uint16_t record_type, const std::string& payload) {
        if (payload.size() > 16u * 1024u * 1024u) throw std::invalid_argument("Payload exceeds 16 MiB");
        size_t frame_bytes = sizeof(FrameHeader) + payload.size() + sizeof(uint32_t);
        if (seg_out_.is_open() && seg_records_ > 0 &&
            (seg_bytes_ + frame_bytes > cfg_.max_segment_bytes ||
             seg_records_ >= cfg_.max_segment_records)) {
            flush();
            seg_out_.close();
            idx_out_.close();
        }
        if (!seg_out_.is_open()) open_new_segment(next_seq_);

        uint32_t offset = static_cast<uint32_t>(seg_bytes_);
        FrameHeader fh{0x46524D31u, static_cast<uint32_t>(payload.size()), next_seq_,
                       timestamp_ns, record_type, 0u, 0u};
        uint32_t crc = compute_frame_crc(fh, payload.data());

        seg_out_.write(reinterpret_cast<const char*>(&fh), sizeof(FrameHeader));
        if (!payload.empty()) seg_out_.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        seg_out_.write(reinterpret_cast<const char*>(&crc), sizeof(uint32_t));
        if (!seg_out_) throw std::runtime_error("Write error on framed log segment");

        if (seg_records_ % cfg_.index_stride == 0) {
            IndexEntry ie{fh.seq, timestamp_ns, offset, 0u};
            ie.entry_crc = crc32c(&ie, offsetof(IndexEntry, entry_crc));
            idx_out_.write(reinterpret_cast<const char*>(&ie), sizeof(IndexEntry));
            if (!idx_out_) throw std::runtime_error("Write error on sparse index");
        }

        seg_bytes_ += frame_bytes;
        total_bytes_ += frame_bytes;
        ++seg_records_;
        return next_seq_++;
    }

    void flush() {
        if (seg_out_.is_open()) {
            seg_out_.flush();
            idx_out_.flush();
        }
    }

    uint32_t segments_created() const noexcept { return segments_created_; }
    uint64_t next_sequence() const noexcept { return next_seq_; }
    uint64_t total_bytes() const noexcept { return total_bytes_; }

private:
    void open_new_segment(uint64_t first_seq) {
        uint32_t id = segments_created_++;
        seg_out_.open(segment_file_path(dir_, id), std::ios::binary | std::ios::trunc);
        idx_out_.open(index_file_path(dir_, id), std::ios::binary | std::ios::trunc);
        if (!seg_out_ || !idx_out_) throw std::runtime_error("Cannot open segment files");

        SegmentHeader sh;
        sh.segment_id = id;
        sh.first_seq = first_seq;
        sh.header_crc = crc32c(&sh, offsetof(SegmentHeader, header_crc));
        seg_out_.write(reinterpret_cast<const char*>(&sh), sizeof(SegmentHeader));

        IndexHeader ih;
        ih.stride = cfg_.index_stride;
        ih.segment_id = id;
        ih.header_crc = crc32c(&ih, offsetof(IndexHeader, header_crc));
        idx_out_.write(reinterpret_cast<const char*>(&ih), sizeof(IndexHeader));

        seg_bytes_ = sizeof(SegmentHeader);
        total_bytes_ += sizeof(SegmentHeader);
        seg_records_ = 0;
    }

    std::filesystem::path dir_;
    WriterConfig cfg_;
    uint64_t next_seq_ = 1;
    uint32_t segments_created_ = 0;
    size_t seg_bytes_ = 0;
    size_t seg_records_ = 0;
    uint64_t total_bytes_ = 0;
    std::ofstream seg_out_;
    std::ofstream idx_out_;
};

inline std::vector<std::pair<uint32_t, std::filesystem::path>> list_segments(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        throw std::runtime_error("Log directory missing: " + dir.string());
    }
    std::vector<std::pair<uint32_t, std::filesystem::path>> segs;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".flog") continue;
        std::string stem = e.path().stem().string();
        if (stem.rfind("segment_", 0) == 0 && stem.size() == 14) {
            segs.emplace_back(static_cast<uint32_t>(std::stoul(stem.substr(8))), e.path());
        }
    }
    std::sort(segs.begin(), segs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return segs;
}

inline void rebuild_index_file(const std::filesystem::path& dir,
                               uint32_t id,
                               uint16_t stride,
                               const std::vector<IndexEntry>& entries) {
    std::ofstream out(index_file_path(dir, id), std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot rewrite index file");
    IndexHeader ih;
    ih.stride = stride;
    ih.segment_id = id;
    ih.header_crc = crc32c(&ih, offsetof(IndexHeader, header_crc));
    out.write(reinterpret_cast<const char*>(&ih), sizeof(IndexHeader));
    for (size_t i = 0; i < entries.size(); i += stride) {
        IndexEntry ie = entries[i];
        ie.entry_crc = crc32c(&ie, offsetof(IndexEntry, entry_crc));
        out.write(reinterpret_cast<const char*>(&ie), sizeof(IndexEntry));
    }
}

inline AuditReport verify_log_directory(const std::filesystem::path& dir, bool recover_tail = false) {
    AuditReport rep;
    auto segs = list_segments(dir);
    if (segs.empty()) {
        rep.valid = false;
        rep.error_detail = "No .flog segments found";
        return rep;
    }

    std::optional<uint64_t> expected_seq;
    for (size_t s_idx = 0; s_idx < segs.size(); ++s_idx) {
        bool is_last = (s_idx + 1 == segs.size());
        uint32_t id = segs[s_idx].first;
        const auto& path = segs[s_idx].second;
        uint64_t fsz = std::filesystem::file_size(path);
        std::ifstream in(path, std::ios::binary);
        SegmentHeader sh;
        if (!in || fsz < sizeof(SegmentHeader) ||
            !in.read(reinterpret_cast<char*>(&sh), sizeof(SegmentHeader)) ||
            std::memcmp(sh.magic, "FLG1", 4) != 0 || sh.version != 1 || sh.segment_id != id ||
            sh.header_crc != crc32c(&sh, offsetof(SegmentHeader, header_crc))) {
            rep.valid = false;
            rep.corrupt_frames++;
            rep.error_detail = "Segment header corruption";
            return rep;
        }
        if (!expected_seq.has_value()) expected_seq = sh.first_seq;
        else if (sh.first_seq != *expected_seq) {
            rep.valid = false;
            rep.corrupt_frames++;
            rep.error_detail = "Segment sequence discontinuity";
            return rep;
        }

        ++rep.segments_checked;
        uint64_t off = sizeof(SegmentHeader);
        std::vector<IndexEntry> entries;
        while (off < fsz) {
            uint64_t rem = fsz - off;
            if (rem < sizeof(FrameHeader) + sizeof(uint32_t)) {
                if (is_last) { rep.truncated_tail_detected = true; break; }
                rep.valid = false; rep.corrupt_frames++; rep.error_detail = "Mid-log truncated header";
                return rep;
            }
            FrameHeader fh;
            in.read(reinterpret_cast<char*>(&fh), sizeof(FrameHeader));
            if (fh.frame_magic != 0x46524D31u || fh.payload_len > 16u * 1024u * 1024u) {
                rep.valid = false; rep.corrupt_frames++; rep.error_detail = "Invalid frame magic/len";
                return rep;
            }
            uint64_t frame_sz = sizeof(FrameHeader) + fh.payload_len + sizeof(uint32_t);
            if (rem < frame_sz) {
                if (is_last) { rep.truncated_tail_detected = true; break; }
                rep.valid = false; rep.corrupt_frames++; rep.error_detail = "Mid-log truncated payload";
                return rep;
            }
            std::string payload(fh.payload_len, '\0');
            if (fh.payload_len > 0) in.read(&payload[0], fh.payload_len);
            uint32_t crc = 0;
            in.read(reinterpret_cast<char*>(&crc), sizeof(uint32_t));
            if (crc != compute_frame_crc(fh, payload.data()) || fh.seq != *expected_seq) {
                rep.valid = false; rep.corrupt_frames++; rep.error_detail = "Frame CRC32C or seq mismatch";
                return rep;
            }
            entries.push_back({fh.seq, fh.timestamp_ns, static_cast<uint32_t>(off), 0u});
            off += frame_sz;
            ++(*expected_seq);
            ++rep.records_verified;
        }
        in.close();

        if (rep.truncated_tail_detected && is_last) {
            if (!recover_tail) {
                rep.valid = false;
                rep.error_detail = "Unrecovered truncated tail frame";
            } else {
                uint16_t stride = 8;
                auto ip = index_file_path(dir, id);
                IndexHeader ih;
                if (std::ifstream idx_in(ip, std::ios::binary);
                    idx_in.read(reinterpret_cast<char*>(&ih), sizeof(IndexHeader)) && ih.stride > 0) {
                    stride = ih.stride;
                }
                std::filesystem::resize_file(path, off);
                rebuild_index_file(dir, id, stride, entries);
                rep.recovered_tail = true;
                rep.truncated_bytes_removed = fsz - off;
            }
        }
    }
    return rep;
}

inline uint32_t lookup_sparse_offset(const std::filesystem::path& dir, uint32_t id, uint64_t target_seq) {
    std::ifstream in(index_file_path(dir, id), std::ios::binary);
    IndexHeader ih;
    if (!in || !in.read(reinterpret_cast<char*>(&ih), sizeof(IndexHeader)) ||
        std::memcmp(ih.magic, "FIDX", 4) != 0 ||
        ih.header_crc != crc32c(&ih, offsetof(IndexHeader, header_crc))) {
        return sizeof(SegmentHeader);
    }
    uint32_t best = sizeof(SegmentHeader);
    IndexEntry ie;
    while (in.read(reinterpret_cast<char*>(&ie), sizeof(IndexEntry))) {
        if (ie.entry_crc != crc32c(&ie, offsetof(IndexEntry, entry_crc))) break;
        if (ie.seq <= target_seq) best = ie.file_offset;
        else break;
    }
    return best;
}

inline std::vector<LogRecord> query_records(const std::filesystem::path& dir,
                                            std::optional<uint64_t> seq_start,
                                            std::optional<uint64_t> seq_end,
                                            std::optional<uint16_t> type_filter) {
    auto segs = list_segments(dir);
    std::vector<SegmentHeader> hdrs(segs.size());
    for (size_t i = 0; i < segs.size(); ++i) {
        std::ifstream in(segs[i].second, std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(&hdrs[i]), sizeof(SegmentHeader)) ||
            std::memcmp(hdrs[i].magic, "FLG1", 4) != 0 ||
            hdrs[i].header_crc != crc32c(&hdrs[i], offsetof(SegmentHeader, header_crc))) {
            throw std::runtime_error("Corrupt segment header in query");
        }
    }

    std::vector<LogRecord> out;
    for (size_t i = 0; i < segs.size(); ++i) {
        if (seq_end.has_value() && hdrs[i].first_seq > *seq_end) break;
        if (seq_start.has_value() && i + 1 < segs.size() && hdrs[i + 1].first_seq <= *seq_start) continue;

        uint32_t start = seq_start.has_value() ? lookup_sparse_offset(dir, segs[i].first, *seq_start)
                                               : static_cast<uint32_t>(sizeof(SegmentHeader));
        uint64_t fsz = std::filesystem::file_size(segs[i].second);
        std::ifstream in(segs[i].second, std::ios::binary);
        in.seekg(start);

        for (uint64_t pos = start; pos < fsz;) {
            if (fsz - pos < sizeof(FrameHeader) + sizeof(uint32_t)) throw std::runtime_error("Truncated frame");
            FrameHeader fh;
            in.read(reinterpret_cast<char*>(&fh), sizeof(FrameHeader));
            uint64_t frame_sz = sizeof(FrameHeader) + fh.payload_len + sizeof(uint32_t);
            if (fh.frame_magic != 0x46524D31u || fsz - pos < frame_sz) throw std::runtime_error("Corrupt frame");
            std::string payload(fh.payload_len, '\0');
            if (fh.payload_len > 0) in.read(&payload[0], fh.payload_len);
            uint32_t crc = 0;
            in.read(reinterpret_cast<char*>(&crc), sizeof(uint32_t));
            if (crc != compute_frame_crc(fh, payload.data())) throw std::runtime_error("CRC32C failure");
            pos += frame_sz;
            if (seq_end.has_value() && fh.seq > *seq_end) return out;
            if ((!seq_start.has_value() || fh.seq >= *seq_start) &&
                (!type_filter.has_value() || fh.record_type == *type_filter)) {
                out.push_back({fh.seq, fh.timestamp_ns, fh.record_type, std::move(payload)});
            }
        }
    }
    return out;
}

}  // namespace flog
